#!/usr/bin/env python3
"""Generate one 1× GReaT/Qwen synthetic release for the unified benchmark.

Benchmark contract
------------------
- Reads one frozen processed real_train.csv and schema.json.
- Fits the LLM generator on real_train only.
- Binary target is used as the GReaT conditional column.
- Generates exactly 1× len(real_train) rows, class-conditioned to the
  real-training target prevalence.
- Preserves raw class-wise chunks so interrupted generation can resume.
- Applies a conservative row-validity filter after generation:
    * target must match the requested class;
    * numeric values must be finite and parseable;
    * categorical values must belong to frozen real-train support;
    * only catastrophic numeric outputs are rejected, defined a priori as
      abs(x) > factor * max(1, max(abs(real_train[col]))), factor=100 by default.
- Rejected rows are resampled; moderate train-support extrapolation is retained.
- NO numeric clipping, category projection, mode imputation, or value repair.
- Writes:
    synthetic_1x.csv
    generation_metadata.json
    generation_chunk_log.csv
    trainer_log_history.json
    saved_model/
    raw_chunks/

Validated reference configuration used in the thesis LLM workflow:
    be-great==0.0.14
    tabularisai/Qwen3-0.3B-distil
    epochs=10
    batch_size=1
    gradient_accumulation_steps=8
    FP32 master parameters + Trainer FP16 autocast
    sampling k=50
    max_length=512
    temperature=0.7

Use evaluate_release.py for utility, fidelity, DCR, MIA and AIA.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import platform
import random
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch


DEFAULT_MODEL = "tabularisai/Qwen3-0.3B-distil"


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_schema(path: Path) -> Dict[str, object]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    for key in ["target", "numeric", "categorical"]:
        if key not in schema:
            raise ValueError(f"schema.json missing '{key}'")
    return schema


def prepare_training_data(
    csv_path: str,
    schema: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    train = pd.read_csv(csv_path)
    expected = list(schema.get("columns", train.columns))
    if list(train.columns) != expected:
        raise ValueError("real_train columns differ from frozen schema order")
    if train.isna().any().any():
        raise ValueError(
            "Main LLM generator input contains missing values; "
            "frozen preprocessing contract was violated"
        )

    target = str(schema["target"])
    numeric = list(schema["numeric"])
    categorical = list(schema["categorical"])

    # Numeric fields remain numeric.
    for col in numeric:
        train[col] = pd.to_numeric(train[col], errors="raise")

    # Keep target integer 0/1 because GReaT conditioning in the validated
    # workflow used integer target values.
    train[target] = pd.to_numeric(train[target], errors="raise").astype(int)
    if sorted(train[target].unique().tolist()) != [0, 1]:
        raise ValueError("Binary target must contain exactly 0 and 1")

    # GReaT serializes table rows as text. Categorical values are converted
    # to strings with an exact restore map so codes such as 0/1 are not
    # accidentally treated as continuous quantities.
    restore: Dict[str, Dict[str, object]] = {}
    for col in categorical:
        values = train[col].dropna().unique().tolist()
        mapping = {str(v): v for v in values}
        if len(mapping) != len(values):
            raise ValueError(
                f"String conversion is not one-to-one for categorical column {col}"
            )
        restore[col] = mapping
        train[col] = train[col].map(str)

    return train.reset_index(drop=True), restore


def expected_chunk_sizes(total: int, chunk_size: int) -> List[int]:
    sizes: List[int] = []
    remaining = total
    while remaining > 0:
        current = min(chunk_size, remaining)
        sizes.append(current)
        remaining -= current
    return sizes


def catastrophic_numeric_thresholds(
    train: pd.DataFrame,
    schema: Dict[str, object],
    factor: float,
) -> Dict[str, float]:
    """Very loose scale guard used only for catastrophic numeric generations.

    The threshold is deliberately much wider than the observed train support.
    Moderate extrapolation remains untouched and is later reflected in fidelity
    metrics. This is not clipping: rows beyond the guard are rejected and the
    generator is asked for replacement rows.
    """
    if factor <= 1:
        raise ValueError("catastrophic_numeric_factor must be > 1")
    thresholds: Dict[str, float] = {}
    for col in schema["numeric"]:
        vals = pd.to_numeric(train[col], errors="raise").to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite values in frozen real_train column {col}")
        reference = max(1.0, float(np.max(np.abs(vals))))
        thresholds[str(col)] = factor * reference
    return thresholds


def filter_and_restore_rows(
    raw: pd.DataFrame,
    *,
    train: pd.DataFrame,
    schema: Dict[str, object],
    restore: Dict[str, Dict[str, object]],
    target_value: int,
    catastrophic_thresholds: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Counter, Counter]:
    """Validate rows without repairing them.

    Returns accepted rows, rejected raw rows, reason counts, and numeric-feature
    rejection counts. A row can have multiple rejection reasons; it is counted
    once in the rejected-row total but every triggered reason is logged.
    """
    expected = list(train.columns)
    target = str(schema["target"])
    numeric = [str(c) for c in schema["numeric"]]
    categorical = [str(c) for c in schema["categorical"]]

    missing = [c for c in expected if c not in raw.columns]
    extra = [c for c in raw.columns if c not in expected]
    if missing or extra:
        raise ValueError(f"Column mismatch. missing={missing}, extra={extra}")

    work = raw[expected].copy()
    invalid = pd.Series(False, index=work.index)
    reasons: Counter = Counter()
    feature_reasons: Counter = Counter()

    # Target: parseable integer and exactly equal to conditioned class.
    y_num = pd.to_numeric(work[target], errors="coerce")
    bad_parse = y_num.isna() | ~np.isfinite(y_num.to_numpy(float))
    if bad_parse.any():
        reasons["target_unparseable_or_nonfinite"] += int(bad_parse.sum())
        invalid |= bad_parse

    y_arr = y_num.fillna(-999999.0).to_numpy(float)
    bad_integer = ~np.isclose(y_arr, np.rint(y_arr), atol=1e-8)
    bad_integer = pd.Series(bad_integer, index=work.index) & ~bad_parse
    if bad_integer.any():
        reasons["target_noninteger"] += int(bad_integer.sum())
        invalid |= bad_integer

    y_int = np.rint(y_arr).astype(np.int64)
    bad_class = pd.Series(y_int != int(target_value), index=work.index) & ~bad_parse & ~bad_integer
    if bad_class.any():
        reasons["target_condition_mismatch"] += int(bad_class.sum())
        invalid |= bad_class

    # Numeric fields: parse, finite, and reject only catastrophic magnitudes.
    numeric_converted: Dict[str, pd.Series] = {}
    for col in numeric:
        vals = pd.to_numeric(work[col], errors="coerce")
        arr = vals.to_numpy(float)
        bad_num = vals.isna() | ~np.isfinite(arr)
        if bad_num.any():
            reasons["numeric_unparseable_or_nonfinite"] += int(bad_num.sum())
            feature_reasons[f"{col}:unparseable_or_nonfinite"] += int(bad_num.sum())
            invalid |= bad_num

        catastrophic = pd.Series(
            np.abs(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
            > float(catastrophic_thresholds[col]),
            index=work.index,
        ) & ~bad_num
        if catastrophic.any():
            reasons["catastrophic_numeric"] += int(catastrophic.sum())
            feature_reasons[f"{col}:catastrophic_numeric"] += int(catastrophic.sum())
            invalid |= catastrophic
        numeric_converted[col] = vals



    # changed: Categorical support is a schema-validity constraint, not a projection.
    # Keep raw string tokens here; restore values only after all invalid rows
    # have been removed. This avoids NaN-induced integer -> float upcasting.
    categorical_tokens: Dict[str, pd.Series] = {}

    for col in categorical:
        raw_tokens = work[col].astype(str)
        mapping = restore[col]

        bad_cat = ~raw_tokens.isin(set(mapping))

        if bad_cat.any():
            reasons["unseen_category"] += int(bad_cat.sum())
            feature_reasons[f"{col}:unseen_category"] += int(bad_cat.sum())
            invalid |= bad_cat

        categorical_tokens[col] = raw_tokens # eoc

    rejected = raw.loc[invalid].copy()
    if len(rejected):
        rejected.insert(0, "_original_index", rejected.index)

    accepted = work.loc[~invalid].copy()
    if len(accepted):
        accepted[target] = y_int[~invalid.to_numpy()]
        for col in numeric:
            accepted[col] = numeric_converted[col].loc[~invalid].to_numpy()
        # changed
        for col in categorical:
            restored = categorical_tokens[col].loc[~invalid].map(restore[col])

            if restored.isna().any():
                raise RuntimeError(
                    f"Internal error: accepted categorical column '{col}' "
                    "could not be restored exactly"
                )  

            accepted[col] = restored.to_numpy() # eoc

    if accepted.isna().any().any():
        raise RuntimeError("Internal error: accepted rows contain missing values")

    return accepted.reset_index(drop=True), rejected.reset_index(drop=True), reasons, feature_reasons


def validate_complete_chunk(
    raw: pd.DataFrame,
    *,
    train: pd.DataFrame,
    schema: Dict[str, object],
    restore: Dict[str, Dict[str, object]],
    target_value: int,
    expected_rows: int,
    catastrophic_thresholds: Dict[str, float],
) -> pd.DataFrame:
    if len(raw) != expected_rows:
        raise ValueError(f"Generated {len(raw)} rows, expected {expected_rows}")
    accepted, rejected, _, _ = filter_and_restore_rows(
        raw,
        train=train,
        schema=schema,
        restore=restore,
        target_value=target_value,
        catastrophic_thresholds=catastrophic_thresholds,
    )
    if len(rejected):
        raise ValueError(f"Existing/final chunk contains {len(rejected)} invalid rows")
    if len(accepted) != expected_rows:
        raise ValueError("Validated row count mismatch")
    return accepted


def validate_existing_chunk(
    path: Path,
    *,
    train: pd.DataFrame,
    schema: Dict[str, object],
    restore: Dict[str, Dict[str, object]],
    target_value: int,
    expected_rows: int,
    catastrophic_thresholds: Dict[str, float],
) -> bool:
    if not path.exists():
        return False
    try:
        chunk = pd.read_csv(path)
        validate_restored_chunk(  # changed from complete to restored
            chunk,
            train=train,
            schema=schema,
            restore=restore,
            target_value=target_value,
            expected_rows=expected_rows,
            catastrophic_thresholds=catastrophic_thresholds,
        )
        return True
    except Exception:
        return False



# new fix 12.08
def validate_restored_chunk(
    df,
    *,
    train,
    schema,
    restore,
    target_value,
    expected_rows,
    catastrophic_thresholds,
):
    if len(df) != expected_rows:
        raise ValueError(
            f"Generated {len(df)} rows, expected {expected_rows}"
        )

    expected = list(train.columns)

    if list(df.columns) != expected:
        raise ValueError("Restored chunk column mismatch")

    if df.isna().any().any():
        raise ValueError("Restored chunk contains missing values")

    work = df.copy()

    # target
    y = pd.to_numeric(work[schema["target"]], errors="coerce")
    if y.isna().any() or not (y.astype(int) == target_value).all():
        raise ValueError("Restored chunk target mismatch")

    # numeric
    for col in schema["numeric"]:
        vals = pd.to_numeric(work[col], errors="coerce")

        if vals.isna().any() or not np.isfinite(vals.to_numpy(float)).all():
            raise ValueError(f"Invalid restored numeric column: {col}")

        if (
            np.abs(vals.to_numpy(float))
            > catastrophic_thresholds[col]
        ).any():
            raise ValueError(
                f"Catastrophic restored numeric values: {col}"
            )

    # categorical: compare against ORIGINAL values,
    # not GReaT string tokens
    for col in schema["categorical"]:
        allowed = list(restore[col].values())

        if not work[col].isin(allowed).all():
            raise ValueError(
                f"Restored categorical outside support: {col}"
            )

    return work.reset_index(drop=True)




    


def generate_class_chunks(
    *,
    model,
    train: pd.DataFrame,
    schema: Dict[str, object],
    restore: Dict[str, Dict[str, object]],
    target_value: int,
    total_rows: int,
    chunk_size: int,
    sampling_batch_size: int,
    max_length: int,
    temperature: float,
    random_seed: int,
    chunks_root: Path,
    attempts_root: Path,
    rejected_root: Path,
    log_rows: List[Dict[str, object]],
    rejection_rows: List[pd.DataFrame],
    rejection_reason_counts: Counter,
    rejection_feature_counts: Counter,
    catastrophic_thresholds: Dict[str, float],
    max_resample_rounds: int,
) -> List[Path]:
    target = str(schema["target"])
    class_dir = chunks_root / f"target_{target_value}"
    attempt_dir = attempts_root / f"target_{target_value}"
    reject_dir = rejected_root / f"target_{target_value}"
    class_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    sizes = expected_chunk_sizes(total_rows, chunk_size)

    for chunk_index, expected_rows in enumerate(sizes):
        chunk_path = class_dir / f"chunk_{chunk_index:04d}.csv"
        paths.append(chunk_path)

        if validate_existing_chunk(
            chunk_path,
            train=train,
            schema=schema,
            restore=restore,
            target_value=target_value,
            expected_rows=expected_rows,
            catastrophic_thresholds=catastrophic_thresholds,
        ):
            log_rows.append({
                "target": target_value,
                "chunk_index": chunk_index,
                "attempt": -1,
                "requested_rows": 0,
                "accepted_rows": expected_rows,
                "rejected_rows": 0,
                "status": "REUSED",
                "seconds": 0.0,
                "peak_gpu_memory_gb": np.nan,
                "path": str(chunk_path),
            })
            print(f"[REUSE] target={target_value} chunk={chunk_index} rows={expected_rows}", flush=True)
            continue

        accepted_parts: List[pd.DataFrame] = []
        accepted_count = 0
        attempt = 0

        while accepted_count < expected_rows:
            if attempt >= max_resample_rounds:
                raise RuntimeError(
                    f"Exceeded max_resample_rounds={max_resample_rounds} for "
                    f"target={target_value}, chunk={chunk_index}; "
                    f"accepted {accepted_count}/{expected_rows}."
                )

            needed = expected_rows - accepted_count
            seed = random_seed + target_value * 1_000_000 + chunk_index * 10_000 + attempt
            set_seeds(seed)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            started = time.time()
            raw = model.sample(
                n_samples=needed,
                start_col=target,
                start_col_dist={int(target_value): 1.0},
                temperature=temperature,
                k=min(sampling_batch_size, needed),
                max_length=max_length,
                drop_nan=False,
                device="cuda",
                guided_sampling=False,
                random_feature_order=True,
                conditions=None,
            )
            elapsed = time.time() - started

            attempt_path = attempt_dir / f"chunk_{chunk_index:04d}_attempt_{attempt:03d}.csv"
            raw.to_csv(attempt_path, index=False)

            accepted, rejected, reasons, feature_reasons = filter_and_restore_rows(
                raw,
                train=train,
                schema=schema,
                restore=restore,
                target_value=target_value,
                catastrophic_thresholds=catastrophic_thresholds,
            )

            if len(rejected):
                rejected.insert(0, "_target_requested", target_value)
                rejected.insert(1, "_chunk_index", chunk_index)
                rejected.insert(2, "_attempt", attempt)
                rejected_path = reject_dir / f"chunk_{chunk_index:04d}_attempt_{attempt:03d}.csv"
                rejected.to_csv(rejected_path, index=False)
                rejection_rows.append(rejected)

            rejection_reason_counts.update(reasons)
            rejection_feature_counts.update(feature_reasons)

            accepted_parts.append(accepted)
            accepted_count += len(accepted)

            row = {
                "target": target_value,
                "chunk_index": chunk_index,
                "attempt": attempt,
                "requested_rows": needed,
                "accepted_rows": len(accepted),
                "rejected_rows": len(rejected),
                "status": "OK" if len(rejected) == 0 else "RESAMPLED",
                "seconds": elapsed,
                "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "path": str(attempt_path),
            }
            log_rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

            attempt += 1
            gc.collect()
            torch.cuda.empty_cache()

        validated = pd.concat(accepted_parts, ignore_index=True).iloc[:expected_rows].copy()
        validated = validate_restored_chunk(  # fix from validated = validate_complete_chunk( to validated = validate_restored_chunk(
            validated,
            train=train,
            schema=schema,
            restore=restore,
            target_value=target_value,
            expected_rows=expected_rows,
            catastrophic_thresholds=catastrophic_thresholds,
        )
        validated.to_csv(chunk_path, index=False)

    return paths

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--real_train", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--sampling_batch_size", type=int, default=50)
    p.add_argument("--chunk_size", type=int, default=250)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--catastrophic_numeric_factor", type=float, default=100.0)
    p.add_argument("--max_resample_rounds", type=int, default=20)

    p.add_argument("--reuse_saved_model", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    schema = load_schema(Path(args.schema))
    train, restore = prepare_training_data(args.real_train, schema)
    target = str(schema["target"])
    n_synthetic = len(train)
    catastrophic_thresholds = catastrophic_numeric_thresholds(
        train, schema, args.catastrophic_numeric_factor
    )

    positive_rate = float(train[target].mean())
    positive_n = int(round(n_synthetic * positive_rate))
    positive_n = max(1, min(positive_n, n_synthetic - 1))
    negative_n = n_synthetic - positive_n

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir = outdir / "saved_model"
    trainer_dir = outdir / "trainer"
    chunks_root = outdir / "raw_chunks"
    attempts_root = outdir / "raw_attempts"
    rejected_root = outdir / "rejected_rows"

    if args.dry_run:
        print(
            "DRY RUN PASS: "
            f"GReaT/Qwen on {len(train)} rows, "
            f"features={len(train.columns)-1}, "
            f"target={target}, "
            f"positive_rate={positive_rate:.6f}, "
            f"release={n_synthetic}"
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the unified GReaT/Qwen benchmark. "
            "Run this script in the validated LLM environment on a GPU node."
        )

    try:
        from be_great import GReaT
    except ImportError as exc:
        raise ImportError(
            'be-great is not importable. The validated thesis environment used '
            '"be-great==0.0.14".'
        ) from exc

    environment = {
        "method": "LLM",
        "method_key": "great_qwen",
        "model": args.model_name_or_path,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "be-great": package_version("be-great"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
        ),
        "seed": args.seed,
        "n_real_train": len(train),
        "n_synthetic": n_synthetic,
        "synthetic_size": "1x",
        "feature_count": len(train.columns) - 1,
        "target": target,
        "target_prevalence_real": positive_rate,
        "negative_requested": negative_n,
        "positive_requested": positive_n,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "sampling_batch_size": args.sampling_batch_size,
        "chunk_size": args.chunk_size,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "fp16": bool(args.fp16),
        "catastrophic_numeric_factor": args.catastrophic_numeric_factor,
        "catastrophic_numeric_thresholds": catastrophic_thresholds,
        "max_resample_rounds": args.max_resample_rounds,
        "postprocessing": (
            "row rejection/resampling for structural/type invalidity and only "
            "catastrophic numeric outputs beyond the predeclared loose scale guard; "
            "moderate support extrapolation retained; no numeric clipping, category "
            "projection, mode imputation or value repair"
        ),
    }

    (outdir / "run_configuration.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    training_seconds = 0.0

    if args.reuse_saved_model:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"--reuse_saved_model requested but {model_dir} does not exist"
            )
        model = GReaT.load_from_dir(str(model_dir))
        model.model.float()
        print("[REUSE] Saved GReaT model loaded; training skipped.", flush=True)

    else:
        model = GReaT(
            llm=args.model_name_or_path,
            experiment_dir=str(trainer_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            fp16=args.fp16,
            bf16=False,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=args.dataloader_num_workers,
            seed=args.seed,
            data_seed=args.seed,
        )

        checkpoint_dtype = str(next(model.model.parameters()).dtype)

        # Validated workaround used in the previous Qwen/Turing experiment:
        # keep FP32 master parameters while Trainer uses FP16 autocast.
        if args.fp16:
            model.model.float()
            if hasattr(model.model.config, "dtype"):
                model.model.config.dtype = "float32"
            if hasattr(model.model.config, "torch_dtype"):
                model.model.config.torch_dtype = torch.float32

        precision = {
            "checkpoint_parameter_dtype": checkpoint_dtype,
            "master_parameter_dtype": str(next(model.model.parameters()).dtype),
            "trainer_fp16": bool(args.fp16),
            "trainer_bf16": False,
        }
        (outdir / "precision_report.json").write_text(
            json.dumps(precision, indent=2),
            encoding="utf-8",
        )

        fit_parameters = inspect.signature(model.fit).parameters
        fit_kwargs = {}
        if "conditional_col" in fit_parameters:
            fit_kwargs["conditional_col"] = target
        if "random_conditional_col" in fit_parameters:
            fit_kwargs["random_conditional_col"] = False

        started = time.time()
        trainer = model.fit(train, **fit_kwargs)
        training_seconds = time.time() - started

        log_history = []
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
            log_history = trainer.state.log_history
        (outdir / "trainer_log_history.json").write_text(
            json.dumps(log_history, indent=2, default=str),
            encoding="utf-8",
        )

        if hasattr(trainer, "save_state"):
            trainer.save_state()

        model.save(str(model_dir))
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        model = GReaT.load_from_dir(str(model_dir))
        model.model.float()

    generation_log: List[Dict[str, object]] = []
    rejection_rows: List[pd.DataFrame] = []
    rejection_reason_counts: Counter = Counter()
    rejection_feature_counts: Counter = Counter()

    negative_paths = generate_class_chunks(
        model=model,
        train=train,
        schema=schema,
        restore=restore,
        target_value=0,
        total_rows=negative_n,
        chunk_size=args.chunk_size,
        sampling_batch_size=args.sampling_batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        random_seed=args.seed,
        chunks_root=chunks_root,
        attempts_root=attempts_root,
        rejected_root=rejected_root,
        log_rows=generation_log,
        rejection_rows=rejection_rows,
        rejection_reason_counts=rejection_reason_counts,
        rejection_feature_counts=rejection_feature_counts,
        catastrophic_thresholds=catastrophic_thresholds,
        max_resample_rounds=args.max_resample_rounds,
    )

    positive_paths = generate_class_chunks(
        model=model,
        train=train,
        schema=schema,
        restore=restore,
        target_value=1,
        total_rows=positive_n,
        chunk_size=args.chunk_size,
        sampling_batch_size=args.sampling_batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        random_seed=args.seed,
        chunks_root=chunks_root,
        attempts_root=attempts_root,
        rejected_root=rejected_root,
        log_rows=generation_log,
        rejection_rows=rejection_rows,
        rejection_reason_counts=rejection_reason_counts,
        rejection_feature_counts=rejection_feature_counts,
        catastrophic_thresholds=catastrophic_thresholds,
        max_resample_rounds=args.max_resample_rounds,
    )

    all_paths = negative_paths + positive_paths
    frames = [pd.read_csv(path) for path in all_paths]
    synthetic = (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

    # Final strict validation and dtype restoration.
    # Validate each class separately because target conditioning is part of
    # the release contract.
    pieces = []
    for target_value in [0, 1]:
        part = synthetic[
            pd.to_numeric(synthetic[target], errors="raise").astype(int)
            == target_value
        ].copy()
        expected_n = negative_n if target_value == 0 else positive_n
        part = validate_restored_chunk( # changed from complete to restored
            part,
            train=train,
            schema=schema,
            restore=restore,
            target_value=target_value,
            expected_rows=expected_n,
            catastrophic_thresholds=catastrophic_thresholds,
        )
        pieces.append(part)

    synthetic = (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

    if len(synthetic) != n_synthetic:
        raise RuntimeError(
            f"Final release has {len(synthetic)} rows; expected {n_synthetic}"
        )

    synthetic.to_csv(outdir / "synthetic_1x.csv", index=False)

    log_df = pd.DataFrame(generation_log)
    log_df.to_csv(outdir / "generation_chunk_log.csv", index=False)

    if rejection_rows:
        rejected_df = pd.concat(rejection_rows, ignore_index=True)
    else:
        rejected_df = pd.DataFrame()
    rejected_df.to_csv(outdir / "generation_rejected_rows.csv", index=False)

    attempted_rows = int(log_df.get("requested_rows", pd.Series(dtype=float)).fillna(0).sum())
    rejected_count = int(log_df.get("rejected_rows", pd.Series(dtype=float)).fillna(0).sum())
    rejection_rate = (rejected_count / attempted_rows) if attempted_rows else 0.0

    numeric_support_violations = {}
    for col in schema["numeric"]:
        r = pd.to_numeric(train[col], errors="raise").to_numpy(float)
        s = pd.to_numeric(synthetic[col], errors="raise").to_numpy(float)
        numeric_support_violations[col] = int(
            np.sum((s < np.min(r)) | (s > np.max(r)))
        )

    final_metadata = {
        **environment,
        "training_seconds": training_seconds,
        "reused_saved_model": bool(args.reuse_saved_model),
        "completed_chunks": len(all_paths),
        "target_prevalence_synthetic": float(
            pd.to_numeric(synthetic[target], errors="raise").mean()
        ),
        "generation_seconds_this_run": (
            float(log_df["seconds"].sum()) if len(log_df) else 0.0
        ),
        "numeric_support_violation_counts": numeric_support_violations,
        "generated_rows_attempted_this_run": attempted_rows,
        "n_rejected_rows_this_run": rejected_count,
        "rejection_rate_this_run": rejection_rate,
        "rejection_reason_counts": dict(rejection_reason_counts),
        "rejection_reason_by_feature": dict(rejection_feature_counts),
        "output_csv": str(outdir / "synthetic_1x.csv"),
        "note": (
            "Frozen real_train only; 1x release; target-conditioned GReaT/Qwen. "
            "Moderate support extrapolation is retained. Only structurally invalid "
            "or catastrophic numeric rows are rejected and resampled; values are "
            "never clipped or repaired."
        ),
    }

    (outdir / "generation_metadata.json").write_text(
        json.dumps(final_metadata, indent=2),
        encoding="utf-8",
    )

    print("\nGReaT/Qwen release complete:")
    print(json.dumps(final_metadata, indent=2))


if __name__ == "__main__":
    main()
