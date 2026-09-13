#!/usr/bin/env python3
"""
Controlled end-to-end smoke test for GReaT on the prepared Diabetes 130-US
reduced representation.

Purpose
-------
This is not a thesis experiment. It checks that the installed be-great stack can:

1. load the Qwen3-0.3B-distil model;
2. fine-tune for one short epoch on a small subset;
3. save and reload the model;
4. generate a small target-conditioned synthetic release;
5. produce structurally valid tabular rows without dominant exact copying.

The real train/validation/test partitions are never recreated here. This script
uses only the frozen reduced training representation produced earlier.

Recommended first run on an RTX 2080 Ti (10.6 GB).

Version 2 explicitly converts the BF16 Qwen checkpoint to FP32 master
parameters before enabling Trainer FP16 autocast. This avoids the error:
"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for
"BFloat16".

Recommended command:
    python smoke_test_diabetes130_great.py \
      --train_csv results/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
      --out_dir results/diabetes130/great_smoke \
      --train_rows 2000 \
      --epochs 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 8 \
      --n_synthetic 100 \
      --max_length 512 \
      --fp16

The script uses public API introspection to support both:
- current `conditions={...}` sampling; and
- the earlier `start_col` / `start_col_dist` API.

Important
---------
Near-zero duplicate rate in this small smoke test is only a technical check and
is not a privacy conclusion.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


TARGET = "readmitted_30d"
MODEL_NAME = "tabularisai/Qwen3-0.3B-distil"
MISSING_TOKEN = "__MISSING__"
RARE_TOKEN = "__RARE__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/great_smoke"),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=MODEL_NAME,
    )
    parser.add_argument("--train_rows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument("--n_synthetic", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sampling_batch_size", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--guided_sampling",
        action="store_true",
        help=(
            "Use only as a fallback. Guided sampling may be substantially "
            "slower for 43 columns."
        ),
    )

    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_training_data(
    path: str,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET}' was not found in {path}."
        )

    if sorted(pd.to_numeric(df[TARGET], errors="raise").unique().tolist()) != [0, 1]:
        raise ValueError("The target must contain exactly the values 0 and 1.")

    # Keep the target as the final column and pass it explicitly as the
    # conditional column during fitting.
    feature_columns = [
        col for col in df.columns if col != TARGET
    ]
    df = df[feature_columns + [TARGET]].copy()
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise",
    ).astype(int)

    if n_rows >= len(df):
        sample = df.sample(
            frac=1.0,
            random_state=seed,
        )
    else:
        positive = df[df[TARGET] == 1]
        negative = df[df[TARGET] == 0]

        positive_n = int(round(n_rows * df[TARGET].mean()))
        positive_n = max(1, min(positive_n, len(positive)))
        negative_n = n_rows - positive_n

        sample = pd.concat(
            [
                positive.sample(
                    n=positive_n,
                    random_state=seed,
                ),
                negative.sample(
                    n=negative_n,
                    random_state=seed + 1,
                ),
            ],
            ignore_index=True,
        ).sample(
            frac=1.0,
            random_state=seed + 2,
        )

    return sample.reset_index(drop=True)


def normalised_row_strings(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.Series:
    normalised = df[list(columns)].copy()

    for col in columns:
        if col == TARGET:
            normalised[col] = pd.to_numeric(
                normalised[col],
                errors="coerce",
            ).round().astype("Int64").astype("string")
        elif pd.api.types.is_numeric_dtype(normalised[col]):
            normalised[col] = pd.to_numeric(
                normalised[col],
                errors="coerce",
            ).round(8).astype("string")
        else:
            normalised[col] = (
                normalised[col]
                .astype("string")
                .fillna(MISSING_TOKEN)
                .str.strip()
            )

    return normalised.astype("string").agg(
        "\x1f".join,
        axis=1,
    )


def generate_conditioned(
    model,
    n_samples: int,
    target_positive_rate: float,
    max_length: int,
    temperature: float,
    sampling_batch_size: int,
    guided_sampling: bool,
) -> Tuple[pd.DataFrame, str]:
    sample_signature = inspect.signature(model.sample)
    parameters = sample_signature.parameters

    positive_n = int(round(n_samples * target_positive_rate))
    positive_n = max(1, min(positive_n, n_samples - 1))
    negative_n = n_samples - positive_n

    common_kwargs = {
        "max_length": max_length,
        "temperature": temperature,
        "device": "cuda",
    }

    if "conditions" in parameters:
        # Current public API. Generate each target class separately so that the
        # strongly imbalanced target is nevertheless represented in the smoke
        # release.
        negative = model.sample(
            n_samples=negative_n,
            conditions={TARGET: "== 0"},
            **common_kwargs,
        )
        positive = model.sample(
            n_samples=positive_n,
            conditions={TARGET: "== 1"},
            **common_kwargs,
        )
        generated = pd.concat(
            [negative, positive],
            ignore_index=True,
        )
        api_used = "conditions"

    elif "start_col" in parameters:
        # Earlier public API. The target distribution is supplied directly.
        generated = model.sample(
            n_samples=n_samples,
            start_col=TARGET,
            start_col_dist={
                0: negative_n / n_samples,
                1: positive_n / n_samples,
            },
            k=sampling_batch_size,
            guided_sampling=guided_sampling,
            random_feature_order=True,
            **common_kwargs,
        )
        api_used = "start_col/start_col_dist"

    else:
        raise RuntimeError(
            "The installed GReaT.sample signature is unsupported:\n"
            f"{sample_signature}"
        )

    generated = generated.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    return generated, api_used


def structural_validation(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    expected_columns = real_train.columns.tolist()
    missing_columns = [
        col for col in expected_columns if col not in synthetic.columns
    ]
    extra_columns = [
        col for col in synthetic.columns if col not in expected_columns
    ]

    per_column_rows: List[Dict[str, object]] = []

    for col in expected_columns:
        if col not in synthetic.columns:
            per_column_rows.append(
                {
                    "feature": col,
                    "kind": "missing_column",
                    "valid_n": 0,
                    "invalid_n": len(synthetic),
                    "invalid_pct": 100.0,
                    "unseen_level_n": np.nan,
                    "unseen_level_pct": np.nan,
                }
            )
            continue

        if col == TARGET:
            parsed = pd.to_numeric(
                synthetic[col],
                errors="coerce",
            )
            valid = parsed.isin([0, 1])
            unseen = pd.Series(False, index=synthetic.index)
            kind = "binary_target"

        elif pd.api.types.is_numeric_dtype(real_train[col]):
            parsed = pd.to_numeric(
                synthetic[col],
                errors="coerce",
            )
            valid = parsed.notna()
            unseen = pd.Series(False, index=synthetic.index)
            kind = "numeric"

        else:
            generated_tokens = (
                synthetic[col]
                .astype("string")
                .fillna(MISSING_TOKEN)
                .str.strip()
            )
            real_levels = set(
                real_train[col]
                .astype("string")
                .fillna(MISSING_TOKEN)
                .str.strip()
                .tolist()
            )
            valid = generated_tokens.ne("")
            unseen = ~generated_tokens.isin(real_levels)
            kind = "categorical"

        per_column_rows.append(
            {
                "feature": col,
                "kind": kind,
                "valid_n": int(valid.sum()),
                "invalid_n": int((~valid).sum()),
                "invalid_pct": float((~valid).mean() * 100),
                "unseen_level_n": (
                    int(unseen.sum())
                    if kind == "categorical"
                    else 0
                ),
                "unseen_level_pct": (
                    float(unseen.mean() * 100)
                    if kind == "categorical"
                    else 0.0
                ),
            }
        )

    per_column = pd.DataFrame(per_column_rows)

    comparable = (
        synthetic[expected_columns].copy()
        if not missing_columns
        else pd.DataFrame()
    )

    exact_duplicate_rate = np.nan
    internal_duplicate_rate = np.nan

    if not comparable.empty:
        train_keys = set(
            normalised_row_strings(
                real_train,
                expected_columns,
            )
        )
        synthetic_keys = normalised_row_strings(
            comparable,
            expected_columns,
        )
        exact_duplicate_rate = float(
            synthetic_keys.isin(train_keys).mean()
        )
        internal_duplicate_rate = float(
            synthetic_keys.duplicated(keep=False).mean()
        )

    target_numeric = (
        pd.to_numeric(
            synthetic[TARGET],
            errors="coerce",
        )
        if TARGET in synthetic.columns
        else pd.Series(dtype=float)
    )

    summary = {
        "expected_columns": len(expected_columns),
        "returned_columns": len(synthetic.columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "n_synthetic_requested": None,
        "n_synthetic_returned": len(synthetic),
        "target_parseable_rate": (
            float(target_numeric.notna().mean())
            if len(target_numeric)
            else 0.0
        ),
        "target_binary_valid_rate": (
            float(target_numeric.isin([0, 1]).mean())
            if len(target_numeric)
            else 0.0
        ),
        "synthetic_positive_rate": (
            float(target_numeric.mean())
            if len(target_numeric)
            and target_numeric.isin([0, 1]).all()
            else None
        ),
        "exact_duplicate_rate_vs_smoke_train": exact_duplicate_rate,
        "internal_duplicate_rate": internal_duplicate_rate,
        "mean_column_invalid_pct": float(
            per_column["invalid_pct"].mean()
        ),
        "maximum_column_invalid_pct": float(
            per_column["invalid_pct"].max()
        ),
        "mean_categorical_unseen_level_pct": float(
            per_column.loc[
                per_column["kind"] == "categorical",
                "unseen_level_pct",
            ].mean()
        ),
        "maximum_categorical_unseen_level_pct": float(
            per_column.loc[
                per_column["kind"] == "categorical",
                "unseen_level_pct",
            ].max()
        ),
    }

    return summary, per_column


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.random_seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This smoke test is configured for GPU."
        )

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "be-great": package_version("be-great"),
        "peft": package_version("peft"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            3,
        ),
        "requested_fp16": args.fp16,
        "requested_train_rows": args.train_rows,
        "requested_epochs": args.epochs,
        "requested_batch_size": args.batch_size,
        "requested_gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "effective_batch_size": (
            args.batch_size
            * args.gradient_accumulation_steps
        ),
    }
    (
        out_dir / "environment.json"
    ).write_text(json.dumps(environment, indent=2))

    try:
        from be_great import GReaT
    except ImportError as exc:
        raise ImportError(
            "be-great is not importable. Install it first, then rerun:\n"
            'python -m pip install "be-great==0.0.14"'
        ) from exc

    train = prepare_training_data(
        args.train_csv,
        args.train_rows,
        args.random_seed,
    )
    train.to_csv(
        out_dir / "smoke_training_subset.csv",
        index=False,
    )

    experiment_dir = out_dir / "trainer"
    model_dir = out_dir / "saved_model"

    constructor_signature = inspect.signature(GReaT.__init__)
    fit_signature = inspect.signature(GReaT.fit)
    sample_signature = inspect.signature(GReaT.sample)

    api_report = {
        "constructor": str(constructor_signature),
        "fit": str(fit_signature),
        "sample": str(sample_signature),
    }
    (
        out_dir / "great_api_signatures.json"
    ).write_text(json.dumps(api_report, indent=2))

    model_kwargs = {
        "llm": args.model_name_or_path,
        "experiment_dir": str(experiment_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        # Qwen3-0.3B-distil is stored as BF16. On a Turing GPU this must
        # not be combined directly with the FP16 GradScaler. The model
        # is explicitly converted to FP32 master weights below, while
        # Trainer uses FP16 autocast for forward/backward computation.
        "fp16": args.fp16,
        "bf16": False,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "logging_steps": 20,
        "save_strategy": "no",
        "report_to": [],
        "dataloader_num_workers": (
            args.dataloader_num_workers
        ),
        "seed": args.random_seed,
        "data_seed": args.random_seed,
    }

    start_time = time.time()

    try:
        model = GReaT(**model_kwargs)

        # Transformers v5 may load this checkpoint using the BF16 dtype
        # declared in its config. FP16 GradScaler cannot unscale BF16
        # gradients on the RTX 2080 Ti. Keep FP32 master parameters and
        # let Trainer perform ordinary FP16 mixed-precision autocast.
        loaded_parameter_dtype = str(
            next(model.model.parameters()).dtype
        )
        if args.fp16:
            model.model.float()
            if hasattr(model.model.config, "dtype"):
                model.model.config.dtype = "float32"
            if hasattr(model.model.config, "torch_dtype"):
                model.model.config.torch_dtype = torch.float32

        training_parameter_dtype = str(
            next(model.model.parameters()).dtype
        )
        precision_report = {
            "checkpoint_parameter_dtype": loaded_parameter_dtype,
            "training_master_parameter_dtype": training_parameter_dtype,
            "trainer_fp16": bool(args.fp16),
            "trainer_bf16": False,
            "reason": (
                "Use FP32 master weights with FP16 autocast because the "
                "checkpoint is stored as BF16 and the RTX 2080 Ti is a "
                "Turing GPU."
            ),
        }
        (
            out_dir / "precision_report.json"
        ).write_text(json.dumps(precision_report, indent=2))

        print("\nPrecision configuration:")
        print(json.dumps(precision_report, indent=2))

        fit_parameters = inspect.signature(
            model.fit
        ).parameters
        fit_kwargs = {}

        if "conditional_col" in fit_parameters:
            fit_kwargs["conditional_col"] = TARGET

        if "random_conditional_col" in fit_parameters:
            fit_kwargs["random_conditional_col"] = False

        trainer = model.fit(
            train,
            **fit_kwargs,
        )

    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA out of memory during smoke training. Rerun with "
            "--batch_size 1 and a larger "
            "--gradient_accumulation_steps, or reduce --train_rows. "
            "Do not change the frozen real-data split."
        ) from exc

    training_seconds = time.time() - start_time

    if hasattr(trainer, "save_state"):
        trainer.save_state()

    model.save(str(model_dir))

    # Explicit save/load check.
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = GReaT.load_from_dir(str(model_dir))

    generation_start = time.time()

    synthetic, sampling_api = generate_conditioned(
        reloaded,
        n_samples=args.n_synthetic,
        target_positive_rate=float(train[TARGET].mean()),
        max_length=args.max_length,
        temperature=args.temperature,
        sampling_batch_size=args.sampling_batch_size,
        guided_sampling=args.guided_sampling,
    )
    generation_seconds = time.time() - generation_start

    synthetic.to_csv(
        out_dir / "synthetic_smoke.csv",
        index=False,
    )

    validation, per_column = structural_validation(
        train,
        synthetic,
    )
    validation["n_synthetic_requested"] = args.n_synthetic
    validation["sampling_api"] = sampling_api
    validation["training_seconds"] = training_seconds
    validation["generation_seconds"] = generation_seconds
    validation["generation_seconds_per_row"] = (
        generation_seconds / max(1, len(synthetic))
    )
    validation["peak_gpu_memory_gb"] = round(
        torch.cuda.max_memory_allocated() / 1024**3,
        3,
    )

    # Smoke-test criteria are deliberately technical and permissive.
    validation["pass_row_count"] = (
        len(synthetic) == args.n_synthetic
    )
    validation["pass_column_set"] = (
        not validation["missing_columns"]
        and not validation["extra_columns"]
    )
    validation["pass_binary_target"] = (
        validation["target_binary_valid_rate"] >= 0.95
    )
    validation["pass_mean_column_validity"] = (
        validation["mean_column_invalid_pct"] <= 5.0
    )
    validation["pass_exact_copy_not_dominant"] = (
        pd.notna(
            validation["exact_duplicate_rate_vs_smoke_train"]
        )
        and validation[
            "exact_duplicate_rate_vs_smoke_train"
        ] < 0.50
    )
    validation["overall_smoke_pass"] = all(
        [
            validation["pass_row_count"],
            validation["pass_column_set"],
            validation["pass_binary_target"],
            validation["pass_mean_column_validity"],
            validation["pass_exact_copy_not_dominant"],
        ]
    )

    (
        out_dir / "smoke_validation.json"
    ).write_text(json.dumps(validation, indent=2))
    per_column.to_csv(
        out_dir / "smoke_column_validation.csv",
        index=False,
    )

    print("\nEnvironment:")
    print(json.dumps(environment, indent=2))

    print("\nInstalled GReaT API:")
    print(json.dumps(api_report, indent=2))

    print("\nSmoke validation:")
    print(json.dumps(validation, indent=2))

    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
