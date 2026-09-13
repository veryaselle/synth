#!/usr/bin/env python3
"""
Train and generate the controlled 20,000-row Diabetes 130-US GReaT pilot.

Design
------
- Uses the already frozen reduced 20,000-row real-training subset.
- Trains Qwen3-0.3B-distil through be-great for one initial epoch.
- Uses FP32 master parameters with Trainer FP16 autocast on RTX 2080 Ti.
- Generates target classes separately with start_col/start_col_dist.
- Generates in resumable chunks and verifies every chunk before accepting it.
- Preserves raw output. Schema enforcement is performed later by the separate
  `constrain_diabetes130_great_schema_v3.py` script.

The script is resumable:
- After training, rerun with --reuse_saved_model to skip training.
- Existing valid generation chunks are reused automatically.
- An interrupted multi-hour generation therefore does not need to restart.

First main run:
    python run_diabetes130_great_main_pilot.py \
      --train_csv results/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
      --out_dir results/diabetes130/great_main_20k \
      --epochs 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 8 \
      --n_synthetic 20000 \
      --sampling_batch_size 50 \
      --chunk_size 500 \
      --max_length 512 \
      --temperature 0.7 \
      --fp16

Resume after interruption:
    python run_diabetes130_great_main_pilot.py \
      --train_csv results/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
      --out_dir results/diabetes130/great_main_20k \
      --n_synthetic 20000 \
      --sampling_batch_size 50 \
      --chunk_size 500 \
      --max_length 512 \
      --temperature 0.7 \
      --fp16 \
      --reuse_saved_model
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import platform
import random
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

TARGET = "readmitted_30d"
MODEL_NAME = "tabularisai/Qwen3-0.3B-distil"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/great_main_20k"),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=MODEL_NAME,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument("--n_synthetic", type=int, default=20000)
    parser.add_argument(
        "--sampling_batch_size",
        type=int,
        default=50,
    )
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--reuse_saved_model",
        action="store_true",
    )
    return parser.parse_args()


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


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target: {TARGET}")

    feature_columns = [
        col for col in df.columns if col != TARGET
    ]
    df = df[feature_columns + [TARGET]].copy()
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise",
    ).astype(int)

    if sorted(df[TARGET].unique().tolist()) != [0, 1]:
        raise ValueError("Target must contain exactly 0 and 1.")

    return df.reset_index(drop=True)


def expected_chunk_sizes(
    total: int,
    chunk_size: int,
) -> List[int]:
    sizes = []
    remaining = total
    while remaining > 0:
        current = min(chunk_size, remaining)
        sizes.append(current)
        remaining -= current
    return sizes


def validate_existing_chunk(
    path: Path,
    expected_rows: int,
    expected_columns: Sequence[str],
    target_value: int,
) -> bool:
    if not path.exists():
        return False

    try:
        chunk = pd.read_csv(path)
    except Exception:
        return False

    if len(chunk) != expected_rows:
        return False

    if chunk.columns.tolist() != list(expected_columns):
        return False

    target = pd.to_numeric(
        chunk[TARGET],
        errors="coerce",
    )
    return bool(
        target.notna().all()
        and (target == target_value).all()
    )


def generate_class_chunks(
    *,
    model,
    target_value: int,
    total_rows: int,
    chunk_size: int,
    sampling_batch_size: int,
    max_length: int,
    temperature: float,
    random_seed: int,
    expected_columns: Sequence[str],
    chunks_root: Path,
    log_rows: List[Dict[str, object]],
) -> List[Path]:
    class_dir = chunks_root / f"target_{target_value}"
    class_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    sizes = expected_chunk_sizes(total_rows, chunk_size)

    for chunk_index, expected_rows in enumerate(sizes):
        chunk_path = class_dir / f"chunk_{chunk_index:04d}.csv"
        paths.append(chunk_path)

        if validate_existing_chunk(
            chunk_path,
            expected_rows,
            expected_columns,
            target_value,
        ):
            log_rows.append(
                {
                    "target": target_value,
                    "chunk_index": chunk_index,
                    "expected_rows": expected_rows,
                    "returned_rows": expected_rows,
                    "status": "REUSED",
                    "seconds": 0.0,
                    "peak_gpu_memory_gb": np.nan,
                    "path": str(chunk_path),
                }
            )
            print(
                f"[REUSE] target={target_value} "
                f"chunk={chunk_index} rows={expected_rows}",
                flush=True,
            )
            continue

        seed = (
            random_seed
            + target_value * 1_000_000
            + chunk_index
        )
        set_seeds(seed)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.time()
        generated = model.sample(
            n_samples=expected_rows,
            start_col=TARGET,
            start_col_dist={int(target_value): 1.0},
            temperature=temperature,
            k=min(sampling_batch_size, expected_rows),
            max_length=max_length,
            drop_nan=False,
            device="cuda",
            guided_sampling=False,
            random_feature_order=True,
            conditions=None,
        )
        elapsed = time.time() - start

        returned_rows = len(generated)
        returned_columns = generated.columns.tolist()
        target = pd.to_numeric(
            generated[TARGET],
            errors="coerce",
        ) if TARGET in generated.columns else pd.Series(dtype=float)

        complete = (
            returned_rows == expected_rows
            and returned_columns == list(expected_columns)
            and len(target) == expected_rows
            and target.notna().all()
            and (target == target_value).all()
        )

        status = "OK" if complete else "FAILED"
        log_row = {
            "target": target_value,
            "chunk_index": chunk_index,
            "expected_rows": expected_rows,
            "returned_rows": returned_rows,
            "status": status,
            "seconds": elapsed,
            "peak_gpu_memory_gb": round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            ),
            "path": str(chunk_path),
        }
        log_rows.append(log_row)
        print(json.dumps(log_row, indent=2), flush=True)

        if not complete:
            partial_path = class_dir / (
                f"chunk_{chunk_index:04d}_PARTIAL.csv"
            )
            generated.to_csv(partial_path, index=False)
            raise RuntimeError(
                "Generation chunk failed validation. Partial output was "
                f"saved to {partial_path}. Completed earlier chunks remain "
                "available and will be reused on the next run."
            )

        generated.to_csv(chunk_path, index=False)
        gc.collect()
        torch.cuda.empty_cache()

    return paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "saved_model"
    trainer_dir = out_dir / "trainer"
    chunks_root = out_dir / "raw_chunks"

    set_seeds(args.random_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    train = load_training_data(args.train_csv)
    expected_columns = train.columns.tolist()
    positive_rate = float(train[TARGET].mean())
    positive_n = int(round(args.n_synthetic * positive_rate))
    positive_n = max(1, min(positive_n, args.n_synthetic - 1))
    negative_n = args.n_synthetic - positive_n

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "be-great": package_version("be-great"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            3,
        ),
        "train_rows": len(train),
        "feature_count": len(expected_columns) - 1,
        "positive_rate": positive_rate,
        "n_synthetic": args.n_synthetic,
        "negative_requested": negative_n,
        "positive_requested": positive_n,
        "sampling_batch_size": args.sampling_batch_size,
        "chunk_size": args.chunk_size,
        "max_length": args.max_length,
        "temperature": args.temperature,
    }
    (out_dir / "run_configuration.json").write_text(
        json.dumps(environment, indent=2)
    )

    from be_great import GReaT

    if args.reuse_saved_model:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No saved model found at {model_dir}"
            )
        model = GReaT.load_from_dir(str(model_dir))
        model.model.float()
        training_seconds = 0.0
        print("[REUSE] Loaded saved model; training skipped.")
    else:
        model = GReaT(
            llm=args.model_name_or_path,
            experiment_dir=str(trainer_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            fp16=args.fp16,
            bf16=False,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=(
                args.dataloader_num_workers
            ),
            seed=args.random_seed,
            data_seed=args.random_seed,
        )

        checkpoint_dtype = str(
            next(model.model.parameters()).dtype
        )
        if args.fp16:
            model.model.float()
            if hasattr(model.model.config, "dtype"):
                model.model.config.dtype = "float32"
            if hasattr(model.model.config, "torch_dtype"):
                model.model.config.torch_dtype = torch.float32

        precision = {
            "checkpoint_dtype": checkpoint_dtype,
            "master_parameter_dtype": str(
                next(model.model.parameters()).dtype
            ),
            "trainer_fp16": bool(args.fp16),
            "trainer_bf16": False,
        }
        (out_dir / "precision_report.json").write_text(
            json.dumps(precision, indent=2)
        )

        fit_parameters = inspect.signature(model.fit).parameters
        fit_kwargs = {}
        if "conditional_col" in fit_parameters:
            fit_kwargs["conditional_col"] = TARGET
        if "random_conditional_col" in fit_parameters:
            fit_kwargs["random_conditional_col"] = False

        start = time.time()
        trainer = model.fit(train, **fit_kwargs)
        training_seconds = time.time() - start

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

    negative_paths = generate_class_chunks(
        model=model,
        target_value=0,
        total_rows=negative_n,
        chunk_size=args.chunk_size,
        sampling_batch_size=args.sampling_batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        random_seed=args.random_seed,
        expected_columns=expected_columns,
        chunks_root=chunks_root,
        log_rows=generation_log,
    )
    positive_paths = generate_class_chunks(
        model=model,
        target_value=1,
        total_rows=positive_n,
        chunk_size=args.chunk_size,
        sampling_batch_size=args.sampling_batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        random_seed=args.random_seed,
        expected_columns=expected_columns,
        chunks_root=chunks_root,
        log_rows=generation_log,
    )

    all_paths = negative_paths + positive_paths
    frames = [pd.read_csv(path) for path in all_paths]
    synthetic = (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=args.random_seed)
        .reset_index(drop=True)
    )

    if len(synthetic) != args.n_synthetic:
        raise RuntimeError(
            f"Final row count is {len(synthetic)}, expected "
            f"{args.n_synthetic}."
        )
    if synthetic.columns.tolist() != expected_columns:
        raise RuntimeError("Final column order differs from training data.")

    raw_path = out_dir / (
        f"synthetic_raw_{args.n_synthetic}.csv"
    )
    synthetic.to_csv(raw_path, index=False)

    log_df = pd.DataFrame(generation_log)
    log_df.to_csv(
        out_dir / "generation_chunk_log.csv",
        index=False,
    )

    metadata = {
        **environment,
        "training_seconds": training_seconds,
        "reused_saved_model": bool(args.reuse_saved_model),
        "completed_chunks": len(all_paths),
        "final_rows": len(synthetic),
        "final_positive_rate": float(
            pd.to_numeric(
                synthetic[TARGET],
                errors="raise",
            ).mean()
        ),
        "raw_output": str(raw_path),
        "generation_seconds_this_run": float(
            log_df["seconds"].sum()
        ),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nMain pilot generation complete:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
