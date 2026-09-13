#!/usr/bin/env python3
"""
Repair CKD DDPM categorical validation failures without retraining the DDPM.

The original CKD encoder always included an explicit categorical missing token,
even when a particular training split contained no missing value for that
feature. During DDPM decoding, this unused category could occasionally win the
argmax. The generated value was then structurally valid as NaN, but the
validator correctly marked it as an unseen category for that training split.

This script:
1. reconstructs the original stratified splits;
2. finds invalid categorical levels in saved DDPM CSV files;
3. replaces only those invalid values with the corresponding training-split mode;
4. backs up the original files;
5. recomputes validation, quality, and TSTR utility only for affected DDPM files;
6. merges the repaired rows into the existing experiment outputs;
7. regenerates all summary CSV files and plots.

It does not retrain the diffusion model.

Example:
    python repair_ckd_ddpm_categories.py \
      --data_path data/kidney_disease_sanitized.csv \
      --experiment_dir results/ckd/synthetic_experiment \
      --source_script run_ckd_synthetic_experiment.py \
      --affected_splits 2 4 \
      --classifiers lr mlp xgb rfc \
      --apply
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List

SCRIPT_VERSION = "2.0-python311-import-fix"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/processed/ckd/kidney_disease_sanitized.csv"))
    parser.add_argument("--experiment_dir", default=str(REPOSITORY_ROOT / "results/ckd/synthetic_experiment"))
    parser.add_argument(
        "--source_script",
        default=str(Path(__file__).resolve().parent / "run_ckd_synthetic_experiment.py"),
        help="Path to the original CKD experiment script.",
    )
    parser.add_argument(
        "--affected_splits",
        nargs="+",
        type=int,
        default=[2, 4],
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["lr", "mlp", "xgb", "rfc"],
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply repairs and recompute outputs. Without this flag, only diagnose.",
    )
    return parser.parse_args()


def import_experiment_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Source script not found: {path}")

    spec = importlib.util.spec_from_file_location(
        "ckd_experiment_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)

    # Required for Python 3.11 dataclasses during dynamic imports.
    # @dataclass looks up cls.__module__ in sys.modules while the
    # imported source file is still being executed.
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Do not leave a partially initialized module registered.
        sys.modules.pop(spec.name, None)
        raise

    return module


def normalized_token(value, is_ordinal: bool) -> str:
    if pd.isna(value):
        return "__MISSING__"

    if is_ordinal:
        try:
            return f"{float(value):.12g}"
        except (TypeError, ValueError):
            pass

    return str(value).strip()


def invalid_mask(
    real_series: pd.Series,
    synthetic_series: pd.Series,
    is_ordinal: bool,
) -> tuple[pd.Series, set[str], set[str]]:
    real_tokens = real_series.map(
        lambda value: normalized_token(value, is_ordinal)
    )
    synthetic_tokens = synthetic_series.map(
        lambda value: normalized_token(value, is_ordinal)
    )

    allowed = set(real_tokens.unique().tolist())
    generated = set(synthetic_tokens.unique().tolist())
    invalid_levels = generated - allowed

    return synthetic_tokens.isin(invalid_levels), allowed, invalid_levels


def parse_saved_filename(path: Path) -> tuple[int, int]:
    match = re.fullmatch(
        r"synthetic_repeat(\d+)_(\d+)x\.csv",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected synthetic filename: {path.name}")

    return int(match.group(1)), int(match.group(2))


def remove_existing_rows(
    df: pd.DataFrame,
    split: int,
    repeat: int,
    multiplier: int,
) -> pd.DataFrame:
    mask = (
        (df["generator"] == "COND_DDPM")
        & (df["split"] == split)
        & (df["repeat"] == repeat)
        & (df["size_multiplier"] == multiplier)
    )
    return df.loc[~mask].copy()


def main() -> None:
    args = parse_args()
    print(f"[INFO] repair script version: {SCRIPT_VERSION}", flush=True)

    experiment_dir = Path(args.experiment_dir)
    source_script = Path(args.source_script)
    module = import_experiment_module(source_script)

    data = module.load_and_validate_data(
        args.data_path,
        module.TARGET,
    )

    X = data[module.FEATURE_COLUMNS]
    y = data[module.TARGET].astype(int)

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )
    split_indices = list(splitter.split(X, y))

    repair_log: List[Dict[str, object]] = []
    repaired_jobs: List[Dict[str, object]] = []

    for split in args.affected_splits:
        if split < 0 or split >= len(split_indices):
            raise ValueError(f"Invalid split index: {split}")

        train_idx, test_idx = split_indices[split]
        real_train = data.iloc[train_idx].copy().reset_index(drop=True)
        real_test = data.iloc[test_idx].copy().reset_index(drop=True)

        folder = (
            experiment_dir
            / "synthetic"
            / "COND_DDPM"
            / f"split_{split}"
        )
        files = sorted(folder.glob("synthetic_repeat*_*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No saved DDPM files found under {folder}. "
                "The original experiment must have been run with --save_synthetic."
            )

        for path in files:
            repeat, multiplier = parse_saved_filename(path)
            synthetic = pd.read_csv(path)

            file_change_count = 0
            feature_details = []

            for col in module.CATEGORICAL_FEATURES:
                is_ordinal = col in module.ORDINAL_CATEGORICAL_FEATURES
                mask, allowed, invalid_levels = invalid_mask(
                    real_train[col],
                    synthetic[col],
                    is_ordinal,
                )

                n_invalid = int(mask.sum())
                if n_invalid == 0:
                    continue

                observed = real_train[col].dropna()
                if observed.empty:
                    raise RuntimeError(
                        f"Training split {split} has no observed value for {col}; "
                        "cannot compute a repair value."
                    )

                replacement = observed.mode(dropna=True).iloc[0]
                synthetic.loc[mask, col] = replacement

                file_change_count += n_invalid
                feature_details.append(
                    {
                        "feature": col,
                        "invalid_levels": "|".join(sorted(invalid_levels)),
                        "n_replaced": n_invalid,
                        "replacement": replacement,
                    }
                )

            validation_after = module.validate_synthetic(
                real_train,
                synthetic,
                module.TARGET,
                len(synthetic),
            )

            repair_log.append(
                {
                    "split": split,
                    "repeat": repeat,
                    "size_multiplier": multiplier,
                    "file": str(path),
                    "n_values_replaced": file_change_count,
                    "features_repaired": "|".join(
                        detail["feature"] for detail in feature_details
                    ),
                    "repair_details": repr(feature_details),
                    "validation_status_after": validation_after["status"],
                    "invalid_category_levels_after": validation_after[
                        "invalid_category_levels"
                    ],
                }
            )

            if file_change_count > 0:
                print(
                    f"[DIAGNOSE] split={split}, repeat={repeat}, "
                    f"size={multiplier}x: repaired {file_change_count} "
                    f"categorical values in "
                    f"{[detail['feature'] for detail in feature_details]}",
                    flush=True,
                )

            if args.apply:
                backup = path.with_suffix(".before_category_repair.csv")
                if not backup.exists():
                    shutil.copy2(path, backup)

                synthetic.to_csv(path, index=False)

                if validation_after["status"] != "OK":
                    raise RuntimeError(
                        f"Validation still failed after repair for {path}: "
                        f"{validation_after}"
                    )

                repaired_jobs.append(
                    {
                        "split": split,
                        "repeat": repeat,
                        "multiplier": multiplier,
                        "real_train": real_train,
                        "real_test": real_test,
                        "synthetic": synthetic,
                        "validation": validation_after,
                    }
                )

    log_path = experiment_dir / "ckd_ddpm_category_repair_log.csv"
    pd.DataFrame(repair_log).to_csv(log_path, index=False)
    print(f"\nSaved diagnostic log: {log_path}")

    if not args.apply:
        print(
            "\nDry run only. Re-run with --apply to back up files, "
            "apply constrained decoding repairs, and recompute results."
        )
        return

    utility_path = experiment_dir / "ckd_synthetic_utility.csv"
    quality_path = experiment_dir / "ckd_synthetic_quality.csv"
    validation_path = experiment_dir / "ckd_synthetic_validation.csv"

    utility = pd.read_csv(utility_path)
    quality = pd.read_csv(quality_path)
    validation = pd.read_csv(validation_path)

    new_utility_rows = []
    new_quality_rows = []
    new_validation_rows = []

    encoder_cache = {}

    for job in repaired_jobs:
        split = int(job["split"])
        repeat = int(job["repeat"])
        multiplier = int(job["multiplier"])
        real_train = job["real_train"]
        real_test = job["real_test"]
        synthetic = job["synthetic"]

        utility = remove_existing_rows(
            utility, split, repeat, multiplier
        )
        quality = remove_existing_rows(
            quality, split, repeat, multiplier
        )
        validation = remove_existing_rows(
            validation, split, repeat, multiplier
        )

        if split not in encoder_cache:
            encoder = module.CKDMixedEncoder().fit(real_train)
            real_encoded = encoder.transform(real_train)
            real_reference = module.real_to_real_reference(real_encoded)
            encoder_cache[split] = (
                encoder,
                real_encoded,
                real_reference,
            )

        encoder, real_encoded, real_reference = encoder_cache[split]

        quality_metrics = module.quality_metrics(
            real_train,
            synthetic,
            module.TARGET,
            encoder,
            real_encoded,
            real_reference,
        )
        new_quality_rows.append(
            {
                "dataset": "CKD",
                "generator": "COND_DDPM",
                "split": split,
                "repeat": repeat,
                "size_multiplier": multiplier,
                "n_synthetic_rows": len(synthetic),
                **quality_metrics,
            }
        )

        new_validation_rows.append(
            {
                "generator": "COND_DDPM",
                "split": split,
                "repeat": repeat,
                "size_multiplier": multiplier,
                **job["validation"],
            }
        )

        classifier_seed = (
            args.random_seed
            + split * 100000
            + repeat * 1000
            + 9000000
            + multiplier * 10
        )

        for classifier_name in args.classifiers:
            metrics = module.evaluate_classifier(
                train_df=synthetic,
                test_df=real_test,
                target_col=module.TARGET,
                classifier_name=classifier_name,
                seed=classifier_seed,
            )

            new_utility_rows.append(
                {
                    "dataset": "CKD",
                    "generator": "COND_DDPM",
                    "split": split,
                    "repeat": repeat,
                    "size_multiplier": multiplier,
                    "n_training_rows": len(synthetic),
                    "classifier": classifier_name,
                    **metrics,
                }
            )

    utility = pd.concat(
        [utility, pd.DataFrame(new_utility_rows)],
        ignore_index=True,
    )
    quality = pd.concat(
        [quality, pd.DataFrame(new_quality_rows)],
        ignore_index=True,
    )
    validation = pd.concat(
        [validation, pd.DataFrame(new_validation_rows)],
        ignore_index=True,
    )

    utility = utility.sort_values(
        ["generator", "split", "repeat", "size_multiplier", "classifier"]
    )
    quality = quality.sort_values(
        ["generator", "split", "repeat", "size_multiplier"]
    )
    validation = validation.sort_values(
        ["generator", "split", "repeat", "size_multiplier"]
    )

    utility.to_csv(utility_path, index=False)
    quality.to_csv(quality_path, index=False)
    validation.to_csv(validation_path, index=False)

    module.create_summaries(
        utility_df=utility,
        quality_df=quality,
        out_dir=experiment_dir,
    )

    print("\nRepaired validation status:")
    print(validation["status"].value_counts(dropna=False).to_string())
    print("\nUpdated summaries and plots were regenerated.")


if __name__ == "__main__":
    main()
