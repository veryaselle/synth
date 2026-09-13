#!/usr/bin/env python3
"""
Validation checks for Cleveland conditional diffusion outputs.

Purpose:
    Verify that saved synthetic Cleveland datasets are structurally valid and
    thesis-ready before writing results.

Checks:
    - expected columns are present
    - no missing values
    - target is binary 0/1
    - categorical columns contain only values seen in the corresponding real train split
    - numeric columns stay within the corresponding real train split ranges
    - exact duplicate rate against real train rows
    - target/class balance compared with real train split

Expected synthetic file layout:
    results/cleveland/diffusion_repeats/synthetic/split_0/synthetic_repeat0_1x.csv
    results/cleveland/diffusion_repeats/synthetic/split_0/synthetic_repeat0_2x.csv
    ...

Example:
    python check_cleveland_diffusion_outputs.py \
      --data_path data/Heart_disease_cleveland_new.csv \
      --diffusion_dir results/cleveland/diffusion_repeats \
      --n_splits 5 \
      --test_size 0.2 \
      --random_seed 42
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/cleveland/Heart_disease_cleveland_new.csv"))
    parser.add_argument("--diffusion_dir", default=str(REPOSITORY_ROOT / "results/cleveland/diffusion"))
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--synthetic_subdir", default="synthetic")
    parser.add_argument("--out_report", default=None)
    parser.add_argument("--out_class_balance", default=None)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def parse_synthetic_filename(path: Path) -> Tuple[int | None, int | None]:
    match = re.search(r"synthetic_repeat(\d+)_(\d+)x\.csv$", path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_split(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"split_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def canonicalize_for_duplicate_check(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(float).round(3)
    return out


def duplicate_rate_vs_train(real_train: pd.DataFrame, syn_df: pd.DataFrame, cols: List[str]) -> float:
    real_tuples = set(map(tuple, canonicalize_for_duplicate_check(real_train, cols).to_numpy()))
    syn_tuples = list(map(tuple, canonicalize_for_duplicate_check(syn_df, cols).to_numpy()))
    if len(syn_tuples) == 0:
        return np.nan
    return sum(row in real_tuples for row in syn_tuples) / len(syn_tuples)


def validate_file(
    syn_path: Path,
    syn_df: pd.DataFrame,
    real_train: pd.DataFrame,
    target_col: str,
    expected_cols: List[str],
) -> Dict:
    row: Dict = {}

    split = parse_split(syn_path)
    repeat, size_multiplier = parse_synthetic_filename(syn_path)

    row["synthetic_file"] = str(syn_path)
    row["split"] = split
    row["repeat"] = repeat
    row["size_multiplier"] = size_multiplier
    row["n_rows"] = len(syn_df)

    missing_expected = [c for c in expected_cols if c not in syn_df.columns]
    extra_cols = [c for c in syn_df.columns if c not in expected_cols]

    row["missing_expected_columns"] = "|".join(missing_expected)
    row["extra_columns"] = "|".join(extra_cols)
    row["has_expected_columns"] = len(missing_expected) == 0

    if missing_expected:
        row["status"] = "FAIL"
        row["n_missing_values"] = np.nan
        row["n_invalid_target"] = np.nan
        row["n_invalid_categorical_total"] = np.nan
        row["n_numeric_out_of_range_total"] = np.nan
        row["duplicate_rate_vs_real_train"] = np.nan
        row["real_train_target_1_rate"] = float(real_train[target_col].mean())
        row["synthetic_target_1_rate"] = np.nan
        row["target_1_rate_abs_diff"] = np.nan
        return row

    syn_df = syn_df[expected_cols].copy()

    row["n_missing_values"] = int(syn_df.isna().sum().sum())

    target_values = set(syn_df[target_col].dropna().unique().tolist())
    row["target_values"] = "|".join(map(str, sorted(target_values)))
    row["n_invalid_target"] = int((~syn_df[target_col].isin([0, 1])).sum())

    invalid_categorical_total = 0
    numeric_out_of_range_total = 0

    for col in CATEGORICAL_FEATURES:
        allowed = set(real_train[col].dropna().unique().tolist())
        invalid_count = int((~syn_df[col].isin(allowed)).sum())
        row[f"invalid_{col}"] = invalid_count
        invalid_categorical_total += invalid_count

    for col in NUMERIC_FEATURES:
        low = float(real_train[col].min())
        high = float(real_train[col].max())
        values = syn_df[col].astype(float)
        out_count = int(((values < low) | (values > high)).sum())
        row[f"{col}_train_min"] = low
        row[f"{col}_train_max"] = high
        row[f"{col}_synthetic_min"] = float(values.min())
        row[f"{col}_synthetic_max"] = float(values.max())
        row[f"out_of_range_{col}"] = out_count
        numeric_out_of_range_total += out_count

    row["n_invalid_categorical_total"] = invalid_categorical_total
    row["n_numeric_out_of_range_total"] = numeric_out_of_range_total

    cols_for_dup = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target_col]
    row["duplicate_rate_vs_real_train"] = duplicate_rate_vs_train(real_train, syn_df, cols_for_dup)

    real_rate = float(real_train[target_col].astype(int).mean())
    syn_rate = float(syn_df[target_col].astype(int).mean())
    row["real_train_target_1_rate"] = real_rate
    row["synthetic_target_1_rate"] = syn_rate
    row["target_1_rate_abs_diff"] = abs(real_rate - syn_rate)

    fail_conditions = [
        row["n_missing_values"] > 0,
        row["n_invalid_target"] > 0,
        row["n_invalid_categorical_total"] > 0,
        row["n_numeric_out_of_range_total"] > 0,
    ]
    row["status"] = "FAIL" if any(fail_conditions) else "OK"

    return row


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    diffusion_dir = Path(args.diffusion_dir)
    synthetic_root = diffusion_dir / args.synthetic_subdir

    out_report = Path(args.out_report) if args.out_report else diffusion_dir / "synthetic_validation_report.csv"
    out_class_balance = (
        Path(args.out_class_balance)
        if args.out_class_balance
        else diffusion_dir / "synthetic_class_balance.csv"
    )

    df = pd.read_csv(data_path)
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [args.target_col]

    missing_real_cols = [c for c in expected_cols if c not in df.columns]
    if missing_real_cols:
        raise ValueError(f"Real dataset missing expected columns: {missing_real_cols}")

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    split_to_train: Dict[int, pd.DataFrame] = {}
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[args.target_col].astype(int)

    for split_idx, (train_idx, _) in enumerate(splitter.split(X, y)):
        split_to_train[split_idx] = df.iloc[train_idx].copy().reset_index(drop=True)

    synthetic_files = sorted(synthetic_root.glob("split_*/synthetic_repeat*_*.csv"))

    if not synthetic_files:
        raise FileNotFoundError(
            f"No synthetic files found under {synthetic_root}. "
            "Did you run diffusion with --save_synthetic?"
        )

    rows = []
    for syn_path in synthetic_files:
        split = parse_split(syn_path)
        if split not in split_to_train:
            raise ValueError(f"Could not map {syn_path} to one of the expected splits.")

        syn_df = pd.read_csv(syn_path)
        real_train = split_to_train[split]

        rows.append(
            validate_file(
                syn_path=syn_path,
                syn_df=syn_df,
                real_train=real_train,
                target_col=args.target_col,
                expected_cols=expected_cols,
            )
        )

    report = pd.DataFrame(rows)
    report.to_csv(out_report, index=False)

    balance_cols = [
        "split",
        "repeat",
        "size_multiplier",
        "n_rows",
        "real_train_target_1_rate",
        "synthetic_target_1_rate",
        "target_1_rate_abs_diff",
        "duplicate_rate_vs_real_train",
        "status",
    ]
    report[balance_cols].to_csv(out_class_balance, index=False)

    status_counts = report["status"].value_counts(dropna=False)

    print("\nValidation status:")
    print(status_counts.to_string())

    print("\nClass-balance summary by size:")
    balance_summary = (
        report
        .groupby("size_multiplier", as_index=False)
        .agg(
            n_files=("synthetic_file", "size"),
            target_1_rate_mean=("synthetic_target_1_rate", "mean"),
            target_1_rate_std=("synthetic_target_1_rate", "std"),
            target_1_abs_diff_mean=("target_1_rate_abs_diff", "mean"),
            duplicate_rate_mean=("duplicate_rate_vs_real_train", "mean"),
            duplicate_rate_max=("duplicate_rate_vs_real_train", "max"),
        )
    )
    print(balance_summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {out_report}")
    print(f"  {out_class_balance}")

    if args.fail_on_error and (report["status"] != "OK").any():
        raise SystemExit("Validation failed for at least one synthetic file.")


if __name__ == "__main__":
    main()
