#!/usr/bin/env python3
"""
Privacy-like evaluation for PIMA synthetic datasets generated in Exp2.

This script evaluates the same synthetic source files used by the adapted clean
utility experiment and produces privacy-like sanity checks for the PIMA section.

Important:
    These metrics detect obvious memorization and unusually close synthetic
    records, but they do NOT constitute a formal privacy guarantee.

Metrics:
    - distance to closest real training record (DCR)
    - class-conditional DCR
    - real-to-real nearest-neighbour reference distance
    - DCR/reference ratio
    - fraction of synthetic records closer than the 5th percentile of the
      real-to-real nearest-neighbour distribution
    - exact and feature-only duplicate rates against the real training split
    - internal synthetic duplicate rate
    - target-balance difference
    - fraction of feature values outside the corresponding real-train range

Outputs:
    pima_privacy_file_level.csv
    pima_privacy_summary.csv
    pima_privacy_thesis_table.csv
    pima_synthetic_validation_report.csv
    pima_dcr_by_generator_size.png
    pima_duplicate_rate_by_generator_size.png

Example:
    python evaluate_pima_privacy.py \
      --data_path data/pima.csv \
      --correlation_dir results/paper/correlation \
      --out_dir results/pima/privacy \
      --splits 0 1 2 \
      --generators TVAE CTGAN COPULA \
      --n_files_per_model 20 \
      --size_multipliers 1 2 3
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PIMA_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/pima/pima.csv"))
    parser.add_argument("--correlation_dir", default=str(REPOSITORY_ROOT / "results/paper/correlation"))
    parser.add_argument("--out_dir", default=str(REPOSITORY_ROOT / "results/pima/privacy"))
    parser.add_argument("--target_col", default="Outcome")
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["TVAE", "CTGAN", "COPULA"],
    )
    parser.add_argument("--n_files_per_model", type=int, default=20)
    parser.add_argument(
        "--size_multipliers",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--starting_seed",
        type=int,
        default=4,
        help="Must match the adapted clean PIMA utility evaluation.",
    )
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument(
        "--no_zero_as_missing",
        action="store_true",
        help="Do not replace implausible zero values with NaN for DCR preprocessing.",
    )
    parser.add_argument(
        "--thesis_size",
        type=float,
        default=3.0,
        help="Synthetic size used in pima_privacy_thesis_table.csv.",
    )
    return parser.parse_args()


def stable_seed(*parts: object, base_seed: int = 42) -> int:
    text = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + base_seed) % (2**32 - 1)


def list_synthetic_files(
    correlation_dir: str,
    split: int,
    generator: str,
    n_files: int,
) -> List[Path]:
    pattern = Path(correlation_dir) / str(split) / "sd" / generator / "*.csv"
    files = [
        Path(p)
        for p in glob.glob(str(pattern))
        if ".ipynb_checkpoints" not in p
    ]

    def sort_key(path: Path):
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.stem)

    files = sorted(files, key=sort_key)
    if n_files > 0:
        files = files[:n_files]
    return files


def load_real_data(data_path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[target_col] = (
        pd.to_numeric(df[target_col], errors="coerce") >= 0.5
    ).astype(int)

    return df


def clean_for_distance(
    df: pd.DataFrame,
    target_col: str,
    zero_as_missing: bool,
) -> pd.DataFrame:
    feature_df = df.drop(columns=[target_col]).copy()

    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    if zero_as_missing:
        for col in PIMA_ZERO_AS_MISSING:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].replace(0, np.nan)

    return feature_df


def make_distance_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def sample_synthetic(
    synthetic_df: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    rng = np.random.default_rng(seed)
    replace = len(synthetic_df) < n_rows
    idx = rng.choice(
        np.arange(len(synthetic_df)),
        size=n_rows,
        replace=replace,
    )
    return synthetic_df.iloc[idx].copy().reset_index(drop=True)


def canonicalize(
    df: pd.DataFrame,
    columns: Sequence[str],
    decimals: int = 6,
) -> pd.DataFrame:
    out = df[list(columns)].copy()

    for col in columns:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all():
            out[col] = numeric.round(decimals)

    return out


def duplicate_rate_against_real(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    real_rows = set(
        map(tuple, canonicalize(real_train, columns).to_numpy())
    )
    synthetic_rows = list(
        map(tuple, canonicalize(synthetic, columns).to_numpy())
    )

    if not synthetic_rows:
        return float("nan")

    n_matches = sum(row in real_rows for row in synthetic_rows)
    return float(n_matches / len(synthetic_rows))


def internal_duplicate_rate(
    synthetic: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    if len(synthetic) == 0:
        return float("nan")

    canonical = canonicalize(synthetic, columns)
    n_unique = len(canonical.drop_duplicates())
    return float(1.0 - n_unique / len(canonical))


def real_to_real_reference(
    real_encoded: np.ndarray,
) -> Dict[str, float | np.ndarray]:
    distances = pairwise_distances(
        real_encoded,
        real_encoded,
        metric="euclidean",
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)

    return {
        "nearest": nearest,
        "mean": float(np.mean(nearest)),
        "median": float(np.median(nearest)),
        "p05": float(np.quantile(nearest, 0.05)),
        "min": float(np.min(nearest)),
    }


def dcr_metrics(
    synthetic_encoded: np.ndarray,
    real_encoded: np.ndarray,
) -> Dict[str, float | np.ndarray]:
    distances = pairwise_distances(
        synthetic_encoded,
        real_encoded,
        metric="euclidean",
    )
    nearest = distances.min(axis=1)

    return {
        "nearest": nearest,
        "mean": float(np.mean(nearest)),
        "median": float(np.median(nearest)),
        "p05": float(np.quantile(nearest, 0.05)),
        "min": float(np.min(nearest)),
    }


def class_conditional_dcr(
    synthetic_encoded: np.ndarray,
    synthetic_y: np.ndarray,
    real_encoded: np.ndarray,
    real_y: np.ndarray,
) -> Dict[str, float]:
    nearest_parts: List[np.ndarray] = []

    for class_value in [0, 1]:
        syn_mask = synthetic_y == class_value
        real_mask = real_y == class_value

        if not np.any(syn_mask) or not np.any(real_mask):
            continue

        distances = pairwise_distances(
            synthetic_encoded[syn_mask],
            real_encoded[real_mask],
            metric="euclidean",
        )
        nearest_parts.append(distances.min(axis=1))

    if not nearest_parts:
        return {
            "class_dcr_mean": float("nan"),
            "class_dcr_median": float("nan"),
            "class_dcr_p05": float("nan"),
            "class_dcr_min": float("nan"),
        }

    nearest = np.concatenate(nearest_parts)

    return {
        "class_dcr_mean": float(np.mean(nearest)),
        "class_dcr_median": float(np.median(nearest)),
        "class_dcr_p05": float(np.quantile(nearest, 0.05)),
        "class_dcr_min": float(np.min(nearest)),
    }


def range_violation_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[str, float]:
    total_values = 0
    total_violations = 0
    rows_with_violation = np.zeros(len(synthetic), dtype=bool)

    for col in feature_cols:
        real_values = pd.to_numeric(real_train[col], errors="coerce")
        syn_values = pd.to_numeric(synthetic[col], errors="coerce")

        low = float(real_values.min())
        high = float(real_values.max())

        invalid = syn_values.isna() | (syn_values < low) | (syn_values > high)
        total_values += len(syn_values)
        total_violations += int(invalid.sum())
        rows_with_violation |= invalid.to_numpy()

    return {
        "out_of_range_value_rate": (
            float(total_violations / total_values)
            if total_values > 0
            else float("nan")
        ),
        "rows_with_out_of_range_rate": (
            float(rows_with_violation.mean())
            if len(rows_with_violation) > 0
            else float("nan")
        ),
    }


def validate_synthetic_file(
    synthetic: pd.DataFrame,
    expected_cols: Sequence[str],
    target_col: str,
) -> Dict[str, object]:
    missing_cols = [c for c in expected_cols if c not in synthetic.columns]
    extra_cols = [c for c in synthetic.columns if c not in expected_cols]

    if missing_cols:
        return {
            "status": "FAIL",
            "missing_columns": "|".join(missing_cols),
            "extra_columns": "|".join(extra_cols),
            "n_missing_values": np.nan,
            "n_invalid_target": np.nan,
        }

    ordered = synthetic[list(expected_cols)].copy()
    target_numeric = pd.to_numeric(
        ordered[target_col],
        errors="coerce",
    )

    n_invalid_target = int(
        (~target_numeric.isin([0, 1])).sum()
    )

    return {
        "status": "OK" if n_invalid_target == 0 else "FAIL",
        "missing_columns": "",
        "extra_columns": "|".join(extra_cols),
        "n_missing_values": int(ordered.isna().sum().sum()),
        "n_invalid_target": n_invalid_target,
    }


def evaluate_sample(
    real_train_raw: pd.DataFrame,
    synthetic_raw: pd.DataFrame,
    target_col: str,
    distance_pipeline: Pipeline,
    real_encoded: np.ndarray,
    real_reference: Dict[str, float | np.ndarray],
    zero_as_missing: bool,
) -> Dict[str, float]:
    feature_cols = [c for c in real_train_raw.columns if c != target_col]
    all_cols = feature_cols + [target_col]

    synthetic_features = clean_for_distance(
        synthetic_raw,
        target_col=target_col,
        zero_as_missing=zero_as_missing,
    )
    synthetic_encoded = distance_pipeline.transform(synthetic_features)

    dcr = dcr_metrics(synthetic_encoded, real_encoded)

    real_y = real_train_raw[target_col].astype(int).to_numpy()
    synthetic_y = synthetic_raw[target_col].astype(int).to_numpy()

    class_dcr = class_conditional_dcr(
        synthetic_encoded=synthetic_encoded,
        synthetic_y=synthetic_y,
        real_encoded=real_encoded,
        real_y=real_y,
    )

    real_nn_mean = float(real_reference["mean"])
    real_nn_p05 = float(real_reference["p05"])
    nearest = np.asarray(dcr["nearest"])

    result = {
        "dcr_mean": float(dcr["mean"]),
        "dcr_median": float(dcr["median"]),
        "dcr_p05": float(dcr["p05"]),
        "dcr_min": float(dcr["min"]),
        **class_dcr,
        "real_nn_mean": real_nn_mean,
        "real_nn_median": float(real_reference["median"]),
        "real_nn_p05": real_nn_p05,
        "real_nn_min": float(real_reference["min"]),
        "dcr_mean_to_real_nn_ratio": (
            float(dcr["mean"]) / real_nn_mean
            if real_nn_mean > 0
            else float("nan")
        ),
        "fraction_below_real_nn_p05": float(
            np.mean(nearest < real_nn_p05)
        ),
        "exact_duplicate_rate": duplicate_rate_against_real(
            real_train_raw,
            synthetic_raw,
            all_cols,
        ),
        "feature_duplicate_rate": duplicate_rate_against_real(
            real_train_raw,
            synthetic_raw,
            feature_cols,
        ),
        "internal_duplicate_rate": internal_duplicate_rate(
            synthetic_raw,
            all_cols,
        ),
        "real_target_1_rate": float(
            real_train_raw[target_col].astype(int).mean()
        ),
        "synthetic_target_1_rate": float(
            synthetic_raw[target_col].astype(int).mean()
        ),
    }

    result["target_1_rate_abs_diff"] = abs(
        result["real_target_1_rate"]
        - result["synthetic_target_1_rate"]
    )

    result.update(
        range_violation_metrics(
            real_train=real_train_raw,
            synthetic=synthetic_raw,
            feature_cols=feature_cols,
        )
    )

    return result


def create_plots(summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is unavailable; plots were not created.")
        return

    plt.figure(figsize=(8, 5))
    for generator, sub in summary.groupby("generator"):
        sub = sub.sort_values("size_multiplier")
        plt.plot(
            sub["size_multiplier"],
            sub["dcr_mean"],
            marker="o",
            label=generator,
        )
    plt.xlabel("Synthetic data size multiplier")
    plt.ylabel("Mean distance to closest real record")
    plt.title("PIMA: DCR by generator and synthetic size")
    plt.xticks(sorted(summary["size_multiplier"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "pima_dcr_by_generator_size.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    for generator, sub in summary.groupby("generator"):
        sub = sub.sort_values("size_multiplier")
        plt.plot(
            sub["size_multiplier"],
            sub["exact_duplicate_rate"],
            marker="o",
            label=generator,
        )
    plt.xlabel("Synthetic data size multiplier")
    plt.ylabel("Exact duplicate rate vs real train")
    plt.title("PIMA: duplicate-rate sanity check")
    plt.xticks(sorted(summary["size_multiplier"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "pima_duplicate_rate_by_generator_size.png",
        dpi=200,
    )
    plt.close()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_df = load_real_data(args.data_path, args.target_col)
    expected_cols = real_df.columns.tolist()
    feature_cols = [
        c for c in expected_cols if c != args.target_col
    ]
    y = real_df[args.target_col].astype(int)

    zero_as_missing = not args.no_zero_as_missing

    file_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []

    for split in args.splits:
        split_seed = args.starting_seed + split
        train_idx, _ = train_test_split(
            np.arange(len(real_df)),
            test_size=args.test_size,
            random_state=split_seed,
            stratify=y,
        )

        real_train_raw = (
            real_df.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        real_train_features = clean_for_distance(
            real_train_raw,
            target_col=args.target_col,
            zero_as_missing=zero_as_missing,
        )

        distance_pipeline = make_distance_pipeline()
        real_encoded = distance_pipeline.fit_transform(
            real_train_features
        )
        real_reference = real_to_real_reference(real_encoded)

        n_real_train = len(real_train_raw)

        print(
            f"[INFO] split={split}: "
            f"real_train={n_real_train}, "
            f"real_nn_mean={real_reference['mean']:.4f}",
            flush=True,
        )

        for generator in args.generators:
            files = list_synthetic_files(
                args.correlation_dir,
                split,
                generator,
                args.n_files_per_model,
            )

            print(
                f"[INFO] split={split}, generator={generator}: "
                f"{len(files)} source files",
                flush=True,
            )

            for synthetic_path in files:
                try:
                    source_df = pd.read_csv(synthetic_path)
                except Exception as exc:
                    print(
                        f"[WARN] Could not read {synthetic_path}: {exc}",
                        flush=True,
                    )
                    continue

                validation = validate_synthetic_file(
                    source_df,
                    expected_cols=expected_cols,
                    target_col=args.target_col,
                )
                validation_rows.append(
                    {
                        "split": split,
                        "generator": generator,
                        "synthetic_file": synthetic_path.name,
                        "n_source_rows": len(source_df),
                        **validation,
                    }
                )

                if validation["status"] != "OK":
                    continue

                source_df = source_df[expected_cols].copy()

                for col in feature_cols:
                    source_df[col] = pd.to_numeric(
                        source_df[col],
                        errors="coerce",
                    )

                source_df[args.target_col] = (
                    pd.to_numeric(
                        source_df[args.target_col],
                        errors="coerce",
                    )
                    >= 0.5
                ).astype(int)

                for multiplier in args.size_multipliers:
                    n_rows = int(round(n_real_train * multiplier))
                    seed = stable_seed(
                        split,
                        generator,
                        synthetic_path.name,
                        multiplier,
                        base_seed=args.sampling_seed,
                    )

                    sampled = sample_synthetic(
                        source_df,
                        n_rows=n_rows,
                        seed=seed,
                    )

                    metrics = evaluate_sample(
                        real_train_raw=real_train_raw,
                        synthetic_raw=sampled,
                        target_col=args.target_col,
                        distance_pipeline=distance_pipeline,
                        real_encoded=real_encoded,
                        real_reference=real_reference,
                        zero_as_missing=zero_as_missing,
                    )

                    file_rows.append(
                        {
                            "dataset": "PIMA",
                            "split": split,
                            "generator": generator,
                            "synthetic_file": synthetic_path.name,
                            "size_multiplier": float(multiplier),
                            "n_synthetic_rows": len(sampled),
                            "n_source_rows": len(source_df),
                            **metrics,
                        }
                    )

                    print(
                        f"[OK] split={split} gen={generator} "
                        f"file={synthetic_path.name} "
                        f"size={multiplier}x "
                        f"dcr={metrics['dcr_mean']:.4f} "
                        f"dup={metrics['exact_duplicate_rate']:.4f}",
                        flush=True,
                    )

    file_level = pd.DataFrame(file_rows)
    validation_df = pd.DataFrame(validation_rows)

    file_level_path = out_dir / "pima_privacy_file_level.csv"
    validation_path = (
        out_dir / "pima_synthetic_validation_report.csv"
    )

    file_level.to_csv(file_level_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    if file_level.empty:
        raise RuntimeError(
            "No privacy results were produced. Check paths and validation output."
        )

    summary = (
        file_level
        .groupby(
            ["generator", "size_multiplier"],
            as_index=False,
        )
        .agg(
            n_files=("synthetic_file", "size"),
            n_splits=("split", "nunique"),
            dcr_mean=("dcr_mean", "mean"),
            dcr_std=("dcr_mean", "std"),
            dcr_median=("dcr_median", "mean"),
            dcr_p05=("dcr_p05", "mean"),
            dcr_min=("dcr_min", "min"),
            class_dcr_mean=("class_dcr_mean", "mean"),
            real_nn_mean=("real_nn_mean", "mean"),
            dcr_mean_to_real_nn_ratio=(
                "dcr_mean_to_real_nn_ratio",
                "mean",
            ),
            fraction_below_real_nn_p05=(
                "fraction_below_real_nn_p05",
                "mean",
            ),
            exact_duplicate_rate=(
                "exact_duplicate_rate",
                "mean",
            ),
            exact_duplicate_rate_max=(
                "exact_duplicate_rate",
                "max",
            ),
            feature_duplicate_rate=(
                "feature_duplicate_rate",
                "mean",
            ),
            internal_duplicate_rate=(
                "internal_duplicate_rate",
                "mean",
            ),
            target_1_rate_abs_diff=(
                "target_1_rate_abs_diff",
                "mean",
            ),
            out_of_range_value_rate=(
                "out_of_range_value_rate",
                "mean",
            ),
            rows_with_out_of_range_rate=(
                "rows_with_out_of_range_rate",
                "mean",
            ),
        )
        .sort_values(
            ["generator", "size_multiplier"]
        )
    )

    summary_path = out_dir / "pima_privacy_summary.csv"
    summary.to_csv(summary_path, index=False)

    thesis_table = summary[
        np.isclose(
            summary["size_multiplier"],
            args.thesis_size,
        )
    ].copy()

    thesis_cols = [
        "generator",
        "size_multiplier",
        "n_files",
        "n_splits",
        "dcr_mean",
        "dcr_p05",
        "dcr_min",
        "real_nn_mean",
        "dcr_mean_to_real_nn_ratio",
        "fraction_below_real_nn_p05",
        "exact_duplicate_rate",
        "feature_duplicate_rate",
        "internal_duplicate_rate",
        "target_1_rate_abs_diff",
        "out_of_range_value_rate",
    ]
    thesis_table = thesis_table[thesis_cols]

    thesis_path = out_dir / "pima_privacy_thesis_table.csv"
    thesis_table.to_csv(thesis_path, index=False)

    create_plots(summary, out_dir)

    print("\nValidation status:")
    print(
        validation_df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nPrivacy summary:")
    print(summary.to_string(index=False))

    print("\nThesis table:")
    print(thesis_table.to_string(index=False))

    print("\nSaved:")
    for path in [
        file_level_path,
        validation_path,
        summary_path,
        thesis_path,
        out_dir / "pima_dcr_by_generator_size.png",
        out_dir / "pima_duplicate_rate_by_generator_size.png",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
