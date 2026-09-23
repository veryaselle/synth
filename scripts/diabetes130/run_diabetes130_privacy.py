#!/usr/bin/env python3
"""
Privacy-related evaluation for the Diabetes 130-US exploratory releases.

This is an empirical audit, not a formal privacy guarantee.

Inputs
------
- Frozen 20K real generator-training source
- Frozen patient-grouped real test partition
- Final GReaT release
- Gaussian and bootstrap releases created by
  run_diabetes130_utility_fidelity.py

Distance representation
-----------------------
A mixed Euclidean representation is fitted only on the real 20K source:
- numerical variables are standardized and jointly weighted to contribute one
  feature block;
- each categorical mismatch contributes equally through one-hot encoding;
- target is not part of the distance vector;
- nearest-neighbour searches are class-conditional on readmitted_30d.

Metrics
-------
- Distance to closest real record (DCR)
- Real-to-real nearest-neighbour reference
- DCR ratio and fraction below the real-NN fifth percentile
- Exact duplicates and internal duplicates
- Proximity-based membership-inference AUROC
- TPR at FPR 1%, 5% and 10%
- Bootstrap AUROC confidence intervals

Example
-------
python run_diabetes130_privacy.py \
  --real_source_csv results/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --test_csv results/diabetes130/llm_pilot_data/reduced/test.csv \
  --great_csv results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv \
  --generated_releases_dir results/diabetes130/final_evaluation/utility_fidelity/generated_releases \
  --out_dir results/diabetes130/final_evaluation/privacy \
  --mia_max_per_group 5000 \
  --bootstrap_iterations 500
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "readmitted_30d"
MISSING_TOKEN = "__MISSING__"

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_source_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--great_csv", required=True)
    parser.add_argument("--generated_releases_dir", required=True)
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/final_evaluation/privacy"),
    )
    parser.add_argument("--mia_max_per_group", type=int, default=5000)
    parser.add_argument("--query_batch_size", type=int, default=512)
    parser.add_argument("--bootstrap_iterations", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def canonical_categorical(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna(MISSING_TOKEN)
        .str.strip()
        .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    )


def load_release(
    path: str | Path,
    expected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if expected_columns is not None:
        missing = [
            col for col in expected_columns if col not in df.columns
        ]
        extra = [
            col for col in df.columns if col not in expected_columns
        ]
        if missing or extra:
            raise ValueError(
                f"Column mismatch for {path}. Missing={missing}; extra={extra}"
            )
        df = df[list(expected_columns)].copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    categorical_columns = [
        col
        for col in df.columns
        if col not in NUMERIC_FEATURES + [TARGET]
    ]
    for col in categorical_columns:
        df[col] = canonical_categorical(df[col])

    return df.reset_index(drop=True)


class MixedDistanceEncoder:
    def __init__(self, categorical_columns: Sequence[str]):
        self.categorical_columns = list(categorical_columns)
        self.numeric_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )

    def fit(self, real: pd.DataFrame):
        self.numeric_scaler.fit(real[NUMERIC_FEATURES])
        self.categorical_encoder.fit(
            real[self.categorical_columns].astype("string")
        )
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self.numeric_scaler.transform(
            df[NUMERIC_FEATURES]
        ).astype(np.float32)
        numeric /= math.sqrt(len(NUMERIC_FEATURES))

        categorical = self.categorical_encoder.transform(
            df[self.categorical_columns].astype("string")
        ).astype(np.float32)

        # A one-hot mismatch has squared Euclidean distance 2. Scaling by
        # sqrt(2 * number_of_original_categorical_features) makes the total
        # categorical squared distance equal to the fraction of mismatched
        # original categorical features.
        categorical /= math.sqrt(
            2.0 * len(self.categorical_columns)
        )

        return sparse.hstack(
            [
                sparse.csr_matrix(numeric),
                categorical,
            ],
            format="csr",
            dtype=np.float32,
        )


def canonical_row_keys(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> pd.Series:
    normalized = pd.DataFrame(index=df.index)

    for col in NUMERIC_FEATURES:
        normalized[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .round(8)
            .astype("string")
        )
    for col in categorical_columns:
        normalized[col] = canonical_categorical(df[col])
    normalized[TARGET] = (
        pd.to_numeric(df[TARGET], errors="coerce")
        .round()
        .astype("Int64")
        .astype("string")
    )

    return normalized[df.columns].astype("string").agg(
        "\x1f".join,
        axis=1,
    )


def nearest_distances(
    reference: sparse.csr_matrix,
    queries: sparse.csr_matrix,
    n_neighbors: int,
    batch_size: int,
    n_jobs: int,
) -> np.ndarray:
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=n_jobs,
    )
    model.fit(reference)

    batches = []
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        distances, _ = model.kneighbors(
            queries[start:stop],
            return_distance=True,
        )
        batches.append(distances)

    return np.vstack(batches)


def tpr_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(fpr <= target_fpr)[0]
    return float(tpr[valid].max()) if len(valid) else 0.0


def bootstrap_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)

    for _ in range(iterations):
        indices = rng.integers(0, n, size=n)
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        aucs.append(
            roc_auc_score(sampled_y, scores[indices])
        )

    if not aucs:
        return np.nan, np.nan

    return (
        float(np.quantile(aucs, 0.025)),
        float(np.quantile(aucs, 0.975)),
    )


def release_privacy_metrics(
    source_name: str,
    release_seed: int,
    real: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame,
    encoder: MixedDistanceEncoder,
    categorical_columns: Sequence[str],
    mia_max_per_group: int,
    query_batch_size: int,
    bootstrap_iterations: int,
    random_seed: int,
    n_jobs: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rng = np.random.default_rng(
        random_seed + max(release_seed, 0)
    )

    real_encoded = encoder.transform(real)
    test_encoded = encoder.transform(test)
    synthetic_encoded = encoder.transform(synthetic)

    dcr_all = []
    real_nn_all = []
    mia_rows: List[Dict[str, object]] = []
    pooled_labels = []
    pooled_scores = []

    for target_value in [0, 1]:
        real_mask = real[TARGET].to_numpy() == target_value
        test_mask = test[TARGET].to_numpy() == target_value
        synthetic_mask = synthetic[TARGET].to_numpy() == target_value

        real_class = real_encoded[real_mask]
        test_class = test_encoded[test_mask]
        synthetic_class = synthetic_encoded[synthetic_mask]

        if synthetic_class.shape[0] == 0:
            raise ValueError(
                f"{source_name} has no synthetic rows for target {target_value}"
            )

        synthetic_to_real = nearest_distances(
            reference=real_class,
            queries=synthetic_class,
            n_neighbors=1,
            batch_size=query_batch_size,
            n_jobs=n_jobs,
        )[:, 0]
        dcr_all.append(synthetic_to_real)

        real_to_real = nearest_distances(
            reference=real_class,
            queries=real_class,
            n_neighbors=2,
            batch_size=query_batch_size,
            n_jobs=n_jobs,
        )[:, 1]
        real_nn_all.append(real_to_real)

        member_count = min(
            real_class.shape[0],
            test_class.shape[0],
            mia_max_per_group,
        )
        member_indices = rng.choice(
            real_class.shape[0],
            size=member_count,
            replace=False,
        )
        nonmember_indices = rng.choice(
            test_class.shape[0],
            size=member_count,
            replace=False,
        )

        member_distances = nearest_distances(
            reference=synthetic_class,
            queries=real_class[member_indices],
            n_neighbors=1,
            batch_size=query_batch_size,
            n_jobs=n_jobs,
        )[:, 0]
        nonmember_distances = nearest_distances(
            reference=synthetic_class,
            queries=test_class[nonmember_indices],
            n_neighbors=1,
            batch_size=query_batch_size,
            n_jobs=n_jobs,
        )[:, 0]

        labels = np.concatenate(
            [
                np.ones(member_count, dtype=int),
                np.zeros(member_count, dtype=int),
            ]
        )
        scores = -np.concatenate(
            [member_distances, nonmember_distances]
        )

        auc = float(roc_auc_score(labels, scores))
        ci_low, ci_high = bootstrap_auc_ci(
            labels,
            scores,
            iterations=bootstrap_iterations,
            seed=random_seed + target_value + max(release_seed, 0),
        )

        mia_rows.append(
            {
                "source": source_name,
                "release_seed": release_seed,
                "target": target_value,
                "member_n": member_count,
                "nonmember_n": member_count,
                "mia_auroc": auc,
                "mia_auroc_ci_low": ci_low,
                "mia_auroc_ci_high": ci_high,
                "tpr_at_fpr_0_01": tpr_at_fpr(
                    labels, scores, 0.01
                ),
                "tpr_at_fpr_0_05": tpr_at_fpr(
                    labels, scores, 0.05
                ),
                "tpr_at_fpr_0_10": tpr_at_fpr(
                    labels, scores, 0.10
                ),
                "member_distance_mean": float(
                    member_distances.mean()
                ),
                "nonmember_distance_mean": float(
                    nonmember_distances.mean()
                ),
                "member_exact_match_rate": float(
                    (member_distances == 0).mean()
                ),
                "nonmember_exact_match_rate": float(
                    (nonmember_distances == 0).mean()
                ),
            }
        )

        pooled_labels.append(labels)
        pooled_scores.append(scores)

    dcr = np.concatenate(dcr_all)
    real_nn = np.concatenate(real_nn_all)

    pooled_labels_array = np.concatenate(pooled_labels)
    pooled_scores_array = np.concatenate(pooled_scores)
    pooled_auc = float(
        roc_auc_score(
            pooled_labels_array,
            pooled_scores_array,
        )
    )
    pooled_ci_low, pooled_ci_high = bootstrap_auc_ci(
        pooled_labels_array,
        pooled_scores_array,
        iterations=bootstrap_iterations,
        seed=random_seed + 10_000 + max(release_seed, 0),
    )

    real_keys = set(
        canonical_row_keys(real, categorical_columns)
    )
    synthetic_keys = canonical_row_keys(
        synthetic,
        categorical_columns,
    )

    exact_duplicate_rate = float(
        synthetic_keys.isin(real_keys).mean()
    )
    internal_duplicate_rate = float(
        synthetic_keys.duplicated(keep=False).mean()
    )

    target_aucs = [
        row["mia_auroc"] for row in mia_rows
    ]

    summary: Dict[str, object] = {
        "source": source_name,
        "release_seed": release_seed,
        "synthetic_rows": len(synthetic),
        "synthetic_positive_rate": float(
            synthetic[TARGET].mean()
        ),
        "dcr_mean": float(dcr.mean()),
        "dcr_median": float(np.median(dcr)),
        "dcr_p05": float(np.quantile(dcr, 0.05)),
        "dcr_min": float(dcr.min()),
        "real_nn_mean": float(real_nn.mean()),
        "real_nn_median": float(np.median(real_nn)),
        "real_nn_p05": float(np.quantile(real_nn, 0.05)),
        "dcr_to_real_nn_mean_ratio": float(
            dcr.mean() / real_nn.mean()
        ) if real_nn.mean() > 0 else np.nan,
        "fraction_dcr_below_real_nn_p05": float(
            (dcr < np.quantile(real_nn, 0.05)).mean()
        ),
        "exact_duplicate_rate_vs_real_source": (
            exact_duplicate_rate
        ),
        "internal_duplicate_rate": internal_duplicate_rate,
        "mia_auroc_macro_target": float(
            np.mean(target_aucs)
        ),
        "mia_auroc_pooled": pooled_auc,
        "mia_auroc_pooled_ci_low": pooled_ci_low,
        "mia_auroc_pooled_ci_high": pooled_ci_high,
        "mia_tpr_at_fpr_0_01_pooled": tpr_at_fpr(
            pooled_labels_array,
            pooled_scores_array,
            0.01,
        ),
        "mia_tpr_at_fpr_0_05_pooled": tpr_at_fpr(
            pooled_labels_array,
            pooled_scores_array,
            0.05,
        ),
        "mia_tpr_at_fpr_0_10_pooled": tpr_at_fpr(
            pooled_labels_array,
            pooled_scores_array,
            0.10,
        ),
    }

    return summary, mia_rows


def parse_release_name(path: Path) -> Tuple[str, int]:
    stem = path.stem
    if "_seed" not in stem:
        return stem, 0
    name, seed_text = stem.rsplit("_seed", 1)
    return name, int(seed_text)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    real = load_release(args.real_source_csv)
    expected_columns = real.columns.tolist()
    test = load_release(args.test_csv, expected_columns)
    great = load_release(args.great_csv, expected_columns)

    categorical_columns = [
        col
        for col in expected_columns
        if col not in NUMERIC_FEATURES + [TARGET]
    ]

    encoder = MixedDistanceEncoder(
        categorical_columns
    ).fit(real)

    releases: List[Tuple[str, int, pd.DataFrame]] = [
        ("GREAT_20K", 0, great)
    ]

    generated_dir = Path(args.generated_releases_dir)
    for path in sorted(generated_dir.glob("*.csv")):
        source_name, release_seed = parse_release_name(path)
        release = load_release(path, expected_columns)
        releases.append(
            (source_name, release_seed, release)
        )

    summary_rows: List[Dict[str, object]] = []
    mia_rows: List[Dict[str, object]] = []

    for source_name, release_seed, release in releases:
        print(
            f"[PRIVACY] source={source_name} seed={release_seed}",
            flush=True,
        )
        summary, targets = release_privacy_metrics(
            source_name=source_name,
            release_seed=release_seed,
            real=real,
            test=test,
            synthetic=release,
            encoder=encoder,
            categorical_columns=categorical_columns,
            mia_max_per_group=args.mia_max_per_group,
            query_batch_size=args.query_batch_size,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
        )
        summary_rows.append(summary)
        mia_rows.extend(targets)

    release_metrics = pd.DataFrame(summary_rows)
    release_metrics.to_csv(
        out_dir / "privacy_release_metrics.csv",
        index=False,
    )

    mia_target_metrics = pd.DataFrame(mia_rows)
    mia_target_metrics.to_csv(
        out_dir / "mia_target_metrics.csv",
        index=False,
    )

    metric_columns = [
        "dcr_mean",
        "dcr_to_real_nn_mean_ratio",
        "fraction_dcr_below_real_nn_p05",
        "exact_duplicate_rate_vs_real_source",
        "internal_duplicate_rate",
        "mia_auroc_macro_target",
        "mia_auroc_pooled",
        "mia_tpr_at_fpr_0_10_pooled",
    ]
    privacy_summary = (
        release_metrics.groupby("source", as_index=False)[
            metric_columns
        ]
        .agg(["mean", "std", "min", "max"])
    )
    privacy_summary.columns = [
        "_".join(
            [part for part in column if part]
        )
        if isinstance(column, tuple)
        else column
        for column in privacy_summary.columns
    ]
    privacy_summary.to_csv(
        out_dir / "privacy_summary_by_source.csv",
        index=False,
    )

    plot_data = release_metrics.copy()
    plot_data["release"] = (
        plot_data["source"]
        + "_"
        + plot_data["release_seed"].astype(str)
    )

    ax = plot_data.set_index("release")[
        "mia_auroc_macro_target"
    ].plot(kind="bar", figsize=(11, 6))
    ax.axhline(0.5, linestyle="--")
    ax.set_ylabel("Class-conditional MIA AUROC")
    ax.set_xlabel("Release")
    ax.set_title("Diabetes 130-US membership-inference audit")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "mia_auroc.png",
        dpi=200,
    )
    plt.close()

    ax = plot_data.set_index("release")[
        "dcr_to_real_nn_mean_ratio"
    ].plot(kind="bar", figsize=(11, 6))
    ax.axhline(1.0, linestyle="--")
    ax.set_ylabel("DCR / real-NN mean ratio")
    ax.set_xlabel("Release")
    ax.set_title("Diabetes 130-US nearest-neighbour privacy indicator")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "dcr_ratio.png",
        dpi=200,
    )
    plt.close()

    run_summary = {
        "status": "complete",
        "releases_evaluated": len(release_metrics),
        "privacy_interpretation": (
            "Empirical privacy-related audit; not a formal privacy guarantee."
        ),
        "distance_is_class_conditional": True,
        "mia_is_class_conditional": True,
        "mia_max_per_group": args.mia_max_per_group,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    print("\nPrivacy release metrics:")
    print(release_metrics.to_string(index=False))
    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
