#!/usr/bin/env python3
"""
Matched-source utility and fidelity evaluation for the Diabetes 130-US
exploratory GReaT case study.

Primary comparison
------------------
REAL_20K
GREAT_20K
GAUSSIAN_EMPIRICAL_20K
EMPIRICAL_BOOTSTRAP_20K

An optional REAL_FULL reference is evaluated separately as an upper reference,
not as the matched-size primary baseline.

Leakage control
---------------
All models are evaluated exclusively on the frozen patient-grouped real test
partition. The script never reconstructs or changes the patient split.

Utility
-------
- Logistic Regression
- Random Forest
- XGBoost, when installed
- MLP, unless --skip_mlp is passed
- AUROC, AUPRC, Brier score, F1, sensitivity, specificity and accuracy
- Fixed threshold 0.5 for threshold-dependent metrics
- Raw test predictions are retained for paired follow-up analyses

Fidelity
--------
- Normalized mean Wasserstein distance for numerical features
- Jensen-Shannon divergence for numerical and categorical features
- Pairwise correlation difference on a real-schema encoding
- Missingness-rate MAE
- Categorical support coverage
- Rare-category coverage
- Numerical range coverage
- Exact-copy and internal-duplicate rates

Simple generators
-----------------
GAUSSIAN_EMPIRICAL samples class-conditionally:
- numerical features from a regularized multivariate Gaussian;
- categorical features independently from class-specific empirical marginals;
- numerical values are rounded when the real feature is integral and projected
  onto the real source support.

EMPIRICAL_BOOTSTRAP samples complete rows class-conditionally with replacement
and serves as a positive memorization control.

Example
-------
python run_diabetes130_utility_fidelity.py \
  --real_source_csv results/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --real_full_train_csv results/diabetes130/llm_pilot_data/reduced/train.csv \
  --validation_csv results/diabetes130/llm_pilot_data/reduced/validation.csv \
  --test_csv results/diabetes130/llm_pilot_data/reduced/test.csv \
  --great_csv results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv \
  --great_audit_json results/diabetes130/great_main_20k/final_release/final_release_audit.json \
  --out_dir results/diabetes130/final_evaluation/utility_fidelity \
  --seeds 41 42 43

Resume after a post-processing failure without retraining classifiers:
    append --resume_existing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

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
    parser.add_argument("--real_full_train_csv")
    parser.add_argument("--validation_csv")
    parser.add_argument("--great_audit_json")
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/final_evaluation/utility_fidelity"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[41, 42, 43],
    )
    parser.add_argument("--skip_mlp", action="store_true")
    parser.add_argument("--skip_xgboost", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--pcd_max_rows", type=int, default=8000)
    parser.add_argument("--js_numeric_bins", type=int, default=20)
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help=(
            "Reuse utility_runs.csv and the already generated Gaussian/"
            "bootstrap releases in <out_dir>. No classifiers are retrained."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def canonical_categorical(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna(MISSING_TOKEN)
        .str.strip()
        .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    )


def load_release(
    path: str,
    expected_columns: Optional[Sequence[str]] = None,
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

    if TARGET not in df.columns:
        raise ValueError(f"Missing target {TARGET} in {path}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"Target is not binary in {path}")

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


def target_counts_for_size(
    source: pd.DataFrame,
    n_rows: int,
) -> Dict[int, int]:
    positive_rate = float(source[TARGET].mean())
    positive_n = int(round(n_rows * positive_rate))
    positive_n = max(1, min(positive_n, n_rows - 1))
    return {0: n_rows - positive_n, 1: positive_n}


def generate_bootstrap(
    source: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = target_counts_for_size(source, n_rows)
    frames = []

    for target_value, count in counts.items():
        group = source.loc[source[TARGET] == target_value]
        indices = rng.choice(group.index.to_numpy(), size=count, replace=True)
        frames.append(group.loc[indices].copy())

    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def is_integral(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    return bool(
        len(values)
        and np.isclose(values, np.round(values)).all()
    )


def generate_gaussian_empirical(
    source: pd.DataFrame,
    n_rows: int,
    seed: int,
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = target_counts_for_size(source, n_rows)
    frames = []

    global_numeric_min = source[NUMERIC_FEATURES].min()
    global_numeric_max = source[NUMERIC_FEATURES].max()
    integral_features = {
        col: is_integral(source[col]) for col in NUMERIC_FEATURES
    }

    for target_value, count in counts.items():
        group = source.loc[source[TARGET] == target_value].reset_index(drop=True)

        numerical = group[NUMERIC_FEATURES].astype(float)
        covariance_model = LedoitWolf().fit(numerical.to_numpy())
        sampled_numeric = rng.multivariate_normal(
            mean=covariance_model.location_,
            cov=covariance_model.covariance_,
            size=count,
            check_valid="ignore",
        )
        sampled_numeric = pd.DataFrame(
            sampled_numeric,
            columns=NUMERIC_FEATURES,
        )

        for col in NUMERIC_FEATURES:
            sampled_numeric[col] = sampled_numeric[col].clip(
                global_numeric_min[col],
                global_numeric_max[col],
            )
            if integral_features[col]:
                sampled_numeric[col] = sampled_numeric[col].round().astype(int)

        sampled = sampled_numeric

        for col in categorical_columns:
            probabilities = group[col].value_counts(normalize=True, dropna=False)
            sampled[col] = rng.choice(
                probabilities.index.to_numpy(),
                size=count,
                replace=True,
                p=probabilities.to_numpy(),
            )

        sampled[TARGET] = target_value
        sampled = sampled[source.columns]
        frames.append(sampled)

    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def make_preprocessor(
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=MISSING_TOKEN,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0.3,
    )


def classifier_factories(
    seed: int,
    n_jobs: int,
    skip_mlp: bool,
    skip_xgboost: bool,
):
    classifiers = {
        "LR": LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        ),
        "RFC": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        ),
    }

    if not skip_mlp:
        classifiers["MLP"] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        )

    if not skip_xgboost:
        try:
            from xgboost import XGBClassifier

            classifiers["XGB"] = XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=seed,
            )
        except ImportError:
            print("[WARN] xgboost is not installed; XGB will be skipped.")

    return classifiers


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan

    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "predicted_positive_rate": float(predictions.mean()),
        "mean_predicted_probability": float(probabilities.mean()),
    }


def normalized_row_keys(
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


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2.0) ** 2)


def categorical_js(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    real = canonical_categorical(real)
    synthetic = canonical_categorical(synthetic)
    levels = sorted(set(real) | set(synthetic))
    real_prob = real.value_counts(normalize=True).reindex(
        levels, fill_value=0.0
    )
    synthetic_prob = synthetic.value_counts(normalize=True).reindex(
        levels, fill_value=0.0
    )
    return js_divergence(real_prob.to_numpy(), synthetic_prob.to_numpy())


def numeric_js(
    real: pd.Series,
    synthetic: pd.Series,
    n_bins: int,
) -> float:
    real_values = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    synthetic_values = (
        pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy()
    )

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(real_values, quantiles))

    if len(edges) < 3:
        return 0.0 if np.allclose(
            np.mean(real_values),
            np.mean(synthetic_values),
        ) else 1.0

    edges[0] = -np.inf
    edges[-1] = np.inf
    real_hist, _ = np.histogram(real_values, bins=edges)
    synthetic_hist, _ = np.histogram(synthetic_values, bins=edges)
    return js_divergence(real_hist, synthetic_hist)


def _safe_correlation_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute a correlation matrix without divide-by-zero warnings.

    Constant encoded dimensions are retained. Their standardized values are
    defined as zero, so correlations involving a constant dimension are zero.
    This keeps the encoded dimensionality identical across releases and makes
    PCD comparable while avoiding NaNs from np.corrcoef.
    """
    values = np.asarray(matrix, dtype=np.float64)

    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "At least two rows are required for correlation estimation."
        )

    centered = values - values.mean(axis=0, keepdims=True)
    standard_deviation = np.sqrt(
        np.mean(centered * centered, axis=0)
    )

    standardized = np.divide(
        centered,
        standard_deviation,
        out=np.zeros_like(centered),
        where=standard_deviation > 1e-12,
    )

    correlation = (
        standardized.T @ standardized
    ) / values.shape[0]

    nonconstant = standard_deviation > 1e-12
    diagonal = np.where(nonconstant, 1.0, 0.0)
    np.fill_diagonal(correlation, diagonal)

    return np.clip(correlation, -1.0, 1.0)


def correlation_difference(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    categorical_columns: Sequence[str],
    max_rows: int,
    seed: int,
) -> Tuple[float, int]:
    real_sample = real.sample(
        n=min(max_rows, len(real)),
        random_state=seed,
    )
    synthetic_sample = synthetic.sample(
        n=min(max_rows, len(synthetic)),
        random_state=seed,
    )

    numeric_mean = real_sample[NUMERIC_FEATURES].mean()
    numeric_std = (
        real_sample[NUMERIC_FEATURES]
        .std(ddof=0)
        .replace(0, 1.0)
    )

    real_numeric = (
        (real_sample[NUMERIC_FEATURES] - numeric_mean)
        / numeric_std
    ).to_numpy(dtype=np.float32)
    synthetic_numeric = (
        (synthetic_sample[NUMERIC_FEATURES] - numeric_mean)
        / numeric_std
    ).to_numpy(dtype=np.float32)

    encoder = OneHotEncoder(
        categories=[
            sorted(
                canonical_categorical(
                    real[col]
                ).unique().tolist()
            )
            for col in categorical_columns
        ],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    real_categorical = encoder.fit_transform(
        real_sample[categorical_columns].astype("string")
    )
    synthetic_categorical = encoder.transform(
        synthetic_sample[categorical_columns].astype("string")
    )

    real_matrix = np.hstack(
        [real_numeric, real_categorical]
    )
    synthetic_matrix = np.hstack(
        [synthetic_numeric, synthetic_categorical]
    )

    real_corr = _safe_correlation_matrix(real_matrix)
    synthetic_corr = _safe_correlation_matrix(
        synthetic_matrix
    )

    upper = np.triu_indices(
        real_corr.shape[0],
        k=1,
    )
    pcd = float(
        np.mean(
            np.abs(
                real_corr[upper]
                - synthetic_corr[upper]
            )
        )
    )
    return pcd, int(real_corr.shape[0])


def compute_fidelity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    source_name: str,
    release_seed: int,
    categorical_columns: Sequence[str],
    pcd_max_rows: int,
    numeric_bins: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    feature_rows: List[Dict[str, object]] = []

    numeric_ws_values = []
    numeric_js_values = []
    numeric_range_coverage = []

    for col in NUMERIC_FEATURES:
        real_values = pd.to_numeric(real[col], errors="coerce").dropna()
        synthetic_values = (
            pd.to_numeric(synthetic[col], errors="coerce").dropna()
        )
        real_range = float(real_values.max() - real_values.min())
        normalized_ws = (
            float(wasserstein_distance(real_values, synthetic_values))
            / real_range
            if real_range > 0
            else 0.0
        )
        js = numeric_js(real_values, synthetic_values, numeric_bins)

        synthetic_range = float(
            synthetic_values.max() - synthetic_values.min()
        )
        range_coverage = (
            min(1.0, synthetic_range / real_range)
            if real_range > 0
            else 1.0
        )

        numeric_ws_values.append(normalized_ws)
        numeric_js_values.append(js)
        numeric_range_coverage.append(range_coverage)

        feature_rows.append(
            {
                "source": source_name,
                "release_seed": release_seed,
                "feature": col,
                "feature_type": "numeric",
                "normalized_wasserstein": normalized_ws,
                "js_divergence": js,
                "missing_rate_real": float(real[col].isna().mean()),
                "missing_rate_synthetic": float(
                    synthetic[col].isna().mean()
                ),
                "missing_rate_abs_diff": float(
                    abs(
                        real[col].isna().mean()
                        - synthetic[col].isna().mean()
                    )
                ),
                "support_coverage": range_coverage,
                "real_unique": int(real[col].nunique(dropna=False)),
                "synthetic_unique": int(
                    synthetic[col].nunique(dropna=False)
                ),
            }
        )

    categorical_js_values = []
    categorical_support_values = []
    rare_support_values = []
    missing_differences = []

    for col in categorical_columns:
        real_values = canonical_categorical(real[col])
        synthetic_values = canonical_categorical(synthetic[col])
        js = categorical_js(real_values, synthetic_values)

        real_levels = set(real_values)
        synthetic_levels = set(synthetic_values)
        support_coverage = (
            len(real_levels & synthetic_levels) / len(real_levels)
            if real_levels
            else 1.0
        )

        real_frequencies = real_values.value_counts(normalize=True)
        rare_levels = set(
            real_frequencies[real_frequencies < 0.01].index
        )
        rare_coverage = (
            len(rare_levels & synthetic_levels) / len(rare_levels)
            if rare_levels
            else 1.0
        )

        missing_real = float((real_values == MISSING_TOKEN).mean())
        missing_synthetic = float(
            (synthetic_values == MISSING_TOKEN).mean()
        )
        missing_abs_diff = abs(missing_real - missing_synthetic)

        categorical_js_values.append(js)
        categorical_support_values.append(support_coverage)
        rare_support_values.append(rare_coverage)
        missing_differences.append(missing_abs_diff)

        feature_rows.append(
            {
                "source": source_name,
                "release_seed": release_seed,
                "feature": col,
                "feature_type": "categorical",
                "normalized_wasserstein": np.nan,
                "js_divergence": js,
                "missing_rate_real": missing_real,
                "missing_rate_synthetic": missing_synthetic,
                "missing_rate_abs_diff": missing_abs_diff,
                "support_coverage": support_coverage,
                "rare_support_coverage": rare_coverage,
                "real_unique": int(real_values.nunique(dropna=False)),
                "synthetic_unique": int(
                    synthetic_values.nunique(dropna=False)
                ),
            }
        )

    pcd, pcd_dimensions = correlation_difference(
        real=real,
        synthetic=synthetic,
        categorical_columns=categorical_columns,
        max_rows=pcd_max_rows,
        seed=release_seed,
    )

    real_keys = set(
        normalized_row_keys(real, categorical_columns)
    )
    synthetic_keys = normalized_row_keys(
        synthetic,
        categorical_columns,
    )

    exact_duplicate_rate = float(
        synthetic_keys.isin(real_keys).mean()
    )
    internal_duplicate_rate = float(
        synthetic_keys.duplicated(keep=False).mean()
    )

    target_difference = abs(
        float(real[TARGET].mean())
        - float(synthetic[TARGET].mean())
    )

    all_feature_js = numeric_js_values + categorical_js_values
    all_missing_differences = [
        row["missing_rate_abs_diff"] for row in feature_rows
    ]

    summary: Dict[str, object] = {
        "source": source_name,
        "release_seed": release_seed,
        "rows": len(synthetic),
        "target_positive_rate": float(synthetic[TARGET].mean()),
        "target_rate_abs_diff": target_difference,
        "normalized_wasserstein_mean_numeric": float(
            np.mean(numeric_ws_values)
        ),
        "js_divergence_mean_numeric": float(
            np.mean(numeric_js_values)
        ),
        "js_divergence_mean_categorical": float(
            np.mean(categorical_js_values)
        ),
        "js_divergence_mean_all_features": float(
            np.mean(all_feature_js)
        ),
        "pairwise_correlation_difference": pcd,
        "pcd_encoded_dimensions": pcd_dimensions,
        "missingness_mae_all_features": float(
            np.mean(all_missing_differences)
        ),
        "categorical_support_coverage_mean": float(
            np.mean(categorical_support_values)
        ),
        "rare_categorical_support_coverage_mean": float(
            np.mean(rare_support_values)
        ),
        "numeric_range_coverage_mean": float(
            np.mean(numeric_range_coverage)
        ),
        "exact_duplicate_rate_vs_real_source": (
            exact_duplicate_rate
        ),
        "internal_duplicate_rate": internal_duplicate_rate,
    }
    return summary, feature_rows


def run_utility_for_release(
    source_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    release_seed: int,
    classifier_seed: int,
    categorical_columns: Sequence[str],
    predictions_dir: Path,
    n_jobs: int,
    skip_mlp: bool,
    skip_xgboost: bool,
) -> List[Dict[str, object]]:
    preprocessor = make_preprocessor(categorical_columns)

    train_features = train_df.drop(columns=[TARGET])
    train_target = train_df[TARGET].to_numpy()
    test_features = test_df.drop(columns=[TARGET])
    test_target = test_df[TARGET].to_numpy()

    transformed_train = preprocessor.fit_transform(train_features)
    transformed_test = preprocessor.transform(test_features)

    rows: List[Dict[str, object]] = []

    for classifier_name, classifier in classifier_factories(
        classifier_seed,
        n_jobs=n_jobs,
        skip_mlp=skip_mlp,
        skip_xgboost=skip_xgboost,
    ).items():
        print(
            f"[UTILITY] source={source_name} "
            f"release_seed={release_seed} "
            f"classifier={classifier_name} "
            f"classifier_seed={classifier_seed}",
            flush=True,
        )

        classifier.fit(transformed_train, train_target)
        probabilities = classifier.predict_proba(
            transformed_test
        )[:, 1]

        metrics = classification_metrics(
            test_target,
            probabilities,
        )
        metrics.update(
            {
                "source": source_name,
                "release_seed": release_seed,
                "classifier": classifier_name,
                "classifier_seed": classifier_seed,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_positive_rate": float(
                    train_df[TARGET].mean()
                ),
                "test_positive_rate": float(
                    test_df[TARGET].mean()
                ),
            }
        )
        rows.append(metrics)

        prediction_path = predictions_dir / (
            f"{source_name}_release{release_seed}_"
            f"{classifier_name}_seed{classifier_seed}.csv"
        )
        pd.DataFrame(
            {
                "test_row_index": np.arange(len(test_df)),
                "y_true": test_target,
                "y_probability": probabilities,
                "y_prediction_0_5": (
                    probabilities >= 0.5
                ).astype(int),
            }
        ).to_csv(prediction_path, index=False)

    return rows


def summarize_utility(
    utility_runs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate utility runs with named aggregation.

    The previous implementation combined groupby(as_index=False), list
    aggregation and reset_index(), which introduced a repeated `index` column.
    Merging those intermediate frames caused pandas.errors.MergeError.
    """
    metric_columns = [
        "auroc",
        "auprc",
        "brier",
        "f1",
        "sensitivity",
        "specificity",
        "accuracy",
        "predicted_positive_rate",
        "mean_predicted_probability",
    ]

    required_columns = {
        "source",
        "classifier",
        *metric_columns,
    }
    missing = sorted(
        required_columns - set(utility_runs.columns)
    )
    if missing:
        raise ValueError(
            "utility_runs is missing required columns: "
            + ", ".join(missing)
        )

    aggregations = {}
    for metric in metric_columns:
        for statistic in ["mean", "std", "min", "max"]:
            aggregations[
                f"{metric}_{statistic}"
            ] = (metric, statistic)

    summary = (
        utility_runs.groupby(
            ["source", "classifier"],
            as_index=False,
            dropna=False,
        )
        .agg(**aggregations)
    )

    counts = (
        utility_runs.groupby(
            ["source", "classifier"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "n_runs"})
    )

    return summary.merge(
        counts,
        on=["source", "classifier"],
        how="left",
        validate="one_to_one",
    )


def utility_deltas(
    utility_summary: pd.DataFrame,
) -> pd.DataFrame:
    reference = utility_summary.loc[
        utility_summary["source"] == "REAL_20K"
    ].set_index("classifier")

    rows = []
    for _, row in utility_summary.iterrows():
        classifier = row["classifier"]
        if classifier not in reference.index:
            continue
        ref = reference.loc[classifier]
        rows.append(
            {
                "source": row["source"],
                "classifier": classifier,
                "delta_auroc_vs_real20k": (
                    row["auroc_mean"] - ref["auroc_mean"]
                ),
                "delta_auprc_vs_real20k": (
                    row["auprc_mean"] - ref["auprc_mean"]
                ),
                "delta_brier_vs_real20k": (
                    row["brier_mean"] - ref["brier_mean"]
                ),
                "delta_f1_vs_real20k": (
                    row["f1_mean"] - ref["f1_mean"]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str,
) -> None:
    pivot = summary.pivot(
        index="classifier",
        columns="source",
        values=f"{metric}_mean",
    )
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Classifier")
    ax.set_title(f"Diabetes 130-US: {ylabel} by training source")
    ax.legend(title="Training source", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    generated_dir = out_dir / "generated_releases"
    predictions_dir = out_dir / "predictions"
    figures_dir = out_dir / "figures"

    for directory in [
        out_dir,
        generated_dir,
        predictions_dir,
        figures_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    real_source = load_release(args.real_source_csv)
    expected_columns = real_source.columns.tolist()
    test = load_release(args.test_csv, expected_columns)
    great = load_release(args.great_csv, expected_columns)

    real_full = (
        load_release(args.real_full_train_csv, expected_columns)
        if args.real_full_train_csv
        else None
    )
    validation = (
        load_release(args.validation_csv, expected_columns)
        if args.validation_csv
        else None
    )

    categorical_columns = [
        col
        for col in expected_columns
        if col not in NUMERIC_FEATURES + [TARGET]
    ]

    input_audit: Dict[str, object] = {
        "real_source_rows": len(real_source),
        "real_full_rows": (
            len(real_full) if real_full is not None else None
        ),
        "validation_rows": (
            len(validation) if validation is not None else None
        ),
        "test_rows": len(test),
        "great_rows": len(great),
        "columns": len(expected_columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": categorical_columns,
        "real_source_positive_rate": float(
            real_source[TARGET].mean()
        ),
        "test_positive_rate": float(test[TARGET].mean()),
        "great_positive_rate": float(great[TARGET].mean()),
        "seeds": args.seeds,
        "fixed_threshold": 0.5,
        "resume_existing": bool(args.resume_existing),
    }

    if args.great_audit_json:
        input_audit["great_release_audit"] = json.loads(
            Path(args.great_audit_json).read_text()
        )

    (out_dir / "input_audit.json").write_text(
        json.dumps(input_audit, indent=2)
    )

    releases: List[Tuple[str, int, pd.DataFrame]] = [
        ("REAL_20K", 0, real_source),
        ("GREAT_20K", 0, great),
    ]

    if real_full is not None:
        releases.append(("REAL_FULL_REFERENCE", 0, real_full))

    for seed in args.seeds:
        set_seed(seed)

        gaussian_path = generated_dir / (
            f"GAUSSIAN_EMPIRICAL_20K_seed{seed}.csv"
        )
        bootstrap_path = generated_dir / (
            f"EMPIRICAL_BOOTSTRAP_20K_seed{seed}.csv"
        )

        if args.resume_existing:
            if not gaussian_path.exists():
                raise FileNotFoundError(
                    "Resume requested, but the Gaussian release "
                    f"is missing: {gaussian_path}"
                )
            if not bootstrap_path.exists():
                raise FileNotFoundError(
                    "Resume requested, but the bootstrap release "
                    f"is missing: {bootstrap_path}"
                )

            gaussian = load_release(
                gaussian_path,
                expected_columns,
            )
            bootstrap = load_release(
                bootstrap_path,
                expected_columns,
            )
            print(
                f"[RESUME] Loaded generated releases for seed={seed}",
                flush=True,
            )
        else:
            gaussian = generate_gaussian_empirical(
                real_source,
                n_rows=len(real_source),
                seed=seed,
                categorical_columns=categorical_columns,
            )
            gaussian.to_csv(
                gaussian_path,
                index=False,
            )

            bootstrap = generate_bootstrap(
                real_source,
                n_rows=len(real_source),
                seed=seed,
            )
            bootstrap.to_csv(
                bootstrap_path,
                index=False,
            )

        releases.append(
            ("GAUSSIAN_EMPIRICAL_20K", seed, gaussian)
        )
        releases.append(
            ("EMPIRICAL_BOOTSTRAP_20K", seed, bootstrap)
        )

    utility_rows: List[Dict[str, object]] = []
    fidelity_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []

    existing_utility_path = out_dir / "utility_runs.csv"

    if args.resume_existing:
        if not existing_utility_path.exists():
            raise FileNotFoundError(
                "--resume_existing was requested, but utility_runs.csv "
                f"does not exist: {existing_utility_path}"
            )

        utility_runs = pd.read_csv(
            existing_utility_path
        )

        key_columns = [
            "source",
            "release_seed",
            "classifier",
            "classifier_seed",
        ]
        missing_key_columns = [
            col
            for col in key_columns
            if col not in utility_runs.columns
        ]
        if missing_key_columns:
            raise ValueError(
                "Existing utility_runs.csv is missing key columns: "
                + ", ".join(missing_key_columns)
            )

        duplicate_keys = utility_runs.duplicated(
            subset=key_columns,
            keep=False,
        )
        if duplicate_keys.any():
            duplicate_path = (
                out_dir
                / "utility_runs_duplicate_keys.csv"
            )
            utility_runs.loc[
                duplicate_keys
            ].to_csv(
                duplicate_path,
                index=False,
            )
            raise ValueError(
                "Existing utility_runs.csv contains duplicate run keys. "
                f"Details were saved to {duplicate_path}."
            )

        expected_sources = {
            "REAL_20K",
            "GREAT_20K",
            "GAUSSIAN_EMPIRICAL_20K",
            "EMPIRICAL_BOOTSTRAP_20K",
        }
        if real_full is not None:
            expected_sources.add(
                "REAL_FULL_REFERENCE"
            )

        missing_sources = sorted(
            expected_sources
            - set(utility_runs["source"])
        )
        if missing_sources:
            raise ValueError(
                "Existing utility_runs.csv is incomplete; missing sources: "
                + ", ".join(missing_sources)
            )

        print(
            "[RESUME] Reusing "
            f"{len(utility_runs)} completed utility runs. "
            "No classifiers will be retrained.",
            flush=True,
        )

    else:
        # Utility: fixed releases use all requested classifier seeds.
        # Independently generated simple releases use their matching seed.
        for source_name, release_seed, release in releases:
            classifier_seeds = (
                args.seeds
                if source_name
                in {
                    "REAL_20K",
                    "GREAT_20K",
                    "REAL_FULL_REFERENCE",
                }
                else [release_seed]
            )

            for classifier_seed in classifier_seeds:
                utility_rows.extend(
                    run_utility_for_release(
                        source_name=source_name,
                        train_df=release,
                        test_df=test,
                        release_seed=release_seed,
                        classifier_seed=classifier_seed,
                        categorical_columns=categorical_columns,
                        predictions_dir=predictions_dir,
                        n_jobs=args.n_jobs,
                        skip_mlp=args.skip_mlp,
                        skip_xgboost=args.skip_xgboost,
                    )
                )

        utility_runs = pd.DataFrame(
            utility_rows
        )
        utility_runs.to_csv(
            existing_utility_path,
            index=False,
        )

    # Fidelity is safe to recompute from the frozen releases and is much
    # cheaper than retraining the downstream classifiers.
    for source_name, release_seed, release in releases:
        if source_name in {
            "REAL_20K",
            "REAL_FULL_REFERENCE",
        }:
            continue

        print(
            f"[FIDELITY] source={source_name} "
            f"release_seed={release_seed}",
            flush=True,
        )
        summary, per_feature = compute_fidelity(
            real=real_source,
            synthetic=release,
            source_name=source_name,
            release_seed=release_seed,
            categorical_columns=categorical_columns,
            pcd_max_rows=args.pcd_max_rows,
            numeric_bins=args.js_numeric_bins,
        )
        fidelity_rows.append(summary)
        feature_rows.extend(per_feature)

    if validation is not None:
        print(
            "[FIDELITY] source=REAL_VALIDATION_REFERENCE "
            "release_seed=0",
            flush=True,
        )
        summary, per_feature = compute_fidelity(
            real=real_source,
            synthetic=validation,
            source_name="REAL_VALIDATION_REFERENCE",
            release_seed=0,
            categorical_columns=categorical_columns,
            pcd_max_rows=args.pcd_max_rows,
            numeric_bins=args.js_numeric_bins,
        )
        fidelity_rows.append(summary)
        feature_rows.extend(per_feature)

    utility_summary = summarize_utility(utility_runs)
    utility_summary.to_csv(
        out_dir / "utility_summary.csv",
        index=False,
    )

    deltas = utility_deltas(utility_summary)
    deltas.to_csv(
        out_dir / "utility_deltas_vs_real20k.csv",
        index=False,
    )

    fidelity_release = pd.DataFrame(fidelity_rows)
    fidelity_release.to_csv(
        out_dir / "fidelity_release_metrics.csv",
        index=False,
    )

    fidelity_feature = pd.DataFrame(feature_rows)
    fidelity_feature.to_csv(
        out_dir / "fidelity_feature_metrics.csv",
        index=False,
    )

    plot_metric(
        utility_summary,
        metric="auroc",
        output_path=figures_dir / "utility_auroc.png",
        ylabel="AUROC",
    )
    plot_metric(
        utility_summary,
        metric="auprc",
        output_path=figures_dir / "utility_auprc.png",
        ylabel="AUPRC",
    )
    plot_metric(
        utility_summary,
        metric="brier",
        output_path=figures_dir / "utility_brier.png",
        ylabel="Brier score",
    )

    if not fidelity_release.empty:
        selected = fidelity_release[
            [
                "source",
                "release_seed",
                "normalized_wasserstein_mean_numeric",
                "js_divergence_mean_all_features",
                "pairwise_correlation_difference",
                "missingness_mae_all_features",
            ]
        ].copy()
        selected["release"] = (
            selected["source"]
            + "_"
            + selected["release_seed"].astype(str)
        )
        selected = selected.set_index("release").drop(
            columns=["source", "release_seed"]
        )
        ax = selected.plot(
            kind="bar",
            figsize=(12, 6),
        )
        ax.set_title("Diabetes 130-US fidelity metrics")
        ax.set_ylabel("Metric value; lower is better")
        ax.set_xlabel("Release")
        ax.legend(bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(
            figures_dir / "fidelity_metrics.png",
            dpi=200,
        )
        plt.close()

    run_summary = {
        "status": "complete",
        "resume_existing": bool(args.resume_existing),
        "utility_run_rows": len(utility_runs),
        "utility_summary_rows": len(utility_summary),
        "fidelity_release_rows": len(fidelity_release),
        "fidelity_feature_rows": len(fidelity_feature),
        "generated_release_files": [
            str(path)
            for path in sorted(generated_dir.glob("*.csv"))
        ],
        "primary_comparison": [
            "REAL_20K",
            "GREAT_20K",
            "GAUSSIAN_EMPIRICAL_20K",
            "EMPIRICAL_BOOTSTRAP_20K",
        ],
        "real_full_is_upper_reference_only": (
            real_full is not None
        ),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    print("\nUtility summary:")
    print(
        utility_summary[
            [
                "source",
                "classifier",
                "n_runs",
                "auroc_mean",
                "auprc_mean",
                "brier_mean",
                "f1_mean",
            ]
        ].to_string(index=False)
    )

    print("\nFidelity release metrics:")
    print(fidelity_release.to_string(index=False))

    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
