#!/usr/bin/env python3
"""
Simple class-conditional synthetic baselines for Cleveland Heart Disease.

Why this script exists:
    The diffusion result should not be compared only against REAL.
    This script adds lightweight non-deep synthetic baselines using the exact
    same Train-Synthetic-Test-Real protocol as the diffusion experiment.

Generators:
    GAUSSIAN_EMPIRICAL:
        For each target class:
        - numeric features are sampled from a class-conditional multivariate Gaussian
        - categorical features are sampled independently from class-conditional empirical distributions

    EMPIRICAL_BOOTSTRAP:
        For each target class:
        - full real training rows are sampled with replacement
        This is a useful memorization/naive baseline, not a privacy-preserving generator.

Outputs:
    simple_baseline_utility.csv
    simple_baseline_utility_summary.csv
    simple_baseline_dataset_summary.csv

Example:
    python run_cleveland_simple_baselines.py \
      --data_path data/Heart_disease_cleveland_new.csv \
      --out_dir results/cleveland/simple_baselines \
      --n_splits 5 \
      --n_repeats 3 \
      --size_multipliers 1 2 3 \
      --generators gaussian empirical_bootstrap \
      --classifiers lr mlp xgb rfc \
      --include_real_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    pairwise_distances,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# ---------------------------------------------------------------------
# Arguments and validation
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/cleveland/Heart_disease_cleveland_new.csv"))
    parser.add_argument("--out_dir", default=str(REPOSITORY_ROOT / "results/cleveland/simple_baselines"))
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--size_multipliers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["gaussian", "empirical_bootstrap"],
        choices=["gaussian", "empirical_bootstrap"],
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["lr", "mlp", "xgb", "rfc"],
        choices=["lr", "mlp", "xgb", "rfc"],
    )
    parser.add_argument("--include_real_baseline", action="store_true")
    parser.add_argument("--save_synthetic", action="store_true")
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame, target_col: str) -> None:
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target_col])
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    target_values = sorted(df[target_col].dropna().unique().tolist())
    if target_values != [0, 1]:
        raise ValueError(f"Expected binary target values [0, 1], got {target_values}")


# ---------------------------------------------------------------------
# Classifiers and utility metrics
# ---------------------------------------------------------------------

def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_classifier(name: str, seed: int):
    name = name.lower()

    if name == "lr":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )

    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        )

    if name == "rfc":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed, but classifier 'xgb' was requested.") from exc

        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier: {name}")


def get_score_vector(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    clf_name: str,
    seed: int,
) -> Dict[str, float]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int).to_numpy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(int).to_numpy()

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("clf", make_classifier(clf_name, seed)),
        ]
    )

    model.fit(X_train, y_train)
    y_score = get_score_vector(model, X_test)
    return evaluate_predictions(y_test, y_score)


# ---------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------

def generate_target_labels(y_train: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    p1 = float(np.mean(y_train == 1))
    return rng.binomial(1, p1, size=n).astype(int)


def sample_empirical_bootstrap(
    real_train: pd.DataFrame,
    target_col: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_train = real_train[target_col].astype(int).to_numpy()
    y_syn = generate_target_labels(y_train, n, rng)

    rows = []
    for y_value in [0, 1]:
        n_class = int((y_syn == y_value).sum())
        class_df = real_train[real_train[target_col] == y_value]
        if n_class == 0:
            continue
        sampled_idx = rng.choice(class_df.index.to_numpy(), size=n_class, replace=True)
        sampled = class_df.loc[sampled_idx].copy()
        rows.append(sampled)

    syn = pd.concat(rows, ignore_index=True)
    syn = syn.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return syn[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target_col]]


def sample_categorical_from_distribution(
    values: pd.Series,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = values.value_counts(normalize=True, dropna=False).sort_index()
    cats = counts.index.to_numpy()
    probs = counts.to_numpy(dtype=float)
    probs = probs / probs.sum()
    return rng.choice(cats, size=n, replace=True, p=probs)


def sample_gaussian_empirical(
    real_train: pd.DataFrame,
    target_col: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Class-conditional baseline:
    - sample target according to real train distribution
    - numeric features: multivariate Gaussian per class
    - categorical features: independent empirical distributions per class
    """
    rng = np.random.default_rng(seed)
    y_train = real_train[target_col].astype(int).to_numpy()
    y_syn = generate_target_labels(y_train, n, rng)

    rows = []

    for y_value in [0, 1]:
        n_class = int((y_syn == y_value).sum())
        if n_class == 0:
            continue

        class_df = real_train[real_train[target_col] == y_value].copy()

        # Numeric multivariate Gaussian
        X_num = class_df[NUMERIC_FEATURES].astype(float).to_numpy()
        mean = X_num.mean(axis=0)
        cov = np.cov(X_num, rowvar=False)

        # Regularize covariance for small sample stability.
        if cov.ndim == 0:
            cov = np.eye(len(NUMERIC_FEATURES)) * float(cov)
        cov = np.asarray(cov, dtype=float)
        cov = cov + np.eye(len(NUMERIC_FEATURES)) * 1e-6

        try:
            num_sample = rng.multivariate_normal(mean, cov, size=n_class)
        except np.linalg.LinAlgError:
            diag = np.diag(np.maximum(np.diag(cov), 1e-6))
            num_sample = rng.multivariate_normal(mean, diag, size=n_class)

        syn_part = pd.DataFrame(num_sample, columns=NUMERIC_FEATURES)

        # Clip numeric values to the class-specific real training range.
        for col in NUMERIC_FEATURES:
            low = float(class_df[col].min())
            high = float(class_df[col].max())
            syn_part[col] = syn_part[col].clip(low, high)

            # Cleveland numeric columns except oldpeak are integer-valued.
            if col != "oldpeak":
                syn_part[col] = syn_part[col].round().astype(int)
            else:
                syn_part[col] = syn_part[col].round(2)

        # Categorical independent empirical sampling
        for col in CATEGORICAL_FEATURES:
            sampled = sample_categorical_from_distribution(class_df[col], n_class, rng)

            # Preserve integer-like categories.
            if pd.api.types.is_integer_dtype(class_df[col]):
                sampled = sampled.astype(int)

            syn_part[col] = sampled

        syn_part[target_col] = int(y_value)
        rows.append(syn_part)

    syn = pd.concat(rows, ignore_index=True)
    syn = syn.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return syn[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target_col]]


def make_synthetic(
    generator: str,
    real_train: pd.DataFrame,
    target_col: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
    if generator == "gaussian":
        return sample_gaussian_empirical(real_train, target_col, n, seed)
    if generator == "empirical_bootstrap":
        return sample_empirical_bootstrap(real_train, target_col, n, seed)
    raise ValueError(f"Unknown generator: {generator}")


# ---------------------------------------------------------------------
# Fidelity and privacy-like metrics
# ---------------------------------------------------------------------

def make_encoded_for_distance(real_train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """
    Fit preprocessing on real_train and transform df.
    Used only for distance/fidelity sanity checks.
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preprocessor = make_preprocessor()
    preprocessor.fit(real_train[feature_cols])
    return preprocessor.transform(df[feature_cols])


def pcd_metric(real_encoded: np.ndarray, syn_encoded: np.ndarray) -> float:
    real_corr = np.corrcoef(real_encoded, rowvar=False)
    syn_corr = np.corrcoef(syn_encoded, rowvar=False)

    real_corr = np.nan_to_num(real_corr)
    syn_corr = np.nan_to_num(syn_corr)

    idx = np.triu_indices_from(real_corr, k=1)
    return float(np.mean(np.abs(real_corr[idx] - syn_corr[idx])))


def wasserstein_metric_numeric_scaled(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return float("nan")

    distances = []
    for col in NUMERIC_FEATURES:
        real_values = real_df[col].astype(float).to_numpy()
        syn_values = syn_df[col].astype(float).to_numpy()

        mean = float(real_values.mean())
        std = float(real_values.std(ddof=0)) or 1.0

        real_scaled = (real_values - mean) / std
        syn_scaled = (syn_values - mean) / std

        distances.append(wasserstein_distance(real_scaled, syn_scaled))

    return float(np.mean(distances))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def js_metric_categorical(real_df: pd.DataFrame, syn_df: pd.DataFrame, target_col: str) -> float:
    values = []
    for col in CATEGORICAL_FEATURES + [target_col]:
        cats = sorted(set(real_df[col].dropna().unique().tolist()) | set(syn_df[col].dropna().unique().tolist()))
        real_counts = np.array([(real_df[col] == c).sum() for c in cats], dtype=float)
        syn_counts = np.array([(syn_df[col] == c).sum() for c in cats], dtype=float)
        values.append(js_divergence(real_counts, syn_counts))
    return float(np.mean(values))


def dcr_metrics(real_encoded: np.ndarray, syn_encoded: np.ndarray) -> Dict[str, float]:
    distances = pairwise_distances(syn_encoded, real_encoded, metric="euclidean")
    nearest = distances.min(axis=1)

    return {
        "dcr_mean": float(np.mean(nearest)),
        "dcr_min": float(np.min(nearest)),
        "dcr_p05": float(np.quantile(nearest, 0.05)),
    }


def canonicalize_for_duplicate(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in NUMERIC_FEATURES:
        out[col] = out[col].astype(float).round(3)
    return out


def duplicate_rate(real_train: pd.DataFrame, syn_df: pd.DataFrame, target_col: str) -> float:
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target_col]
    real_tuples = set(map(tuple, canonicalize_for_duplicate(real_train, cols).to_numpy()))
    syn_tuples = list(map(tuple, canonicalize_for_duplicate(syn_df, cols).to_numpy()))

    if not syn_tuples:
        return float("nan")

    return float(sum(row in real_tuples for row in syn_tuples) / len(syn_tuples))


def compute_fidelity_privacy(
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    target_col: str,
) -> Dict[str, float]:
    real_encoded = make_encoded_for_distance(real_train, real_train)
    syn_encoded = make_encoded_for_distance(real_train, syn_df)

    out = {
        "pcd": pcd_metric(real_encoded, syn_encoded),
        "ws_numeric_scaled": wasserstein_metric_numeric_scaled(real_train, syn_df),
        "js_categorical_target": js_metric_categorical(real_train, syn_df, target_col),
        "duplicate_rate_vs_real_train": duplicate_rate(real_train, syn_df, target_col),
        "synthetic_target_1_rate": float(syn_df[target_col].astype(int).mean()),
        "real_train_target_1_rate": float(real_train[target_col].astype(int).mean()),
    }
    out["target_1_rate_abs_diff"] = abs(out["synthetic_target_1_rate"] - out["real_train_target_1_rate"])
    out.update(dcr_metrics(real_encoded, syn_encoded))
    return out


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_root = out_dir / "synthetic"
    if args.save_synthetic:
        syn_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    validate_dataset(df, args.target_col)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[args.target_col].astype(int).copy()

    dataset_summary = pd.DataFrame(
        [
            {"key": "n_rows", "value": len(df)},
            {"key": "n_features", "value": len(feature_cols)},
            {"key": "target_0_count", "value": int((y == 0).sum())},
            {"key": "target_1_count", "value": int((y == 1).sum())},
            {"key": "target_1_rate", "value": float((y == 1).mean())},
            {"key": "numeric_features", "value": json.dumps(NUMERIC_FEATURES)},
            {"key": "categorical_features", "value": json.dumps(CATEGORICAL_FEATURES)},
            {"key": "generators", "value": json.dumps(args.generators)},
        ]
    )
    dataset_summary.to_csv(out_dir / "simple_baseline_dataset_summary.csv", index=False)

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    rows = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"\n=== Split {split_idx}/{args.n_splits - 1} ===", flush=True)

        real_train = df.iloc[train_idx].copy().reset_index(drop=True)
        real_test = df.iloc[test_idx].copy().reset_index(drop=True)

        split_seed = args.random_seed + split_idx

        if args.include_real_baseline:
            print("Evaluating REAL baseline...", flush=True)
            for clf_name in args.classifiers:
                metrics = evaluate_classifier(
                    train_df=real_train,
                    test_df=real_test,
                    target_col=args.target_col,
                    clf_name=clf_name,
                    seed=split_seed,
                )

                rows.append(
                    {
                        "dataset": "cleveland",
                        "generator": "REAL",
                        "split": split_idx,
                        "repeat": 0,
                        "size_multiplier": 1,
                        "n_synthetic_rows": np.nan,
                        "classifier": clf_name,
                        "pcd": np.nan,
                        "ws_numeric_scaled": np.nan,
                        "js_categorical_target": np.nan,
                        "dcr_mean": np.nan,
                        "dcr_min": np.nan,
                        "dcr_p05": np.nan,
                        "duplicate_rate_vs_real_train": np.nan,
                        "synthetic_target_1_rate": np.nan,
                        "real_train_target_1_rate": float(real_train[args.target_col].astype(int).mean()),
                        "target_1_rate_abs_diff": np.nan,
                        **metrics,
                    }
                )

        for generator in args.generators:
            for repeat in range(args.n_repeats):
                for multiplier in args.size_multipliers:
                    n_syn = int(len(real_train) * multiplier)
                    syn_seed = args.random_seed + 100000 * split_idx + 1000 * repeat + 10 * multiplier
                    if generator == "empirical_bootstrap":
                        syn_seed += 777

                    print(
                        f"Generator={generator}, repeat={repeat}, size={multiplier}x ({n_syn} rows)",
                        flush=True,
                    )

                    syn_df = make_synthetic(
                        generator=generator,
                        real_train=real_train,
                        target_col=args.target_col,
                        n=n_syn,
                        seed=syn_seed,
                    )

                    if args.save_synthetic:
                        split_dir = syn_root / generator / f"split_{split_idx}"
                        split_dir.mkdir(parents=True, exist_ok=True)
                        syn_df.to_csv(split_dir / f"synthetic_repeat{repeat}_{multiplier}x.csv", index=False)

                    fp_metrics = compute_fidelity_privacy(real_train, syn_df, args.target_col)

                    generator_name = {
                        "gaussian": "GAUSSIAN_EMPIRICAL",
                        "empirical_bootstrap": "EMPIRICAL_BOOTSTRAP",
                    }[generator]

                    for clf_name in args.classifiers:
                        metrics = evaluate_classifier(
                            train_df=syn_df,
                            test_df=real_test,
                            target_col=args.target_col,
                            clf_name=clf_name,
                            seed=syn_seed,
                        )

                        rows.append(
                            {
                                "dataset": "cleveland",
                                "generator": generator_name,
                                "split": split_idx,
                                "repeat": repeat,
                                "size_multiplier": multiplier,
                                "n_synthetic_rows": n_syn,
                                "classifier": clf_name,
                                **fp_metrics,
                                **metrics,
                            }
                        )

        pd.DataFrame(rows).to_csv(out_dir / "simple_baseline_utility.csv", index=False)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "simple_baseline_utility.csv", index=False)

    summary_df = (
        result_df
        .groupby(["generator", "classifier", "size_multiplier"], dropna=False, as_index=False)
        .agg(
            n_rows=("auc", "size"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"),
            brier_mean=("brier", "mean"),
            pcd_mean=("pcd", "mean"),
            ws_mean=("ws_numeric_scaled", "mean"),
            js_mean=("js_categorical_target", "mean"),
            dcr_mean=("dcr_mean", "mean"),
            dcr_min=("dcr_min", "min"),
            duplicate_rate_mean=("duplicate_rate_vs_real_train", "mean"),
            target_1_rate_abs_diff_mean=("target_1_rate_abs_diff", "mean"),
        )
        .sort_values(["generator", "classifier", "size_multiplier"])
    )
    summary_df.to_csv(out_dir / "simple_baseline_utility_summary.csv", index=False)

    print("\nSaved:")
    print(f"  {out_dir / 'simple_baseline_dataset_summary.csv'}")
    print(f"  {out_dir / 'simple_baseline_utility.csv'}")
    print(f"  {out_dir / 'simple_baseline_utility_summary.csv'}")

    print("\nAUC summary:")
    print(
        summary_df[
            ["generator", "classifier", "size_multiplier", "n_rows", "auc_mean", "auc_std", "duplicate_rate_mean"]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
