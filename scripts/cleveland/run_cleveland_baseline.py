#!/usr/bin/env python3
"""
Clean real-data baseline for the Cleveland Heart Disease dataset.

Purpose:
    Step 1 of the clean thesis pipeline.
    Train classifiers on real Cleveland training data and evaluate on held-out real test data.

Outputs:
    results/cleveland/baseline_real/baseline_real.csv
    results/cleveland/baseline_real/dataset_summary.csv

Example:
    python run_cleveland_baseline.py \
      --data_path data/Heart_disease_cleveland_new.csv \
      --out_dir results/cleveland/baseline_real \
      --n_splits 5 \
      --classifiers lr mlp xgb rfc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DEFAULT_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
DEFAULT_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/cleveland/Heart_disease_cleveland_new.csv"), help="Path to Cleveland CSV file.")
    parser.add_argument("--out_dir", default=str(REPOSITORY_ROOT / "results/cleveland/baseline_real"))
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["lr", "mlp", "xgb", "rfc"],
        choices=["lr", "mlp", "xgb", "rfc"],
    )
    return parser.parse_args()


def validate_dataset(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> None:
    expected = set(numeric_features + categorical_features + [target_col])
    missing_cols = sorted(expected - set(df.columns))
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    target_values = sorted(df[target_col].dropna().unique().tolist())
    if target_values != [0, 1]:
        raise ValueError(
            f"Expected binary target values [0, 1], but got {target_values}. "
            "If your target is 0/1/2/3/4, create a binary target first."
        )


def make_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            # Kept for robustness; has no effect if there are no missing values.
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Kept for robustness; has no effect if there are no missing values.
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_classifier(name: str, seed: int):
    name = name.lower()

    if name == "lr":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight=None,
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
            class_weight=None,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed, but classifier 'xgb' was requested. "
                "Install it or rerun without --classifiers xgb."
            ) from exc

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
    # Last fallback, not ideal for AUC/Brier but prevents hard failure.
    return model.predict(X).astype(float)


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

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


def run_baseline(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    numeric_features = DEFAULT_NUMERIC_FEATURES
    categorical_features = DEFAULT_CATEGORICAL_FEATURES

    validate_dataset(df, args.target_col, numeric_features, categorical_features)

    X = df[numeric_features + categorical_features].copy()
    y = df[args.target_col].astype(int).copy()

    summary_rows = [
        {"key": "n_rows", "value": len(df)},
        {"key": "n_features", "value": X.shape[1]},
        {"key": "n_missing_total", "value": int(df.isna().sum().sum())},
        {"key": "target_0_count", "value": int((y == 0).sum())},
        {"key": "target_1_count", "value": int((y == 1).sum())},
        {"key": "target_1_rate", "value": float((y == 1).mean())},
        {"key": "numeric_features", "value": json.dumps(numeric_features)},
        {"key": "categorical_features", "value": json.dumps(categorical_features)},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "dataset_summary.csv", index=False)

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    rows: List[Dict] = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()

        for clf_name in args.classifiers:
            seed = args.random_seed + split_idx
            model = Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                    ("clf", make_classifier(clf_name, seed)),
                ]
            )

            model.fit(X_train, y_train)
            y_score = get_score_vector(model, X_test)
            metrics = evaluate_predictions(y_test, y_score)

            rows.append(
                {
                    "dataset": "cleveland",
                    "experiment": "real_baseline",
                    "split": split_idx,
                    "classifier": clf_name,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    **metrics,
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "baseline_real.csv", index=False)

    aggregate_df = (
        result_df
        .groupby("classifier", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"),
            brier_mean=("brier", "mean"),
        )
        .sort_values("auc_mean", ascending=False)
    )
    aggregate_df.to_csv(out_dir / "baseline_real_summary.csv", index=False)

    return result_df, aggregate_df


def main() -> None:
    args = parse_args()
    result_df, aggregate_df = run_baseline(args)

    print("\nSaved:")
    print(f"  {Path(args.out_dir) / 'dataset_summary.csv'}")
    print(f"  {Path(args.out_dir) / 'baseline_real.csv'}")
    print(f"  {Path(args.out_dir) / 'baseline_real_summary.csv'}")

    print("\nAUC summary:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
