#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


NUMERIC_FEATURES = [
    "age", "bp", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wc", "rc",
]
CATEGORICAL_FEATURES = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane",
]
TARGET = "target"


def clean_input(df):
    if TARGET in df.columns:
        clean = df.copy()
    else:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"?": np.nan, "": np.nan})

        for col in ["pcv", "wc", "rc"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["classification"] = (
            df["classification"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        df[TARGET] = (
            df["classification"]
            .map({"ckd": 1, "notckd": 0})
            .astype(int)
        )
        clean = df.drop(columns=["id", "classification"])
        clean.loc[
            pd.to_numeric(clean["sod"], errors="coerce") < 100,
            "sod",
        ] = np.nan
        clean.loc[
            pd.to_numeric(clean["pot"], errors="coerce") > 10,
            "pot",
        ] = np.nan

    for col in NUMERIC_FEATURES:
        clean[col] = pd.to_numeric(
            clean[col], errors="coerce"
        ).astype(float)

    for col in CATEGORICAL_FEATURES:
        clean[col] = clean[col].astype(object)
        clean[col] = clean[col].where(
            pd.notna(clean[col]), np.nan
        )

    clean[TARGET] = clean[TARGET].astype(int)
    return clean


def make_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_preprocessor():
    return ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                ("onehot", make_ohe()),
            ]),
            CATEGORICAL_FEATURES,
        ),
    ])


def make_models(seed):
    models = {
        "LR": LogisticRegression(
            max_iter=2000,
            random_state=seed,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        ),
        "RFC": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
    }

    if XGBClassifier is not None:
        models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/processed/ckd/kidney_disease_sanitized.csv"))
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/ckd/baseline_real"),
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = clean_input(pd.read_csv(args.data_path))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    rows = []
    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    for split, (train_idx, test_idx) in enumerate(
        splitter.split(X, y)
    ):
        for name, estimator in make_models(
            args.random_seed + split
        ).items():
            model = Pipeline([
                ("preprocess", make_preprocessor()),
                ("classifier", estimator),
            ])
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            y_true = y.iloc[test_idx].to_numpy()
            y_score = model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]
            y_pred = (y_score >= 0.5).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).ravel()

            rows.append({
                "dataset": "CKD",
                "split": split,
                "classifier": name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "auc": roc_auc_score(y_true, y_score),
                "accuracy": accuracy_score(
                    y_true, y_pred
                ),
                "sensitivity": (
                    tp / (tp + fn)
                    if (tp + fn)
                    else np.nan
                ),
                "specificity": (
                    tn / (tn + fp)
                    if (tn + fp)
                    else np.nan
                ),
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "brier": brier_score_loss(
                    y_true, y_score
                ),
            })

    split_df = pd.DataFrame(rows)
    split_df.to_csv(
        out_dir / "ckd_real_baseline_split_results.csv",
        index=False,
    )

    summary = (
        split_df.groupby("classifier", as_index=False)
        .agg(
            n_splits=("auc", "size"),
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
    summary.to_csv(
        out_dir / "ckd_real_baseline_summary.csv",
        index=False,
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
