#!/usr/bin/env python3
"""Prepare the three frozen real train/test datasets for the main benchmark.

The split is performed before any fitted preprocessing. All imputation parameters
are learned on the real training split only and then applied to the held-out test
split. The resulting processed CSVs contain no missing values and are the only
real tables that main generators should consume.

Dataset-specific compatibility choices:
- PIMA uses the legacy split-0 seed (4) and treats physiologically implausible
  zeros as missing in Glucose, BloodPressure, SkinThickness, Insulin and BMI.
  Missing values are filled by a train-fitted sklearn IterativeImputer (MICE-like).
- Cleveland uses split 0 from the previously used StratifiedShuffleSplit(seed=42)
  and needs no imputation after cleaning.
- CKD uses split 0 from StratifiedShuffleSplit(seed=42); numeric columns receive
  train medians and categorical/ordinal columns receive train modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "unified_benchmark_v3"
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

PIMA_ZERO_AS_MISSING = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
PIMA_NUMERIC = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age",
]
CLEVELAND_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CLEVELAND_CATEGORICAL = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
CKD_NUMERIC = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
]
CKD_CATEGORICAL = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
    "appet", "pe", "ane"
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(
    out_root: Path,
    dataset: str,
    split_id: int,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
    target: str,
    numeric: List[str],
    categorical: List[str],
    metadata: Dict[str, object],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    out = out_root / dataset / f"split_{split_id}"
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "real_train_raw": out / "real_train_raw.csv",
        "real_test_raw": out / "real_test_raw.csv",
        "real_train": out / "real_train.csv",
        "real_test": out / "real_test.csv",
    }
    train_raw.to_csv(paths["real_train_raw"], index=False)
    test_raw.to_csv(paths["real_test_raw"], index=False)
    train_processed.to_csv(paths["real_train"], index=False)
    test_processed.to_csv(paths["real_test"], index=False)

    pd.DataFrame({"row_index": train_indices}).to_csv(out / "train_indices.csv", index=False)
    pd.DataFrame({"row_index": test_indices}).to_csv(out / "test_indices.csv", index=False)

    schema = {
        "dataset": dataset,
        "split_id": split_id,
        "target": target,
        "numeric": numeric,
        "categorical": categorical,
        "columns": list(train_processed.columns),
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    summary = {
        **metadata,
        "split_id": split_id,
        "n_total": int(len(train_processed) + len(test_processed)),
        "n_train": int(len(train_processed)),
        "n_test": int(len(test_processed)),
        "train_target_prevalence": float(pd.to_numeric(train_processed[target]).mean()),
        "test_target_prevalence": float(pd.to_numeric(test_processed[target]).mean()),
        "missing_train_processed": int(train_processed.isna().sum().sum()),
        "missing_test_processed": int(test_processed.isna().sum().sum()),
        "files": {k: {"path": p.name, "sha256": sha256(p)} for k, p in paths.items()},
    }
    (out / "preparation_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary["missing_train_processed"] or summary["missing_test_processed"]:
        raise RuntimeError(f"{dataset}: processed data still contains missing values")
    if set(train_processed.columns) != set(test_processed.columns):
        raise RuntimeError(f"{dataset}: train/test schemas differ")
    if set(pd.unique(train_processed[target])) != {0, 1}:
        raise RuntimeError(f"{dataset}: train target is not binary 0/1")


def prepare_pima(path: Path, out_root: Path) -> None:
    df = pd.read_csv(path).copy()
    target = "Outcome"
    df[target] = pd.to_numeric(df[target], errors="raise").astype(int)
    idx = np.arange(len(df))
    for split_id, split_seed in enumerate([4, 5, 6, 7, 8]):
        tr_idx, te_idx = train_test_split(
            idx, test_size=0.2, random_state=split_seed, stratify=df[target]
        )
        tr_idx, te_idx = np.asarray(tr_idx), np.asarray(te_idx)
        train_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
        test_raw = df.iloc[te_idx].copy().reset_index(drop=True)

        for c in PIMA_ZERO_AS_MISSING:
            train_raw[c] = pd.to_numeric(train_raw[c], errors="coerce").replace(0, np.nan)
            test_raw[c] = pd.to_numeric(test_raw[c], errors="coerce").replace(0, np.nan)
        for c in PIMA_NUMERIC:
            train_raw[c] = pd.to_numeric(train_raw[c], errors="coerce")
            test_raw[c] = pd.to_numeric(test_raw[c], errors="coerce")

        imputer = IterativeImputer(
            random_state=42 + split_id, max_iter=20, initial_strategy="median",
            sample_posterior=False, skip_complete=True,
        )
        train_processed, test_processed = train_raw.copy(), test_raw.copy()
        train_processed[PIMA_NUMERIC] = imputer.fit_transform(train_raw[PIMA_NUMERIC])
        test_processed[PIMA_NUMERIC] = imputer.transform(test_raw[PIMA_NUMERIC])
        train_processed[target] = train_raw[target].to_numpy()
        test_processed[target] = test_raw[target].to_numpy()

        save_bundle(
            out_root, "pima", split_id, train_raw, test_raw, train_processed, test_processed,
            target, PIMA_NUMERIC, [],
            metadata={
                "split_strategy": "stratified train_test_split",
                "test_size": 0.2, "split_seed": split_seed,
                "preprocessing": {
                    "zero_as_missing_columns": PIMA_ZERO_AS_MISSING,
                    "imputation": "sklearn IterativeImputer (MICE-like), train-fitted only",
                    "imputer_random_seed": 42 + split_id, "max_iter": 20,
                    "initial_strategy": "median", "sample_posterior": False,
                },
            },
            train_indices=tr_idx, test_indices=te_idx,
        )


def five_stratified_splits(df: pd.DataFrame, target: str, seed: int = 42):
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    for split_id, (tr_idx, te_idx) in enumerate(splitter.split(df, df[target])):
        yield split_id, np.asarray(tr_idx), np.asarray(te_idx)


def prepare_cleveland(path: Path, out_root: Path) -> None:
    df = pd.read_csv(path).copy()
    target = "target"
    for c in CLEVELAND_NUMERIC + CLEVELAND_CATEGORICAL + [target]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df[target] = df[target].astype(int)
    if df.isna().any().any():
        raise ValueError("Clean Cleveland input unexpectedly contains missing values")
    for split_id, tr_idx, te_idx in five_stratified_splits(df, target, seed=42):
        train_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
        test_raw = df.iloc[te_idx].copy().reset_index(drop=True)
        save_bundle(
            out_root, "cleveland", split_id, train_raw, test_raw, train_raw.copy(), test_raw.copy(),
            target, CLEVELAND_NUMERIC, CLEVELAND_CATEGORICAL,
            metadata={
                "split_strategy": "StratifiedShuffleSplit", "n_splits": 5,
                "test_size": 0.2, "split_seed": 42,
                "preprocessing": {"imputation": "none; cleaned input has zero missing values"},
            },
            train_indices=tr_idx, test_indices=te_idx,
        )


def prepare_ckd(path: Path, out_root: Path) -> None:
    df = pd.read_csv(path).copy()
    target = "target"
    missing = set(CKD_NUMERIC + CKD_CATEGORICAL + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"CKD input missing columns: {sorted(missing)}")
    df = df[CKD_NUMERIC + CKD_CATEGORICAL + [target]].copy()
    for c in CKD_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="raise").astype(int)

    for split_id, tr_idx, te_idx in five_stratified_splits(df, target, seed=42):
        train_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
        test_raw = df.iloc[te_idx].copy().reset_index(drop=True)
        train_processed, test_processed = train_raw.copy(), test_raw.copy()
        numeric_fill, categorical_fill = {}, {}
        for c in CKD_NUMERIC:
            val = float(pd.to_numeric(train_raw[c], errors="coerce").median())
            numeric_fill[c] = val
            train_processed[c] = pd.to_numeric(train_raw[c], errors="coerce").fillna(val)
            test_processed[c] = pd.to_numeric(test_raw[c], errors="coerce").fillna(val)
        for c in CKD_CATEGORICAL:
            mode = train_raw[c].dropna().mode()
            if len(mode) == 0:
                raise ValueError(f"CKD categorical column {c} has no observed training value")
            val = mode.iloc[0]
            categorical_fill[c] = val.item() if hasattr(val, "item") else val
            train_processed[c] = train_raw[c].fillna(val)
            test_processed[c] = test_raw[c].fillna(val)
        save_bundle(
            out_root, "ckd", split_id, train_raw, test_raw, train_processed, test_processed,
            target, CKD_NUMERIC, CKD_CATEGORICAL,
            metadata={
                "split_strategy": "StratifiedShuffleSplit", "n_splits": 5,
                "test_size": 0.2, "split_seed": 42,
                "preprocessing": {
                    "numeric_imputation": "training-split median",
                    "categorical_imputation": "training-split mode",
                    "numeric_fill_values": numeric_fill,
                    "categorical_fill_values": categorical_fill,
                },
            },
            train_indices=tr_idx, test_indices=te_idx,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pima", default=str(REPOSITORY_ROOT / "data/raw/pima/pima.csv"))
    p.add_argument("--cleveland", default=str(REPOSITORY_ROOT / "data/raw/cleveland/Heart_disease_cleveland_new.csv"))
    p.add_argument("--ckd", default=str(REPOSITORY_ROOT / "data/processed/ckd/kidney_disease_sanitized.csv"))
    p.add_argument("--out_root", default=str(BENCHMARK_ROOT / "frozen_data"))
    args = p.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prepare_pima(Path(args.pima), out_root)
    prepare_cleveland(Path(args.cleveland), out_root)
    prepare_ckd(Path(args.ckd), out_root)
    print(f"Frozen datasets written to {out_root}")


if __name__ == "__main__":
    main()
