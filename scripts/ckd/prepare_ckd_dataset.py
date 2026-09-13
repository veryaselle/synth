#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "age", "bp", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wc", "rc",
]
ORDINAL_CATEGORICAL_FEATURES = ["sg", "al", "su"]
NOMINAL_CATEGORICAL_FEATURES = [
    "rbc", "pc", "pcc", "ba", "htn", "dm",
    "cad", "appet", "pe", "ane",
]
CATEGORICAL_FEATURES = (
    ORDINAL_CATEGORICAL_FEATURES
    + NOMINAL_CATEGORICAL_FEATURES
)
TARGET = "target"


def sanitize_ckd_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

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

    target_map = {"ckd": 1, "notckd": 0}
    unknown = sorted(
        set(df["classification"].dropna().unique())
        - set(target_map)
    )
    if unknown:
        raise ValueError(f"Unknown target values: {unknown}")

    df[TARGET] = df["classification"].map(target_map).astype(int)
    df = df.drop(columns=["id", "classification"])

    # Three isolated values are separated by a large empirical gap from
    # the remaining electrolyte observations and are treated as missing.
    df.loc[pd.to_numeric(df["sod"], errors="coerce") < 100, "sod"] = np.nan
    df.loc[pd.to_numeric(df["pot"], errors="coerce") > 10, "pot"] = np.nan

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object)
        df[col] = df[col].where(pd.notna(df[col]), np.nan)

    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    return df[expected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/ckd/kidney_disease.csv"))
    parser.add_argument(
        "--out_path",
        default=str(REPOSITORY_ROOT / "data/processed/ckd/kidney_disease_sanitized.csv"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.data_path)
    clean = sanitize_ckd_dataframe(raw)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    print(f"Shape: {clean.shape}")
    print(clean["target"].value_counts().sort_index())


if __name__ == "__main__":
    main()
