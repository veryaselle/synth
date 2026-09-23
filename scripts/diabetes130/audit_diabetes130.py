#!/usr/bin/env python3
"""
Reproduce the initial audit of the Diabetes 130-US Hospitals dataset.

Usage:
    python audit_diabetes130.py \
      --data_path data/diabetic_data.csv \
      --mapping_path data/IDS_mapping.csv \
      --out_dir results/diabetes130/audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/diabetes130/diabetic_data.csv"))
    parser.add_argument("--mapping_path", default=str(REPOSITORY_ROOT / "data/raw/diabetes130/IDS_mapping.csv"))
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/audit"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"?": pd.NA, "": pd.NA})

    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)

    patient_counts = df["patient_nbr"].value_counts()
    repeated = patient_counts[patient_counts > 1]

    print(f"Encounters: {len(df):,}")
    print(f"Unique patients: {df['patient_nbr'].nunique():,}")
    print(f"Patients with repeated encounters: {len(repeated):,}")
    print(
        "Encounters from repeated patients: "
        f"{df['patient_nbr'].isin(repeated.index).mean() * 100:.2f}%"
    )
    print(
        "Early readmission (<30 days): "
        f"{df['readmitted_30d'].sum():,} "
        f"({df['readmitted_30d'].mean() * 100:.2f}%)"
    )

    missingness = pd.DataFrame({
        "feature": [
            col
            for col in df.columns
            if col not in ["encounter_id", "patient_nbr", "readmitted", "readmitted_30d"]
        ],
    })
    missingness["missing_n"] = [
        int(df[col].isna().sum())
        for col in missingness["feature"]
    ]
    missingness["missing_pct"] = [
        float(df[col].isna().mean() * 100)
        for col in missingness["feature"]
    ]
    missingness["n_unique_nonmissing"] = [
        int(df[col].nunique(dropna=True))
        for col in missingness["feature"]
    ]
    missingness.sort_values(
        "missing_pct",
        ascending=False,
    ).to_csv(
        out_dir / "diabetes130_missingness.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
