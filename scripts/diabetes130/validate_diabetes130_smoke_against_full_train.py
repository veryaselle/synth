#!/usr/bin/env python3
"""
Validate a Diabetes 130-US GReaT smoke release against the complete frozen
reduced real-training schema rather than only the 2,000-row smoke subset.

This distinction matters because a category absent from the smoke subset may
still be valid in the full 79,473-row real training partition.

Example:
    python validate_diabetes130_smoke_against_full_train.py \
      --real_train_csv results/diabetes130/llm_pilot_data/reduced/train.csv \
      --synthetic_csv results/diabetes130/great_smoke_v2/synthetic_smoke.csv \
      --out_dir results/diabetes130/great_smoke_v2/full_schema_validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


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
    parser.add_argument("--real_train_csv", required=True)
    parser.add_argument("--synthetic_csv", required=True)
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/great_smoke_v2/full_schema_validation"),
    )
    return parser.parse_args()


def canonical_categorical(series: pd.Series) -> pd.Series:
    values = (
        series.astype("string")
        .fillna(MISSING_TOKEN)
        .str.strip()
    )

    # CSV round-tripping may turn integer-coded categories into strings such as
    # "1.0". Strip only a terminal ".0"; diagnosis codes such as "250.13"
    # remain unchanged.
    values = values.str.replace(
        r"^(-?\d+)\.0$",
        r"\1",
        regex=True,
    )
    return values


def canonical_row_keys(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.Series:
    normalised = pd.DataFrame(index=df.index)

    for col in columns:
        if col == TARGET:
            normalised[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .round()
                .astype("Int64")
                .astype("string")
            )
        elif col in NUMERIC_FEATURES:
            normalised[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .round(8)
                .astype("string")
            )
        else:
            normalised[col] = canonical_categorical(df[col])

    return normalised.astype("string").agg(
        "\x1f".join,
        axis=1,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(args.real_train_csv)
    synthetic = pd.read_csv(args.synthetic_csv)

    expected_columns = real.columns.tolist()
    missing_columns = [
        col for col in expected_columns
        if col not in synthetic.columns
    ]
    extra_columns = [
        col for col in synthetic.columns
        if col not in expected_columns
    ]

    column_rows: List[Dict[str, object]] = []
    unseen_rows: List[Dict[str, object]] = []
    range_rows: List[Dict[str, object]] = []

    for col in expected_columns:
        if col not in synthetic.columns:
            column_rows.append(
                {
                    "feature": col,
                    "kind": "missing_column",
                    "invalid_n": len(synthetic),
                    "invalid_pct": 100.0,
                    "unseen_n": np.nan,
                    "unseen_pct": np.nan,
                    "range_violation_n": np.nan,
                    "range_violation_pct": np.nan,
                }
            )
            continue

        if col == TARGET:
            generated = pd.to_numeric(
                synthetic[col],
                errors="coerce",
            )
            invalid = ~generated.isin([0, 1])
            column_rows.append(
                {
                    "feature": col,
                    "kind": "binary_target",
                    "invalid_n": int(invalid.sum()),
                    "invalid_pct": float(
                        invalid.mean() * 100
                    ),
                    "unseen_n": 0,
                    "unseen_pct": 0.0,
                    "range_violation_n": 0,
                    "range_violation_pct": 0.0,
                }
            )
            continue

        if col in NUMERIC_FEATURES:
            real_numeric = pd.to_numeric(
                real[col],
                errors="coerce",
            )
            generated = pd.to_numeric(
                synthetic[col],
                errors="coerce",
            )
            invalid = generated.isna()

            real_min = float(real_numeric.min())
            real_max = float(real_numeric.max())
            range_violation = (
                generated.notna()
                & (
                    (generated < real_min)
                    | (generated > real_max)
                )
            )

            range_rows.append(
                {
                    "feature": col,
                    "real_min": real_min,
                    "real_max": real_max,
                    "synthetic_min": (
                        float(generated.min())
                        if generated.notna().any()
                        else np.nan
                    ),
                    "synthetic_max": (
                        float(generated.max())
                        if generated.notna().any()
                        else np.nan
                    ),
                    "range_violation_n": int(
                        range_violation.sum()
                    ),
                    "range_violation_pct": float(
                        range_violation.mean() * 100
                    ),
                }
            )

            column_rows.append(
                {
                    "feature": col,
                    "kind": "numeric",
                    "invalid_n": int(invalid.sum()),
                    "invalid_pct": float(
                        invalid.mean() * 100
                    ),
                    "unseen_n": 0,
                    "unseen_pct": 0.0,
                    "range_violation_n": int(
                        range_violation.sum()
                    ),
                    "range_violation_pct": float(
                        range_violation.mean() * 100
                    ),
                }
            )
            continue

        real_tokens = canonical_categorical(real[col])
        generated_tokens = canonical_categorical(
            synthetic[col]
        )
        valid_levels = set(real_tokens.tolist())
        unseen = ~generated_tokens.isin(valid_levels)

        if unseen.any():
            counts = (
                generated_tokens[unseen]
                .value_counts(dropna=False)
            )
            for value, count in counts.items():
                unseen_rows.append(
                    {
                        "feature": col,
                        "generated_value": str(value),
                        "count": int(count),
                        "pct_of_synthetic_rows": float(
                            count / len(synthetic) * 100
                        ),
                    }
                )

        column_rows.append(
            {
                "feature": col,
                "kind": "categorical",
                "invalid_n": 0,
                "invalid_pct": 0.0,
                "unseen_n": int(unseen.sum()),
                "unseen_pct": float(
                    unseen.mean() * 100
                ),
                "range_violation_n": 0,
                "range_violation_pct": 0.0,
            }
        )

    column_validation = pd.DataFrame(column_rows)
    unseen_values = pd.DataFrame(unseen_rows)
    numeric_ranges = pd.DataFrame(range_rows)

    exact_duplicate_rate = np.nan
    internal_duplicate_rate = np.nan

    if not missing_columns:
        comparable = synthetic[
            expected_columns
        ].copy()
        real_keys = set(
            canonical_row_keys(real, expected_columns)
        )
        synthetic_keys = canonical_row_keys(
            comparable,
            expected_columns,
        )
        exact_duplicate_rate = float(
            synthetic_keys.isin(real_keys).mean()
        )
        internal_duplicate_rate = float(
            synthetic_keys.duplicated(
                keep=False
            ).mean()
        )

    categorical_mask = (
        column_validation["kind"] == "categorical"
    )
    numeric_mask = (
        column_validation["kind"] == "numeric"
    )

    summary = {
        "real_train_rows": len(real),
        "synthetic_rows": len(synthetic),
        "expected_columns": len(expected_columns),
        "returned_columns": len(synthetic.columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "target_invalid_n": int(
            column_validation.loc[
                column_validation["feature"] == TARGET,
                "invalid_n",
            ].sum()
        ),
        "numeric_invalid_cells": int(
            column_validation.loc[
                numeric_mask,
                "invalid_n",
            ].sum()
        ),
        "numeric_range_violation_cells": int(
            column_validation.loc[
                numeric_mask,
                "range_violation_n",
            ].sum()
        ),
        "categorical_unseen_cells": int(
            column_validation.loc[
                categorical_mask,
                "unseen_n",
            ].sum()
        ),
        "categorical_unseen_cell_pct": float(
            column_validation.loc[
                categorical_mask,
                "unseen_n",
            ].sum()
            / max(
                1,
                len(synthetic)
                * int(categorical_mask.sum()),
            )
            * 100
        ),
        "categorical_columns_with_unseen_values": int(
            (
                column_validation.loc[
                    categorical_mask,
                    "unseen_n",
                ]
                > 0
            ).sum()
        ),
        "maximum_categorical_unseen_pct": float(
            column_validation.loc[
                categorical_mask,
                "unseen_pct",
            ].max()
        ),
        "exact_duplicate_rate_vs_full_real_train": (
            exact_duplicate_rate
        ),
        "internal_duplicate_rate": (
            internal_duplicate_rate
        ),
    }

    summary["pass_column_set"] = (
        not missing_columns
        and not extra_columns
    )
    summary["pass_binary_target"] = (
        summary["target_invalid_n"] == 0
    )
    summary["pass_numeric_parseability"] = (
        summary["numeric_invalid_cells"] == 0
    )
    summary["pass_closed_categorical_schema"] = (
        summary["categorical_unseen_cells"] == 0
    )
    summary["pass_numeric_training_ranges"] = (
        summary["numeric_range_violation_cells"] == 0
    )
    summary["pass_no_exact_copy"] = (
        pd.notna(exact_duplicate_rate)
        and exact_duplicate_rate == 0.0
    )

    (
        out_dir / "full_schema_validation.json"
    ).write_text(json.dumps(summary, indent=2))
    column_validation.to_csv(
        out_dir / "full_schema_column_validation.csv",
        index=False,
    )
    unseen_values.to_csv(
        out_dir / "full_schema_unseen_values.csv",
        index=False,
    )
    numeric_ranges.to_csv(
        out_dir / "full_schema_numeric_ranges.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2))

    if not unseen_values.empty:
        print("\nUnseen categorical values:")
        print(
            unseen_values.sort_values(
                ["feature", "count"],
                ascending=[True, False],
            ).to_string(index=False)
        )

    if not numeric_ranges.empty:
        print("\nNumeric ranges:")
        print(numeric_ranges.to_string(index=False))

    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
