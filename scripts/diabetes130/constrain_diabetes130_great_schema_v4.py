#!/usr/bin/env python3
"""
Audit and apply the frozen reduced-schema policy to a Diabetes 130-US GReaT
release.

This v4 script fixes the repaired-row counter and produces exact diagnostics
for every remaining out-of-schema categorical value and numerical range
violation.

Current repair policy
---------------------
1. Only `diag_1`, `diag_2`, and `diag_3` are automatically repaired.
2. Any diagnosis value outside the frozen reduced real-training schema is
   mapped to the already defined `__RARE__` token.
3. No other categorical value is silently repaired.
4. Numerical values are not clipped; violations are reported with row indices.

Example
-------
python constrain_diabetes130_great_schema_v4.py \
  --real_train_csv results/diabetes130/llm_pilot_data/reduced/train.csv \
  --synthetic_csv results/diabetes130/great_main_20k/synthetic_raw_20000.csv \
  --out_dir results/diabetes130/great_main_20k/schema_constrained_v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET = "readmitted_30d"
MISSING_TOKEN = "__MISSING__"
RARE_TOKEN = "__RARE__"

DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]

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
        default=str(REPOSITORY_ROOT / "results/diabetes130/great_main_20k/schema_constrained_v4"),
    )
    return parser.parse_args()


def canonical_categorical(series: pd.Series) -> pd.Series:
    values = (
        series.astype("string")
        .fillna(MISSING_TOKEN)
        .str.strip()
    )

    # Preserve genuine decimal diagnosis codes such as 250.13, while
    # normalising CSV round-trips such as 3.0 -> 3.
    return values.str.replace(
        r"^(-?\d+)\.0$",
        r"\1",
        regex=True,
    )


def canonical_row_keys(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.Series:
    normalized = pd.DataFrame(index=df.index)

    for col in columns:
        if col == TARGET:
            normalized[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .round()
                .astype("Int64")
                .astype("string")
            )
        elif col in NUMERIC_FEATURES:
            normalized[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .round(8)
                .astype("string")
            )
        else:
            normalized[col] = canonical_categorical(df[col])

    return normalized.astype("string").agg("\x1f".join, axis=1)


def evaluate(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> Tuple[
    Dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
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
                    "invalid_pct": float(invalid.mean() * 100),
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
            out_of_range = (
                generated.notna()
                & (
                    (generated < real_min)
                    | (generated > real_max)
                )
            )

            for row_index in synthetic.index[out_of_range]:
                range_rows.append(
                    {
                        "row_index": int(row_index),
                        "feature": col,
                        "generated_value": float(
                            generated.loc[row_index]
                        ),
                        "real_train_min": real_min,
                        "real_train_max": real_max,
                        "target": (
                            int(
                                pd.to_numeric(
                                    synthetic.loc[row_index, TARGET],
                                    errors="coerce",
                                )
                            )
                            if TARGET in synthetic.columns
                            and pd.notna(
                                pd.to_numeric(
                                    synthetic.loc[row_index, TARGET],
                                    errors="coerce",
                                )
                            )
                            else np.nan
                        ),
                    }
                )

            column_rows.append(
                {
                    "feature": col,
                    "kind": "numeric",
                    "invalid_n": int(invalid.sum()),
                    "invalid_pct": float(invalid.mean() * 100),
                    "unseen_n": 0,
                    "unseen_pct": 0.0,
                    "range_violation_n": int(out_of_range.sum()),
                    "range_violation_pct": float(
                        out_of_range.mean() * 100
                    ),
                }
            )
            continue

        allowed = set(
            canonical_categorical(real[col]).tolist()
        )
        generated = canonical_categorical(synthetic[col])
        unseen = ~generated.isin(allowed)

        if unseen.any():
            counts = generated[unseen].value_counts(
                dropna=False
            )
            for value, count in counts.items():
                example_indices = (
                    generated.index[
                        unseen & generated.eq(value)
                    ]
                    .tolist()[:10]
                )
                unseen_rows.append(
                    {
                        "feature": col,
                        "generated_value": str(value),
                        "count": int(count),
                        "pct_of_synthetic_rows": float(
                            count / len(synthetic) * 100
                        ),
                        "example_row_indices": ",".join(
                            str(int(i)) for i in example_indices
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
                "unseen_pct": float(unseen.mean() * 100),
                "range_violation_n": 0,
                "range_violation_pct": 0.0,
            }
        )

    columns = pd.DataFrame(column_rows)
    unseen_values = pd.DataFrame(unseen_rows)
    range_violations = pd.DataFrame(range_rows)

    exact_duplicate_rate = np.nan
    internal_duplicate_rate = np.nan

    if not missing_columns:
        aligned = synthetic[expected_columns].copy()
        real_keys = set(
            canonical_row_keys(real, expected_columns)
        )
        synthetic_keys = canonical_row_keys(
            aligned,
            expected_columns,
        )
        exact_duplicate_rate = float(
            synthetic_keys.isin(real_keys).mean()
        )
        internal_duplicate_rate = float(
            synthetic_keys.duplicated(keep=False).mean()
        )

    categorical_mask = columns["kind"] == "categorical"
    numeric_mask = columns["kind"] == "numeric"

    categorical_feature_count = int(categorical_mask.sum())
    numeric_feature_count = int(numeric_mask.sum())

    categorical_unseen_cells = int(
        columns.loc[categorical_mask, "unseen_n"].sum()
    )
    numeric_invalid_cells = int(
        columns.loc[numeric_mask, "invalid_n"].sum()
    )
    numeric_range_violation_cells = int(
        columns.loc[
            numeric_mask,
            "range_violation_n",
        ].sum()
    )

    summary = {
        "real_train_rows": len(real),
        "synthetic_rows": len(synthetic),
        "expected_columns": len(expected_columns),
        "returned_columns": len(synthetic.columns),
        "categorical_feature_count": categorical_feature_count,
        "numeric_feature_count": numeric_feature_count,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "target_invalid_n": int(
            columns.loc[
                columns["feature"] == TARGET,
                "invalid_n",
            ].sum()
        ),
        "numeric_invalid_cells": numeric_invalid_cells,
        "numeric_invalid_cell_pct": float(
            numeric_invalid_cells
            / max(1, len(synthetic) * numeric_feature_count)
            * 100
        ),
        "numeric_range_violation_cells": (
            numeric_range_violation_cells
        ),
        "numeric_range_violation_cell_pct": float(
            numeric_range_violation_cells
            / max(1, len(synthetic) * numeric_feature_count)
            * 100
        ),
        "categorical_unseen_cells": categorical_unseen_cells,
        "categorical_unseen_cell_pct": float(
            categorical_unseen_cells
            / max(
                1,
                len(synthetic) * categorical_feature_count,
            )
            * 100
        ),
        "categorical_columns_with_unseen_values": int(
            (
                columns.loc[
                    categorical_mask,
                    "unseen_n",
                ] > 0
            ).sum()
        ),
        "exact_duplicate_rate_vs_full_real_train": (
            exact_duplicate_rate
        ),
        "internal_duplicate_rate": internal_duplicate_rate,
    }

    summary["pass_column_set"] = (
        not missing_columns and not extra_columns
    )
    summary["pass_binary_target"] = (
        summary["target_invalid_n"] == 0
    )
    summary["pass_numeric_parseability"] = (
        numeric_invalid_cells == 0
    )
    summary["pass_numeric_training_ranges"] = (
        numeric_range_violation_cells == 0
    )
    summary["pass_closed_categorical_schema"] = (
        categorical_unseen_cells == 0
    )
    summary["pass_no_exact_copy"] = (
        pd.notna(exact_duplicate_rate)
        and exact_duplicate_rate == 0.0
    )

    return (
        summary,
        columns,
        unseen_values,
        range_violations,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(args.real_train_csv)
    raw = pd.read_csv(args.synthetic_csv)

    (
        raw_summary,
        raw_columns,
        raw_unseen,
        raw_ranges,
    ) = evaluate(real, raw)

    repaired = raw.copy()

    # Cast diagnosis columns before inserting textual tokens.
    for col in DIAGNOSIS_COLUMNS:
        repaired[col] = canonical_categorical(
            repaired[col]
        ).astype("string")

    repaired_row_mask = pd.Series(
        False,
        index=repaired.index,
        dtype=bool,
    )
    repair_rows: List[Dict[str, object]] = []
    repaired_cell_count = 0

    for col in DIAGNOSIS_COLUMNS:
        allowed = set(
            canonical_categorical(real[col]).tolist()
        )
        generated = canonical_categorical(repaired[col])
        unseen = ~generated.isin(allowed)

        if not unseen.any():
            continue

        repaired_row_mask.loc[unseen] = True
        repaired_cell_count += int(unseen.sum())

        counts = generated[unseen].value_counts(
            dropna=False
        )
        for value, count in counts.items():
            repair_rows.append(
                {
                    "feature": col,
                    "original_value": str(value),
                    "replacement_value": RARE_TOKEN,
                    "repaired_cells": int(count),
                    "rule": (
                        "Out-of-schema diagnosis value mapped "
                        "to frozen reduced-schema __RARE__."
                    ),
                }
            )

        repaired.loc[unseen, col] = RARE_TOKEN

    (
        repaired_summary,
        repaired_columns,
        repaired_unseen,
        repaired_ranges,
    ) = evaluate(real, repaired)

    if repaired_unseen.empty:
        non_diag_unseen = pd.DataFrame(
            columns=[
                "feature",
                "generated_value",
                "count",
                "pct_of_synthetic_rows",
                "example_row_indices",
            ]
        )
    else:
        non_diag_unseen = repaired_unseen.loc[
            ~repaired_unseen["feature"].isin(
                DIAGNOSIS_COLUMNS
            )
        ].copy()

    non_diag_unseen_cells = (
        int(non_diag_unseen["count"].sum())
        if not non_diag_unseen.empty
        else 0
    )

    audit = {
        "policy": (
            "Only out-of-schema diagnosis values are mapped "
            "to __RARE__; all other violations are reported."
        ),
        "diagnosis_columns": DIAGNOSIS_COLUMNS,
        "raw_validation": raw_summary,
        "diagnosis_constrained_validation": repaired_summary,
        "repaired_cell_count": repaired_cell_count,
        "repaired_cell_pct_of_all_table_cells": float(
            repaired_cell_count
            / max(1, len(repaired) * len(repaired.columns))
            * 100
        ),
        "rows_with_at_least_one_diagnosis_repair": int(
            repaired_row_mask.sum()
        ),
        "non_diagnosis_unseen_cells_after_diagnosis_repair": (
            non_diag_unseen_cells
        ),
        "numeric_range_violation_cells_after_diagnosis_repair": (
            int(
                repaired_summary[
                    "numeric_range_violation_cells"
                ]
            )
        ),
        "overall_schema_pass_after_diagnosis_repair": all(
            [
                repaired_summary["pass_column_set"],
                repaired_summary["pass_binary_target"],
                repaired_summary[
                    "pass_numeric_parseability"
                ],
                repaired_summary[
                    "pass_numeric_training_ranges"
                ],
                repaired_summary[
                    "pass_closed_categorical_schema"
                ],
            ]
        ),
    }

    repaired.to_csv(
        out_dir / "synthetic_diagnosis_constrained.csv",
        index=False,
    )
    pd.DataFrame(repair_rows).to_csv(
        out_dir / "diagnosis_repair_log.csv",
        index=False,
    )

    raw_columns.to_csv(
        out_dir / "column_validation_raw.csv",
        index=False,
    )
    repaired_columns.to_csv(
        out_dir / "column_validation_after_diagnosis_repair.csv",
        index=False,
    )

    raw_unseen.to_csv(
        out_dir / "unseen_values_raw.csv",
        index=False,
    )
    repaired_unseen.to_csv(
        out_dir / "unseen_values_after_diagnosis_repair.csv",
        index=False,
    )
    non_diag_unseen.to_csv(
        out_dir / "non_diagnosis_unseen_values.csv",
        index=False,
    )

    raw_ranges.to_csv(
        out_dir / "numeric_range_violations_raw.csv",
        index=False,
    )
    repaired_ranges.to_csv(
        out_dir / "numeric_range_violations_after_diagnosis_repair.csv",
        index=False,
    )

    (
        out_dir / "schema_diagnostic_audit.json"
    ).write_text(json.dumps(audit, indent=2))

    print(json.dumps(audit, indent=2))

    print("\nRemaining non-diagnosis unseen values:")
    if non_diag_unseen.empty:
        print("None")
    else:
        print(
            non_diag_unseen.sort_values(
                ["feature", "count"],
                ascending=[True, False],
            ).to_string(index=False)
        )

    print("\nNumeric range violations:")
    if repaired_ranges.empty:
        print("None")
    else:
        print(repaired_ranges.to_string(index=False))

    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
