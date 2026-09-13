#!/usr/bin/env python3
"""
Finalize the 20,000-row Diabetes 130-US GReaT release using a transparent,
frozen schema-projection policy.

Policy
------
1. Diagnosis columns:
   Out-of-schema values in diag_1, diag_2, and diag_3 are mapped to the
   predefined reduced-schema token `__RARE__`.

2. Recoverable categorical field spillover:
   For A1Cresult, max_glu_serum, age, and race only, an out-of-schema value is
   replaced by a training-valid category when that category is the unique
   longest prefix of the generated string. Examples:
       "__MISSING__MISSING__..." -> "__MISSING__"
       "[60-70) diabetesMed"     -> "[60-70)"
   This addresses serialization spillover without guessing a category.

3. Numeric support:
   Parseable values outside the real training support are projected to the
   corresponding training minimum or maximum. Every affected cell is logged.

4. Irrecoverable rows:
   Rows with an invalid target, an unparsable numeric value, or any remaining
   out-of-schema categorical value are rejected rather than imputed.

The raw synthetic release is never overwritten.

Example
-------
python finalize_diabetes130_great_release.py \
  --real_train_csv results/diabetes130/llm_pilot_data/reduced/train.csv \
  --synthetic_csv results/diabetes130/great_main_20k/synthetic_raw_20000.csv \
  --out_dir results/diabetes130/great_main_20k/final_release
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET = "readmitted_30d"
MISSING_TOKEN = "__MISSING__"
RARE_TOKEN = "__RARE__"

DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# Prefix recovery is deliberately restricted to columns for which the raw
# 20K diagnostic output demonstrated field-spillover strings beginning with a
# valid category. It is not applied generically to all categorical columns.
PREFIX_RECOVERY_COLUMNS = [
    "A1Cresult",
    "max_glu_serum",
    "age",
    "race",
]

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
        default=str(REPOSITORY_ROOT / "results/diabetes130/great_main_20k/final_release"),
    )
    return parser.parse_args()


def canonical_categorical(series: pd.Series) -> pd.Series:
    values = (
        series.astype("string")
        .fillna(MISSING_TOKEN)
        .str.strip()
    )

    # Normalize integer-coded categorical values after CSV round-tripping.
    # Genuine decimal diagnosis codes such as 250.13 remain unchanged.
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

    return normalized.astype("string").agg(
        "\x1f".join,
        axis=1,
    )


def unique_longest_valid_prefix(
    value: str,
    allowed_levels: Sequence[str],
) -> Tuple[str | None, str]:
    matches = [
        level
        for level in allowed_levels
        if level and value.startswith(level)
    ]

    if not matches:
        return None, ""

    maximum_length = max(len(level) for level in matches)
    longest = sorted(
        {
            level
            for level in matches
            if len(level) == maximum_length
        }
    )

    if len(longest) != 1:
        return None, ""

    prefix = longest[0]

    # Exact categories are not repairs.
    if value == prefix:
        return None, ""

    return prefix, value[len(prefix):]


def is_integral_training_feature(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool(
        np.isclose(
            numeric.to_numpy(),
            np.round(numeric.to_numpy()),
        ).all()
    )


def validate_final_release(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    expected_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Dict[str, object]:
    missing_columns = [
        col
        for col in expected_columns
        if col not in synthetic.columns
    ]
    extra_columns = [
        col
        for col in synthetic.columns
        if col not in expected_columns
    ]

    target = pd.to_numeric(
        synthetic[TARGET],
        errors="coerce",
    )
    target_invalid = int(
        (~target.isin([0, 1])).sum()
    )

    numeric_invalid_cells = 0
    numeric_range_violation_cells = 0
    numeric_details: List[Dict[str, object]] = []

    for col in NUMERIC_FEATURES:
        real_numeric = pd.to_numeric(
            real[col],
            errors="coerce",
        )
        generated = pd.to_numeric(
            synthetic[col],
            errors="coerce",
        )

        invalid = generated.isna()
        out_of_range = (
            generated.notna()
            & (
                (generated < real_numeric.min())
                | (generated > real_numeric.max())
            )
        )

        numeric_invalid_cells += int(invalid.sum())
        numeric_range_violation_cells += int(
            out_of_range.sum()
        )

        numeric_details.append(
            {
                "feature": col,
                "invalid_n": int(invalid.sum()),
                "range_violation_n": int(
                    out_of_range.sum()
                ),
                "real_min": float(real_numeric.min()),
                "real_max": float(real_numeric.max()),
                "synthetic_min": (
                    float(generated.min())
                    if generated.notna().any()
                    else None
                ),
                "synthetic_max": (
                    float(generated.max())
                    if generated.notna().any()
                    else None
                ),
            }
        )

    categorical_unseen_cells = 0
    categorical_details: List[Dict[str, object]] = []

    for col in categorical_columns:
        allowed = set(
            canonical_categorical(real[col]).tolist()
        )
        generated = canonical_categorical(
            synthetic[col]
        )
        unseen = ~generated.isin(allowed)

        categorical_unseen_cells += int(unseen.sum())
        categorical_details.append(
            {
                "feature": col,
                "unseen_n": int(unseen.sum()),
                "unseen_pct": float(
                    unseen.mean() * 100
                ),
            }
        )

    exact_duplicate_rate = np.nan
    internal_duplicate_rate = np.nan
    exact_duplicate_n = 0
    internal_duplicate_row_n = 0

    if not missing_columns and not extra_columns:
        real_keys = set(
            canonical_row_keys(real, expected_columns)
        )
        synthetic_keys = canonical_row_keys(
            synthetic[list(expected_columns)],
            expected_columns,
        )

        exact_mask = synthetic_keys.isin(real_keys)
        internal_mask = synthetic_keys.duplicated(
            keep=False
        )

        exact_duplicate_n = int(exact_mask.sum())
        internal_duplicate_row_n = int(
            internal_mask.sum()
        )
        exact_duplicate_rate = float(
            exact_mask.mean()
        )
        internal_duplicate_rate = float(
            internal_mask.mean()
        )

    summary = {
        "rows": len(synthetic),
        "expected_columns": len(expected_columns),
        "returned_columns": len(synthetic.columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "target_invalid_n": target_invalid,
        "numeric_invalid_cells": numeric_invalid_cells,
        "numeric_range_violation_cells": (
            numeric_range_violation_cells
        ),
        "categorical_unseen_cells": (
            categorical_unseen_cells
        ),
        "exact_duplicate_n_vs_full_real_train": (
            exact_duplicate_n
        ),
        "exact_duplicate_rate_vs_full_real_train": (
            exact_duplicate_rate
        ),
        "internal_duplicate_row_n": (
            internal_duplicate_row_n
        ),
        "internal_duplicate_rate": (
            internal_duplicate_rate
        ),
        "positive_rate": (
            float(target.mean())
            if len(target) and target.isin([0, 1]).all()
            else None
        ),
        "pass_column_set": (
            not missing_columns and not extra_columns
        ),
        "pass_binary_target": target_invalid == 0,
        "pass_numeric_parseability": (
            numeric_invalid_cells == 0
        ),
        "pass_numeric_training_ranges": (
            numeric_range_violation_cells == 0
        ),
        "pass_closed_categorical_schema": (
            categorical_unseen_cells == 0
        ),
        "numeric_details": numeric_details,
        "categorical_details": categorical_details,
    }

    summary["overall_schema_pass"] = all(
        [
            summary["pass_column_set"],
            summary["pass_binary_target"],
            summary["pass_numeric_parseability"],
            summary["pass_numeric_training_ranges"],
            summary["pass_closed_categorical_schema"],
        ]
    )

    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(args.real_train_csv)
    raw = pd.read_csv(args.synthetic_csv)

    expected_columns = real.columns.tolist()

    missing_columns = [
        col for col in expected_columns
        if col not in raw.columns
    ]
    extra_columns = [
        col for col in raw.columns
        if col not in expected_columns
    ]

    if missing_columns or extra_columns:
        raise ValueError(
            "Raw synthetic column set differs from the frozen real "
            f"training schema. Missing={missing_columns}; "
            f"extra={extra_columns}"
        )

    raw = raw[expected_columns].copy()

    categorical_columns = [
        col
        for col in expected_columns
        if col not in NUMERIC_FEATURES + [TARGET]
    ]

    allowed_levels: Dict[str, List[str]] = {
        col: sorted(
            set(
                canonical_categorical(
                    real[col]
                ).tolist()
            ),
            key=lambda value: (
                -len(value),
                value,
            ),
        )
        for col in categorical_columns
    }

    projected = raw.copy()

    # Store all categorical features as pandas strings before textual
    # replacements. This avoids incompatible-dtype assignments.
    for col in categorical_columns:
        projected[col] = canonical_categorical(
            projected[col]
        ).astype("string")

    diagnosis_log: List[Dict[str, object]] = []
    prefix_log: List[Dict[str, object]] = []
    numeric_log: List[Dict[str, object]] = []
    rejection_log: List[Dict[str, object]] = []

    diagnosis_repaired_rows: set[int] = set()
    prefix_repaired_rows: set[int] = set()
    numerically_clipped_rows: set[int] = set()

    # ------------------------------------------------------------------
    # Stage 1: diagnosis-code projection to the frozen __RARE__ category.
    # ------------------------------------------------------------------
    for col in DIAGNOSIS_COLUMNS:
        if RARE_TOKEN not in allowed_levels[col]:
            raise ValueError(
                f"{RARE_TOKEN} is not present in the real training "
                f"schema for {col}."
            )

        generated = canonical_categorical(
            projected[col]
        )
        allowed = set(allowed_levels[col])
        unseen = ~generated.isin(allowed)

        for row_index in projected.index[unseen]:
            original = str(
                generated.loc[row_index]
            )
            diagnosis_log.append(
                {
                    "row_index": int(row_index),
                    "feature": col,
                    "original_value": original,
                    "replacement_value": RARE_TOKEN,
                    "target": int(
                        pd.to_numeric(
                            projected.loc[row_index, TARGET],
                            errors="coerce",
                        )
                    ),
                }
            )
            diagnosis_repaired_rows.add(
                int(row_index)
            )

        projected.loc[unseen, col] = RARE_TOKEN

    # ------------------------------------------------------------------
    # Stage 2: unique valid-prefix recovery for documented field spillover.
    # ------------------------------------------------------------------
    for col in PREFIX_RECOVERY_COLUMNS:
        generated = canonical_categorical(
            projected[col]
        )
        allowed = set(allowed_levels[col])
        unseen_indices = generated.index[
            ~generated.isin(allowed)
        ]

        for row_index in unseen_indices:
            original = str(
                generated.loc[row_index]
            )
            prefix, suffix = (
                unique_longest_valid_prefix(
                    original,
                    allowed_levels[col],
                )
            )

            if prefix is None:
                continue

            projected.loc[row_index, col] = prefix
            prefix_repaired_rows.add(int(row_index))
            prefix_log.append(
                {
                    "row_index": int(row_index),
                    "feature": col,
                    "original_value": original,
                    "replacement_value": prefix,
                    "removed_suffix": suffix,
                    "target": int(
                        pd.to_numeric(
                            projected.loc[row_index, TARGET],
                            errors="coerce",
                        )
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Stage 3: project parseable numerical values onto real-training support.
    # ------------------------------------------------------------------
    irrecoverable_rows: set[int] = set()

    for col in NUMERIC_FEATURES:
        real_numeric = pd.to_numeric(
            real[col],
            errors="coerce",
        )
        generated = pd.to_numeric(
            projected[col],
            errors="coerce",
        )

        parse_failure = generated.isna()
        for row_index in projected.index[
            parse_failure
        ]:
            irrecoverable_rows.add(int(row_index))
            rejection_log.append(
                {
                    "row_index": int(row_index),
                    "reason": "unparsable_numeric_value",
                    "feature": col,
                    "value": str(
                        projected.loc[row_index, col]
                    ),
                }
            )

        lower = float(real_numeric.min())
        upper = float(real_numeric.max())

        low_mask = generated.notna() & (
            generated < lower
        )
        high_mask = generated.notna() & (
            generated > upper
        )

        for row_index in projected.index[
            low_mask | high_mask
        ]:
            original = float(
                generated.loc[row_index]
            )
            replacement = float(
                np.clip(original, lower, upper)
            )

            numeric_log.append(
                {
                    "row_index": int(row_index),
                    "feature": col,
                    "original_value": original,
                    "replacement_value": replacement,
                    "real_train_min": lower,
                    "real_train_max": upper,
                    "target": int(
                        pd.to_numeric(
                            projected.loc[row_index, TARGET],
                            errors="coerce",
                        )
                    ),
                }
            )
            numerically_clipped_rows.add(
                int(row_index)
            )
            generated.loc[row_index] = replacement

        if is_integral_training_feature(
            real[col]
        ):
            projected[col] = (
                generated.round()
                .astype("Int64")
            )
        else:
            projected[col] = generated

    # ------------------------------------------------------------------
    # Stage 4: reject any remaining schema-invalid rows instead of imputing.
    # ------------------------------------------------------------------
    target_numeric = pd.to_numeric(
        projected[TARGET],
        errors="coerce",
    )
    invalid_target = ~target_numeric.isin([0, 1])

    for row_index in projected.index[
        invalid_target
    ]:
        irrecoverable_rows.add(int(row_index))
        rejection_log.append(
            {
                "row_index": int(row_index),
                "reason": "invalid_target",
                "feature": TARGET,
                "value": str(
                    projected.loc[row_index, TARGET]
                ),
            }
        )

    projected[TARGET] = (
        target_numeric.round().astype("Int64")
    )

    for col in categorical_columns:
        generated = canonical_categorical(
            projected[col]
        )
        allowed = set(allowed_levels[col])
        unseen = ~generated.isin(allowed)

        for row_index in projected.index[unseen]:
            irrecoverable_rows.add(int(row_index))
            rejection_log.append(
                {
                    "row_index": int(row_index),
                    "reason": (
                        "remaining_out_of_schema_category"
                    ),
                    "feature": col,
                    "value": str(
                        generated.loc[row_index]
                    ),
                }
            )

    rejected_indices = sorted(
        irrecoverable_rows
    )
    final = projected.drop(
        index=rejected_indices
    ).reset_index(drop=True)

    validation = validate_final_release(
        real=real,
        synthetic=final,
        expected_columns=expected_columns,
        categorical_columns=categorical_columns,
    )

    # Aggregate repair counts by feature for the audit.
    diagnosis_by_feature = dict(
        Counter(
            row["feature"]
            for row in diagnosis_log
        )
    )
    prefix_by_feature = dict(
        Counter(
            row["feature"]
            for row in prefix_log
        )
    )
    numeric_by_feature = dict(
        Counter(
            row["feature"]
            for row in numeric_log
        )
    )
    rejection_by_reason = dict(
        Counter(
            row["reason"]
            for row in rejection_log
        )
    )

    raw_rows = len(raw)
    final_rows = len(final)

    audit = {
        "policy": {
            "diagnosis_projection": (
                "Out-of-schema diag_1/diag_2/diag_3 "
                "values -> __RARE__."
            ),
            "field_spillover_recovery": (
                "Unique longest training-valid prefix for "
                "A1Cresult, max_glu_serum, age and race."
            ),
            "numeric_support_projection": (
                "Parseable out-of-range values clipped to "
                "real-training minimum or maximum."
            ),
            "irrecoverable_rows": (
                "Rejected; no mode or arbitrary category "
                "imputation."
            ),
        },
        "raw_rows": raw_rows,
        "final_rows": final_rows,
        "retained_row_rate": float(
            final_rows / raw_rows
        ),
        "rejected_rows": len(
            rejected_indices
        ),
        "rejected_row_rate": float(
            len(rejected_indices) / raw_rows
        ),
        "rejected_original_row_indices": (
            rejected_indices
        ),
        "diagnosis_repaired_cells": len(
            diagnosis_log
        ),
        "diagnosis_repaired_rows": len(
            diagnosis_repaired_rows
        ),
        "diagnosis_repairs_by_feature": (
            diagnosis_by_feature
        ),
        "prefix_recovered_cells": len(
            prefix_log
        ),
        "prefix_recovered_rows": len(
            prefix_repaired_rows
        ),
        "prefix_repairs_by_feature": (
            prefix_by_feature
        ),
        "numeric_clipped_cells": len(
            numeric_log
        ),
        "numeric_clipped_rows": len(
            numerically_clipped_rows
        ),
        "numeric_clips_by_feature": (
            numeric_by_feature
        ),
        "rejections_by_reason": (
            rejection_by_reason
        ),
        "final_validation": validation,
    }

    final.to_csv(
        out_dir / "synthetic_final_valid.csv",
        index=False,
    )
    pd.DataFrame(diagnosis_log).to_csv(
        out_dir / "diagnosis_projection_log.csv",
        index=False,
    )
    pd.DataFrame(prefix_log).to_csv(
        out_dir / "categorical_prefix_recovery_log.csv",
        index=False,
    )
    pd.DataFrame(numeric_log).to_csv(
        out_dir / "numeric_support_projection_log.csv",
        index=False,
    )
    pd.DataFrame(rejection_log).to_csv(
        out_dir / "row_rejection_log.csv",
        index=False,
    )

    (
        out_dir / "final_release_audit.json"
    ).write_text(json.dumps(audit, indent=2))

    print(json.dumps(audit, indent=2))
    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
