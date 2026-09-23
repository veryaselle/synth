#!/usr/bin/env python3
"""
Prepare the Diabetes 130-US Hospitals dataset for a controlled local
tabular-LLM pilot.

The script deliberately stops before model training. It freezes:
    - the cohort definition,
    - the binary target,
    - patient-grouped train/validation/test partitions,
    - full and reduced pilot representations,
    - explicit missing and rare-category handling,
    - a 20,000-row training pilot,
    - validation and serialization diagnostics.

Primary target:
    readmitted_30d = 1 if readmitted == "<30", else 0.

Cohort rule:
    Exclude discharge dispositions associated with death or hospice:
    11, 13, 14, 19, 20, 21.

Leakage control:
    encounter_id and patient_nbr are never model features.
    patient_nbr is used only to keep every patient in exactly one partition.

Representations:
    full_raw:
        Keeps 44 features after removing identifiers, the source target,
        weight, examide and citoglipton.

    reduced:
        Additionally removes payer_code and medical_specialty and collapses
        rare diagnosis codes according to counts fitted on the real training
        partition only.

Example:
    python prepare_diabetes130_llm_pilot.py \
      --data_path data/diabetic_data.csv \
      --out_dir results/diabetes130/llm_pilot_data \
      --pilot_rows 20000 \
      --rare_min_count 20 \
      --random_seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ID_COLUMNS = ["encounter_id", "patient_nbr"]
SOURCE_TARGET = "readmitted"
TARGET = "readmitted_30d"

DEATH_OR_HOSPICE_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]

PRIMARY_DROP_COLUMNS = [
    "weight",       # approximately 97% missing
    "examide",      # constant in supplied data
    "citoglipton",  # constant in supplied data
]

REDUCED_ADDITIONAL_DROP_COLUMNS = [
    "payer_code",         # administrative and substantially missing
    "medical_specialty",  # high-cardinality and approximately 49% missing
]

HIGH_CARDINALITY_DIAGNOSIS_COLUMNS = [
    "diag_1",
    "diag_2",
    "diag_3",
]

MISSING_TOKEN = "__MISSING__"
RARE_TOKEN = "__RARE__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/diabetes130/diabetic_data.csv"))
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "data/processed/diabetes130/llm_pilot_data"),
    )
    parser.add_argument("--pilot_rows", type=int, default=20000)
    parser.add_argument("--rare_min_count", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--n_group_folds",
        type=int,
        default=10,
        help=(
            "Ten folds produce approximately 80/10/10 train/validation/test "
            "partitions by assigning fold 0 to test, fold 1 to validation and "
            "the remaining folds to training."
        ),
    )
    return parser.parse_args()


def normalise_source(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    for col in output.select_dtypes(include="object").columns:
        output[col] = output[col].astype("string").str.strip()
        output[col] = output[col].replace(
            {
                "?": pd.NA,
                "": pd.NA,
            }
        )

    output[TARGET] = (
        output[SOURCE_TARGET] == "<30"
    ).astype(int)

    return output


def apply_cohort_rule(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    excluded_mask = df["discharge_disposition_id"].isin(
        DEATH_OR_HOSPICE_DISPOSITION_IDS
    )

    exclusions = (
        df.loc[
            excluded_mask,
            ["discharge_disposition_id", TARGET],
        ]
        .groupby("discharge_disposition_id", as_index=False)
        .agg(
            excluded_encounters=(TARGET, "size"),
            positive_targets=(TARGET, "sum"),
        )
    )
    exclusions["reason"] = (
        "Death or hospice disposition; excluded from the primary "
        "subsequent-readmission cohort."
    )

    eligible = (
        df.loc[~excluded_mask]
        .copy()
        .reset_index(drop=True)
    )

    return eligible, exclusions


def assign_grouped_partitions(
    df: pd.DataFrame,
    n_splits: int,
    random_seed: int,
) -> pd.Series:
    if n_splits < 3:
        raise ValueError("n_group_folds must be at least 3.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    fold_assignment = np.full(
        len(df),
        fill_value=-1,
        dtype=int,
    )

    dummy_x = np.zeros((len(df), 1))
    y = df[TARGET].to_numpy()
    groups = df["patient_nbr"].to_numpy()

    for fold, (_, held_out_idx) in enumerate(
        splitter.split(dummy_x, y, groups)
    ):
        fold_assignment[held_out_idx] = fold

    if (fold_assignment < 0).any():
        raise RuntimeError("At least one row was not assigned to a fold.")

    partition = np.full(
        len(df),
        fill_value="train",
        dtype=object,
    )
    partition[fold_assignment == 0] = "test"
    partition[fold_assignment == 1] = "validation"

    return pd.Series(
        partition,
        index=df.index,
        name="partition",
    )


def validate_group_split(
    manifest: pd.DataFrame,
) -> Dict[str, object]:
    patient_sets = {
        partition: set(
            manifest.loc[
                manifest["partition"] == partition,
                "patient_nbr",
            ].tolist()
        )
        for partition in [
            "train",
            "validation",
            "test",
        ]
    }

    overlaps = {
        "train_validation": len(
            patient_sets["train"]
            & patient_sets["validation"]
        ),
        "train_test": len(
            patient_sets["train"]
            & patient_sets["test"]
        ),
        "validation_test": len(
            patient_sets["validation"]
            & patient_sets["test"]
        ),
    }

    if any(overlaps.values()):
        raise RuntimeError(
            f"Patient leakage detected between partitions: {overlaps}"
        )

    return {
        "patient_overlap_train_validation": overlaps[
            "train_validation"
        ],
        "patient_overlap_train_test": overlaps["train_test"],
        "patient_overlap_validation_test": overlaps[
            "validation_test"
        ],
    }


def explicit_missing_tokens(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    output = df[list(feature_columns) + [TARGET]].copy()

    for col in feature_columns:
        if (
            pd.api.types.is_object_dtype(output[col])
            or pd.api.types.is_string_dtype(output[col])
        ):
            output[col] = (
                output[col]
                .astype("string")
                .fillna(MISSING_TOKEN)
            )

    return output


def fit_rare_levels(
    training_df: pd.DataFrame,
    columns: Sequence[str],
    minimum_count: int,
) -> Dict[str, List[str]]:
    rare_levels: Dict[str, List[str]] = {}

    for col in columns:
        tokens = (
            training_df[col]
            .astype("string")
            .fillna(MISSING_TOKEN)
        )
        counts = tokens.value_counts(dropna=False)
        rare_levels[col] = sorted(
            counts[counts < minimum_count]
            .index.astype(str)
            .tolist()
        )

    return rare_levels


def apply_rare_levels(
    df: pd.DataFrame,
    rare_levels: Dict[str, List[str]],
) -> pd.DataFrame:
    output = df.copy()

    for col, levels in rare_levels.items():
        level_set = set(levels)
        tokens = (
            output[col]
            .astype("string")
            .fillna(MISSING_TOKEN)
        )
        output[col] = tokens.map(
            lambda value: (
                RARE_TOKEN
                if str(value) in level_set
                else str(value)
            )
        )

    return output


def stratified_pilot_sample(
    training_df: pd.DataFrame,
    n_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    if n_rows >= len(training_df):
        return training_df.sample(
            frac=1.0,
            random_state=random_seed,
        ).reset_index(drop=True)

    positive = training_df[
        training_df[TARGET] == 1
    ]
    negative = training_df[
        training_df[TARGET] == 0
    ]

    positive_count = int(
        round(n_rows * training_df[TARGET].mean())
    )
    positive_count = min(
        max(1, positive_count),
        len(positive),
    )
    negative_count = min(
        n_rows - positive_count,
        len(negative),
    )

    sample = pd.concat(
        [
            positive.sample(
                n=positive_count,
                random_state=random_seed,
            ),
            negative.sample(
                n=negative_count,
                random_state=random_seed + 1,
            ),
        ],
        ignore_index=True,
    )

    return sample.sample(
        frac=1.0,
        random_state=random_seed + 2,
    ).reset_index(drop=True)


def serialise_row(
    row: pd.Series,
    feature_columns: Sequence[str],
) -> str:
    fields = [f"{TARGET}={int(row[TARGET])}"]

    for col in feature_columns:
        value = row[col]
        if pd.isna(value):
            displayed = MISSING_TOKEN
        else:
            displayed = str(value).strip()

        fields.append(f"{col}={displayed}")

    return " | ".join(fields)


def write_representation(
    *,
    name: str,
    source_df: pd.DataFrame,
    feature_columns: Sequence[str],
    partition_column: str,
    output_dir: Path,
    pilot_rows: int,
    random_seed: int,
) -> Dict[str, object]:
    representation_dir = output_dir / name
    representation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    partition_stats = []

    for partition in [
        "train",
        "validation",
        "test",
    ]:
        subset = (
            source_df[
                source_df[partition_column] == partition
            ][list(feature_columns) + [TARGET]]
            .copy()
            .reset_index(drop=True)
        )
        subset.to_csv(
            representation_dir / f"{partition}.csv",
            index=False,
        )

        partition_stats.append(
            {
                "representation": name,
                "partition": partition,
                "rows": len(subset),
                "features": len(feature_columns),
                "positive_n": int(subset[TARGET].sum()),
                "positive_pct": float(
                    subset[TARGET].mean() * 100
                ),
            }
        )

    train = (
        source_df[
            source_df[partition_column] == "train"
        ][list(feature_columns) + [TARGET]]
        .copy()
        .reset_index(drop=True)
    )
    pilot = stratified_pilot_sample(
        train,
        pilot_rows,
        random_seed,
    )
    pilot.to_csv(
        representation_dir / f"train_pilot_{len(pilot)}.csv",
        index=False,
    )

    sample_n = min(200, len(pilot))
    serialised = pilot.head(sample_n).apply(
        lambda row: serialise_row(
            row,
            feature_columns,
        ),
        axis=1,
    )
    serialised.to_frame("text").to_csv(
        representation_dir / "serialization_examples.csv",
        index=False,
    )

    character_lengths = serialised.str.len().to_numpy()
    whitespace_segments = (
        serialised.str.split().str.len().to_numpy()
    )

    length_summary = pd.DataFrame(
        [
            {
                "representation": name,
                "measure": "characters",
                "n_rows": sample_n,
                "min": float(np.min(character_lengths)),
                "median": float(
                    np.median(character_lengths)
                ),
                "p90": float(
                    np.quantile(character_lengths, 0.90)
                ),
                "p95": float(
                    np.quantile(character_lengths, 0.95)
                ),
                "p99": float(
                    np.quantile(character_lengths, 0.99)
                ),
                "max": float(np.max(character_lengths)),
            },
            {
                "representation": name,
                "measure": "whitespace_segments",
                "n_rows": sample_n,
                "min": float(np.min(whitespace_segments)),
                "median": float(
                    np.median(whitespace_segments)
                ),
                "p90": float(
                    np.quantile(whitespace_segments, 0.90)
                ),
                "p95": float(
                    np.quantile(whitespace_segments, 0.95)
                ),
                "p99": float(
                    np.quantile(whitespace_segments, 0.99)
                ),
                "max": float(np.max(whitespace_segments)),
            },
        ]
    )
    length_summary.to_csv(
        representation_dir / "serialization_length_summary.csv",
        index=False,
    )

    return {
        "partition_stats": partition_stats,
        "pilot_rows": len(pilot),
        "serialization_rows": sample_n,
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data_path)
    cleaned = normalise_source(raw)
    cohort, exclusions = apply_cohort_rule(cleaned)

    cohort["partition"] = assign_grouped_partitions(
        cohort,
        n_splits=args.n_group_folds,
        random_seed=args.random_seed,
    )

    manifest = cohort[
        [
            "encounter_id",
            "patient_nbr",
            TARGET,
            "partition",
        ]
    ].copy()
    manifest.to_csv(
        out_dir / "patient_group_split_manifest.csv",
        index=False,
    )

    overlap_checks = validate_group_split(manifest)

    candidate_features = [
        col
        for col in raw.columns
        if col
        not in ID_COLUMNS + [SOURCE_TARGET]
    ]

    full_features = [
        col
        for col in candidate_features
        if col not in PRIMARY_DROP_COLUMNS
    ]

    reduced_features = [
        col
        for col in full_features
        if col not in REDUCED_ADDITIONAL_DROP_COLUMNS
    ]

    # Prepare full representation with explicit categorical missing tokens.
    full = explicit_missing_tokens(
        cohort,
        full_features,
    )
    full["partition"] = cohort["partition"].to_numpy()

    # Fit diagnosis-code rare-level rules on the real training partition only.
    training_mask = cohort["partition"] == "train"
    rare_levels = fit_rare_levels(
        cohort.loc[training_mask],
        HIGH_CARDINALITY_DIAGNOSIS_COLUMNS,
        minimum_count=args.rare_min_count,
    )

    reduced_source = apply_rare_levels(
        cohort,
        rare_levels,
    )
    reduced = explicit_missing_tokens(
        reduced_source,
        reduced_features,
    )
    reduced["partition"] = cohort["partition"].to_numpy()

    full_result = write_representation(
        name="full_raw",
        source_df=full,
        feature_columns=full_features,
        partition_column="partition",
        output_dir=out_dir,
        pilot_rows=args.pilot_rows,
        random_seed=args.random_seed,
    )
    reduced_result = write_representation(
        name="reduced",
        source_df=reduced,
        feature_columns=reduced_features,
        partition_column="partition",
        output_dir=out_dir,
        pilot_rows=args.pilot_rows,
        random_seed=args.random_seed,
    )

    partition_summary = (
        manifest.groupby("partition", as_index=False)
        .agg(
            encounters=("encounter_id", "size"),
            unique_patients=("patient_nbr", "nunique"),
            positive_n=(TARGET, "sum"),
            positive_pct=(TARGET, lambda x: x.mean() * 100),
        )
        .sort_values("partition")
    )
    partition_summary.to_csv(
        out_dir / "partition_summary.csv",
        index=False,
    )

    exclusions.to_csv(
        out_dir / "cohort_exclusions.csv",
        index=False,
    )

    rare_level_rows = []
    for col, levels in rare_levels.items():
        rare_level_rows.append(
            {
                "feature": col,
                "rare_min_count": args.rare_min_count,
                "n_collapsed_levels": len(levels),
                "levels": "|".join(levels),
            }
        )
    pd.DataFrame(rare_level_rows).to_csv(
        out_dir / "reduced_rare_level_rules.csv",
        index=False,
    )

    representation_summary = pd.DataFrame(
        full_result["partition_stats"]
        + reduced_result["partition_stats"]
    )
    representation_summary.to_csv(
        out_dir / "representation_partition_summary.csv",
        index=False,
    )

    validation = {
        "source_rows": int(len(raw)),
        "cohort_rows": int(len(cohort)),
        "excluded_rows": int(len(raw) - len(cohort)),
        "unique_cohort_patients": int(
            cohort["patient_nbr"].nunique()
        ),
        "source_positive_pct": float(
            cleaned[TARGET].mean() * 100
        ),
        "cohort_positive_pct": float(
            cohort[TARGET].mean() * 100
        ),
        "full_feature_count": len(full_features),
        "reduced_feature_count": len(reduced_features),
        "full_features": full_features,
        "reduced_features": reduced_features,
        "primary_drops": PRIMARY_DROP_COLUMNS,
        "reduced_additional_drops": (
            REDUCED_ADDITIONAL_DROP_COLUMNS
        ),
        "cohort_excluded_discharge_ids": (
            DEATH_OR_HOSPICE_DISPOSITION_IDS
        ),
        "rare_level_fit_partition": "train",
        "rare_min_count": args.rare_min_count,
        "pilot_rows_requested": args.pilot_rows,
        "full_pilot_rows": full_result["pilot_rows"],
        "reduced_pilot_rows": reduced_result["pilot_rows"],
        **overlap_checks,
    }

    (
        out_dir / "preparation_metadata.json"
    ).write_text(
        json.dumps(validation, indent=2)
    )

    checks = [
        {
            "check": "No patient overlap between partitions",
            "status": (
                "PASS"
                if all(value == 0 for value in overlap_checks.values())
                else "FAIL"
            ),
        },
        {
            "check": "No identifiers in full representation",
            "status": (
                "PASS"
                if not set(ID_COLUMNS).intersection(full_features)
                else "FAIL"
            ),
        },
        {
            "check": "No identifiers in reduced representation",
            "status": (
                "PASS"
                if not set(ID_COLUMNS).intersection(reduced_features)
                else "FAIL"
            ),
        },
        {
            "check": "Binary target only",
            "status": (
                "PASS"
                if sorted(cohort[TARGET].unique().tolist())
                == [0, 1]
                else "FAIL"
            ),
        },
        {
            "check": "All eligible rows assigned exactly once",
            "status": (
                "PASS"
                if manifest["partition"].notna().all()
                and len(manifest) == len(cohort)
                else "FAIL"
            ),
        },
    ]
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(
        out_dir / "preparation_validation.csv",
        index=False,
    )

    if (checks_df["status"] != "PASS").any():
        raise RuntimeError(
            "At least one preparation validation check failed."
        )

    print("\nCohort and grouped split prepared successfully.")
    print(partition_summary.to_string(index=False))

    print("\nRepresentations:")
    print(
        representation_summary.to_string(index=False)
    )

    print("\nValidation:")
    print(checks_df.to_string(index=False))

    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
