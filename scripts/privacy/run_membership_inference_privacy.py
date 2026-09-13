#!/usr/bin/env python3
"""
Empirical proximity-based membership inference evaluation for the completed
PIMA, Cleveland and CKD synthetic-data experiments.

Primary attack:
    Members     = real rows used to train the synthetic generator.
    Non-members = held-out real test rows from the same stratified split.
    Attack score = negative distance to the closest released synthetic row
                   with the same target class.

A larger score means that a real row is closer to the released synthetic data
and is therefore predicted to be a member.

The attack is deliberately simple and transparent. It provides attack-based
empirical evidence about membership leakage, but it does not provide a formal
privacy guarantee and it is not equivalent to differential privacy.

Outputs:
    membership_attack_file_level.csv
    membership_attack_split_level.csv
    membership_attack_summary.csv
    membership_attack_thesis_table.csv
    membership_attack_validation.csv
    membership_attack_auc_by_generator_size.png
    membership_attack_tpr_at_10pct_fpr.png
    membership_attack_protocol.json

Recommended full run:
    python run_membership_inference_privacy.py \
      --pima_data_path data/pima.csv \
      --pima_correlation_dir results/paper/correlation \
      --cleveland_data_path data/Heart_disease_cleveland_new.csv \
      --cleveland_ddpm_dir results/cleveland/diffusion_repeats \
      --cleveland_simple_dir results/cleveland/simple_baselines \
      --ckd_data_path data/kidney_disease_sanitized.csv \
      --ckd_experiment_dir results/ckd/synthetic_experiment \
      --out_dir results/privacy_membership_attack \
      --datasets pima cleveland ckd \
      --size_multipliers 1 2 3 \
      --pima_generators TVAE COPULA \
      --pima_n_files_per_model 20 \
      --include_pima_bootstrap_control
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PIMA_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

CLEVELAND_NUMERIC = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]
CLEVELAND_CATEGORICAL = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

CKD_NUMERIC = [
    "age",
    "bp",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
]
CKD_CATEGORICAL = [
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]

MISSING_TOKEN = "__MISSING__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["pima", "cleveland", "ckd"],
        default=["pima", "cleveland", "ckd"],
    )
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/privacy_membership_attack"),
    )
    parser.add_argument(
        "--size_multipliers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)

    # PIMA.
    parser.add_argument("--pima_data_path", default=str(REPOSITORY_ROOT / "data/raw/pima/pima.csv"))
    parser.add_argument(
        "--pima_correlation_dir",
        default=str(REPOSITORY_ROOT / "results/paper/correlation"),
    )
    parser.add_argument(
        "--pima_generators",
        nargs="+",
        default=["TVAE", "COPULA"],
    )
    parser.add_argument(
        "--pima_splits",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--pima_n_files_per_model",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--pima_starting_seed",
        type=int,
        default=4,
        help="Must match the adapted PIMA utility/privacy split protocol.",
    )
    parser.add_argument(
        "--pima_sampling_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--include_pima_bootstrap_control",
        action="store_true",
    )
    parser.add_argument(
        "--pima_bootstrap_repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--pima_no_zero_as_missing",
        action="store_true",
    )

    # Cleveland.
    parser.add_argument(
        "--cleveland_data_path",
        default=str(REPOSITORY_ROOT / "data/raw/cleveland/Heart_disease_cleveland_new.csv"),
    )
    parser.add_argument(
        "--cleveland_ddpm_dir",
        default=str(REPOSITORY_ROOT / "results/cleveland/diffusion_repeats"),
    )
    parser.add_argument(
        "--cleveland_simple_dir",
        default=str(REPOSITORY_ROOT / "results/cleveland/simple_baselines"),
    )
    parser.add_argument(
        "--cleveland_generators",
        nargs="+",
        choices=[
            "COND_DDPM",
            "GAUSSIAN_EMPIRICAL",
            "EMPIRICAL_BOOTSTRAP",
        ],
        default=[
            "COND_DDPM",
            "GAUSSIAN_EMPIRICAL",
            "EMPIRICAL_BOOTSTRAP",
        ],
    )
    parser.add_argument(
        "--cleveland_n_splits",
        type=int,
        default=5,
    )

    # CKD.
    parser.add_argument(
        "--ckd_data_path",
        default=str(REPOSITORY_ROOT / "data/processed/ckd/kidney_disease_sanitized.csv"),
    )
    parser.add_argument(
        "--ckd_experiment_dir",
        default=str(REPOSITORY_ROOT / "results/ckd/synthetic_experiment"),
    )
    parser.add_argument(
        "--ckd_generators",
        nargs="+",
        choices=[
            "COND_DDPM",
            "GAUSSIAN_EMPIRICAL",
            "EMPIRICAL_BOOTSTRAP",
        ],
        default=[
            "COND_DDPM",
            "GAUSSIAN_EMPIRICAL",
            "EMPIRICAL_BOOTSTRAP",
        ],
    )
    parser.add_argument("--ckd_n_splits", type=int, default=5)

    parser.add_argument(
        "--distance_batch_size",
        type=int,
        default=2048,
        help="Query batch size for nearest-synthetic distance computation.",
    )
    parser.add_argument(
        "--exact_match_tolerance",
        type=float,
        default=1e-10,
    )

    return parser.parse_args()


def stable_seed(*parts: object, base_seed: int = 42) -> int:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + base_seed) % (2**32 - 1)


def make_ohe() -> OneHotEncoder:
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


class AttackEncoder:
    """
    Training-only mixed-type encoder used only by the attack.

    Numerical variables:
        median imputation with missingness indicators, then standardisation.

    Categorical variables:
        explicit missing token, then one-hot encoding.

    The target is not included in the vector. The primary attack is conditioned
    on the target class when selecting the nearest synthetic record.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
    ):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.transformer: ColumnTransformer | None = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df[
            self.numeric_features + self.categorical_features
        ].copy()

        for col in self.numeric_features:
            prepared[col] = pd.to_numeric(
                prepared[col],
                errors="coerce",
            )

        for col in self.categorical_features:
            prepared[col] = prepared[col].map(
                lambda value: (
                    MISSING_TOKEN
                    if pd.isna(value)
                    else str(value).strip()
                )
            )

        return prepared

    def fit(self, real_train: pd.DataFrame) -> "AttackEncoder":
        transformers = []

        if self.numeric_features:
            numeric_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_features,
                )
            )

        if self.categorical_features:
            categorical_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value=MISSING_TOKEN,
                        ),
                    ),
                    ("onehot", make_ohe()),
                ]
            )
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features,
                )
            )

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        self.transformer.fit(self._prepare(real_train))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("AttackEncoder must be fitted first.")

        encoded = self.transformer.transform(
            self._prepare(df)
        )
        return np.asarray(encoded, dtype=float)


def clean_pima(
    df: pd.DataFrame,
    target_col: str = "Outcome",
    zero_as_missing: bool = True,
) -> pd.DataFrame:
    output = df.copy()

    for col in output.columns:
        output[col] = pd.to_numeric(
            output[col],
            errors="coerce",
        )

    output[target_col] = (
        output[target_col] >= 0.5
    ).astype(int)

    if zero_as_missing:
        for col in PIMA_ZERO_AS_MISSING:
            if col in output.columns:
                output[col] = output[col].replace(0, np.nan)

    return output


def clean_cleveland(
    df: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:
    output = df.copy()

    for col in CLEVELAND_NUMERIC:
        output[col] = pd.to_numeric(
            output[col],
            errors="coerce",
        )

    for col in CLEVELAND_CATEGORICAL:
        output[col] = output[col].where(
            output[col].notna(),
            np.nan,
        )

    output[target_col] = pd.to_numeric(
        output[target_col],
        errors="raise",
    ).astype(int)

    return output[
        CLEVELAND_NUMERIC
        + CLEVELAND_CATEGORICAL
        + [target_col]
    ]


def clean_ckd(
    df: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:
    output = df.copy()

    for col in CKD_NUMERIC:
        output[col] = pd.to_numeric(
            output[col],
            errors="coerce",
        )

    for col in CKD_CATEGORICAL:
        output[col] = output[col].where(
            output[col].notna(),
            np.nan,
        )

    output[target_col] = pd.to_numeric(
        output[target_col],
        errors="raise",
    ).astype(int)

    return output[
        CKD_NUMERIC
        + CKD_CATEGORICAL
        + [target_col]
    ]


def validate_synthetic(
    synthetic: pd.DataFrame,
    expected_columns: Sequence[str],
    target_col: str,
) -> Dict[str, object]:
    missing_columns = [
        col
        for col in expected_columns
        if col not in synthetic.columns
    ]

    if missing_columns:
        return {
            "status": "FAIL",
            "missing_columns": "|".join(missing_columns),
            "invalid_target_count": np.nan,
            "n_rows": len(synthetic),
        }

    target = pd.to_numeric(
        synthetic[target_col],
        errors="coerce",
    )
    invalid_target_count = int(
        (target.isna() | ~target.isin([0, 1])).sum()
    )

    return {
        "status": (
            "OK"
            if invalid_target_count == 0
            else "FAIL"
        ),
        "missing_columns": "",
        "invalid_target_count": invalid_target_count,
        "n_rows": len(synthetic),
    }


def balanced_attack_samples(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    target_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use equal member/non-member counts separately within each target class.

    This prevents the disease-label distribution from acting as a membership
    shortcut and gives every held-out row a matching member counterpart when
    possible.
    """
    rng = np.random.default_rng(seed)

    member_parts = []
    nonmember_parts = []

    target_values = sorted(
        set(real_train[target_col].unique())
        | set(real_test[target_col].unique())
    )

    for target_value in target_values:
        members = real_train[
            real_train[target_col] == target_value
        ]
        nonmembers = real_test[
            real_test[target_col] == target_value
        ]

        count = min(len(members), len(nonmembers))
        if count == 0:
            continue

        member_indices = rng.choice(
            members.index.to_numpy(),
            size=count,
            replace=False,
        )
        nonmember_indices = rng.choice(
            nonmembers.index.to_numpy(),
            size=count,
            replace=False,
        )

        member_parts.append(
            members.loc[member_indices].copy()
        )
        nonmember_parts.append(
            nonmembers.loc[nonmember_indices].copy()
        )

    if not member_parts or not nonmember_parts:
        raise ValueError(
            "Could not construct class-balanced member/non-member sets."
        )

    balanced_members = pd.concat(
        member_parts,
        ignore_index=True,
    )
    balanced_nonmembers = pd.concat(
        nonmember_parts,
        ignore_index=True,
    )

    return balanced_members, balanced_nonmembers


def nearest_distances_batched(
    query_encoded: np.ndarray,
    synthetic_encoded: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(synthetic_encoded) == 0:
        raise ValueError("Synthetic dataset is empty.")

    nearest = []

    for start in range(0, len(query_encoded), batch_size):
        stop = start + batch_size
        distances = pairwise_distances(
            query_encoded[start:stop],
            synthetic_encoded,
            metric="euclidean",
        )
        nearest.append(distances.min(axis=1))

    return np.concatenate(nearest)


def nearest_distances_class_conditional(
    query_encoded: np.ndarray,
    query_targets: np.ndarray,
    synthetic_encoded: np.ndarray,
    synthetic_targets: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, int]:
    result = np.empty(len(query_encoded), dtype=float)
    fallback_count = 0

    for target_value in sorted(
        np.unique(query_targets).tolist()
    ):
        query_mask = query_targets == target_value
        synthetic_mask = synthetic_targets == target_value

        if synthetic_mask.sum() == 0:
            fallback_count += int(query_mask.sum())
            result[query_mask] = nearest_distances_batched(
                query_encoded[query_mask],
                synthetic_encoded,
                batch_size,
            )
        else:
            result[query_mask] = nearest_distances_batched(
                query_encoded[query_mask],
                synthetic_encoded[synthetic_mask],
                batch_size,
            )

    return result, fallback_count


def tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    maximum_fpr: float,
) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= maximum_fpr]

    if len(eligible) == 0:
        return 0.0

    return float(np.max(eligible))


def calculate_attack_statistics(
    member_distances: np.ndarray,
    nonmember_distances: np.ndarray,
    tolerance: float,
    prefix: str,
) -> Dict[str, float]:
    labels = np.concatenate(
        [
            np.ones(len(member_distances), dtype=int),
            np.zeros(len(nonmember_distances), dtype=int),
        ]
    )
    distances = np.concatenate(
        [member_distances, nonmember_distances]
    )
    scores = -distances

    directed_auc = float(
        roc_auc_score(labels, scores)
    )
    distinguishability_auc = max(
        directed_auc,
        1.0 - directed_auc,
    )

    return {
        f"{prefix}_attack_auc": directed_auc,
        f"{prefix}_distinguishability_auc": (
            distinguishability_auc
        ),
        f"{prefix}_distinguishability_advantage": (
            2.0 * abs(directed_auc - 0.5)
        ),
        f"{prefix}_tpr_at_fpr_01": tpr_at_fpr(
            labels,
            scores,
            0.01,
        ),
        f"{prefix}_tpr_at_fpr_05": tpr_at_fpr(
            labels,
            scores,
            0.05,
        ),
        f"{prefix}_tpr_at_fpr_10": tpr_at_fpr(
            labels,
            scores,
            0.10,
        ),
        f"{prefix}_member_distance_mean": float(
            np.mean(member_distances)
        ),
        f"{prefix}_member_distance_median": float(
            np.median(member_distances)
        ),
        f"{prefix}_nonmember_distance_mean": float(
            np.mean(nonmember_distances)
        ),
        f"{prefix}_nonmember_distance_median": float(
            np.median(nonmember_distances)
        ),
        f"{prefix}_distance_gap_nonmember_minus_member": float(
            np.mean(nonmember_distances)
            - np.mean(member_distances)
        ),
        f"{prefix}_member_exact_match_rate": float(
            np.mean(member_distances <= tolerance)
        ),
        f"{prefix}_nonmember_exact_match_rate": float(
            np.mean(nonmember_distances <= tolerance)
        ),
    }


def evaluate_membership_attack(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    seed: int,
    batch_size: int,
    exact_match_tolerance: float,
) -> Dict[str, float]:
    balanced_members, balanced_nonmembers = (
        balanced_attack_samples(
            real_train,
            real_test,
            target_col,
            seed,
        )
    )

    encoder = AttackEncoder(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    ).fit(real_train)

    member_encoded = encoder.transform(
        balanced_members
    )
    nonmember_encoded = encoder.transform(
        balanced_nonmembers
    )
    synthetic_encoded = encoder.transform(
        synthetic
    )

    synthetic_targets = synthetic[
        target_col
    ].astype(int).to_numpy()
    member_targets = balanced_members[
        target_col
    ].astype(int).to_numpy()
    nonmember_targets = balanced_nonmembers[
        target_col
    ].astype(int).to_numpy()

    member_unconditional = nearest_distances_batched(
        member_encoded,
        synthetic_encoded,
        batch_size,
    )
    nonmember_unconditional = nearest_distances_batched(
        nonmember_encoded,
        synthetic_encoded,
        batch_size,
    )

    member_conditional, member_fallback = (
        nearest_distances_class_conditional(
            member_encoded,
            member_targets,
            synthetic_encoded,
            synthetic_targets,
            batch_size,
        )
    )
    nonmember_conditional, nonmember_fallback = (
        nearest_distances_class_conditional(
            nonmember_encoded,
            nonmember_targets,
            synthetic_encoded,
            synthetic_targets,
            batch_size,
        )
    )

    metrics = {
        "n_attack_members": len(balanced_members),
        "n_attack_nonmembers": len(balanced_nonmembers),
        "class_conditional_fallback_count": (
            member_fallback + nonmember_fallback
        ),
    }

    metrics.update(
        calculate_attack_statistics(
            member_unconditional,
            nonmember_unconditional,
            exact_match_tolerance,
            prefix="unconditional",
        )
    )
    metrics.update(
        calculate_attack_statistics(
            member_conditional,
            nonmember_conditional,
            exact_match_tolerance,
            prefix="class_conditional",
        )
    )

    return metrics


def sample_dataframe(
    source: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    replace = len(source) < n_rows
    selected = rng.choice(
        np.arange(len(source)),
        size=n_rows,
        replace=replace,
    )
    return source.iloc[selected].copy().reset_index(
        drop=True
    )


def empirical_bootstrap(
    real_train: pd.DataFrame,
    target_col: str,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target_probability = float(
        real_train[target_col].mean()
    )
    generated_targets = rng.binomial(
        1,
        target_probability,
        size=n_rows,
    )

    parts = []

    for target_value in [0, 1]:
        count = int(
            (generated_targets == target_value).sum()
        )
        if count == 0:
            continue

        class_rows = real_train[
            real_train[target_col] == target_value
        ]
        selected = rng.choice(
            class_rows.index.to_numpy(),
            size=count,
            replace=True,
        )
        parts.append(
            class_rows.loc[selected].copy()
        )

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def list_pima_source_files(
    correlation_dir: str,
    split: int,
    generator: str,
    n_files: int,
) -> List[Path]:
    pattern = (
        Path(correlation_dir)
        / str(split)
        / "sd"
        / generator
        / "*.csv"
    )
    files = [
        Path(path)
        for path in glob.glob(str(pattern))
        if ".ipynb_checkpoints" not in path
    ]

    def key(path: Path):
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.stem)

    files = sorted(files, key=key)

    if n_files > 0:
        files = files[:n_files]

    return files


def parse_synthetic_filename(
    path: Path,
) -> Tuple[int, int]:
    match = re.fullmatch(
        r"synthetic_repeat(\d+)_(\d+)x\.csv",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"Unexpected synthetic filename: {path}"
        )

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def evaluate_release(
    *,
    dataset: str,
    generator: str,
    split: int,
    repeat: int,
    size_multiplier: int,
    release_id: str,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    args: argparse.Namespace,
    file_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
) -> None:
    expected_columns = (
        list(numeric_features)
        + list(categorical_features)
        + [target_col]
    )

    validation = validate_synthetic(
        synthetic,
        expected_columns,
        target_col,
    )
    validation_rows.append(
        {
            "dataset": dataset,
            "generator": generator,
            "split": split,
            "repeat": repeat,
            "size_multiplier": size_multiplier,
            "release_id": release_id,
            **validation,
        }
    )

    if validation["status"] != "OK":
        print(
            f"[WARN] Skipping invalid release: "
            f"{dataset} {generator} split={split} "
            f"repeat={repeat} size={size_multiplier}x "
            f"id={release_id}",
            flush=True,
        )
        return

    synthetic = synthetic[
        expected_columns
    ].copy()
    synthetic[target_col] = pd.to_numeric(
        synthetic[target_col],
        errors="raise",
    ).astype(int)

    attack_seed = stable_seed(
        dataset,
        split,
        "balanced_attack_set",
        base_seed=args.random_seed,
    )

    metrics = evaluate_membership_attack(
        real_train=real_train,
        real_test=real_test,
        synthetic=synthetic,
        target_col=target_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        seed=attack_seed,
        batch_size=args.distance_batch_size,
        exact_match_tolerance=args.exact_match_tolerance,
    )

    row = {
        "dataset": dataset,
        "generator": generator,
        "split": split,
        "repeat": repeat,
        "size_multiplier": size_multiplier,
        "release_id": release_id,
        "n_real_train": len(real_train),
        "n_real_test": len(real_test),
        "n_synthetic": len(synthetic),
        **metrics,
    }
    file_rows.append(row)

    print(
        f"[OK] {dataset} {generator} "
        f"split={split} repeat={repeat} "
        f"size={size_multiplier}x "
        f"AUCcc={metrics['class_conditional_attack_auc']:.4f} "
        f"TPR@10%FPR="
        f"{metrics['class_conditional_tpr_at_fpr_10']:.4f}",
        flush=True,
    )


def run_pima(
    args: argparse.Namespace,
    file_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
) -> None:
    target_col = "Outcome"
    real = clean_pima(
        pd.read_csv(args.pima_data_path),
        target_col=target_col,
        zero_as_missing=(
            not args.pima_no_zero_as_missing
        ),
    )
    numeric_features = [
        col
        for col in real.columns
        if col != target_col
    ]

    y = real[target_col].astype(int)

    for split in args.pima_splits:
        split_seed = args.pima_starting_seed + split
        train_idx, test_idx = train_test_split(
            np.arange(len(real)),
            test_size=args.test_size,
            random_state=split_seed,
            stratify=y,
        )

        real_train = (
            real.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )
        real_test = (
            real.iloc[test_idx]
            .copy()
            .reset_index(drop=True)
        )

        for generator in args.pima_generators:
            files = list_pima_source_files(
                args.pima_correlation_dir,
                split,
                generator,
                args.pima_n_files_per_model,
            )

            if not files:
                print(
                    f"[WARN] No PIMA files for split={split}, "
                    f"generator={generator}",
                    flush=True,
                )

            for source_path in files:
                source = clean_pima(
                    pd.read_csv(source_path),
                    target_col=target_col,
                    zero_as_missing=(
                        not args.pima_no_zero_as_missing
                    ),
                )

                for multiplier in args.size_multipliers:
                    n_rows = len(real_train) * multiplier
                    sample_seed = stable_seed(
                        "PIMA",
                        split,
                        generator,
                        source_path.name,
                        multiplier,
                        base_seed=args.pima_sampling_seed,
                    )
                    synthetic = sample_dataframe(
                        source,
                        n_rows,
                        sample_seed,
                    )

                    evaluate_release(
                        dataset="PIMA",
                        generator=generator,
                        split=split,
                        repeat=int(
                            re.sub(r"\D", "", source_path.stem)
                            or 0
                        ),
                        size_multiplier=multiplier,
                        release_id=str(source_path),
                        real_train=real_train,
                        real_test=real_test,
                        synthetic=synthetic,
                        target_col=target_col,
                        numeric_features=numeric_features,
                        categorical_features=[],
                        args=args,
                        file_rows=file_rows,
                        validation_rows=validation_rows,
                    )

        if args.include_pima_bootstrap_control:
            for repeat in range(
                args.pima_bootstrap_repeats
            ):
                for multiplier in args.size_multipliers:
                    n_rows = len(real_train) * multiplier
                    seed = stable_seed(
                        "PIMA",
                        split,
                        "EMPIRICAL_BOOTSTRAP",
                        repeat,
                        multiplier,
                        base_seed=args.random_seed,
                    )
                    synthetic = empirical_bootstrap(
                        real_train,
                        target_col,
                        n_rows,
                        seed,
                    )

                    evaluate_release(
                        dataset="PIMA",
                        generator="EMPIRICAL_BOOTSTRAP",
                        split=split,
                        repeat=repeat,
                        size_multiplier=multiplier,
                        release_id=(
                            f"generated_bootstrap_split{split}_"
                            f"repeat{repeat}_{multiplier}x"
                        ),
                        real_train=real_train,
                        real_test=real_test,
                        synthetic=synthetic,
                        target_col=target_col,
                        numeric_features=numeric_features,
                        categorical_features=[],
                        args=args,
                        file_rows=file_rows,
                        validation_rows=validation_rows,
                    )


def run_cleveland(
    args: argparse.Namespace,
    file_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
) -> None:
    target_col = "target"
    real = clean_cleveland(
        pd.read_csv(args.cleveland_data_path),
        target_col=target_col,
    )

    splitter = StratifiedShuffleSplit(
        n_splits=args.cleveland_n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    X = real[
        CLEVELAND_NUMERIC
        + CLEVELAND_CATEGORICAL
    ]
    y = real[target_col]

    split_map = {
        split: (train_idx, test_idx)
        for split, (train_idx, test_idx)
        in enumerate(splitter.split(X, y))
    }

    generator_paths = {
        "COND_DDPM": (
            Path(args.cleveland_ddpm_dir)
            / "synthetic"
        ),
        "GAUSSIAN_EMPIRICAL": (
            Path(args.cleveland_simple_dir)
            / "synthetic"
            / "gaussian"
        ),
        "EMPIRICAL_BOOTSTRAP": (
            Path(args.cleveland_simple_dir)
            / "synthetic"
            / "empirical_bootstrap"
        ),
    }

    for generator in args.cleveland_generators:
        root = generator_paths[generator]

        for split, (
            train_idx,
            test_idx,
        ) in split_map.items():
            real_train = (
                real.iloc[train_idx]
                .copy()
                .reset_index(drop=True)
            )
            real_test = (
                real.iloc[test_idx]
                .copy()
                .reset_index(drop=True)
            )

            split_dir = root / f"split_{split}"

            for path in sorted(
                split_dir.glob(
                    "synthetic_repeat*_?x.csv"
                )
            ):
                repeat, multiplier = (
                    parse_synthetic_filename(path)
                )

                if (
                    multiplier
                    not in args.size_multipliers
                ):
                    continue

                synthetic = clean_cleveland(
                    pd.read_csv(path),
                    target_col=target_col,
                )

                evaluate_release(
                    dataset="Cleveland",
                    generator=generator,
                    split=split,
                    repeat=repeat,
                    size_multiplier=multiplier,
                    release_id=str(path),
                    real_train=real_train,
                    real_test=real_test,
                    synthetic=synthetic,
                    target_col=target_col,
                    numeric_features=CLEVELAND_NUMERIC,
                    categorical_features=(
                        CLEVELAND_CATEGORICAL
                    ),
                    args=args,
                    file_rows=file_rows,
                    validation_rows=validation_rows,
                )


def run_ckd(
    args: argparse.Namespace,
    file_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
) -> None:
    target_col = "target"
    real = clean_ckd(
        pd.read_csv(args.ckd_data_path),
        target_col=target_col,
    )

    splitter = StratifiedShuffleSplit(
        n_splits=args.ckd_n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    X = real[CKD_NUMERIC + CKD_CATEGORICAL]
    y = real[target_col]

    split_map = {
        split: (train_idx, test_idx)
        for split, (train_idx, test_idx)
        in enumerate(splitter.split(X, y))
    }

    root = (
        Path(args.ckd_experiment_dir)
        / "synthetic"
    )

    for generator in args.ckd_generators:
        for split, (
            train_idx,
            test_idx,
        ) in split_map.items():
            real_train = (
                real.iloc[train_idx]
                .copy()
                .reset_index(drop=True)
            )
            real_test = (
                real.iloc[test_idx]
                .copy()
                .reset_index(drop=True)
            )

            split_dir = (
                root
                / generator
                / f"split_{split}"
            )

            for path in sorted(
                split_dir.glob(
                    "synthetic_repeat*_?x.csv"
                )
            ):
                repeat, multiplier = (
                    parse_synthetic_filename(path)
                )

                if (
                    multiplier
                    not in args.size_multipliers
                ):
                    continue

                synthetic = clean_ckd(
                    pd.read_csv(path),
                    target_col=target_col,
                )

                evaluate_release(
                    dataset="CKD",
                    generator=generator,
                    split=split,
                    repeat=repeat,
                    size_multiplier=multiplier,
                    release_id=str(path),
                    real_train=real_train,
                    real_test=real_test,
                    synthetic=synthetic,
                    target_col=target_col,
                    numeric_features=CKD_NUMERIC,
                    categorical_features=CKD_CATEGORICAL,
                    args=args,
                    file_rows=file_rows,
                    validation_rows=validation_rows,
                )


def mean_confidence_interval(
    values: pd.Series,
) -> Tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)

    if len(clean) == 0:
        return np.nan, np.nan

    mean = float(np.mean(clean))

    if len(clean) == 1:
        return mean, mean

    standard_error = float(
        np.std(clean, ddof=1) / math.sqrt(len(clean))
    )

    try:
        from scipy.stats import t

        critical = float(
            t.ppf(0.975, df=len(clean) - 1)
        )
    except ImportError:
        critical = 1.96

    return (
        mean - critical * standard_error,
        mean + critical * standard_error,
    )


def risk_band(auc: float) -> str:
    if pd.isna(auc):
        return "not available"
    if auc < 0.55:
        return "no or very weak signal"
    if auc < 0.60:
        return "weak signal"
    if auc < 0.70:
        return "moderate signal"
    return "strong signal"


def create_summaries(
    file_level: pd.DataFrame,
    out_dir: Path,
) -> None:
    metric_columns = [
        col
        for col in file_level.columns
        if (
            col.startswith("unconditional_")
            or col.startswith(
                "class_conditional_"
            )
        )
    ]

    # First average multiple releases/repeats within each real split.
    split_level = (
        file_level
        .groupby(
            [
                "dataset",
                "generator",
                "split",
                "size_multiplier",
            ],
            as_index=False,
        )
        .agg(
            n_releases=("release_id", "size"),
            **{
                f"{col}_mean_within_split": (
                    col,
                    "mean",
                )
                for col in metric_columns
            },
        )
    )
    split_level.to_csv(
        out_dir / "membership_attack_split_level.csv",
        index=False,
    )

    primary_auc = (
        "class_conditional_attack_auc_"
        "mean_within_split"
    )

    summary_rows = []

    for (
        dataset,
        generator,
        multiplier,
    ), group in split_level.groupby(
        [
            "dataset",
            "generator",
            "size_multiplier",
        ]
    ):
        row: Dict[str, object] = {
            "dataset": dataset,
            "generator": generator,
            "size_multiplier": multiplier,
            "n_splits": group["split"].nunique(),
            "n_releases": int(
                group["n_releases"].sum()
            ),
        }

        for column in [
            col
            for col in split_level.columns
            if col.endswith("_mean_within_split")
        ]:
            output_name = column.replace(
                "_mean_within_split",
                "_mean",
            )
            row[output_name] = float(
                group[column].mean()
            )
            row[
                output_name.replace(
                    "_mean",
                    "_std",
                )
            ] = float(
                group[column].std(ddof=1)
            )

        ci_low, ci_high = mean_confidence_interval(
            group[primary_auc]
        )
        row[
            "class_conditional_attack_auc_ci95_low"
        ] = ci_low
        row[
            "class_conditional_attack_auc_ci95_high"
        ] = ci_high
        row["risk_interpretation"] = risk_band(
            row[
                "class_conditional_attack_auc_mean"
            ]
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "dataset",
            "generator",
            "size_multiplier",
        ]
    )
    summary.to_csv(
        out_dir / "membership_attack_summary.csv",
        index=False,
    )

    thesis = summary[
        summary["size_multiplier"]
        == summary.groupby(
            ["dataset", "generator"]
        )["size_multiplier"].transform("max")
    ].copy()

    thesis = thesis[
        [
            "dataset",
            "generator",
            "size_multiplier",
            "n_splits",
            "n_releases",
            "class_conditional_attack_auc_mean",
            "class_conditional_attack_auc_std",
            "class_conditional_attack_auc_ci95_low",
            "class_conditional_attack_auc_ci95_high",
            "class_conditional_tpr_at_fpr_10_mean",
            "class_conditional_member_distance_mean_mean",
            "class_conditional_nonmember_distance_mean_mean",
            "class_conditional_distance_gap_nonmember_minus_member_mean",
            "class_conditional_member_exact_match_rate_mean",
            "class_conditional_nonmember_exact_match_rate_mean",
            "risk_interpretation",
        ]
    ].sort_values(
        ["dataset", "generator"]
    )
    thesis.to_csv(
        out_dir / "membership_attack_thesis_table.csv",
        index=False,
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for (
            dataset,
            generator,
        ), group in summary.groupby(
            ["dataset", "generator"]
        ):
            group = group.sort_values(
                "size_multiplier"
            )
            label = f"{dataset}: {generator}"
            plt.plot(
                group["size_multiplier"],
                group[
                    "class_conditional_attack_auc_mean"
                ],
                marker="o",
                label=label,
            )

        plt.axhline(
            0.5,
            linestyle="--",
            linewidth=1,
        )
        plt.xlabel("Synthetic data size multiplier")
        plt.ylabel(
            "Class-conditional membership attack AUROC"
        )
        plt.title(
            "Proximity-based membership inference across datasets"
        )
        plt.xticks(
            sorted(
                summary[
                    "size_multiplier"
                ].unique()
            )
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            out_dir
            / "membership_attack_auc_by_generator_size.png",
            dpi=200,
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        for (
            dataset,
            generator,
        ), group in summary.groupby(
            ["dataset", "generator"]
        ):
            group = group.sort_values(
                "size_multiplier"
            )
            label = f"{dataset}: {generator}"
            plt.plot(
                group["size_multiplier"],
                group[
                    "class_conditional_tpr_at_fpr_10_mean"
                ],
                marker="o",
                label=label,
            )

        plt.xlabel("Synthetic data size multiplier")
        plt.ylabel("TPR at FPR ≤ 10%")
        plt.title(
            "Membership attack sensitivity at a constrained false-positive rate"
        )
        plt.xticks(
            sorted(
                summary[
                    "size_multiplier"
                ].unique()
            )
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            out_dir
            / "membership_attack_tpr_at_10pct_fpr.png",
            dpi=200,
        )
        plt.close()

    except ImportError:
        print(
            "[WARN] matplotlib unavailable; plots were not created.",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "attack": (
            "Nearest-synthetic-record membership inference"
        ),
        "primary_score": (
            "Negative Euclidean distance to the nearest "
            "synthetic record with the same target class"
        ),
        "members": (
            "Class-balanced subset of the real generator "
            "training split"
        ),
        "non_members": (
            "Class-balanced held-out real test rows"
        ),
        "encoder": {
            "numerical": (
                "Training-only median imputation with "
                "missingness indicators and standardisation"
            ),
            "categorical": (
                "Explicit missing token and training-only "
                "one-hot encoding"
            ),
        },
        "primary_metric": (
            "Class-conditional attack AUROC"
        ),
        "secondary_metrics": [
            "Unconditional attack AUROC",
            "TPR at FPR <= 1%, 5% and 10%",
            "Member/non-member nearest-synthetic distances",
            "Exact-match rates",
        ],
        "interpretation": (
            "AUROC near 0.5 indicates little detectable "
            "membership signal for this attack. Higher AUROC "
            "indicates greater empirical membership leakage."
        ),
        "limitation": (
            "This is an empirical attack against one release "
            "mechanism and does not provide a formal privacy "
            "guarantee."
        ),
        "arguments": vars(args),
    }
    (
        out_dir / "membership_attack_protocol.json"
    ).write_text(json.dumps(protocol, indent=2))

    file_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []

    if "pima" in args.datasets:
        print("\n========== PIMA ==========", flush=True)
        run_pima(
            args,
            file_rows,
            validation_rows,
        )

    if "cleveland" in args.datasets:
        print(
            "\n========== Cleveland ==========",
            flush=True,
        )
        run_cleveland(
            args,
            file_rows,
            validation_rows,
        )

    if "ckd" in args.datasets:
        print("\n========== CKD ==========", flush=True)
        run_ckd(
            args,
            file_rows,
            validation_rows,
        )

    file_level = pd.DataFrame(file_rows)
    validation = pd.DataFrame(validation_rows)

    validation.to_csv(
        out_dir / "membership_attack_validation.csv",
        index=False,
    )

    if file_level.empty:
        raise RuntimeError(
            "No valid synthetic releases were evaluated. "
            "Check the supplied data and synthetic-output paths."
        )

    file_level.to_csv(
        out_dir / "membership_attack_file_level.csv",
        index=False,
    )

    create_summaries(
        file_level,
        out_dir,
    )

    print("\nValidation status:")
    print(
        validation["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    summary = pd.read_csv(
        out_dir / "membership_attack_summary.csv"
    )

    display_columns = [
        "dataset",
        "generator",
        "size_multiplier",
        "n_splits",
        "n_releases",
        "class_conditional_attack_auc_mean",
        "class_conditional_attack_auc_std",
        "class_conditional_tpr_at_fpr_10_mean",
        "risk_interpretation",
    ]

    print("\nMembership attack summary:")
    print(
        summary[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print(f"\nSaved results under: {out_dir}")


if __name__ == "__main__":
    main()
