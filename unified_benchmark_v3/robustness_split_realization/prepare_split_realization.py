#!/usr/bin/env python3
"""Prepare an independent realization of the benchmark train/test splits.

Purpose
-------
This supplementary robustness experiment changes the *data-partition realization*
while keeping the frozen generator/evaluator policies unchanged. It is intentionally
kept separate from the primary frozen benchmark.

Design
------
- The CLI seed identifies one supplementary partition realization.
- PIMA mirrors the primary legacy-compatible scheme by using five independent
  stratified train_test_split calls with seeds SEED..SEED+4.
- Cleveland and CKD mirror the primary scheme by using StratifiedShuffleSplit
  with n_splits=5 and random_state=SEED.
- PIMA imputation settings, Cleveland cleaning assumptions, and CKD train-fitted
  imputation rules are identical to the primary benchmark.
- Generator and evaluator seeds are *not* changed here; the run scripts retain
  the primary seed schedule. This keeps the intervention focused on partitioning.

Outputs
-------
robustness_split_realization/data/seed_<SEED>/<dataset>/split_0..4
plus a realization manifest and an overlap audit against the primary frozen splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent
REPOSITORY_ROOT = BENCHMARK_ROOT.parent
sys.path.insert(0, str(BENCHMARK_ROOT))

import prepare_frozen_datasets as F  # noqa: E402

DEFAULT_REALIZATION_SEED = 2026
N_SPLITS = 5
TEST_SIZE = 0.20


def _prepare_pima(path: Path, out_root: Path, realization_seed: int) -> None:
    df = pd.read_csv(path).copy()
    target = "Outcome"
    df[target] = pd.to_numeric(df[target], errors="raise").astype(int)
    idx = np.arange(len(df))

    for split_id, split_seed in enumerate(range(realization_seed, realization_seed + N_SPLITS)):
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=TEST_SIZE,
            random_state=split_seed,
            stratify=df[target],
        )
        tr_idx, te_idx = np.asarray(tr_idx), np.asarray(te_idx)
        train_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
        test_raw = df.iloc[te_idx].copy().reset_index(drop=True)

        for c in F.PIMA_ZERO_AS_MISSING:
            train_raw[c] = pd.to_numeric(train_raw[c], errors="coerce").replace(0, np.nan)
            test_raw[c] = pd.to_numeric(test_raw[c], errors="coerce").replace(0, np.nan)
        for c in F.PIMA_NUMERIC:
            train_raw[c] = pd.to_numeric(train_raw[c], errors="coerce")
            test_raw[c] = pd.to_numeric(test_raw[c], errors="coerce")

        # Keep the primary preprocessing seed schedule fixed. sample_posterior=False,
        # but preserving the same declared seed avoids introducing another planned change.
        imputer_seed = 42 + split_id
        imputer = IterativeImputer(
            random_state=imputer_seed,
            max_iter=20,
            initial_strategy="median",
            sample_posterior=False,
            skip_complete=True,
        )
        train_processed, test_processed = train_raw.copy(), test_raw.copy()
        train_processed[F.PIMA_NUMERIC] = imputer.fit_transform(train_raw[F.PIMA_NUMERIC])
        test_processed[F.PIMA_NUMERIC] = imputer.transform(test_raw[F.PIMA_NUMERIC])
        train_processed[target] = train_raw[target].to_numpy()
        test_processed[target] = test_raw[target].to_numpy()

        F.save_bundle(
            out_root,
            "pima",
            split_id,
            train_raw,
            test_raw,
            train_processed,
            test_processed,
            target,
            F.PIMA_NUMERIC,
            [],
            metadata={
                "experiment": "independent split-realization robustness check",
                "realization_seed": realization_seed,
                "split_strategy": "stratified train_test_split",
                "test_size": TEST_SIZE,
                "split_seed": split_seed,
                "generator_seed_policy": "unchanged from primary: 42 + split_id",
                "evaluator_seed_policy": "unchanged from primary: 42",
                "preprocessing": {
                    "zero_as_missing_columns": F.PIMA_ZERO_AS_MISSING,
                    "imputation": "sklearn IterativeImputer (MICE-like), train-fitted only",
                    "imputer_random_seed": imputer_seed,
                    "max_iter": 20,
                    "initial_strategy": "median",
                    "sample_posterior": False,
                },
            },
            train_indices=tr_idx,
            test_indices=te_idx,
        )


def _five_stratified_splits(df: pd.DataFrame, target: str, seed: int):
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=seed,
    )
    for split_id, (tr_idx, te_idx) in enumerate(splitter.split(df, df[target])):
        yield split_id, np.asarray(tr_idx), np.asarray(te_idx)


def _prepare_cleveland(path: Path, out_root: Path, realization_seed: int) -> None:
    df = pd.read_csv(path).copy()
    target = "target"
    for c in F.CLEVELAND_NUMERIC + F.CLEVELAND_CATEGORICAL + [target]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df[target] = df[target].astype(int)
    if df.isna().any().any():
        raise ValueError("Clean Cleveland input unexpectedly contains missing values")

    for split_id, tr_idx, te_idx in _five_stratified_splits(df, target, realization_seed):
        train_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
        test_raw = df.iloc[te_idx].copy().reset_index(drop=True)
        F.save_bundle(
            out_root,
            "cleveland",
            split_id,
            train_raw,
            test_raw,
            train_raw.copy(),
            test_raw.copy(),
            target,
            F.CLEVELAND_NUMERIC,
            F.CLEVELAND_CATEGORICAL,
            metadata={
                "experiment": "independent split-realization robustness check",
                "realization_seed": realization_seed,
                "split_strategy": "StratifiedShuffleSplit",
                "n_splits": N_SPLITS,
                "test_size": TEST_SIZE,
                "split_seed": realization_seed,
                "generator_seed_policy": "unchanged from primary: 42 + split_id",
                "evaluator_seed_policy": "unchanged from primary: 42",
                "preprocessing": {"imputation": "none; cleaned input has zero missing values"},
            },
            train_indices=tr_idx,
            test_indices=te_idx,
        )


def _prepare_ckd(path: Path, out_root: Path, realization_seed: int) -> None:
    df = pd.read_csv(path).copy()
    target = "target"
    missing = set(F.CKD_NUMERIC + F.CKD_CATEGORICAL + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"CKD input missing columns: {sorted(missing)}")
    df = df[F.CKD_NUMERIC + F.CKD_CATEGORICAL + [target]].copy()
    for c in F.CKD_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="raise").astype(int)

    for split_id, tr_idx, te_idx in _five_stratified_splits(df, target, realization_seed):
        train_raw = df.iloc[tr_idx].copy().reset_index(drop=True)
        test_raw = df.iloc[te_idx].copy().reset_index(drop=True)
        train_processed, test_processed = train_raw.copy(), test_raw.copy()
        numeric_fill: Dict[str, float] = {}
        categorical_fill: Dict[str, object] = {}

        for c in F.CKD_NUMERIC:
            val = float(pd.to_numeric(train_raw[c], errors="coerce").median())
            numeric_fill[c] = val
            train_processed[c] = pd.to_numeric(train_raw[c], errors="coerce").fillna(val)
            test_processed[c] = pd.to_numeric(test_raw[c], errors="coerce").fillna(val)

        for c in F.CKD_CATEGORICAL:
            mode = train_raw[c].dropna().mode()
            if len(mode) == 0:
                raise ValueError(f"CKD categorical column {c} has no observed training value")
            val = mode.iloc[0]
            categorical_fill[c] = val.item() if hasattr(val, "item") else val
            train_processed[c] = train_raw[c].fillna(val)
            test_processed[c] = test_raw[c].fillna(val)

        F.save_bundle(
            out_root,
            "ckd",
            split_id,
            train_raw,
            test_raw,
            train_processed,
            test_processed,
            target,
            F.CKD_NUMERIC,
            F.CKD_CATEGORICAL,
            metadata={
                "experiment": "independent split-realization robustness check",
                "realization_seed": realization_seed,
                "split_strategy": "StratifiedShuffleSplit",
                "n_splits": N_SPLITS,
                "test_size": TEST_SIZE,
                "split_seed": realization_seed,
                "generator_seed_policy": "unchanged from primary: 42 + split_id",
                "evaluator_seed_policy": "unchanged from primary: 42",
                "preprocessing": {
                    "numeric_imputation": "training-split median",
                    "categorical_imputation": "training-split mode",
                    "numeric_fill_values": numeric_fill,
                    "categorical_fill_values": categorical_fill,
                },
            },
            train_indices=tr_idx,
            test_indices=te_idx,
        )


def _index_set(path: Path) -> frozenset[int]:
    return frozenset(pd.read_csv(path)["row_index"].astype(int).tolist())


def _audit_against_primary(out_root: Path, primary_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in ["pima", "cleveland", "ckd"]:
        primary_sets = {
            i: _index_set(primary_root / dataset / f"split_{i}" / "test_indices.csv")
            for i in range(N_SPLITS)
        }
        robust_sets = {
            i: _index_set(out_root / dataset / f"split_{i}" / "test_indices.csv")
            for i in range(N_SPLITS)
        }
        for r_id, r_set in robust_sets.items():
            for p_id, p_set in primary_sets.items():
                intersection = len(r_set & p_set)
                union = len(r_set | p_set)
                rows.append(
                    {
                        "dataset": dataset,
                        "robust_split": r_id,
                        "primary_split": p_id,
                        "robust_test_n": len(r_set),
                        "primary_test_n": len(p_set),
                        "intersection_n": intersection,
                        "jaccard": intersection / union if union else 1.0,
                        "exact_duplicate_test_partition": r_set == p_set,
                    }
                )

    audit = pd.DataFrame(rows)
    duplicates = audit[audit["exact_duplicate_test_partition"]]
    if not duplicates.empty:
        raise RuntimeError(
            "Second realization accidentally duplicates a primary test partition:\n"
            + duplicates.to_string(index=False)
        )
    return audit


def _validate_split_bundles(out_root: Path) -> None:
    expected_sizes = {
        "pima": (614, 154),
        "cleveland": (242, 61),
        "ckd": (320, 80),
    }
    for dataset, (n_train, n_test) in expected_sizes.items():
        for split_id in range(N_SPLITS):
            d = out_root / dataset / f"split_{split_id}"
            required = [
                "real_train.csv",
                "real_test.csv",
                "schema.json",
                "train_indices.csv",
                "test_indices.csv",
                "preparation_metadata.json",
            ]
            for name in required:
                if not (d / name).exists():
                    raise FileNotFoundError(d / name)

            tr = pd.read_csv(d / "real_train.csv")
            te = pd.read_csv(d / "real_test.csv")
            tr_idx = _index_set(d / "train_indices.csv")
            te_idx = _index_set(d / "test_indices.csv")
            if (len(tr), len(te)) != (n_train, n_test):
                raise RuntimeError(
                    f"{dataset} split_{split_id}: expected {(n_train, n_test)}, "
                    f"got {(len(tr), len(te))}"
                )
            if tr_idx & te_idx:
                raise RuntimeError(f"{dataset} split_{split_id}: train/test index overlap")
            if tr.isna().any().any() or te.isna().any().any():
                raise RuntimeError(f"{dataset} split_{split_id}: processed missing values remain")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_REALIZATION_SEED)
    p.add_argument("--pima", default=str(REPOSITORY_ROOT / "data/raw/pima/pima.csv"))
    p.add_argument(
        "--cleveland",
        default=str(REPOSITORY_ROOT / "data/raw/cleveland/Heart_disease_cleveland_new.csv"),
    )
    p.add_argument(
        "--ckd",
        default=str(REPOSITORY_ROOT / "data/processed/ckd/kidney_disease_sanitized.csv"),
    )
    p.add_argument("--primary_root", default=str(BENCHMARK_ROOT / "frozen_data"))
    p.add_argument("--out_root", default=None)
    args = p.parse_args()

    out_root = (
        Path(args.out_root)
        if args.out_root
        else THIS_DIR / "data" / f"seed_{args.seed}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    _prepare_pima(Path(args.pima), out_root, args.seed)
    _prepare_cleveland(Path(args.cleveland), out_root, args.seed)
    _prepare_ckd(Path(args.ckd), out_root, args.seed)
    _validate_split_bundles(out_root)

    audit = _audit_against_primary(out_root, Path(args.primary_root))
    audit.to_csv(out_root / "split_overlap_with_primary.csv", index=False)

    manifest = {
        "experiment": "independent split-realization robustness check",
        "status": "supplementary robustness experiment; does not replace primary frozen benchmark",
        "realization_seed": args.seed,
        "n_splits_per_dataset": N_SPLITS,
        "test_size": TEST_SIZE,
        "partition_policy": {
            "pima": f"stratified train_test_split seeds {args.seed}..{args.seed + N_SPLITS - 1}",
            "cleveland": f"StratifiedShuffleSplit(n_splits=5, random_state={args.seed})",
            "ckd": f"StratifiedShuffleSplit(n_splits=5, random_state={args.seed})",
        },
        "held_constant": {
            "generator_seed_schedule": "42 + split_id",
            "evaluator_seed": 42,
            "release_size": "1x",
            "generator_hyperparameters": "same frozen configurations as primary benchmark",
            "evaluation_pipeline": "same evaluate_release.py / evaluate_real_reference.py",
        },
        "interpretation": (
            "Primarily probes sensitivity to an independently shuffled train/test partition "
            "realization. Retraining stochastic generators can still contribute residual run "
            "variation; this experiment is not a complete multi-seed generator-stochasticity study."
        ),
    }
    (out_root / "realization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Split realization written to: {out_root}")
    print("Validation: PASS")
    print("No supplementary-realization test partition exactly duplicates a primary frozen test partition.")
    print(audit.groupby("dataset")["jaccard"].agg(["min", "mean", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
