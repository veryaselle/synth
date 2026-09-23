#!/usr/bin/env python3
"""Strictly finalize the independent split-realization robustness experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent
DEFAULT_SEED = 2026
DATASETS = ["pima", "cleveland", "ckd"]
METHODS = [
    "REAL",
    "TVAE",
    "CTGAN",
    "CopulaGAN",
    "ARF",
    "Gaussian Copula",
    "Conditional DDPM",
    "LLM",
]
EXPECTED_SPLITS = set(range(5))


def split_from_path(path: Path):
    for part in path.parts:
        if part.startswith("split_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                pass
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--results_root", default=None)
    p.add_argument("--tables_out", default=None)
    p.add_argument("--skip_comparison", action="store_true")
    args = p.parse_args()

    root = Path(args.results_root) if args.results_root else THIS_DIR / "results" / f"seed_{args.seed}"
    tables = Path(args.tables_out) if args.tables_out else root / "tables"
    files = sorted(root.rglob("main_results_row.csv"))
    if not files:
        raise SystemExit(f"No main_results_row.csv files found under {root}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) != 1:
            raise SystemExit(f"{f}: expected exactly one row, got {len(df)}")
        r = df.iloc[0].to_dict()
        r["_split"] = split_from_path(f)
        r["_path"] = str(f)
        rows.append(r)
    d = pd.DataFrame(rows)
    d["dataset"] = d["dataset"].astype(str).str.lower()

    errors = []
    for dataset in DATASETS:
        for method in METHODS:
            sub = d[(d["dataset"] == dataset) & (d["method"] == method)]
            splits = sub["_split"].tolist()
            if len(sub) != 5:
                errors.append(f"{dataset}/{method}: expected 5 rows, found {len(sub)}")
            if set(splits) != EXPECTED_SPLITS:
                errors.append(f"{dataset}/{method}: expected splits 0..4, found {sorted(splits)}")
            if len(splits) != len(set(splits)):
                errors.append(f"{dataset}/{method}: duplicate split rows")

    expected_pairs = {(ds, m) for ds in DATASETS for m in METHODS}
    observed_pairs = set(zip(d["dataset"], d["method"]))
    extra_pairs = sorted(observed_pairs - expected_pairs)
    if extra_pairs:
        errors.append(f"Unexpected dataset/method pairs: {extra_pairs}")
    if len(d) != 120:
        errors.append(f"Expected exactly 120 split-level rows, found {len(d)}")

    if errors:
        print("FINALIZATION REFUSED:")
        for e in errors:
            print(" -", e)
        raise SystemExit(2)

    tables.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_ROOT / "assemble_main_tables.py"),
            "--results_root", str(root),
            "--outdir", str(tables),
        ],
        check=True,
    )

    inventory = (
        d.groupby(["dataset", "method"], as_index=False)
        .agg(n_rows=("_split", "size"), n_splits=("_split", "nunique"))
        .sort_values(["dataset", "method"])
    )
    inventory.to_csv(tables / "validated_inventory.csv", index=False)

    manifest_path = THIS_DIR / "data" / f"seed_{args.seed}" / "realization_manifest.json"
    final_meta = {
        "experiment": "independent split-realization robustness check",
        "realization_seed": args.seed,
        "validated_split_level_rows": int(len(d)),
        "validated_dataset_method_cells": int(len(inventory)),
        "expected": "3 datasets x 8 training sources x 5 splits = 120 rows",
        "primary_benchmark_replaced": False,
        "realization_manifest": str(manifest_path),
    }
    (tables / "finalization_metadata.json").write_text(
        json.dumps(final_meta, indent=2), encoding="utf-8"
    )

    if not args.skip_comparison:
        subprocess.run(
            [
                sys.executable,
                str(THIS_DIR / "compare_with_primary.py"),
                "--seed", str(args.seed),
                "--robust_summary", str(tables / "main_results_summary_numeric.csv"),
            ],
            check=True,
        )

    print("SECOND REALIZATION FINALIZATION PASS")
    print(f"Validated 120 rows and wrote tables to {tables}")


if __name__ == "__main__":
    main()
