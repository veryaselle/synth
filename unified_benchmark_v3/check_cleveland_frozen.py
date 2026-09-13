#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "unified_benchmark_v3"

EXPECTED_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
EXPECTED_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
EXPECTED_TARGET = "target"
EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen_root", default=str(BENCHMARK_ROOT / "frozen_data/cleveland"))
    p.add_argument("--results_root", default=str(BENCHMARK_ROOT / "results/main_benchmark/cleveland"))
    p.add_argument("--allow_existing_results", action="store_true")
    args = p.parse_args()

    frozen = Path(args.frozen_root)
    if not frozen.exists():
        raise SystemExit(f"Missing frozen Cleveland root: {frozen}")

    print("Cleveland frozen-data preflight")
    print("=" * 80)

    for split in range(5):
        d = frozen / f"split_{split}"
        required = [
            d / "real_train.csv",
            d / "real_test.csv",
            d / "schema.json",
            d / "preparation_metadata.json",
        ]
        missing = [str(x) for x in required if not x.exists()]
        if missing:
            raise SystemExit(f"split_{split}: missing files: {missing}")

        tr = pd.read_csv(d / "real_train.csv")
        te = pd.read_csv(d / "real_test.csv")
        schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))

        if list(tr.columns) != EXPECTED_COLUMNS:
            raise SystemExit(f"split_{split}: unexpected train columns: {list(tr.columns)}")
        if list(te.columns) != EXPECTED_COLUMNS:
            raise SystemExit(f"split_{split}: unexpected test columns: {list(te.columns)}")
        if schema.get("target") != EXPECTED_TARGET:
            raise SystemExit(f"split_{split}: target mismatch in schema")
        if schema.get("numeric") != EXPECTED_NUMERIC:
            raise SystemExit(f"split_{split}: numeric schema mismatch: {schema.get('numeric')}")
        if schema.get("categorical") != EXPECTED_CATEGORICAL:
            raise SystemExit(f"split_{split}: categorical schema mismatch: {schema.get('categorical')}")
        if list(schema.get("columns", [])) != EXPECTED_COLUMNS:
            raise SystemExit(f"split_{split}: schema column order mismatch")
        if tr.isna().any().any() or te.isna().any().any():
            raise SystemExit(f"split_{split}: processed data contains missing values")
        if set(tr[EXPECTED_TARGET].unique()) != {0, 1}:
            raise SystemExit(f"split_{split}: train target is not binary 0/1")
        if set(te[EXPECTED_TARGET].unique()) != {0, 1}:
            raise SystemExit(f"split_{split}: test target is not binary 0/1")

        print(
            f"split_{split}: train={tr.shape}, test={te.shape}, "
            f"train prevalence={tr[EXPECTED_TARGET].mean():.6f}, "
            f"test prevalence={te[EXPECTED_TARGET].mean():.6f}, missing=0"
        )

    results = Path(args.results_root)
    existing = list(results.rglob("main_results_row.csv")) if results.exists() else []
    if existing and not args.allow_existing_results:
        print("\nExisting Cleveland main benchmark rows detected:")
        for x in existing:
            print(" ", x)
        raise SystemExit(
            "\nRefusing to mix new Cleveland runs with existing primary rows. "
            "Move/archive the old Cleveland results first, or rerun with "
            "--allow_existing_results only if those files are intentionally resumable."
        )

    print("\nPRE-FLIGHT PASS")
    print("Frozen Cleveland contract: 5 splits, 242 train / 61 test each, 5 numeric + 8 categorical features.")

if __name__ == "__main__":
    main()
