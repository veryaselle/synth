#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

EXPECTED_NUMERIC = ["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
EXPECTED_CATEGORICAL = ["sg","al","su","rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
EXPECTED_TARGET = "target"
EXPECTED_COLUMNS = EXPECTED_NUMERIC + EXPECTED_CATEGORICAL + [EXPECTED_TARGET]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen_root", default="frozen_data/ckd")
    p.add_argument("--results_root", default=str(Path(__file__).resolve().parent / "results" / "main_benchmark" / "ckd"))
    p.add_argument("--allow_existing_results", action="store_true")
    args = p.parse_args()

    frozen = Path(args.frozen_root)
    if not frozen.exists():
        raise SystemExit(f"Missing frozen CKD root: {frozen}")

    print("CKD frozen-data preflight")
    print("=" * 90)

    for split in range(5):
        d = frozen / f"split_{split}"
        required = [d/"real_train.csv", d/"real_test.csv", d/"real_train_raw.csv", d/"real_test_raw.csv", d/"schema.json", d/"preparation_metadata.json"]
        missing = [str(x) for x in required if not x.exists()]
        if missing:
            raise SystemExit(f"split_{split}: missing files: {missing}")

        tr = pd.read_csv(d / "real_train.csv")
        te = pd.read_csv(d / "real_test.csv")
        schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
        metadata = json.loads((d / "preparation_metadata.json").read_text(encoding="utf-8"))

        if len(tr) != 320 or len(te) != 80:
            raise SystemExit(f"split_{split}: expected train/test=320/80, got {len(tr)}/{len(te)}")
        if list(tr.columns) != EXPECTED_COLUMNS or list(te.columns) != EXPECTED_COLUMNS:
            raise SystemExit(f"split_{split}: processed column order mismatch")
        if schema.get("dataset") != "ckd" or schema.get("split_id") != split:
            raise SystemExit(f"split_{split}: schema dataset/split mismatch")
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
        if set(tr[EXPECTED_TARGET].unique()) != {0,1} or set(te[EXPECTED_TARGET].unique()) != {0,1}:
            raise SystemExit(f"split_{split}: target is not binary 0/1")

        prep = metadata.get("preprocessing", {})
        if prep.get("numeric_imputation") != "training-split median":
            raise SystemExit(f"split_{split}: unexpected numeric imputation metadata")
        if prep.get("categorical_imputation") != "training-split mode":
            raise SystemExit(f"split_{split}: unexpected categorical imputation metadata")
        if metadata.get("split_seed") != 42:
            raise SystemExit(f"split_{split}: expected split_seed=42")

        print(f"split_{split}: train={tr.shape}, test={te.shape}, train_counts={tr[target].value_counts().sort_index().to_dict() if (target := EXPECTED_TARGET) else {}}, test_counts={te[target].value_counts().sort_index().to_dict()}, train prevalence={tr[target].mean():.6f}, test prevalence={te[target].mean():.6f}, missing=0")

    results = Path(args.results_root)
    existing = list(results.rglob("main_results_row.csv")) if results.exists() else []
    if existing and not args.allow_existing_results:
        print("\nExisting CKD main benchmark rows detected:")
        for x in existing:
            print(" ", x)
        raise SystemExit("\nRefusing to mix a new CKD primary run with existing rows. Archive/remove results/main_benchmark/ckd first, or use --allow_existing_results only for an intentional resume.")

    print("\nPRE-FLIGHT PASS")
    print("Frozen CKD contract: 5 splits, 320 train / 80 test each, 11 numeric + 13 categorical/ordinal features; zero missing values after training-split-only median/mode imputation.")

if __name__ == "__main__":
    main()
