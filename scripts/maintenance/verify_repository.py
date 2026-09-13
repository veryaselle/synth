#!/usr/bin/env python3
"""Verify repository structure, Python syntax, frozen release integrity, and data presence."""
from __future__ import annotations

import argparse
import compileall
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    required = [
        root / "README.md",
        root / "data/raw/pima/pima.csv",
        root / "data/raw/cleveland/Heart_disease_cleveland_new.csv",
        root / "data/raw/ckd/kidney_disease.csv",
        root / "data/raw/diabetes130/diabetic_data.csv",
        root / "scripts/diabetes130/run_diabetes130_utility_fidelity_v2.py",
        root / "scripts/diabetes130/run_diabetes130_privacy.py",
        root / "results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv",
        root / "results/diabetes130/great_main_20k/final_release/final_release_audit.json",
    ]

    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    if not compileall.compile_dir(root / "scripts", quiet=1):
        raise SystemExit("Python syntax verification failed.")

    audit_path = root / "results/diabetes130/great_main_20k/final_release/final_release_audit.json"
    audit = json.loads(audit_path.read_text())
    validation = audit["final_validation"]
    if not validation["overall_schema_pass"]:
        raise SystemExit("Frozen GReaT release audit does not pass.")
    if validation["rows"] != 19999:
        raise SystemExit(f"Unexpected final row count: {validation['rows']}")
    if validation["exact_duplicate_rate_vs_full_real_train"] != 0.0:
        raise SystemExit("Frozen release contains exact real-data copies.")
    if validation["internal_duplicate_rate"] != 0.0:
        raise SystemExit("Frozen release contains internal duplicates.")

    print("Repository verification: PASS")
    print(f"Root: {root}")
    print("Python scripts: syntax PASS")
    print(f"Frozen GReaT rows: {validation['rows']}")
    print("Frozen GReaT schema: PASS")
    if not args.fast:
        release_path = root / "results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv"
        print(f"Frozen release SHA-256: {sha256(release_path)}")


if __name__ == "__main__":
    main()
