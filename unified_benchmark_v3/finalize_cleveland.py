#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import pandas as pd

EXPECTED_METHODS = [
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default=str(Path(__file__).resolve().parent / "results" / "main_benchmark"))
    p.add_argument("--tables_out", default=str(Path(__file__).resolve().parent / "results" / "main_benchmark" / "tables"))
    p.add_argument("--assembler", default="assemble_main_tables.py")
    args = p.parse_args()

    root = Path(args.results_root)
    clev = root / "cleveland"
    files = list(clev.rglob("main_results_row.csv"))
    if not files:
        raise SystemExit("No Cleveland main_results_row.csv files found")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) != 1:
            raise SystemExit(f"{f}: expected exactly one row, got {len(df)}")
        r = df.iloc[0].to_dict()
        r["_path"] = str(f)
        r["_split"] = split_from_path(f)
        rows.append(r)

    d = pd.DataFrame(rows)
    if not (d["dataset"].astype(str).str.lower() == "cleveland").all():
        bad = d[d["dataset"].astype(str).str.lower() != "cleveland"]
        raise SystemExit("Non-Cleveland row under Cleveland results tree:\n" + bad.to_string())

    print("Cleveland primary-row inventory")
    print("=" * 90)

    errors = []
    for method in EXPECTED_METHODS:
        m = d[d["method"] == method].copy()
        splits = m["_split"].tolist()
        print(f"{method:20s}: n={len(m)} splits={sorted(splits)}")
        if len(m) != 5:
            errors.append(f"{method}: expected 5 rows, found {len(m)}")
        if set(splits) != EXPECTED_SPLITS:
            errors.append(f"{method}: expected splits 0..4, found {sorted(splits)}")
        if len(splits) != len(set(splits)):
            errors.append(f"{method}: duplicate split-level rows detected")

    unexpected = sorted(set(d["method"]) - set(EXPECTED_METHODS))
    if unexpected:
        errors.append(f"Unexpected methods under Cleveland main benchmark: {unexpected}")

    if len(d) != 40:
        errors.append(f"Expected exactly 40 Cleveland primary rows, found {len(d)}")

    if errors:
        print("\nFINALIZATION REFUSED:")
        for e in errors:
            print(" -", e)
        print("\nAll detected files:")
        print(d[["method", "_split", "_path"]].sort_values(["method", "_split"]).to_string(index=False))
        raise SystemExit(2)

    subprocess.run([
        "python", args.assembler,
        "--results_root", str(root),
        "--outdir", args.tables_out,
    ], check=True)

    table = Path(args.tables_out) / "cleveland_main_table_mean_sd.csv"
    print("\nCLEVELAND FINAL TABLE")
    print("=" * 90)
    print(pd.read_csv(table).to_string(index=False))
    print(f"\nSaved: {table}")

if __name__ == "__main__":
    main()
