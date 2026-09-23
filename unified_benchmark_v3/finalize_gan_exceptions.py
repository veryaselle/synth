#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import pandas as pd

DATASETS = ["pima", "cleveland", "ckd"]
METHODS = ["CTGAN", "CopulaGAN"]
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
    p.add_argument(
        "--results_root",
        default=str(Path(__file__).resolve().parent / "results" / "gan_sensitivity"),
        help="Root containing the GAN exception runs."
    )
    p.add_argument(
        "--tables_out",
        default=str(Path(__file__).resolve().parent / "results" / "gan_sensitivity" / "tables"),
    )
    p.add_argument(
        "--assembler",
        default="assemble_main_tables.py",
    )
    args = p.parse_args()

    root = Path(args.results_root)
    files = list(root.rglob("main_results_row.csv"))
    if not files:
        raise SystemExit(
            f"No main_results_row.csv files found under {root}"
        )

    rows = []
    for f in files:
        d = pd.read_csv(f)
        if len(d) != 1:
            raise SystemExit(
                f"{f}: expected exactly one row, found {len(d)}"
            )
        r = d.iloc[0].to_dict()
        r["_path"] = str(f)
        r["_split"] = split_from_path(f)
        rows.append(r)

    df = pd.DataFrame(rows)

    print("GAN sensitivity inventory")
    print("=" * 90)

    errors = []
    for dataset in DATASETS:
        for method in METHODS:
            m = df[
                (df["dataset"].astype(str).str.lower() == dataset)
                & (df["method"] == method)
            ].copy()

            splits = sorted(
                int(x) for x in m["_split"].dropna().tolist()
            )
            print(
                f"{dataset:12s} {method:10s}: "
                f"n={len(m)} splits={splits}"
            )

            if len(m) != 5:
                errors.append(
                    f"{dataset}/{method}: expected 5 rows, "
                    f"found {len(m)}"
                )
            if set(splits) != EXPECTED_SPLITS:
                errors.append(
                    f"{dataset}/{method}: expected splits 0..4, "
                    f"found {splits}"
                )
            if len(splits) != len(set(splits)):
                errors.append(
                    f"{dataset}/{method}: duplicate split rows"
                )

    expected_rows = len(DATASETS) * len(METHODS) * 5
    relevant = df[
        df["dataset"].astype(str).str.lower().isin(DATASETS)
        & df["method"].isin(METHODS)
    ].copy()

    if len(relevant) != expected_rows:
        errors.append(
            f"Expected {expected_rows} relevant GAN rows total, "
            f"found {len(relevant)}"
        )

    if errors:
        print("\nFINALIZATION REFUSED:")
        for e in errors:
            print(" -", e)

        print("\nDetected relevant rows:")
        if not relevant.empty:
            print(
                relevant[
                    ["dataset", "method", "_split", "_path"]
                ]
                .sort_values(["dataset", "method", "_split"])
                .to_string(index=False)
            )
        raise SystemExit(2)

    out = Path(args.tables_out)
    out.mkdir(parents=True, exist_ok=True)

    # Use the same assembler as the frozen primary benchmark,
    # so formatting and metric aggregation remain identical.
    subprocess.run(
        [
            "python",
            args.assembler,
            "--results_root",
            str(root),
            "--outdir",
            str(out),
        ],
        check=True,
    )

    print("\nGAN SENSITIVITY FINAL TABLES")
    print("=" * 90)

    for ds in DATASETS:
        table = out / f"{ds}_main_table_mean_sd.csv"
        if not table.exists():
            raise SystemExit(f"Expected table not created: {table}")

        print(f"\n{ds.upper()}")
        print(pd.read_csv(table).to_string(index=False))
        print(f"Saved: {table}")

    print("\nAll 30 GAN sensitivity rows validated and aggregated.")


if __name__ == "__main__":
    main()
