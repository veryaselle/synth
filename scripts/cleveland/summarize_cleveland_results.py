#!/usr/bin/env python3
"""
Create thesis-ready summary tables and simple plots for Cleveland diffusion results.

Inputs:
    - diffusion_utility_summary.csv from run_cleveland_diffusion.py
    - optionally baseline_real_summary.csv from run_cleveland_baseline.py

Outputs:
    tables:
        cleveland_real_vs_best_ddpm.csv
        cleveland_auc_by_size.csv
        cleveland_fidelity_by_size.csv
        cleveland_privacy_by_size.csv
        cleveland_best_ddpm_by_classifier.csv

    plots:
        auc_by_size.png
        fidelity_by_size.png
        dcr_by_size.png

Example:
    python summarize_cleveland_results.py \
      --diffusion_summary results/cleveland/diffusion_repeats/diffusion_utility_summary.csv \
      --baseline_summary results/cleveland/baseline_real/baseline_real_summary.csv \
      --out_dir results/cleveland/thesis_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_summary", default=str(REPOSITORY_ROOT / "results/cleveland/diffusion/diffusion_utility_summary.csv"))
    parser.add_argument("--baseline_summary", default=str(REPOSITORY_ROOT / "results/cleveland/baseline_real/baseline_real_summary.csv"))
    parser.add_argument("--out_dir", default=str(REPOSITORY_ROOT / "results/cleveland/thesis_tables"))
    return parser.parse_args()


def maybe_read_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def standardize_diffusion_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ["generator", "classifier", "size_multiplier", "auc_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Diffusion summary missing required columns: {missing}")
    return df.copy()


def create_best_ddpm_table(diff_df: pd.DataFrame) -> pd.DataFrame:
    ddpm = diff_df[diff_df["generator"] == "COND_DDPM"].copy()
    if ddpm.empty:
        raise ValueError("No COND_DDPM rows found in diffusion summary.")

    idx = ddpm.groupby("classifier")["auc_mean"].idxmax()
    best = ddpm.loc[idx].copy().sort_values("auc_mean", ascending=False)

    cols = [
        "classifier",
        "size_multiplier",
        "n_rows",
        "auc_mean",
        "auc_std",
        "accuracy_mean",
        "sensitivity_mean",
        "specificity_mean",
        "precision_mean",
        "f1_mean",
        "brier_mean",
        "pcd_mean",
        "ws_mean",
        "js_mean",
        "dcr_mean",
        "dcr_min",
        "duplicate_rate_mean",
    ]
    cols = [c for c in cols if c in best.columns]
    return best[cols]


def create_real_vs_best_table(diff_df: pd.DataFrame, baseline_df: pd.DataFrame | None) -> pd.DataFrame:
    best = create_best_ddpm_table(diff_df).copy()
    best = best.rename(
        columns={
            "size_multiplier": "best_ddpm_size",
            "auc_mean": "ddpm_auc_mean",
            "auc_std": "ddpm_auc_std",
            "accuracy_mean": "ddpm_accuracy_mean",
            "sensitivity_mean": "ddpm_sensitivity_mean",
            "specificity_mean": "ddpm_specificity_mean",
            "precision_mean": "ddpm_precision_mean",
            "f1_mean": "ddpm_f1_mean",
            "brier_mean": "ddpm_brier_mean",
        }
    )

    real_rows = diff_df[diff_df["generator"] == "REAL"].copy()

    if not real_rows.empty:
        real_cols = [
            "classifier",
            "auc_mean",
            "auc_std",
            "accuracy_mean",
            "sensitivity_mean",
            "specificity_mean",
            "precision_mean",
            "f1_mean",
            "brier_mean",
        ]
        real_cols = [c for c in real_cols if c in real_rows.columns]
        real = real_rows[real_cols].rename(
            columns={
                "auc_mean": "real_auc_mean",
                "auc_std": "real_auc_std",
                "accuracy_mean": "real_accuracy_mean",
                "sensitivity_mean": "real_sensitivity_mean",
                "specificity_mean": "real_specificity_mean",
                "precision_mean": "real_precision_mean",
                "f1_mean": "real_f1_mean",
                "brier_mean": "real_brier_mean",
            }
        )
    elif baseline_df is not None:
        real = baseline_df.copy().rename(
            columns={
                "auc_mean": "real_auc_mean",
                "auc_std": "real_auc_std",
                "accuracy_mean": "real_accuracy_mean",
                "sensitivity_mean": "real_sensitivity_mean",
                "specificity_mean": "real_specificity_mean",
                "precision_mean": "real_precision_mean",
                "f1_mean": "real_f1_mean",
                "brier_mean": "real_brier_mean",
            }
        )
    else:
        real = pd.DataFrame({"classifier": best["classifier"]})

    merged = best.merge(real, on="classifier", how="left")

    if "real_auc_mean" in merged.columns:
        merged["auc_gap_ddpm_minus_real"] = merged["ddpm_auc_mean"] - merged["real_auc_mean"]
    if "real_brier_mean" in merged.columns:
        merged["brier_gap_ddpm_minus_real"] = merged["ddpm_brier_mean"] - merged["real_brier_mean"]

    preferred_order = [
        "classifier",
        "best_ddpm_size",
        "ddpm_auc_mean",
        "ddpm_auc_std",
        "real_auc_mean",
        "real_auc_std",
        "auc_gap_ddpm_minus_real",
        "ddpm_accuracy_mean",
        "real_accuracy_mean",
        "ddpm_sensitivity_mean",
        "real_sensitivity_mean",
        "ddpm_specificity_mean",
        "real_specificity_mean",
        "ddpm_f1_mean",
        "real_f1_mean",
        "ddpm_brier_mean",
        "real_brier_mean",
        "brier_gap_ddpm_minus_real",
        "pcd_mean",
        "ws_mean",
        "js_mean",
        "dcr_mean",
        "dcr_min",
        "duplicate_rate_mean",
    ]
    cols = [c for c in preferred_order if c in merged.columns]
    return merged[cols].sort_values("ddpm_auc_mean", ascending=False)


def create_auc_by_size(diff_df: pd.DataFrame) -> pd.DataFrame:
    ddpm = diff_df[diff_df["generator"] == "COND_DDPM"].copy()
    return (
        ddpm
        .groupby("size_multiplier", as_index=False)
        .agg(
            auc_mean=("auc_mean", "mean"),
            auc_std_across_classifiers=("auc_mean", "std"),
            accuracy_mean=("accuracy_mean", "mean"),
            sensitivity_mean=("sensitivity_mean", "mean"),
            specificity_mean=("specificity_mean", "mean"),
            f1_mean=("f1_mean", "mean"),
            brier_mean=("brier_mean", "mean"),
        )
    )


def create_fidelity_by_size(diff_df: pd.DataFrame) -> pd.DataFrame:
    ddpm = diff_df[diff_df["generator"] == "COND_DDPM"].copy()
    cols = [c for c in ["pcd_mean", "ws_mean", "js_mean"] if c in ddpm.columns]
    return ddpm.groupby("size_multiplier", as_index=False).agg(**{c: (c, "mean") for c in cols})


def create_privacy_by_size(diff_df: pd.DataFrame) -> pd.DataFrame:
    ddpm = diff_df[diff_df["generator"] == "COND_DDPM"].copy()
    cols = [c for c in ["dcr_mean", "dcr_min", "duplicate_rate_mean"] if c in ddpm.columns]
    return ddpm.groupby("size_multiplier", as_index=False).agg(**{c: (c, "mean") for c in cols})


def create_plots(diff_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ddpm = diff_df[diff_df["generator"] == "COND_DDPM"].copy()

    plt.figure(figsize=(8, 5))
    for clf, sub in ddpm.groupby("classifier"):
        sub = sub.sort_values("size_multiplier")
        plt.plot(sub["size_multiplier"], sub["auc_mean"], marker="o", label=clf)
    plt.xlabel("Synthetic data size multiplier")
    plt.ylabel("Mean AUC")
    plt.title("Cleveland conditional diffusion: AUC by synthetic size")
    plt.xticks(sorted(ddpm["size_multiplier"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "auc_by_size.png", dpi=200)
    plt.close()

    fidelity = create_fidelity_by_size(diff_df)
    plt.figure(figsize=(8, 5))
    for col in ["pcd_mean", "ws_mean", "js_mean"]:
        if col in fidelity.columns:
            plt.plot(fidelity["size_multiplier"], fidelity[col], marker="o", label=col)
    plt.xlabel("Synthetic data size multiplier")
    plt.ylabel("Mean metric value")
    plt.title("Cleveland conditional diffusion: fidelity by synthetic size")
    plt.xticks(sorted(ddpm["size_multiplier"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fidelity_by_size.png", dpi=200)
    plt.close()

    privacy = create_privacy_by_size(diff_df)
    plt.figure(figsize=(8, 5))
    for col in ["dcr_mean", "dcr_min"]:
        if col in privacy.columns:
            plt.plot(privacy["size_multiplier"], privacy[col], marker="o", label=col)
    plt.xlabel("Synthetic data size multiplier")
    plt.ylabel("Distance to closest real record")
    plt.title("Cleveland conditional diffusion: DCR by synthetic size")
    plt.xticks(sorted(ddpm["size_multiplier"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dcr_by_size.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_df = standardize_diffusion_summary(pd.read_csv(args.diffusion_summary))
    baseline_df = maybe_read_csv(args.baseline_summary)

    best_ddpm = create_best_ddpm_table(diff_df)
    real_vs_best = create_real_vs_best_table(diff_df, baseline_df)
    auc_by_size = create_auc_by_size(diff_df)
    fidelity_by_size = create_fidelity_by_size(diff_df)
    privacy_by_size = create_privacy_by_size(diff_df)

    best_ddpm.to_csv(out_dir / "cleveland_best_ddpm_by_classifier.csv", index=False)
    real_vs_best.to_csv(out_dir / "cleveland_real_vs_best_ddpm.csv", index=False)
    auc_by_size.to_csv(out_dir / "cleveland_auc_by_size.csv", index=False)
    fidelity_by_size.to_csv(out_dir / "cleveland_fidelity_by_size.csv", index=False)
    privacy_by_size.to_csv(out_dir / "cleveland_privacy_by_size.csv", index=False)

    create_plots(diff_df, out_dir)

    print("\nSaved tables:")
    for name in [
        "cleveland_best_ddpm_by_classifier.csv",
        "cleveland_real_vs_best_ddpm.csv",
        "cleveland_auc_by_size.csv",
        "cleveland_fidelity_by_size.csv",
        "cleveland_privacy_by_size.csv",
    ]:
        print(f"  {out_dir / name}")

    print("\nSaved plots:")
    for name in ["auc_by_size.png", "fidelity_by_size.png", "dcr_by_size.png"]:
        print(f"  {out_dir / name}")

    print("\nREAL vs best DDPM:")
    print(real_vs_best.to_string(index=False))


if __name__ == "__main__":
    main()
