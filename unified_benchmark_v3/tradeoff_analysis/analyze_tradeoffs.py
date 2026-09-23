#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADEOFF_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRADEOFF_DIR.parent
OUTPUT_DIR = TRADEOFF_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# dir_suffix controls the actual method-folder name on disk.
CONFIGS = {
    "baseline_b500_d2e4": {
        "root": PROJECT_ROOT / "results" / "main_benchmark",
        "batch_size": 500,
        "discriminator_lr": 2e-4,
        "is_baseline": True,
        "dir_suffix": "",
    },
    "b100_d2e4": {
        "root": PROJECT_ROOT / "results" / "gan_sensitivity",
        "batch_size": 100,
        "discriminator_lr": 2e-4,
        "is_baseline": False,
        "dir_suffix": "_b100",
    },
    "b200_d2e4": {
        "root": PROJECT_ROOT / "results" / "gan_sensitivity" / "b200",
        "batch_size": 200,
        "discriminator_lr": 2e-4,
        "is_baseline": False,
        "dir_suffix": "",
    },
    "b100_d5e4": {
        "root": PROJECT_ROOT / "results" / "gan_sensitivity" / "lr5e4_b100",
        "batch_size": 100,
        "discriminator_lr": 5e-4,
        "is_baseline": False,
        "dir_suffix": "",
    },
}

DATASETS = ["pima", "cleveland", "ckd"]
METHODS = ["CTGAN", "CopulaGAN"]
SPLITS = range(5)

METRICS = [
    "utility_auroc",
    "utility_f1",
    "utility_brier",
    "fidelity_pcd",
    "fidelity_ws",
    "fidelity_js",
    "privacy_dcr_mean",
    "privacy_mia_auc",
    "privacy_aia_risk",
]

IMPROVEMENT_RULES = {
    "utility_auroc": "high",
    "utility_f1": "high",
    "utility_brier": "low",
    "fidelity_pcd": "low",
    "fidelity_ws": "low",
    "fidelity_js": "low",
    "privacy_mia_auc": "mia",
    "privacy_aia_risk": "low",
}


def result_file(cfg: dict, dataset: str, split: int, method: str) -> Path:
    method_dir = method + cfg.get("dir_suffix", "")
    return (
        cfg["root"] / dataset / f"split_{split}" / method_dir
        / "evaluation" / "main_results_row.csv"
    )


def validate_paths():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("TRADEOFF_DIR:", TRADEOFF_DIR)
    print()
    missing = []
    for config_id, cfg in CONFIGS.items():
        print(f"[{config_id}] root={cfg['root']} suffix='{cfg['dir_suffix']}'")
        found = 0
        for ds in DATASETS:
            for method in METHODS:
                for split in SPLITS:
                    f = result_file(cfg, ds, split, method)
                    if f.exists():
                        found += 1
                    else:
                        missing.append((config_id, f))
        print(f"  found {found}/30 expected GAN result rows")

    if missing:
        print("\nMISSING FILES:")
        for config_id, f in missing:
            print(f"  {config_id}: {f}")
        raise SystemExit(
            f"\nRefusing analysis: {len(missing)} GAN result files missing."
        )


def load_all():
    rows = []
    for config_id, cfg in CONFIGS.items():
        for ds in DATASETS:
            for method in METHODS:
                for split in SPLITS:
                    f = result_file(cfg, ds, split, method)
                    d = pd.read_csv(f)
                    if len(d) != 1:
                        raise SystemExit(f"{f}: expected 1 row, got {len(d)}")
                    r = d.iloc[0].to_dict()
                    if str(r.get("dataset", "")).lower() != ds:
                        raise SystemExit(
                            f"{f}: dataset={r.get('dataset')} but expected {ds}"
                        )
                    if str(r.get("method", "")) != method:
                        raise SystemExit(
                            f"{f}: method={r.get('method')} but expected {method}"
                        )
                    r.update({
                        "config_id": config_id,
                        "split_id": split,
                        "batch_size": cfg["batch_size"],
                        "discriminator_lr": cfg["discriminator_lr"],
                        "is_baseline": cfg["is_baseline"],
                        "_path": f.relative_to(PROJECT_ROOT).as_posix(),
                    })
                    rows.append(r)

    df = pd.DataFrame(rows)
    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["privacy_mia_distance"] = (df["privacy_mia_auc"] - 0.5).abs()
    return df


def aggregate(df):
    metrics = METRICS + ["privacy_mia_distance"]
    agg = {}
    for c in metrics:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_sd"] = (c, "std")
    out = (
        df.groupby(["dataset", "method", "config_id"], as_index=False)
        .agg(**agg)
    )
    meta = (
        df.groupby(["dataset", "method", "config_id"], as_index=False)
        .agg(
            batch_size=("batch_size", "first"),
            discriminator_lr=("discriminator_lr", "first"),
            is_baseline=("is_baseline", "first"),
        )
    )
    return out.merge(
        meta, on=["dataset", "method", "config_id"],
        validate="one_to_one"
    )


def paired_deltas(df):
    base = df[df["is_baseline"]].copy()
    base_cols = ["dataset", "method", "split_id"] + METRICS
    base = base[base_cols].rename(
        columns={m: m + "_baseline" for m in METRICS}
    )

    x = df.merge(
        base,
        on=["dataset", "method", "split_id"],
        how="left",
        validate="many_to_one"
    )

    delta_names = []
    for metric, rule in IMPROVEMENT_RULES.items():
        col = f"delta_{metric}_improvement"
        delta_names.append(col)
        cur = x[metric]
        ref = x[metric + "_baseline"]
        if rule == "high":
            x[col] = cur - ref
        elif rule == "low":
            x[col] = ref - cur
        elif rule == "mia":
            x[col] = (ref - 0.5).abs() - (cur - 0.5).abs()

    x["delta_privacy_dcr_mean_raw"] = (
        x["privacy_dcr_mean"] - x["privacy_dcr_mean_baseline"]
    )

    nonbase = x[~x["is_baseline"]].copy()

    records = []
    for keys, sub in nonbase.groupby(
        ["dataset", "method", "config_id"], sort=False
    ):
        ds, method, config_id = keys
        row = {
            "dataset": ds,
            "method": method,
            "config_id": config_id,
            "n_splits": len(sub),
        }
        for c in delta_names:
            row[c + "_mean"] = sub[c].mean()
            row[c + "_sd"] = sub[c].std()
            row[c + "_n_improved"] = int((sub[c] > 0).sum())
            row[c + "_n_worse"] = int((sub[c] < 0).sum())
        row["delta_privacy_dcr_mean_raw_mean"] = (
            sub["delta_privacy_dcr_mean_raw"].mean()
        )
        row["delta_privacy_dcr_mean_raw_sd"] = (
            sub["delta_privacy_dcr_mean_raw"].std()
        )
        records.append(row)

    return x, pd.DataFrame(records)


def rank_scores(summary):
    d = summary.copy()

    defs = {
        "utility_auroc_mean": "high",
        "utility_f1_mean": "high",
        "utility_brier_mean": "low",
        "fidelity_pcd_mean": "low",
        "fidelity_ws_mean": "low",
        "fidelity_js_mean": "low",
        "privacy_mia_distance_mean": "low",
        "privacy_aia_risk_mean": "low",
    }

    for col, direction in defs.items():
        d[col + "_rank"] = d.groupby(["dataset", "method"])[col].rank(
            method="average",
            ascending=(direction == "low")
        )

    d["utility_rank"] = d[
        ["utility_auroc_mean_rank", "utility_f1_mean_rank", "utility_brier_mean_rank"]
    ].mean(axis=1)
    d["fidelity_rank"] = d[
        ["fidelity_pcd_mean_rank", "fidelity_ws_mean_rank", "fidelity_js_mean_rank"]
    ].mean(axis=1)
    d["privacy_rank"] = d[
        ["privacy_mia_distance_mean_rank", "privacy_aia_risk_mean_rank"]
    ].mean(axis=1)

    for dim in ["utility", "fidelity", "privacy"]:
        d[dim + "_score"] = 1.0 - (d[dim + "_rank"] - 1.0) / 3.0

    d["balanced_mean_score"] = d[
        ["utility_score", "fidelity_score", "privacy_score"]
    ].mean(axis=1)
    d["balanced_min_score"] = d[
        ["utility_score", "fidelity_score", "privacy_score"]
    ].min(axis=1)

    return d


def pareto_flags(scored):
    parts = []
    cols = ["utility_score", "fidelity_score", "privacy_score"]

    for _, sub in scored.groupby(["dataset", "method"], sort=False):
        s = sub.copy()
        vals = s[cols].to_numpy(float)
        flags = []
        for i, a in enumerate(vals):
            dominated = False
            for j, b in enumerate(vals):
                if i == j:
                    continue
                if np.all(b >= a) and np.any(b > a):
                    dominated = True
                    break
            flags.append(not dominated)
        s["pareto_efficient"] = flags
        parts.append(s)

    return pd.concat(parts, ignore_index=True)


def strategy_table(scored):
    rows = []
    for (ds, method), sub in scored.groupby(["dataset", "method"]):
        selections = {
            "utility_first": sub.sort_values(
                ["utility_score", "balanced_mean_score"], ascending=False
            ).iloc[0],
            "fidelity_first": sub.sort_values(
                ["fidelity_score", "balanced_mean_score"], ascending=False
            ).iloc[0],
            "privacy_first": sub.sort_values(
                ["privacy_score", "balanced_mean_score"], ascending=False
            ).iloc[0],
        }

        p = sub[sub["pareto_efficient"]]
        selections["balanced_pareto"] = p.sort_values(
            ["balanced_min_score", "balanced_mean_score"], ascending=False
        ).iloc[0]

        for strategy, r in selections.items():
            rows.append({
                "dataset": ds,
                "method": method,
                "strategy": strategy,
                "config_id": r["config_id"],
                "batch_size": r["batch_size"],
                "discriminator_lr": r["discriminator_lr"],
                "utility_score": r["utility_score"],
                "fidelity_score": r["fidelity_score"],
                "privacy_score": r["privacy_score"],
                "balanced_min_score": r["balanced_min_score"],
                "pareto_efficient": r["pareto_efficient"],
            })
    return pd.DataFrame(rows)


def make_plots(scored):
    pairs = [
        ("utility_score", "fidelity_score", "Utility", "Fidelity"),
        ("utility_score", "privacy_score", "Utility", "Privacy"),
        ("fidelity_score", "privacy_score", "Fidelity", "Privacy"),
    ]

    for ds in DATASETS:
        for method in METHODS:
            sub = scored[
                (scored["dataset"] == ds) & (scored["method"] == method)
            ].copy()

            for x, y, xl, yl in pairs:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(sub[x], sub[y], s=80)
                for _, r in sub.iterrows():
                    label = r["config_id"]
                    if r["pareto_efficient"]:
                        label += " *"
                    ax.annotate(
                        label, (r[x], r[y]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8
                    )
                ax.set_xlim(-0.08, 1.08)
                ax.set_ylim(-0.08, 1.08)
                ax.set_xlabel(f"{xl} score (higher = better)")
                ax.set_ylabel(f"{yl} score (higher = better)")
                ax.set_title(f"{ds.upper()} — {method}: {xl} vs {yl}")
                fig.tight_layout()
                fig.savefig(
                    FIG_DIR / f"{ds}_{method.lower()}_{xl.lower()}_vs_{yl.lower()}.png",
                    dpi=200
                )
                plt.close(fig)


def main():
    print("=" * 90)
    print("GAN TRADE-OFF ANALYSIS")
    print("=" * 90)

    validate_paths()

    df = load_all()
    if len(df) != 120:
        raise SystemExit(f"Expected 120 rows after loading, found {len(df)}")

    df.to_csv(OUTPUT_DIR / "gan_tradeoff_split_level.csv", index=False)

    summary = aggregate(df)
    summary.to_csv(OUTPUT_DIR / "gan_tradeoff_config_summary.csv", index=False)

    paired_split, paired_summary = paired_deltas(df)
    paired_split.to_csv(
        OUTPUT_DIR / "gan_paired_deltas_split_level.csv", index=False
    )
    paired_summary.to_csv(
        OUTPUT_DIR / "gan_paired_deltas_summary.csv", index=False
    )

    scored = rank_scores(summary)
    scored = pareto_flags(scored)
    scored.to_csv(
        OUTPUT_DIR / "gan_tradeoff_dimension_scores.csv", index=False
    )
    scored[scored["pareto_efficient"]].to_csv(
        OUTPUT_DIR / "gan_pareto_front.csv", index=False
    )

    strategies = strategy_table(scored)
    strategies.to_csv(
        OUTPUT_DIR / "gan_strategy_recommendations.csv", index=False
    )

    make_plots(scored)

    print("\nLoaded rows:", len(df))
    print("\nPARETO-EFFICIENT CONFIGURATIONS")
    print("-" * 90)
    print(
        scored.loc[
            scored["pareto_efficient"],
            [
                "dataset", "method", "config_id",
                "utility_score", "fidelity_score", "privacy_score",
                "batch_size", "discriminator_lr",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nSTRATEGY TABLE")
    print("-" * 90)
    print(strategies.round(3).to_string(index=False))

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
