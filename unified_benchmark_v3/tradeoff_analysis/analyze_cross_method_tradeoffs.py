#!/usr/bin/env python3
"""
Cross-method utility–fidelity–privacy trade-off analysis.

Main scientific separation
--------------------------
1. Primary layer:
   frozen main benchmark only (TVAE, CTGAN, CopulaGAN, ARF,
   Gaussian Copula, Conditional DDPM, LLM).

2. Exploratory overlay:
   only the non-baseline GAN configurations that were Pareto-efficient
   in the preceding within-GAN sensitivity analysis.

The tuned GAN points NEVER replace the frozen CTGAN/CopulaGAN primary rows.

Scoring
-------
Every raw metric is retained. Two descriptive scoring schemes are computed:
- within-dataset min–max normalization
- within-dataset rank normalization

A strategy is called "robust across scoring" only if both schemes select the
same candidate. Dimension/Pareto scores are summaries, not universal quality
metrics.

Privacy
-------
Composite privacy uses:
- |MIA AUROC - 0.5|, lower is better
- AIA risk, lower is better

DCR is retained descriptively but excluded from the composite privacy score.
No privacy metric here is a formal privacy guarantee.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADEOFF_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPOSITORY_ROOT / "unified_benchmark_v3"
PRIMARY_ROOT = BENCHMARK_ROOT / "results" / "main_benchmark"
GAN_OUTPUT = BENCHMARK_ROOT / "results" / "tradeoff_analysis"

OUT = BENCHMARK_ROOT / "results" / "cross_method_tradeoffs"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["pima", "cleveland", "ckd"]
PRIMARY_METHODS = [
    "REAL",
    "TVAE",
    "CTGAN",
    "CopulaGAN",
    "ARF",
    "Gaussian Copula",
    "Conditional DDPM",
    "LLM",
]
SYNTHETIC_PRIMARY_METHODS = [m for m in PRIMARY_METHODS if m != "REAL"]
EXPECTED_SPLITS = set(range(5))

RAW_METRICS = [
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

OBJECTIVES = {
    "utility_auroc": "high",
    "utility_f1": "high",
    "utility_brier": "low",
    "fidelity_pcd": "low",
    "fidelity_ws": "low",
    "fidelity_js": "low",
    "privacy_mia_distance": "low",
    "privacy_aia_risk": "low",
}

DIMENSIONS = {
    "utility": ["utility_auroc", "utility_f1", "utility_brier"],
    "fidelity": ["fidelity_pcd", "fidelity_ws", "fidelity_js"],
    "privacy": ["privacy_mia_distance", "privacy_aia_risk"],
}


def split_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("split_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                pass
    raise ValueError(f"Could not derive split from {path}")


def read_single(path: Path) -> dict:
    d = pd.read_csv(path)
    if len(d) != 1:
        raise SystemExit(f"{path}: expected exactly one row, found {len(d)}")
    return d.iloc[0].to_dict()


def load_primary_split_level() -> pd.DataFrame:
    rows = []

    for ds in DATASETS:
        for split in range(5):
            split_dir = PRIMARY_ROOT / ds / f"split_{split}"
            if not split_dir.exists():
                raise SystemExit(f"Missing primary split directory: {split_dir}")

            files = list(split_dir.rglob("main_results_row.csv"))
            local = []
            for f in files:
                r = read_single(f)
                method = str(r.get("method", ""))
                dataset = str(r.get("dataset", "")).lower()

                # Ignore anything that is not one of the frozen primary methods.
                if method not in PRIMARY_METHODS:
                    continue
                if dataset != ds:
                    raise SystemExit(
                        f"{f}: dataset={r.get('dataset')} but expected {ds}"
                    )

                r.update({
                    "dataset": ds,
                    "method": method,
                    "split_id": split,
                    "candidate_id": f"{method}::frozen",
                    "config_id": "frozen",
                    "source_layer": "primary_frozen",
                    "_path": str(f),
                })
                local.append(r)

            d = pd.DataFrame(local)
            if len(d) != 8:
                raise SystemExit(
                    f"{ds} split {split}: expected exactly 8 primary rows, "
                    f"found {len(d)}\n"
                    + (d[["method", "_path"]].to_string(index=False)
                       if len(d) else "No rows")
                )

            methods = set(d["method"])
            if methods != set(PRIMARY_METHODS):
                raise SystemExit(
                    f"{ds} split {split}: primary method inventory mismatch. "
                    f"Found {sorted(methods)}"
                )
            if d["method"].duplicated().any():
                raise SystemExit(
                    f"{ds} split {split}: duplicate primary methods found"
                )

            rows.extend(local)

    df = pd.DataFrame(rows)
    if len(df) != 120:
        raise SystemExit(f"Expected 120 frozen primary rows, found {len(df)}")
    return df


def load_exploratory_gan_pareto_split_level() -> pd.DataFrame:
    split_path = GAN_OUTPUT / "gan_tradeoff_split_level.csv"
    pareto_path = GAN_OUTPUT / "gan_pareto_front.csv"

    if not split_path.exists():
        raise SystemExit(
            f"Missing {split_path}. Run the within-GAN trade-off job first."
        )
    if not pareto_path.exists():
        raise SystemExit(
            f"Missing {pareto_path}. Run the within-GAN trade-off job first."
        )

    split = pd.read_csv(split_path)
    pareto = pd.read_csv(pareto_path)

    needed = {"dataset", "method", "config_id", "split_id", *RAW_METRICS}
    missing = needed - set(split.columns)
    if missing:
        raise SystemExit(
            f"{split_path} missing columns: {sorted(missing)}"
        )

    pareto_keys = pareto[["dataset", "method", "config_id"]].drop_duplicates()
    # Baseline stays represented by the frozen primary rows only.
    pareto_keys = pareto_keys[
        pareto_keys["config_id"].astype(str) != "baseline_b500_d2e4"
    ].copy()

    x = split.merge(
        pareto_keys,
        on=["dataset", "method", "config_id"],
        how="inner",
        validate="many_to_one",
    )

    if x.empty:
        raise SystemExit("No non-baseline GAN Pareto rows were selected.")

    x["candidate_id"] = (
        x["method"].astype(str) + "::" + x["config_id"].astype(str)
    )
    x["source_layer"] = "exploratory_tuned_gan"

    # Each selected candidate must have all five frozen splits.
    inventory = (
        x.groupby(["dataset", "method", "config_id"], as_index=False)
        .agg(
            n_rows=("split_id", "size"),
            n_splits=("split_id", "nunique"),
        )
    )
    bad = inventory[
        (inventory["n_rows"] != 5) | (inventory["n_splits"] != 5)
    ]
    if not bad.empty:
        raise SystemExit(
            "Incomplete exploratory GAN Pareto candidates:\n"
            + bad.to_string(index=False)
        )

    return x


def numericize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in RAW_METRICS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["privacy_mia_distance"] = (d["privacy_mia_auc"] - 0.5).abs()
    return d


def summarize_candidates(
    primary: pd.DataFrame,
    tuned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = numericize(primary)
    tuned = numericize(tuned)

    # REAL is a utility reference only.
    real = primary[primary["method"] == "REAL"].copy()
    synthetic_primary = primary[primary["method"] != "REAL"].copy()

    all_synth = pd.concat(
        [synthetic_primary, tuned],
        ignore_index=True,
        sort=False,
    )

    # No duplicate candidate/split keys.
    key = ["dataset", "candidate_id", "split_id"]
    if all_synth.duplicated(key).any():
        bad = all_synth[all_synth.duplicated(key, keep=False)]
        raise SystemExit(
            "Duplicate candidate/split rows:\n"
            + bad[key + ["source_layer"]].to_string(index=False)
        )

    agg = {}
    for c in RAW_METRICS + ["privacy_mia_distance"]:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_sd"] = (c, "std")

    summary = (
        all_synth.groupby(
            ["dataset", "method", "candidate_id", "config_id", "source_layer"],
            as_index=False,
        )
        .agg(**agg)
    )

    counts = (
        all_synth.groupby(
            ["dataset", "candidate_id"],
            as_index=False,
        )
        .agg(
            n_rows=("split_id", "size"),
            n_splits=("split_id", "nunique"),
        )
    )
    bad = counts[
        (counts["n_rows"] != 5) | (counts["n_splits"] != 5)
    ]
    if not bad.empty:
        raise SystemExit(
            "Cross-method candidate is not based on exactly five splits:\n"
            + bad.to_string(index=False)
        )

    real_summary = (
        real.groupby("dataset", as_index=False)
        .agg(
            real_auroc_mean=("utility_auroc", "mean"),
            real_auroc_sd=("utility_auroc", "std"),
            real_f1_mean=("utility_f1", "mean"),
            real_f1_sd=("utility_f1", "std"),
            real_brier_mean=("utility_brier", "mean"),
            real_brier_sd=("utility_brier", "std"),
        )
    )

    summary = summary.merge(
        real_summary,
        on="dataset",
        how="left",
        validate="many_to_one",
    )

    summary["auroc_gap_vs_real"] = (
        summary["utility_auroc_mean"] - summary["real_auroc_mean"]
    )
    summary["auroc_retention_vs_real"] = (
        summary["utility_auroc_mean"] / summary["real_auroc_mean"]
    )
    summary["f1_gap_vs_real"] = (
        summary["utility_f1_mean"] - summary["real_f1_mean"]
    )
    summary["brier_excess_vs_real"] = (
        summary["utility_brier_mean"] - summary["real_brier_mean"]
    )

    return summary, real_summary


def minmax_high_better(series: pd.Series, direction: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min()
    hi = s.max()

    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(np.nan, index=s.index)
    if np.isclose(hi, lo):
        return pd.Series(1.0, index=s.index)

    if direction == "high":
        return (s - lo) / (hi - lo)
    if direction == "low":
        return (hi - s) / (hi - lo)
    raise ValueError(direction)


def rank_high_better(
    df: pd.DataFrame,
    col: str,
    direction: str,
) -> pd.Series:
    ascending = direction == "low"
    rank = df[col].rank(method="average", ascending=ascending)
    n = len(df)
    if n <= 1:
        return pd.Series(1.0, index=df.index)
    return 1.0 - (rank - 1.0) / (n - 1.0)


def score_within_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for ds, sub in summary.groupby("dataset", sort=False):
        s = sub.copy()

        # Metric-level normalized scores.
        for metric, direction in OBJECTIVES.items():
            mean_col = metric + "_mean"

            s[metric + "_minmax_score"] = minmax_high_better(
                s[mean_col], direction
            )
            s[metric + "_rank_score"] = rank_high_better(
                s, mean_col, direction
            )

        # Dimension scores.
        for dim, metrics in DIMENSIONS.items():
            s[dim + "_minmax_score"] = s[
                [m + "_minmax_score" for m in metrics]
            ].mean(axis=1)
            s[dim + "_rank_score"] = s[
                [m + "_rank_score" for m in metrics]
            ].mean(axis=1)

        # Balanced summaries.
        for scheme in ["minmax", "rank"]:
            dims = [
                "utility_" + scheme + "_score",
                "fidelity_" + scheme + "_score",
                "privacy_" + scheme + "_score",
            ]
            s["balanced_" + scheme + "_mean"] = s[dims].mean(axis=1)
            s["balanced_" + scheme + "_min"] = s[dims].min(axis=1)

        parts.append(s)

    return pd.concat(parts, ignore_index=True)


def pareto_mask(
    df: pd.DataFrame,
    cols: list[str],
) -> np.ndarray:
    x = df[cols].to_numpy(float)
    flags = np.ones(len(df), dtype=bool)

    for i, a in enumerate(x):
        if not np.all(np.isfinite(a)):
            flags[i] = False
            continue
        for j, b in enumerate(x):
            if i == j or not np.all(np.isfinite(b)):
                continue
            if np.all(b >= a) and np.any(b > a):
                flags[i] = False
                break
    return flags


def add_dimension_pareto(scored: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ds, sub in scored.groupby("dataset", sort=False):
        s = sub.copy()
        for scheme in ["minmax", "rank"]:
            cols = [
                "utility_" + scheme + "_score",
                "fidelity_" + scheme + "_score",
                "privacy_" + scheme + "_score",
            ]
            s["pareto_" + scheme] = pareto_mask(s, cols)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def choose_one(
    sub: pd.DataFrame,
    scheme: str,
    strategy: str,
) -> pd.Series:
    if strategy == "utility_first":
        return sub.sort_values(
            [
                f"utility_{scheme}_score",
                f"balanced_{scheme}_mean",
                "utility_auroc_mean",
            ],
            ascending=False,
        ).iloc[0]

    if strategy == "fidelity_first":
        return sub.sort_values(
            [
                f"fidelity_{scheme}_score",
                f"balanced_{scheme}_mean",
                "utility_auroc_mean",
            ],
            ascending=False,
        ).iloc[0]

    if strategy == "privacy_first":
        return sub.sort_values(
            [
                f"privacy_{scheme}_score",
                f"balanced_{scheme}_mean",
                "utility_auroc_mean",
            ],
            ascending=False,
        ).iloc[0]

    if strategy == "balanced_pareto":
        p = sub[sub[f"pareto_{scheme}"]].copy()
        if p.empty:
            p = sub.copy()
        return p.sort_values(
            [
                f"balanced_{scheme}_min",
                f"balanced_{scheme}_mean",
                "utility_auroc_mean",
            ],
            ascending=False,
        ).iloc[0]

    raise ValueError(strategy)


def strategy_tables(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = [
        "utility_first",
        "fidelity_first",
        "privacy_first",
        "balanced_pareto",
    ]

    rows = []
    robust = []

    for ds, sub in scored.groupby("dataset", sort=False):
        chosen = {}
        for scheme in ["minmax", "rank"]:
            for strategy in strategies:
                r = choose_one(sub, scheme, strategy)
                chosen[(scheme, strategy)] = r["candidate_id"]

                rows.append({
                    "dataset": ds,
                    "scoring_scheme": scheme,
                    "strategy": strategy,
                    "candidate_id": r["candidate_id"],
                    "method": r["method"],
                    "config_id": r["config_id"],
                    "source_layer": r["source_layer"],
                    "AUROC": r["utility_auroc_mean"],
                    "F1": r["utility_f1_mean"],
                    "Brier": r["utility_brier_mean"],
                    "PCD": r["fidelity_pcd_mean"],
                    "WS": r["fidelity_ws_mean"],
                    "JS": r["fidelity_js_mean"],
                    "MIA_AUROC": r["privacy_mia_auc_mean"],
                    "MIA_distance": r["privacy_mia_distance_mean"],
                    "AIA_risk": r["privacy_aia_risk_mean"],
                    "DCR_descriptive": r["privacy_dcr_mean_mean"],
                    "AUROC_retention_vs_REAL": r["auroc_retention_vs_real"],
                    "utility_score": r[f"utility_{scheme}_score"],
                    "fidelity_score": r[f"fidelity_{scheme}_score"],
                    "privacy_score": r[f"privacy_{scheme}_score"],
                    "balanced_min_score": r[f"balanced_{scheme}_min"],
                    "pareto": bool(r[f"pareto_{scheme}"]),
                })

        for strategy in strategies:
            a = chosen[("minmax", strategy)]
            b = chosen[("rank", strategy)]
            robust.append({
                "dataset": ds,
                "strategy": strategy,
                "minmax_candidate": a,
                "rank_candidate": b,
                "robust_across_scoring": a == b,
            })

    return pd.DataFrame(rows), pd.DataFrame(robust)


def score_primary_only(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute normalization using ONLY frozen primary candidates.
    This prevents exploratory tuned GAN points from changing the primary
    benchmark scores or strategy recommendations.
    """
    primary_summary = summary[
        summary["source_layer"] == "primary_frozen"
    ].copy()
    scored = score_within_dataset(primary_summary)
    return add_dimension_pareto(scored)


def constraint_scenarios(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit what-if tables.
    - If utility must retain at least X% of REAL AUROC, what is the best
      empirical privacy or fidelity available?
    - If fidelity score must be at least X, what utility/privacy is available?
    - If privacy score must be at least X, what utility/fidelity is available?

    Min-max scores are used only as descriptive within-dataset constraints.
    """
    rows = []

    for ds, sub in scored.groupby("dataset", sort=False):
        # Utility retention constraints are interpretable in raw AUROC terms.
        for threshold in [0.80, 0.85, 0.90, 0.95, 0.98]:
            eligible = sub[
                sub["auroc_retention_vs_real"] >= threshold
            ].copy()
            if eligible.empty:
                continue

            best_priv = eligible.sort_values(
                ["privacy_minmax_score", "utility_minmax_score"],
                ascending=False,
            ).iloc[0]
            best_fid = eligible.sort_values(
                ["fidelity_minmax_score", "utility_minmax_score"],
                ascending=False,
            ).iloc[0]

            for objective, r in [
                ("best_privacy", best_priv),
                ("best_fidelity", best_fid),
            ]:
                rows.append({
                    "dataset": ds,
                    "constraint_type": "minimum_AUROC_retention_vs_REAL",
                    "threshold": threshold,
                    "objective_within_constraint": objective,
                    "candidate_id": r["candidate_id"],
                    "source_layer": r["source_layer"],
                    "AUROC_retention_vs_REAL": r["auroc_retention_vs_real"],
                    "AUROC": r["utility_auroc_mean"],
                    "utility_score": r["utility_minmax_score"],
                    "fidelity_score": r["fidelity_minmax_score"],
                    "privacy_score": r["privacy_minmax_score"],
                    "MIA_AUROC": r["privacy_mia_auc_mean"],
                    "AIA_risk": r["privacy_aia_risk_mean"],
                })

        # Descriptive normalized fidelity/privacy constraints.
        for threshold in [0.50, 0.70, 0.90]:
            eligible_f = sub[
                sub["fidelity_minmax_score"] >= threshold
            ].copy()
            if not eligible_f.empty:
                r = eligible_f.sort_values(
                    ["utility_minmax_score", "privacy_minmax_score"],
                    ascending=False,
                ).iloc[0]
                rows.append({
                    "dataset": ds,
                    "constraint_type": "minimum_fidelity_score",
                    "threshold": threshold,
                    "objective_within_constraint": "best_utility",
                    "candidate_id": r["candidate_id"],
                    "source_layer": r["source_layer"],
                    "AUROC_retention_vs_REAL": r["auroc_retention_vs_real"],
                    "AUROC": r["utility_auroc_mean"],
                    "utility_score": r["utility_minmax_score"],
                    "fidelity_score": r["fidelity_minmax_score"],
                    "privacy_score": r["privacy_minmax_score"],
                    "MIA_AUROC": r["privacy_mia_auc_mean"],
                    "AIA_risk": r["privacy_aia_risk_mean"],
                })

            eligible_p = sub[
                sub["privacy_minmax_score"] >= threshold
            ].copy()
            if not eligible_p.empty:
                r = eligible_p.sort_values(
                    ["utility_minmax_score", "fidelity_minmax_score"],
                    ascending=False,
                ).iloc[0]
                rows.append({
                    "dataset": ds,
                    "constraint_type": "minimum_privacy_score",
                    "threshold": threshold,
                    "objective_within_constraint": "best_utility",
                    "candidate_id": r["candidate_id"],
                    "source_layer": r["source_layer"],
                    "AUROC_retention_vs_REAL": r["auroc_retention_vs_real"],
                    "AUROC": r["utility_auroc_mean"],
                    "utility_score": r["utility_minmax_score"],
                    "fidelity_score": r["fidelity_minmax_score"],
                    "privacy_score": r["privacy_minmax_score"],
                    "MIA_AUROC": r["privacy_mia_auc_mean"],
                    "AIA_risk": r["privacy_aia_risk_mean"],
                })

    return pd.DataFrame(rows)


def make_plots(scored: pd.DataFrame):
    pairs = [
        ("utility_minmax_score", "fidelity_minmax_score", "Utility", "Fidelity"),
        ("utility_minmax_score", "privacy_minmax_score", "Utility", "Privacy"),
        ("fidelity_minmax_score", "privacy_minmax_score", "Fidelity", "Privacy"),
    ]

    for ds, sub in scored.groupby("dataset", sort=False):
        for x, y, xl, yl in pairs:
            fig, ax = plt.subplots(figsize=(8.5, 6.0))

            primary = sub[sub["source_layer"] == "primary_frozen"]
            tuned = sub[sub["source_layer"] == "exploratory_tuned_gan"]

            if not primary.empty:
                ax.scatter(
                    primary[x], primary[y],
                    marker="o", s=70, label="Primary frozen"
                )
            if not tuned.empty:
                ax.scatter(
                    tuned[x], tuned[y],
                    marker="^", s=85, label="Exploratory tuned GAN"
                )

            for _, r in sub.iterrows():
                label = r["candidate_id"]
                if r["pareto_minmax"]:
                    label += " *"
                ax.annotate(
                    label,
                    (r[x], r[y]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

            ax.set_xlim(-0.06, 1.06)
            ax.set_ylim(-0.06, 1.06)
            ax.set_xlabel(f"{xl} min–max score (higher = better)")
            ax.set_ylabel(f"{yl} min–max score (higher = better)")
            ax.set_title(
                f"{ds.upper()}: {xl} vs {yl}\n"
                "Primary benchmark + exploratory GAN Pareto overlay"
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(
                FIG / f"{ds}_{xl.lower()}_vs_{yl.lower()}.png",
                dpi=200,
            )
            plt.close(fig)

        # Raw AUROC retention vs privacy indicators.
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        ax.scatter(
            sub["auroc_retention_vs_real"],
            sub["privacy_aia_risk_mean"],
            s=75,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["candidate_id"],
                (r["auroc_retention_vs_real"], r["privacy_aia_risk_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("AUROC retention vs REAL")
        ax.set_ylabel("AIA risk (lower = better)")
        ax.set_title(f"{ds.upper()}: raw utility retention vs AIA risk")
        fig.tight_layout()
        fig.savefig(FIG / f"{ds}_auroc_retention_vs_aia.png", dpi=200)
        plt.close(fig)


def main():
    print("=" * 100)
    print("CROSS-METHOD TRADE-OFF ANALYSIS")
    print("=" * 100)
    print("REPOSITORY_ROOT:", REPOSITORY_ROOT)
    print("PRIMARY_ROOT:", PRIMARY_ROOT)
    print("GAN_OUTPUT:", GAN_OUTPUT)
    print()

    primary = load_primary_split_level()
    tuned = load_exploratory_gan_pareto_split_level()

    print(f"Frozen primary rows loaded: {len(primary)}")
    print(f"Exploratory tuned GAN split-level rows loaded: {len(tuned)}")

    primary.to_csv(OUT / "primary_frozen_split_level.csv", index=False)
    tuned.to_csv(OUT / "tuned_gan_pareto_split_level.csv", index=False)

    summary, real_summary = summarize_candidates(primary, tuned)
    summary.to_csv(OUT / "cross_method_raw_summary.csv", index=False)
    real_summary.to_csv(OUT / "real_reference_summary.csv", index=False)

    scored = score_within_dataset(summary)
    scored = add_dimension_pareto(scored)
    scored.to_csv(OUT / "cross_method_scores.csv", index=False)

    scored[scored["pareto_minmax"]].to_csv(
        OUT / "cross_method_dimension_pareto_minmax.csv",
        index=False,
    )
    scored[scored["pareto_rank"]].to_csv(
        OUT / "cross_method_dimension_pareto_rank.csv",
        index=False,
    )

    # Primary-only analysis is independently normalized.
    primary_only = score_primary_only(summary)
    primary_only.to_csv(
        OUT / "primary_only_tradeoff_scores.csv",
        index=False,
    )
    primary_only[primary_only["pareto_minmax"]].to_csv(
        OUT / "primary_only_dimension_pareto_minmax.csv",
        index=False,
    )
    primary_only[primary_only["pareto_rank"]].to_csv(
        OUT / "primary_only_dimension_pareto_rank.csv",
        index=False,
    )

    primary_strategies, primary_robustness = strategy_tables(primary_only)
    primary_strategies.to_csv(
        OUT / "primary_only_strategies.csv",
        index=False,
    )
    primary_robustness.to_csv(
        OUT / "primary_only_strategy_robustness.csv",
        index=False,
    )

    # Overlay analysis: frozen primary + exploratory tuned GAN Pareto points.
    strategies, robustness = strategy_tables(scored)
    strategies.to_csv(OUT / "cross_method_strategies.csv", index=False)
    robustness.to_csv(
        OUT / "cross_method_strategy_robustness.csv",
        index=False,
    )

    scenarios = constraint_scenarios(scored)
    scenarios.to_csv(
        OUT / "cross_method_constraint_scenarios.csv",
        index=False,
    )

    make_plots(scored)

    print("\nCANDIDATE INVENTORY")
    print("-" * 100)
    inv = (
        summary.groupby(
            ["dataset", "source_layer"], as_index=False
        )
        .size()
        .rename(columns={"size": "n_candidates"})
    )
    print(inv.to_string(index=False))

    print("\nPRIMARY-ONLY STRATEGY ROBUSTNESS: MINMAX vs RANK")
    print("-" * 100)
    print(primary_robustness.to_string(index=False))

    print("\nOVERLAY STRATEGY ROBUSTNESS: MINMAX vs RANK")
    print("-" * 100)
    print(robustness.to_string(index=False))

    print("\nOVERLAY MINMAX STRATEGIES")
    print("-" * 100)
    show = strategies[strategies["scoring_scheme"] == "minmax"][
        [
            "dataset", "strategy", "candidate_id", "source_layer",
            "AUROC", "AUROC_retention_vs_REAL",
            "utility_score", "fidelity_score", "privacy_score",
            "MIA_AUROC", "AIA_risk",
        ]
    ].copy()
    print(show.round(4).to_string(index=False))

    print("\nSaved to:", OUT)
    print("\nInterpretation discipline:")
    print(" - frozen primary benchmark remains unchanged")
    print(" - tuned GANs are exploratory overlay only")
    print(" - raw metrics are primary evidence")
    print(" - DCR is descriptive and excluded from privacy composite")
    print(" - minmax/rank scores are descriptive within-dataset summaries")
    print(" - disagreement between scoring schemes must be reported")


if __name__ == "__main__":
    main()
