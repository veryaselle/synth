#!/usr/bin/env python3
"""Compare one supplementary split realization with the primary frozen benchmark.

The default primary reference is the thesis-reported mean table already tracked in
results/final_thesis/reported_primary_means.csv. If the unrounded primary numeric
summary is available on the execution system, pass it with --primary_summary for
more precise rank comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent
DEFAULT_SEED = 2026
SYNTHETIC_METHODS = [
    "TVAE",
    "CTGAN",
    "CopulaGAN",
    "ARF",
    "Gaussian Copula",
    "Conditional DDPM",
    "LLM",
]

METRICS: Dict[str, str] = {
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


def _normalize_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either the thesis-reported table or assembler numeric summary."""
    out = df.copy()
    if "AUROC" in out.columns:
        rename = {
            "AUROC": "utility_auroc",
            "F1": "utility_f1",
            "Brier": "utility_brier",
            "PCD": "fidelity_pcd",
            "WS": "fidelity_ws",
            "JS": "fidelity_js",
            "DCR": "privacy_dcr_mean",
            "MIA_AUROC": "privacy_mia_auc",
            "AIA_risk": "privacy_aia_risk",
        }
        out = out.rename(columns=rename)
    elif "utility_auroc_mean" in out.columns:
        rename = {
            c: c[:-5]
            for c in out.columns
            if c.endswith("_mean")
        }
        out = out.rename(columns=rename)

    needed = {
        "dataset", "method", "utility_auroc", "utility_f1", "utility_brier",
        "fidelity_pcd", "fidelity_ws", "fidelity_js", "privacy_dcr_mean",
        "privacy_mia_auc", "privacy_aia_risk",
    }
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Primary summary missing columns: {sorted(missing)}")
    return out[list(needed)].copy()


def _normalize_robust(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "utility_auroc_mean" not in out.columns:
        raise ValueError("Robust summary must be assemble_main_tables.py numeric output")
    rename = {c: c[:-5] for c in out.columns if c.endswith("_mean")}
    out = out.rename(columns=rename)
    return out


def _add_privacy_distance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["privacy_mia_distance"] = (pd.to_numeric(out["privacy_mia_auc"], errors="coerce") - 0.5).abs()
    return out


def _metric_rank(series: pd.Series, direction: str) -> pd.Series:
    ascending = direction == "low"
    return pd.to_numeric(series, errors="coerce").rank(method="average", ascending=ascending)


def _high_better_minmax(series: pd.Series, direction: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(np.nan, index=x.index)
    if np.isclose(lo, hi):
        return pd.Series(1.0, index=x.index)
    return (x - lo) / (hi - lo) if direction == "high" else (hi - x) / (hi - lo)


def _dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dataset, sub in df.groupby("dataset", sort=False):
        s = sub.copy()
        for metric, direction in METRICS.items():
            s[f"{metric}_minmax_score"] = _high_better_minmax(s[metric], direction)
            rank = _metric_rank(s[metric], direction)
            n = len(s)
            s[f"{metric}_rank_score"] = 1.0 if n <= 1 else 1.0 - (rank - 1.0) / (n - 1.0)
        for dim, metrics in DIMENSIONS.items():
            s[f"{dim}_minmax_score"] = s[[f"{m}_minmax_score" for m in metrics]].mean(axis=1)
            s[f"{dim}_rank_score"] = s[[f"{m}_rank_score" for m in metrics]].mean(axis=1)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def _leader(series: pd.Series) -> str:
    mx = series.max()
    winners = series.index[np.isclose(series.to_numpy(float), float(mx), rtol=0, atol=1e-12)].tolist()
    return " / ".join(sorted(str(x) for x in winners))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--robust_summary", default=None)
    p.add_argument("--primary_summary", default=None)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    robust_summary = (
        Path(args.robust_summary)
        if args.robust_summary
        else THIS_DIR / "results" / f"seed_{args.seed}" / "tables" / "main_results_summary_numeric.csv"
    )
    if args.primary_summary:
        primary_summary = Path(args.primary_summary)
        primary_source = "user-supplied primary summary"
    else:
        exact_candidate = BENCHMARK_ROOT / "results" / "main_benchmark" / "tables" / "main_results_summary_numeric.csv"
        if exact_candidate.exists():
            primary_summary = exact_candidate
            primary_source = "unrounded primary numeric summary"
        else:
            primary_summary = BENCHMARK_ROOT / "results" / "final_thesis" / "reported_primary_means.csv"
            primary_source = "thesis-reported rounded primary means"

    outdir = Path(args.outdir) if args.outdir else robust_summary.parent / "primary_comparison"
    outdir.mkdir(parents=True, exist_ok=True)

    primary = _add_privacy_distance(_normalize_primary(pd.read_csv(primary_summary)))
    robust = _add_privacy_distance(_normalize_robust(pd.read_csv(robust_summary)))

    primary = primary[primary["method"].isin(SYNTHETIC_METHODS)].copy()
    robust = robust[robust["method"].isin(SYNTHETIC_METHODS)].copy()

    expected = {(d, m) for d in ["pima", "cleveland", "ckd"] for m in SYNTHETIC_METHODS}
    got = set(zip(robust["dataset"].str.lower(), robust["method"]))
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise SystemExit(f"Robust summary inventory mismatch. missing={missing}, extra={extra}")

    primary["dataset"] = primary["dataset"].str.lower()
    robust["dataset"] = robust["dataset"].str.lower()

    joined = primary.merge(
        robust,
        on=["dataset", "method"],
        suffixes=("_primary", "_robust"),
        validate="one_to_one",
    )

    detail_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    leader_rows: List[Dict[str, object]] = []

    for dataset, sub in joined.groupby("dataset", sort=False):
        for metric, direction in METRICS.items():
            pcol = f"{metric}_primary"
            rcol = f"{metric}_robust"
            p_rank = _metric_rank(sub[pcol], direction)
            r_rank = _metric_rank(sub[rcol], direction)
            rho = spearmanr(p_rank, r_rank, nan_policy="omit").statistic

            for idx, row in sub.iterrows():
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "method": row["method"],
                        "metric": metric,
                        "primary_mean": row[pcol],
                        "robust_mean": row[rcol],
                        "delta_robust_minus_primary": row[rcol] - row[pcol],
                        "primary_rank": float(p_rank.loc[idx]),
                        "robust_rank": float(r_rank.loc[idx]),
                        "rank_change": float(r_rank.loc[idx] - p_rank.loc[idx]),
                    }
                )

            rank_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "direction": direction,
                    "spearman_primary_vs_robust": rho,
                    "primary_source": primary_source,
                }
            )

            # Metric-level winner(s), respecting direction.
            p_best = sub[pcol].max() if direction == "high" else sub[pcol].min()
            r_best = sub[rcol].max() if direction == "high" else sub[rcol].min()
            p_methods = sorted(sub.loc[np.isclose(sub[pcol], p_best), "method"].tolist())
            r_methods = sorted(sub.loc[np.isclose(sub[rcol], r_best), "method"].tolist())
            leader_rows.append(
                {
                    "dataset": dataset,
                    "level": "metric",
                    "criterion": metric,
                    "primary_leader": " / ".join(p_methods),
                    "robust_leader": " / ".join(r_methods),
                    "same_leader_set": p_methods == r_methods,
                }
            )

    pd.DataFrame(detail_rows).to_csv(outdir / "metric_method_comparison.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(outdir / "metric_rank_agreement.csv", index=False)

    # Compare thesis-style dimension leaders under both min-max and rank normalization.
    # For the primary side, prefer the stored strategy table produced from the unrounded
    # primary split-level results. This avoids creating artificial ties from the 3-decimal
    # thesis display table.
    r_scored = _dimension_scores(robust)
    strategy_path = BENCHMARK_ROOT / "results" / "final_thesis" / "primary_frozen_strategy_robustness.csv"
    strategy_df = pd.read_csv(strategy_path) if strategy_path.exists() else None
    strategy_map = {
        "utility": "utility_first",
        "fidelity": "fidelity_first",
        "privacy": "privacy_first",
    }
    p_scored = _dimension_scores(primary) if strategy_df is None else None

    def clean_candidate(value: object) -> str:
        text = str(value)
        return text.replace("::frozen", "")

    dimension_rows: List[Dict[str, object]] = []
    for dataset in ["pima", "cleveland", "ckd"]:
        rs = r_scored[r_scored["dataset"] == dataset].set_index("method")
        ps = p_scored[p_scored["dataset"] == dataset].set_index("method") if p_scored is not None else None
        for dim in ["utility", "fidelity", "privacy"]:
            for scheme in ["minmax", "rank"]:
                col = f"{dim}_{scheme}_score"
                if strategy_df is not None:
                    row = strategy_df[
                        (strategy_df["dataset"].astype(str).str.lower() == dataset)
                        & (strategy_df["strategy"] == strategy_map[dim])
                    ]
                    if len(row) != 1:
                        raise ValueError(f"Primary strategy table missing {dataset}/{dim}")
                    p_leader = clean_candidate(row.iloc[0][f"{scheme}_candidate"])
                else:
                    p_leader = _leader(ps[col])
                r_leader = _leader(rs[col])
                dimension_rows.append(
                    {
                        "dataset": dataset,
                        "dimension": dim,
                        "scoring_scheme": scheme,
                        "primary_leader": p_leader,
                        "robust_leader": r_leader,
                        "same_leader": p_leader == r_leader,
                    }
                )
    pd.DataFrame(dimension_rows).to_csv(outdir / "dimension_leader_stability.csv", index=False)
    leader_df = pd.DataFrame(leader_rows)
    leader_df.to_csv(outdir / "metric_leader_stability.csv", index=False)

    # Real-reference AUROC and retention under this supplementary realization.
    robust_all = _normalize_robust(pd.read_csv(robust_summary))
    real = robust_all[robust_all["method"] == "REAL"]["dataset utility_auroc".split()].rename(
        columns={"utility_auroc": "real_auroc_robust"}
    )
    synth = robust_all[robust_all["method"].isin(SYNTHETIC_METHODS)].copy()
    retention = synth.merge(real, on="dataset", validate="many_to_one")
    retention["auroc_retention_vs_robust_real"] = retention["utility_auroc"] / retention["real_auroc_robust"]
    retention[["dataset", "method", "utility_auroc", "real_auroc_robust", "auroc_retention_vs_robust_real"]].to_csv(
        outdir / "robust_auroc_retention.csv", index=False
    )

    rank_df = pd.DataFrame(rank_rows)
    dim_df = pd.DataFrame(dimension_rows)
    summary = {
        "experiment": "independent split-realization robustness check",
        "realization_seed": args.seed,
        "primary_source": primary_source,
        "primary_summary": str(primary_summary),
        "robust_summary": str(robust_summary),
        "mean_metric_rank_spearman": float(rank_df["spearman_primary_vs_robust"].mean()),
        "median_metric_rank_spearman": float(rank_df["spearman_primary_vs_robust"].median()),
        "dimension_leader_agreement_fraction": float(dim_df["same_leader"].mean()),
        "interpretation_note": (
            "This experiment probes robustness to an independently shuffled supplementary split realization. "
            "It does not replace the primary frozen benchmark and does not fully isolate generator "
            "stochasticity. Metric-level rank agreement may use thesis-reported rounded means when the "
            "unrounded primary summary is unavailable; stored primary strategy selections are used for "
            "dimension-leader comparison when present."
        ),
    }
    (outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Independent split-realization robustness summary",
        "",
        f"- Realization seed: `{args.seed}`",
        f"- Primary source: {primary_source}",
        f"- Mean metric-level Spearman agreement: `{summary['mean_metric_rank_spearman']:.3f}`",
        f"- Median metric-level Spearman agreement: `{summary['median_metric_rank_spearman']:.3f}`",
        f"- Dimension-leader agreement across dataset × dimension × scoring scheme: `{summary['dimension_leader_agreement_fraction']:.1%}`",
        "",
        "## Dimension leaders",
        "",
    ]
    for _, r in dim_df.iterrows():
        marker = "stable" if r["same_leader"] else "changed"
        lines.append(
            f"- {r['dataset']} / {r['dimension']} / {r['scoring_scheme']}: "
            f"{r['primary_leader']} -> {r['robust_leader']} ({marker})"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is supplementary robustness evidence. The primary frozen benchmark remains the confirmatory reference. "
        "A changed leader is informative rather than a failure: it identifies conclusions that are sensitive to the train/test partition realization.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Comparison outputs written to {outdir}")


if __name__ == "__main__":
    main()
