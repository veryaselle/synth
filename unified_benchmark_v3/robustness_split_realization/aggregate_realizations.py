#!/usr/bin/env python3
"""Aggregate multiple supplementary split-realization robustness runs.

Each realization is first compared independently with the frozen primary benchmark by
``compare_with_primary.py``. This script then summarizes those already-finalized
comparisons without pooling split-level observations into a larger primary benchmark.

The intended thesis use is descriptive robustness evidence: quantify whether broad
method rankings and winner-level conclusions persist across independently shuffled
train/test partition realizations while the generator-seed schedule and evaluator seed
remain fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SEEDS = [2026, 3719704]
EXPECTED_DATASETS = ["pima", "cleveland", "ckd"]
EXPECTED_METRICS = [
    "utility_auroc",
    "utility_f1",
    "utility_brier",
    "fidelity_pcd",
    "fidelity_ws",
    "fidelity_js",
    "privacy_mia_distance",
    "privacy_aia_risk",
]
EXPECTED_DIMENSIONS = ["utility", "fidelity", "privacy"]
EXPECTED_SCHEMES = ["minmax", "rank"]
EXPECTED_METHODS = [
    "TVAE",
    "CTGAN",
    "CopulaGAN",
    "ARF",
    "Gaussian Copula",
    "Conditional DDPM",
    "LLM",
]
METRIC_TO_DIMENSION = {
    "utility_auroc": "utility",
    "utility_f1": "utility",
    "utility_brier": "utility",
    "fidelity_pcd": "fidelity",
    "fidelity_ws": "fidelity",
    "fidelity_js": "fidelity",
    "privacy_mia_distance": "privacy",
    "privacy_aia_risk": "privacy",
}


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = sorted(s[out.isna()].astype(str).unique().tolist())
        raise ValueError(f"Cannot parse boolean values: {bad}")
    return out.astype(bool)


def _leader_set(value: object) -> tuple[str, ...]:
    """Canonicalize 'A / B' leader strings so ordering cannot create false changes."""
    parts = [x.strip() for x in str(value).split("/") if x.strip()]
    return tuple(sorted(set(parts)))


def _leader_string(value: object) -> str:
    return " / ".join(_leader_set(value))


def _validate_rank(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    label = f"seed {seed} metric_rank_agreement.csv"
    _require_columns(
        df,
        ["dataset", "metric", "direction", "spearman_primary_vs_robust", "primary_source"],
        label,
    )
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.lower()
    expected = {(d, m) for d in EXPECTED_DATASETS for m in EXPECTED_METRICS}
    got = set(zip(out["dataset"], out["metric"]))
    if got != expected or len(out) != len(expected):
        raise ValueError(
            f"{label} inventory mismatch: missing={sorted(expected-got)}, "
            f"extra={sorted(got-expected)}, rows={len(out)}"
        )
    if out.duplicated(["dataset", "metric"]).any():
        raise ValueError(f"{label} contains duplicate dataset/metric rows")
    rho = pd.to_numeric(out["spearman_primary_vs_robust"], errors="coerce")
    if rho.isna().any() or ((rho < -1) | (rho > 1)).any():
        raise ValueError(f"{label} contains invalid Spearman coefficients")
    out["spearman_primary_vs_robust"] = rho
    out["dimension"] = out["metric"].map(METRIC_TO_DIMENSION)
    out["realization_seed"] = seed
    return out


def _validate_dimension_leaders(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    label = f"seed {seed} dimension_leader_stability.csv"
    _require_columns(
        df,
        ["dataset", "dimension", "scoring_scheme", "primary_leader", "robust_leader", "same_leader"],
        label,
    )
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.lower()
    expected = {
        (d, dim, scheme)
        for d in EXPECTED_DATASETS
        for dim in EXPECTED_DIMENSIONS
        for scheme in EXPECTED_SCHEMES
    }
    got = set(zip(out["dataset"], out["dimension"], out["scoring_scheme"]))
    if got != expected or len(out) != len(expected):
        raise ValueError(
            f"{label} inventory mismatch: missing={sorted(expected-got)}, "
            f"extra={sorted(got-expected)}, rows={len(out)}"
        )
    if out.duplicated(["dataset", "dimension", "scoring_scheme"]).any():
        raise ValueError(f"{label} contains duplicate keys")
    out["same_leader"] = _bool_series(out["same_leader"])
    out["primary_leader"] = out["primary_leader"].map(_leader_string)
    out["robust_leader"] = out["robust_leader"].map(_leader_string)
    out["realization_seed"] = seed
    return out


def _validate_metric_leaders(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    label = f"seed {seed} metric_leader_stability.csv"
    _require_columns(
        df,
        ["dataset", "level", "criterion", "primary_leader", "robust_leader", "same_leader_set"],
        label,
    )
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.lower()
    expected = {(d, m) for d in EXPECTED_DATASETS for m in EXPECTED_METRICS}
    got = set(zip(out["dataset"], out["criterion"]))
    if got != expected or len(out) != len(expected):
        raise ValueError(
            f"{label} inventory mismatch: missing={sorted(expected-got)}, "
            f"extra={sorted(got-expected)}, rows={len(out)}"
        )
    if out.duplicated(["dataset", "criterion"]).any():
        raise ValueError(f"{label} contains duplicate keys")
    out["same_leader_set"] = _bool_series(out["same_leader_set"])
    out["primary_leader"] = out["primary_leader"].map(_leader_string)
    out["robust_leader"] = out["robust_leader"].map(_leader_string)
    out["dimension"] = out["criterion"].map(METRIC_TO_DIMENSION)
    out["realization_seed"] = seed
    return out


def _validate_retention(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    label = f"seed {seed} robust_auroc_retention.csv"
    _require_columns(
        df,
        ["dataset", "method", "utility_auroc", "real_auroc_robust", "auroc_retention_vs_robust_real"],
        label,
    )
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.lower()
    expected = {(d, m) for d in EXPECTED_DATASETS for m in EXPECTED_METHODS}
    got = set(zip(out["dataset"], out["method"]))
    if got != expected or len(out) != len(expected):
        raise ValueError(
            f"{label} inventory mismatch: missing={sorted(expected-got)}, "
            f"extra={sorted(got-expected)}, rows={len(out)}"
        )
    if out.duplicated(["dataset", "method"]).any():
        raise ValueError(f"{label} contains duplicate dataset/method rows")
    for c in ["utility_auroc", "real_auroc_robust", "auroc_retention_vs_robust_real"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].isna().any():
            raise ValueError(f"{label} contains non-numeric values in {c}")
    out["realization_seed"] = seed
    return out


def _check_primary_consistency(df: pd.DataFrame, keys: list[str], label: str) -> None:
    counts = df.groupby(keys, dropna=False)["primary_leader"].nunique(dropna=False)
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(f"Primary leader is inconsistent across realizations in {label}: {bad.to_dict()}")


def _cross_leader_table(
    df: pd.DataFrame,
    *,
    keys: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    _check_primary_consistency(df, keys, "/".join(keys))
    rows: list[dict[str, object]] = []
    for key_vals, sub in df.groupby(keys, sort=False, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals))
        primary = _leader_string(sub["primary_leader"].iloc[0])
        row["primary_leader"] = primary
        robust_values: list[str] = []
        for seed in seeds:
            one = sub[sub["realization_seed"] == seed]
            if len(one) != 1:
                raise ValueError(f"Expected one row for {keys}={key_vals}, seed={seed}; found {len(one)}")
            leader = _leader_string(one["robust_leader"].iloc[0])
            row[f"seed_{seed}_leader"] = leader
            robust_values.append(leader)
        row["n_unique_supplementary_leader_sets"] = len(set(robust_values))
        row["all_supplementary_exact_same"] = len(set(robust_values)) == 1
        row["supplementary_consensus_leader"] = robust_values[0] if len(set(robust_values)) == 1 else ""
        row["any_supplementary_matches_primary"] = any(x == primary for x in robust_values)
        row["all_supplementary_match_primary"] = all(x == primary for x in robust_values)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_rho(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Supplementary realization seeds to aggregate (default: 2026 3719704).",
    )
    p.add_argument(
        "--results_root",
        default=str(THIS_DIR / "results"),
        help="Root containing seed_<SEED>/tables/primary_comparison directories.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <results_root>/across_realizations).",
    )
    args = p.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise SystemExit("At least two distinct supplementary realization seeds are required")

    results_root = Path(args.results_root)
    outdir = Path(args.outdir) if args.outdir else results_root / "across_realizations"
    outdir.mkdir(parents=True, exist_ok=True)

    ranks: list[pd.DataFrame] = []
    dims: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    retentions: list[pd.DataFrame] = []

    for seed in seeds:
        base = results_root / f"seed_{seed}" / "tables" / "primary_comparison"
        required = {
            "rank": base / "metric_rank_agreement.csv",
            "dimension": base / "dimension_leader_stability.csv",
            "metric": base / "metric_leader_stability.csv",
            "retention": base / "robust_auroc_retention.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise SystemExit(f"Seed {seed}: missing finalized comparison files: {missing}")
        ranks.append(_validate_rank(pd.read_csv(required["rank"]), seed))
        dims.append(_validate_dimension_leaders(pd.read_csv(required["dimension"]), seed))
        metrics.append(_validate_metric_leaders(pd.read_csv(required["metric"]), seed))
        retentions.append(_validate_retention(pd.read_csv(required["retention"]), seed))

    rank_df = pd.concat(ranks, ignore_index=True)
    dim_df = pd.concat(dims, ignore_index=True)
    metric_df = pd.concat(metrics, ignore_index=True)
    retention_df = pd.concat(retentions, ignore_index=True)

    # The per-seed comparisons must all use the same primary source description.
    primary_sources = sorted(rank_df["primary_source"].dropna().astype(str).unique().tolist())
    if len(primary_sources) != 1:
        raise ValueError(f"Realizations use inconsistent primary sources: {primary_sources}")

    overview_rows = []
    for seed in seeds:
        r = rank_df[rank_df["realization_seed"] == seed]
        d = dim_df[dim_df["realization_seed"] == seed]
        m = metric_df[metric_df["realization_seed"] == seed]
        overview_rows.append(
            {
                "realization_seed": seed,
                "mean_metric_rank_spearman": r["spearman_primary_vs_robust"].mean(),
                "median_metric_rank_spearman": r["spearman_primary_vs_robust"].median(),
                "dimension_leader_primary_agreement_fraction": d["same_leader"].mean(),
                "metric_leader_primary_agreement_fraction": m["same_leader_set"].mean(),
            }
        )
    overview = pd.DataFrame(overview_rows)
    overview.to_csv(outdir / "realization_overview.csv", index=False)

    by_ds_dim = (
        rank_df.groupby(["realization_seed", "dataset", "dimension"], as_index=False)
        .agg(
            n_metrics=("metric", "size"),
            mean_spearman=("spearman_primary_vs_robust", "mean"),
            median_spearman=("spearman_primary_vs_robust", "median"),
            min_spearman=("spearman_primary_vs_robust", "min"),
            max_spearman=("spearman_primary_vs_robust", "max"),
        )
        .sort_values(["realization_seed", "dataset", "dimension"])
    )
    by_ds_dim.to_csv(outdir / "rank_agreement_by_dataset_dimension.csv", index=False)

    by_dim = (
        rank_df.groupby("dimension", as_index=False)
        .agg(
            n_coefficients=("spearman_primary_vs_robust", "size"),
            mean_spearman=("spearman_primary_vs_robust", "mean"),
            median_spearman=("spearman_primary_vs_robust", "median"),
            min_spearman=("spearman_primary_vs_robust", "min"),
            max_spearman=("spearman_primary_vs_robust", "max"),
        )
        .sort_values("mean_spearman", ascending=False)
    )
    by_dim.to_csv(outdir / "rank_agreement_by_dimension.csv", index=False)

    dim_cross = _cross_leader_table(
        dim_df,
        keys=["dataset", "dimension", "scoring_scheme"],
        seeds=seeds,
    )
    dim_cross.to_csv(outdir / "dimension_leader_cross_realization.csv", index=False)

    metric_cross = _cross_leader_table(
        metric_df,
        keys=["dataset", "dimension", "criterion"],
        seeds=seeds,
    )
    metric_cross.to_csv(outdir / "metric_leader_cross_realization.csv", index=False)

    retention_df = retention_df[
        [
            "realization_seed",
            "dataset",
            "method",
            "utility_auroc",
            "real_auroc_robust",
            "auroc_retention_vs_robust_real",
        ]
    ].sort_values(["realization_seed", "dataset", "method"])
    retention_df.to_csv(outdir / "auroc_retention_all_realizations.csv", index=False)
    retention_summary = (
        retention_df.groupby(["dataset", "method"], as_index=False)
        .agg(
            n_realizations=("realization_seed", "nunique"),
            mean_auroc_retention=("auroc_retention_vs_robust_real", "mean"),
            min_auroc_retention=("auroc_retention_vs_robust_real", "min"),
            max_auroc_retention=("auroc_retention_vs_robust_real", "max"),
        )
        .sort_values(["dataset", "method"])
    )
    retention_summary.to_csv(outdir / "auroc_retention_summary.csv", index=False)

    pooled = rank_df["spearman_primary_vs_robust"]
    dim_same_n = int(dim_cross["all_supplementary_exact_same"].sum())
    metric_same_n = int(metric_cross["all_supplementary_exact_same"].sum())
    dim_total = len(dim_cross)
    metric_total = len(metric_cross)

    dimension_order = by_dim["dimension"].tolist()
    summary = {
        "experiment": "cross-realization summary of supplementary split robustness checks",
        "realization_seeds": seeds,
        "primary_source": primary_sources[0],
        "n_metric_rank_coefficients": int(len(rank_df)),
        "pooled_mean_metric_rank_spearman": float(pooled.mean()),
        "pooled_median_metric_rank_spearman": float(pooled.median()),
        "rank_agreement_by_dimension": {
            row["dimension"]: {
                "mean": float(row["mean_spearman"]),
                "median": float(row["median_spearman"]),
                "min": float(row["min_spearman"]),
                "max": float(row["max_spearman"]),
            }
            for _, row in by_dim.iterrows()
        },
        "supplementary_dimension_leader_exact_agreement_fraction": float(dim_same_n / dim_total),
        "supplementary_metric_leader_exact_agreement_fraction": float(metric_same_n / metric_total),
        "dimension_leader_primary_agreement_by_seed": {
            str(int(row["realization_seed"])): float(row["dimension_leader_primary_agreement_fraction"])
            for _, row in overview.iterrows()
        },
        "interpretation_boundary": (
            "The frozen primary benchmark remains the confirmatory reference. Supplementary "
            "realizations are summarized separately and are not pooled into a larger primary "
            "benchmark. Generator seeds are not independently varied in this analysis."
        ),
    }
    (outdir / "cross_realization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Cross-realization split-robustness summary",
        "",
        f"- Supplementary realization seeds: `{', '.join(str(s) for s in seeds)}`",
        f"- Primary source: {primary_sources[0]}",
        f"- Pooled mean metric-level Spearman agreement with primary: `{pooled.mean():.3f}`",
        f"- Pooled median metric-level Spearman agreement with primary: `{pooled.median():.3f}`",
        f"- Exact dimension-leader agreement between supplementary realizations: "
        f"`{dim_same_n}/{dim_total} = {dim_same_n/dim_total:.1%}`",
        f"- Exact metric-leader agreement between supplementary realizations: "
        f"`{metric_same_n}/{metric_total} = {metric_same_n/metric_total:.1%}`",
        "",
        "## Per-realization agreement with the frozen primary benchmark",
        "",
    ]
    for _, row in overview.iterrows():
        lines.append(
            f"- seed {int(row['realization_seed'])}: mean rho="
            f"`{row['mean_metric_rank_spearman']:.3f}`, median rho="
            f"`{row['median_metric_rank_spearman']:.3f}`, dimension-leader agreement="
            f"`{row['dimension_leader_primary_agreement_fraction']:.1%}`"
        )

    lines += ["", "## Rank agreement by evaluation dimension", ""]
    for _, row in by_dim.iterrows():
        lines.append(
            f"- {row['dimension']}: mean rho=`{row['mean_spearman']:.3f}`, "
            f"median rho=`{row['median_spearman']:.3f}`, "
            f"range=`{row['min_spearman']:.3f}` to `{row['max_spearman']:.3f}`"
        )

    if dimension_order:
        strongest = dimension_order[0]
        weakest = dimension_order[-1]
        lines += [
            "",
            "## Descriptive interpretation",
            "",
            f"Across the supplied supplementary realizations, `{strongest}` showed the strongest "
            f"average rank agreement with the frozen primary benchmark, whereas `{weakest}` showed "
            "the weakest. Winner identity was less stable than the broader rank structure, so the "
            "robustness evidence supports broad performance patterns more strongly than claims about "
            "a single universally best generator.",
            "",
            "The frozen primary benchmark remains the confirmatory reference. These supplementary "
            "realizations are not pooled into a new primary benchmark, and they do not constitute a "
            "complete repeated-generator-seed study.",
        ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Cross-realization outputs written to {outdir}")


if __name__ == "__main__":
    main()
