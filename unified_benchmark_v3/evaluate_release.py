#!/usr/bin/env python3
"""Unified evaluator for the 3-dataset × 7-method thesis benchmark.

Contract
--------
Inputs are a frozen processed real training split, its untouched processed real
held-out test split, and one 1× synthetic release generated from the training
split only.

Primary metrics
---------------
Utility (TSTR): AUROC ↑, F1@0.5 ↑, Brier ↓
Fidelity:       PCD ↓, normalized Wasserstein ↓, Jensen-Shannon ↓
Privacy:        DCR, MIA AUROC (0.5 ~ chance), AIA normalized excess risk (0 best)

Important
---------
* Utility preprocessing is fitted on SYNTHETIC data only. This avoids leaking
  real-train preprocessing statistics into TSTR classifier training.
* Fidelity/privacy encoders are fitted on REAL TRAIN only, because real train is
  the reference distribution for these audits.
* DCR, MIA and AIA are empirical privacy-risk indicators, not formal guarantees.
* AIA uses the target/diagnosis as the hidden attribute by default. Its main risk
  statistic compares attack success on training members with success on held-out
  controls, reducing confusion between population-level predictability and
  train-specific leakage.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

RANDOM_SEED = 42


@dataclass
class FeatureSchema:
    target: str
    numeric: List[str]
    categorical: List[str]


def parse_columns(text: Optional[str]) -> Optional[List[str]]:
    if text is None or text.strip() == "":
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def infer_schema(real_train: pd.DataFrame, target: str,
                 numeric_cols: Optional[List[str]] = None,
                 categorical_cols: Optional[List[str]] = None) -> FeatureSchema:
    features = [c for c in real_train.columns if c != target]
    if numeric_cols is None and categorical_cols is None:
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(real_train[c])]
        categorical = [c for c in features if c not in numeric]
    else:
        numeric = numeric_cols or []
        categorical = categorical_cols or []
        specified = set(numeric) | set(categorical)
        missing = set(features) - specified
        extra = specified - set(features)
        overlap = set(numeric) & set(categorical)
        if missing:
            raise ValueError("Schema leaves features unspecified: " + ", ".join(sorted(missing)))
        if extra:
            raise ValueError("Schema contains unknown features: " + ", ".join(sorted(extra)))
        if overlap:
            raise ValueError("Columns listed as both numeric and categorical: " + ", ".join(sorted(overlap)))
    return FeatureSchema(target, list(numeric), list(categorical))


def validate_frames(real_train: pd.DataFrame, real_test: pd.DataFrame,
                    synthetic: pd.DataFrame, target: str,
                    schema: FeatureSchema) -> None:
    expected = list(real_train.columns)
    for name, df in [("real_train", real_train), ("real_test", real_test), ("synthetic", synthetic)]:
        if target not in df.columns:
            raise ValueError(f"Target '{target}' absent from {name}")
        if set(df.columns) != set(expected):
            raise ValueError(
                f"Schema mismatch in {name}: missing={sorted(set(expected)-set(df.columns))}, "
                f"extra={sorted(set(df.columns)-set(expected))}"
            )
        if df[target].isna().any():
            raise ValueError(f"Missing target values in {name}")

    if real_train[target].nunique() != 2:
        raise ValueError("Binary target required")
    if len(synthetic) == 0:
        raise ValueError("Synthetic release is empty")

    # Main benchmark uses already processed real data and requires generator output
    # to respect the frozen categorical support.
    for col in schema.categorical:
        train_levels = set(real_train[col].astype(str).unique())
        syn_levels = set(synthetic[col].astype(str).unique())
        unseen = syn_levels - train_levels
        if unseen:
            raise ValueError(f"Synthetic column '{col}' contains unseen categories: {sorted(unseen)[:10]}")


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, schema.numeric),
        ("cat", cat_pipe, schema.categorical),
    ], remainder="drop", sparse_threshold=0.0)


def encode_reference_space(real_train: pd.DataFrame, real_test: pd.DataFrame,
                           synthetic: pd.DataFrame, schema: FeatureSchema):
    prep = build_preprocessor(schema)
    Xr = np.asarray(prep.fit_transform(real_train.drop(columns=[schema.target])), float)
    Xt = np.asarray(prep.transform(real_test.drop(columns=[schema.target])), float)
    Xs = np.asarray(prep.transform(synthetic.drop(columns=[schema.target])), float)
    return Xr, Xt, Xs


def encode_tstr_space(synthetic: pd.DataFrame, real_test: pd.DataFrame,
                      schema: FeatureSchema):
    """Fit all classifier preprocessing on synthetic data only."""
    prep = build_preprocessor(schema)
    Xs = np.asarray(prep.fit_transform(synthetic.drop(columns=[schema.target])), float)
    Xt = np.asarray(prep.transform(real_test.drop(columns=[schema.target])), float)
    return Xs, Xt


def minmax_fit_transform(X_train: np.ndarray, *others: np.ndarray):
    lo = np.nanmin(X_train, axis=0)
    hi = np.nanmax(X_train, axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    outs = [(X_train - lo) / span]
    outs.extend((X - lo) / span for X in others)
    return tuple(outs)


def encode_binary_target(train_y: pd.Series, *others: pd.Series):
    levels = sorted(train_y.dropna().unique().tolist(), key=lambda x: str(x))
    if len(levels) != 2:
        raise ValueError(f"Binary target required; levels={levels}")
    mapping = {levels[0]: 0, levels[1]: 1}
    outs = []
    for s in (train_y,) + others:
        if s.isna().any():
            raise ValueError("Target contains missing values")
        unknown = set(s.unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Target contains unknown levels: {unknown}")
        outs.append(s.map(mapping).astype(int).to_numpy())
    return tuple(outs)


# ---------------- Utility ----------------

def classifier_factories(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "LR": LogisticRegression(max_iter=3000, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed),
        "RFC": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
    }
    if XGBClassifier is not None:
        models["XGB"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
    return models


def utility_tstr(X_syn: np.ndarray, y_syn: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, seed: int) -> pd.DataFrame:
    if len(np.unique(y_syn)) < 2:
        raise ValueError("Synthetic target has fewer than two classes")
    rows = []
    for name, model in classifier_factories(seed).items():
        m = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_syn, y_syn)
        p = m.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({
            "classifier": name,
            "auroc": float(roc_auc_score(y_test, p)),
            "f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
            "brier": float(brier_score_loss(y_test, p)),
        })
    return pd.DataFrame(rows)


# ---------------- Fidelity ----------------

def safe_corr(X: np.ndarray) -> np.ndarray:
    if X.shape[1] < 2:
        return np.zeros((X.shape[1], X.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.corrcoef(X, rowvar=False)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


def pcd_metric(X_real: np.ndarray, y_real: np.ndarray,
               X_syn: np.ndarray, y_syn: np.ndarray) -> float:
    # Include the target so PCD also captures feature-target relationships.
    R = np.column_stack([X_real, y_real.astype(float)])
    S = np.column_stack([X_syn, y_syn.astype(float)])
    Cr, Cs = safe_corr(R), safe_corr(S)
    mask = ~np.eye(Cr.shape[0], dtype=bool)
    return float(np.mean(np.abs(Cr[mask] - Cs[mask]))) if mask.any() else 0.0


def normalized_ws(real_train: pd.DataFrame, synthetic: pd.DataFrame,
                  numeric_cols: Sequence[str]):
    vals: Dict[str, float] = {}
    for col in numeric_cols:
        r = pd.to_numeric(real_train[col], errors="coerce").dropna().to_numpy(float)
        s = pd.to_numeric(synthetic[col], errors="coerce").dropna().to_numpy(float)
        if len(r) == 0 or len(s) == 0:
            vals[col] = np.nan
            continue
        scale = float(np.std(r, ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        vals[col] = float(wasserstein_distance(r, s) / scale)
    finite = [v for v in vals.values() if np.isfinite(v)]
    return (float(np.mean(finite)) if finite else np.nan), vals


def js_discrete(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    d = jensenshannon(p, q, base=2.0)
    return float(d * d)


def js_feature(real: pd.Series, syn: pd.Series, numeric: bool, bins: int = 20) -> float:
    if numeric:
        r = pd.to_numeric(real, errors="coerce").dropna().to_numpy(float)
        s = pd.to_numeric(syn, errors="coerce").dropna().to_numpy(float)
        if len(r) == 0 or len(s) == 0:
            return np.nan
        edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        pr, _ = np.histogram(r, bins=edges)
        ps, _ = np.histogram(s, bins=edges)
        return js_discrete(pr, ps)
    r = real.astype("object").where(real.notna(), "__MISSING__").astype(str)
    s = syn.astype("object").where(syn.notna(), "__MISSING__").astype(str)
    levels = sorted(set(r.unique()) | set(s.unique()))
    pr = r.value_counts().reindex(levels, fill_value=0).to_numpy()
    ps = s.value_counts().reindex(levels, fill_value=0).to_numpy()
    return js_discrete(pr, ps)


def js_metric(real_train: pd.DataFrame, synthetic: pd.DataFrame, schema: FeatureSchema):
    vals: Dict[str, float] = {}
    for c in schema.numeric:
        vals[c] = js_feature(real_train[c], synthetic[c], numeric=True)
    for c in schema.categorical:
        vals[c] = js_feature(real_train[c], synthetic[c], numeric=False)
    # Include target prevalence/distribution as a categorical fidelity component.
    vals[schema.target] = js_feature(real_train[schema.target], synthetic[schema.target], numeric=False)
    finite = [v for v in vals.values() if np.isfinite(v)]
    return (float(np.mean(finite)) if finite else np.nan), vals


# ---------------- Privacy ----------------

def nearest_distance(A: np.ndarray, B: np.ndarray, chunk: int = 2000) -> np.ndarray:
    if len(B) == 0:
        raise ValueError("Empty reference matrix")
    out = np.empty(len(A), float)
    for i in range(0, len(A), chunk):
        D = cdist(A[i:i+chunk], B, metric="euclidean")
        out[i:i+chunk] = np.min(D, axis=1)
    return out


def dcr_metrics(X_train: np.ndarray, X_syn: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    syn_to_real = nearest_distance(X_syn, X_train)
    max_ref = min(len(X_train), 5000)
    idx = rng.choice(len(X_train), size=max_ref, replace=False) if len(X_train) > max_ref else np.arange(len(X_train))
    R = X_train[idx]
    D = cdist(R, R, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    real_nn = np.min(D, axis=1)
    real_mean = float(np.mean(real_nn))
    syn_mean = float(np.mean(syn_to_real))
    p05 = float(np.quantile(real_nn, 0.05))
    return {
        "dcr_mean": syn_mean,
        "dcr_median": float(np.median(syn_to_real)),
        "dcr_p05": float(np.quantile(syn_to_real, 0.05)),
        "dcr_min": float(np.min(syn_to_real)),
        "real_nn_mean": real_mean,
        "dcr_ratio_to_real_nn": float(syn_mean / real_mean) if real_mean > 0 else np.nan,
        "fraction_syn_below_real_nn_p05": float(np.mean(syn_to_real < p05)),
    }


def balanced_member_nonmember_indices(y_member: np.ndarray, y_nonmember: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    mem_sel, non_sel = [], []
    for cls in sorted(set(np.unique(y_member)) & set(np.unique(y_nonmember))):
        mi, ni = np.where(y_member == cls)[0], np.where(y_nonmember == cls)[0]
        n = min(len(mi), len(ni))
        if n:
            mem_sel.extend(rng.choice(mi, n, replace=False))
            non_sel.extend(rng.choice(ni, n, replace=False))
    if not mem_sel:
        raise ValueError("Could not construct class-matched member/control samples")
    return np.asarray(mem_sel, int), np.asarray(non_sel, int)


def mia_auc(X_member: np.ndarray, y_member: np.ndarray,
            X_nonmember: np.ndarray, y_nonmember: np.ndarray,
            X_syn: np.ndarray, y_syn: np.ndarray, seed: int):
    mi, ni = balanced_member_nonmember_indices(y_member, y_nonmember, seed)
    scores, labels, per_class = [], [], []
    for cls in sorted(np.unique(np.concatenate([y_member[mi], y_nonmember[ni]]))):
        syn_cls = X_syn[y_syn == cls]
        if len(syn_cls) == 0:
            continue
        Xm = X_member[mi][y_member[mi] == cls]
        Xn = X_nonmember[ni][y_nonmember[ni] == cls]
        dm, dn = nearest_distance(Xm, syn_cls), nearest_distance(Xn, syn_cls)
        sc = np.concatenate([-dm, -dn])
        lb = np.concatenate([np.ones(len(dm)), np.zeros(len(dn))])
        per_class.append(float(roc_auc_score(lb, sc)))
        scores.append(sc); labels.append(lb)
    if not scores:
        return {"mia_auc_pooled": np.nan, "mia_auc_macro": np.nan, "mia_n": 0}
    scores_all, labels_all = np.concatenate(scores), np.concatenate(labels)
    return {
        "mia_auc_pooled": float(roc_auc_score(labels_all, scores_all)),
        "mia_auc_macro": float(np.mean(per_class)),
        "mia_n": int(len(labels_all)),
    }


def aia_scores(X_victim: np.ndarray, X_syn: np.ndarray,
               y_syn_sensitive: np.ndarray, k: int, chunk: int = 1000) -> np.ndarray:
    k_eff = min(k, len(X_syn))
    if k_eff < 1:
        raise ValueError("AIA requires at least one synthetic row")
    scores = np.empty(len(X_victim), float)
    for i in range(0, len(X_victim), chunk):
        D = cdist(X_victim[i:i+chunk], X_syn, metric="euclidean")
        idx = np.argpartition(D, kth=k_eff-1, axis=1)[:, :k_eff]
        scores[i:i+chunk] = np.mean(y_syn_sensitive[idx] == 1, axis=1)
    return scores


def aia_excess_risk(X_member: np.ndarray, y_member: np.ndarray,
                    X_control: np.ndarray, y_control: np.ndarray,
                    X_syn: np.ndarray, y_syn: np.ndarray,
                    k: int, seed: int):
    """Nearest-synthetic attribute inference with member-vs-control correction.

    The hidden attribute is binary. Member and control targets are class-matched.
    The main risk is normalized excess success:
        (success_member - success_control) / (1 - success_control)
    and is clipped at zero for the main privacy-risk table. A negative raw value
    means no evidence of train-specific inference advantage.
    """
    mi, ni = balanced_member_nonmember_indices(y_member, y_control, seed)
    Xm, ym = X_member[mi], y_member[mi]
    Xc, yc = X_control[ni], y_control[ni]
    sm = aia_scores(Xm, X_syn, y_syn, k)
    sc = aia_scores(Xc, X_syn, y_syn, k)
    pred_m, pred_c = (sm >= 0.5).astype(int), (sc >= 0.5).astype(int)
    acc_m = float(np.mean(pred_m == ym))
    acc_c = float(np.mean(pred_c == yc))
    denom = 1.0 - acc_c
    raw = float((acc_m - acc_c) / denom) if denom > 1e-12 else 0.0
    clipped = float(np.clip(raw, 0.0, 1.0))
    return {
        "aia_risk": clipped,
        "aia_risk_raw": raw,
        "aia_member_success": acc_m,
        "aia_control_success": acc_c,
        "aia_k": int(min(k, len(X_syn))),
        "aia_n_members": int(len(ym)),
        "aia_n_controls": int(len(yc)),
        "aia_secret": "target",
    }


def exact_duplicate_rates(real_train: pd.DataFrame, synthetic: pd.DataFrame):
    # Supplementary diagnostic, not one of the nine primary metrics.
    r = real_train.astype(str).agg("||".join, axis=1)
    s = synthetic.astype(str).agg("||".join, axis=1)
    real_set = set(r)
    exact = float(np.mean([x in real_set for x in s]))
    internal = float(1.0 - pd.Series(s).nunique() / len(s))
    return {"exact_duplicate_rate": exact, "internal_duplicate_rate": internal}


# ---------------- Orchestration ----------------

def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    real_train = pd.read_csv(args.real_train)
    real_test = pd.read_csv(args.real_test)
    synthetic = pd.read_csv(args.synthetic)

    # Reorder before validation.
    expected = list(real_train.columns)
    real_test = real_test[expected].copy()
    synthetic = synthetic[expected].copy()
    schema = infer_schema(real_train, args.target, parse_columns(args.numeric_cols), parse_columns(args.categorical_cols))
    validate_frames(real_train, real_test, synthetic, args.target, schema)

    yr, yt, ys = encode_binary_target(real_train[args.target], real_test[args.target], synthetic[args.target])

    # Utility: synthetic-fitted preprocessing only.
    Xs_u, Xt_u = encode_tstr_space(synthetic, real_test, schema)
    utility = utility_tstr(Xs_u, ys, Xt_u, yt, args.seed)
    utility.to_csv(outdir / "utility_by_classifier.csv", index=False)
    utility_summary = {
        "auroc": float(utility.auroc.mean()),
        "f1": float(utility.f1_at_0_5.mean()),
        "brier": float(utility.brier.mean()),
        "n_classifiers": int(len(utility)),
    }

    # Shared reference encoding for fidelity/privacy.
    Xr, Xt, Xs = encode_reference_space(real_train, real_test, synthetic, schema)
    Xr_d, Xt_d, Xs_d = minmax_fit_transform(Xr, Xt, Xs)

    pcd = pcd_metric(Xr, yr, Xs, ys)
    ws, ws_detail = normalized_ws(real_train, synthetic, schema.numeric)
    js, js_detail = js_metric(real_train, synthetic, schema)
    fidelity_summary = {"pcd": pcd, "ws": ws, "js": js}
    pd.DataFrame({"feature": list(ws_detail), "ws": list(ws_detail.values())}).to_csv(outdir / "fidelity_ws_by_feature.csv", index=False)
    pd.DataFrame({"feature": list(js_detail), "js": list(js_detail.values())}).to_csv(outdir / "fidelity_js_by_feature.csv", index=False)

    dcr = dcr_metrics(Xr_d, Xs_d, args.seed)
    mia = mia_auc(Xr_d, yr, Xt_d, yt, Xs_d, ys, args.seed)
    aia = aia_excess_risk(Xr_d, yr, Xt_d, yt, Xs_d, ys, args.aia_k, args.seed)
    duplicates = exact_duplicate_rates(real_train, synthetic)
    privacy_summary = {**dcr, **mia, **aia, **duplicates}

    main_row = {
        "dataset": args.dataset,
        "method": args.method,
        "synthetic_size_label": args.synthetic_size_label,
        "n_real_train": len(real_train),
        "n_real_test": len(real_test),
        "n_synthetic": len(synthetic),
        "utility_auroc": utility_summary["auroc"],
        "utility_f1": utility_summary["f1"],
        "utility_brier": utility_summary["brier"],
        "fidelity_pcd": pcd,
        "fidelity_ws": ws,
        "fidelity_js": js,
        "privacy_dcr_mean": dcr["dcr_mean"],
        "privacy_dcr_ratio": dcr["dcr_ratio_to_real_nn"],
        "privacy_mia_auc": mia["mia_auc_pooled"],
        "privacy_aia_risk": aia["aia_risk"],
    }
    pd.DataFrame([main_row]).to_csv(outdir / "main_results_row.csv", index=False)

    payload = {
        "schema": {"target": schema.target, "numeric": schema.numeric, "categorical": schema.categorical},
        "protocol": {
            "utility": "TSTR; preprocessing fitted on synthetic training release only",
            "fidelity": "real_train vs synthetic; PCD and JS include target",
            "privacy_members": "real_train",
            "privacy_controls": "real_test",
            "mia": "class-conditional nearest-synthetic distance AUROC",
            "aia": "target hidden; nearest-synthetic kNN; member/control normalized excess success",
            "aia_k": args.aia_k,
            "seed": args.seed,
        },
        "utility": utility_summary,
        "fidelity": fidelity_summary,
        "privacy": privacy_summary,
        "main_results_row": main_row,
    }
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Evaluation complete")
    print(pd.DataFrame([main_row]).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified thesis benchmark evaluator v2")
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--real_train", required=True)
    p.add_argument("--real_test", required=True)
    p.add_argument("--synthetic", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--synthetic_size_label", default="1x")
    p.add_argument("--numeric_cols", default=None)
    p.add_argument("--categorical_cols", default=None)
    p.add_argument("--aia_k", type=int, default=1, help="Primary AIA follows nearest-synthetic inference; default k=1")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
