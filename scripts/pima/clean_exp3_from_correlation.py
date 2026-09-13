#!/usr/bin/env python3
"""
Clean Exp3 from already generated Exp2/correlation synthetic CSV files.

Purpose:
  Train classifiers on existing synthetic data (TSTR: Train Synthetic, Test Real)
  and evaluate them on a held-out real test split.

This avoids the original exp3 Optuna/generator bottleneck and produces one clean
clf_performance.csv that is easy to analyse.

Example:
  python clean_exp3_from_correlation.py \
    --data_path data/pima.csv \
    --correlation_dir results/paper/correlation \
    --out_path results/clean_exp3/clf_performance.csv \
    --splits 0 1 2 \
    --generators TVAE CTGAN COPULA \
    --classifiers lr xgb \
    --n_files_per_model 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None


PIMA_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/pima/pima.csv"))
    parser.add_argument("--correlation_dir", default=str(REPOSITORY_ROOT / "results/paper/correlation"))
    parser.add_argument("--out_path", default=str(REPOSITORY_ROOT / "results/clean_exp3/clf_performance.csv"))
    parser.add_argument("--target_col", default="Outcome")
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--generators", nargs="+", default=["TVAE", "CTGAN", "COPULA"])
    parser.add_argument("--classifiers", nargs="+", default=["lr", "xgb"])
    parser.add_argument("--size_multipliers", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--n_files_per_model", type=int, default=20,
                        help="Number of synthetic CSV files per split/generator. Use 0 for all files.")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--starting_seed", type=int, default=4,
                        help="Use the same seed family as exp2 correlation config.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--include_real_baseline", action="store_true",
                        help="Also train classifiers on the real train split and test on real test split.")
    parser.add_argument("--no_zero_as_missing", action="store_true",
                        help="Do not convert biologically implausible PIMA zeros to NaN.")
    return parser.parse_args()


def normalize_path(path: str) -> str:
    return str(path).replace("\\", "/")


def extract_generator_and_filename(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract generator and filename from .../sd/{GEN}/{file}.csv."""
    p = normalize_path(path)
    m = re.search(r"/sd/([^/]+)/([^/]+\.csv)$", p)
    if m:
        return m.group(1), m.group(2)
    return None, os.path.basename(p)


def load_metric_map(correlation_dir: str, split: int) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load pcd/js/ws values from exp2 result table for a split.

    Supports both results.csv and results{split}.csv naming.
    Key: (generator, filename.csv)
    """
    split_dir = Path(correlation_dir) / str(split)
    candidates = [split_dir / f"results{split}.csv", split_dir / "results.csv"]
    result_path = next((p for p in candidates if p.exists()), None)

    metric_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    if result_path is None:
        print(f"[WARN] No exp2 results table found for split {split}: {candidates}", file=sys.stderr)
        return metric_map

    df = pd.read_csv(result_path)
    if "csv_path" not in df.columns:
        print(f"[WARN] {result_path} has no csv_path column.", file=sys.stderr)
        return metric_map

    for _, row in df.iterrows():
        gen, filename = extract_generator_and_filename(str(row["csv_path"]))
        if gen is None or filename is None:
            continue
        if (gen, filename) in metric_map:
            continue
        metric_map[(gen, filename)] = {
            "pcd": pd.to_numeric(row.get("pcd", np.nan), errors="coerce"),
            "js": pd.to_numeric(row.get("js", np.nan), errors="coerce"),
            "ws": pd.to_numeric(row.get("ws", np.nan), errors="coerce"),
        }
    return metric_map


def load_real_data(data_path: str, target_col: str, zero_as_missing: bool = True) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")

    # For PIMA, zeros in these medical columns are commonly treated as missing values.
    if zero_as_missing:
        for col in PIMA_ZERO_AS_MISSING:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)

    df[target_col] = (pd.to_numeric(df[target_col], errors="coerce") >= 0.5).astype(int)
    return df


def make_classifier(name: str, seed: int):
    name = name.lower()
    if name == "lr":
        return LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        )
    if name == "rfc":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=seed,
        )
    if name == "xgb":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed in this environment, but classifier 'xgb' was requested.")
        return XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier: {name}")


def make_pipeline(clf_name: str, seed: int) -> Pipeline:
    clf = make_classifier(clf_name, seed)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def get_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores)
    return model.predict(X)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["sensitivity"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    out["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    out["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        out["auc"] = roc_auc_score(y_true, y_score)
    except Exception:
        out["auc"] = np.nan

    # Brier needs probabilities in [0, 1]. If decision scores are outside range, skip.
    if np.nanmin(y_score) >= 0 and np.nanmax(y_score) <= 1:
        try:
            out["brier"] = brier_score_loss(y_true, y_score)
        except Exception:
            out["brier"] = np.nan
    else:
        out["brier"] = np.nan
    return out


def list_synthetic_files(correlation_dir: str, split: int, generator: str, n_files: int) -> List[Path]:
    pattern = Path(correlation_dir) / str(split) / "sd" / generator / "*.csv"
    files = [Path(p) for p in glob.glob(str(pattern)) if ".ipynb_checkpoints" not in p]

    def sort_key(p: Path):
        try:
            return int(p.stem)
        except ValueError:
            return p.stem

    files = sorted(files, key=sort_key)
    if n_files and n_files > 0:
        files = files[:n_files]
    return files


def output_fieldnames() -> List[str]:
    return [
        "split",
        "generator",
        "synthetic_file",
        "source",
        "classifier",
        "size_multiplier",
        "n_train_rows",
        "n_synthetic_available",
        "pcd",
        "js",
        "ws",
        "accuracy",
        "auc",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier",
    ]


def append_row(out_path: str, row: Dict):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(out_path).exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames())
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in output_fieldnames()})


def load_done_keys(out_path: str) -> set:
    if not Path(out_path).exists():
        return set()
    df = pd.read_csv(out_path)
    done = set()
    required = ["split", "generator", "synthetic_file", "classifier", "size_multiplier", "source"]
    if not set(required).issubset(df.columns):
        return done
    for _, r in df.iterrows():
        done.add((
            int(r["split"]), str(r["generator"]), str(r["synthetic_file"]),
            str(r["classifier"]), float(r["size_multiplier"]), str(r["source"])
        ))
    return done


def evaluate_one(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    clf_name: str,
    seed: int,
) -> Dict[str, float]:
    X_train = train_df.drop(columns=[target_col])
    y_train = (pd.to_numeric(train_df[target_col], errors="coerce") >= 0.5).astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = (pd.to_numeric(test_df[target_col], errors="coerce") >= 0.5).astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training data contains only one class; skipping.")

    model = make_pipeline(clf_name, seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)
    return compute_metrics(y_test.values, y_pred, y_score)


def main():
    args = parse_args()
    print("[INFO] Clean Exp3 from correlation synthetic CSVs")
    print(f"[INFO] data_path={args.data_path}")
    print(f"[INFO] correlation_dir={args.correlation_dir}")
    print(f"[INFO] out_path={args.out_path}")

    real_df = load_real_data(
        args.data_path,
        args.target_col,
        zero_as_missing=not args.no_zero_as_missing,
    )
    y = real_df[args.target_col]
    feature_cols = [c for c in real_df.columns if c != args.target_col]
    done = load_done_keys(args.out_path)

    for split in args.splits:
        split_seed = args.starting_seed + split
        train_idx, test_idx = train_test_split(
            np.arange(len(real_df)),
            test_size=args.test_size,
            random_state=split_seed,
            stratify=y,
        )
        real_train = real_df.iloc[train_idx].reset_index(drop=True)
        real_test = real_df.iloc[test_idx].reset_index(drop=True)
        n_real_train = len(real_train)

        print(f"[INFO] Split {split}: real_train={len(real_train)}, real_test={len(real_test)}")

        # Optional real-data baseline.
        if args.include_real_baseline:
            for clf_name in args.classifiers:
                key = (split, "REAL", "real_train", clf_name, 1.0, "real_baseline")
                if key in done:
                    continue
                try:
                    metrics = evaluate_one(real_train, real_test, args.target_col, clf_name, split_seed)
                    row = {
                        "split": split,
                        "generator": "REAL",
                        "synthetic_file": "real_train",
                        "source": "real_baseline",
                        "classifier": clf_name,
                        "size_multiplier": 1.0,
                        "n_train_rows": len(real_train),
                        "n_synthetic_available": len(real_train),
                        "pcd": np.nan,
                        "js": np.nan,
                        "ws": np.nan,
                        **metrics,
                    }
                    append_row(args.out_path, row)
                    done.add(key)
                    print(f"[OK] split={split} REAL clf={clf_name} auc={metrics.get('auc'):.4f}")
                except Exception as e:
                    print(f"[WARN] REAL baseline failed for split={split}, clf={clf_name}: {e}", file=sys.stderr)

        metric_map = load_metric_map(args.correlation_dir, split)

        for generator in args.generators:
            files = list_synthetic_files(args.correlation_dir, split, generator, args.n_files_per_model)
            print(f"[INFO] Split {split}, {generator}: using {len(files)} synthetic CSV files")
            if not files:
                continue

            for path in files:
                filename = path.name
                try:
                    sd = pd.read_csv(path)
                except Exception as e:
                    print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
                    continue

                if args.target_col not in sd.columns:
                    print(f"[WARN] {path} has no target column {args.target_col}; skipping.", file=sys.stderr)
                    continue

                # Keep same column order as real data and drop unknown extra columns.
                missing_cols = [c for c in real_df.columns if c not in sd.columns]
                if missing_cols:
                    print(f"[WARN] {path} missing columns {missing_cols}; skipping.", file=sys.stderr)
                    continue
                sd = sd[real_df.columns].copy()
                sd[args.target_col] = (pd.to_numeric(sd[args.target_col], errors="coerce") >= 0.5).astype(int)

                metrics_from_exp2 = metric_map.get((generator, filename), {"pcd": np.nan, "js": np.nan, "ws": np.nan})

                for mult in args.size_multipliers:
                    n_rows = int(round(n_real_train * mult))
                    if n_rows <= 0:
                        continue
                    replace = len(sd) < n_rows
                    sampled_sd = sd.sample(
                        n=n_rows,
                        replace=replace,
                        random_state=args.random_seed + split * 100000 + hash((generator, filename, mult)) % 100000,
                    ).reset_index(drop=True)

                    for clf_name in args.classifiers:
                        key = (split, generator, filename, clf_name, float(mult), "synthetic")
                        if key in done:
                            continue
                        try:
                            perf = evaluate_one(sampled_sd, real_test, args.target_col, clf_name, args.random_seed + split)
                            row = {
                                "split": split,
                                "generator": generator,
                                "synthetic_file": filename,
                                "source": "synthetic",
                                "classifier": clf_name,
                                "size_multiplier": float(mult),
                                "n_train_rows": len(sampled_sd),
                                "n_synthetic_available": len(sd),
                                "pcd": metrics_from_exp2.get("pcd", np.nan),
                                "js": metrics_from_exp2.get("js", np.nan),
                                "ws": metrics_from_exp2.get("ws", np.nan),
                                **perf,
                            }
                            append_row(args.out_path, row)
                            done.add(key)
                            print(
                                f"[OK] split={split} gen={generator} file={filename} "
                                f"mult={mult} clf={clf_name} auc={perf.get('auc'):.4f}"
                            )
                        except Exception as e:
                            print(
                                f"[WARN] failed split={split} gen={generator} file={filename} "
                                f"mult={mult} clf={clf_name}: {e}",
                                file=sys.stderr,
                            )

    print(f"[DONE] Results written to {args.out_path}")


if __name__ == "__main__":
    main()
