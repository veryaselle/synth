#!/usr/bin/env python3
"""
Clean conditional diffusion experiment for the Cleveland Heart Disease dataset.

This script is intentionally standalone and transparent.

Goal:
    Train a simple conditional DDPM-style tabular diffusion model on real Cleveland
    training data, generate synthetic training data conditioned on the binary target,
    and evaluate downstream utility using Train-Synthetic-Test-Real.

Pipeline:
    1. Load cleaned Cleveland CSV.
    2. Use fixed stratified train/test splits.
    3. Fit a simple tabular encoder on the real training split.
       - numeric columns -> standardized continuous values
       - categorical columns -> one-hot vectors
    4. Train conditional diffusion model on encoded features X | target y.
    5. Generate synthetic data sizes 1x, 2x, 3x.
    6. Decode synthetic rows back to raw tabular format.
    7. Train classifiers on synthetic data and test on real held-out test data.
    8. Save utility, fidelity, and simple privacy-like metrics.

Example:
    python run_cleveland_diffusion.py \
      --data_path data/Heart_disease_cleveland_new.csv \
      --out_dir results/cleveland/diffusion \
      --n_splits 5 \
      --epochs 300 \
      --timesteps 100 \
      --classifiers lr mlp xgb rfc \
      --size_multipliers 1 2 3 \
      --include_real_baseline

For a quick smoke test:
    python run_cleveland_diffusion.py \
      --data_path data/Heart_disease_cleveland_new.csv \
      --out_dir results/cleveland/diffusion_smoke \
      --n_splits 1 \
      --epochs 50 \
      --timesteps 50 \
      --classifiers lr xgb \
      --size_multipliers 1 \
      --include_real_baseline
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    pairwise_distances,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DEFAULT_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
DEFAULT_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data and output
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/raw/cleveland/Heart_disease_cleveland_new.csv"))
    parser.add_argument("--out_dir", default=str(REPOSITORY_ROOT / "results/cleveland/diffusion"))
    parser.add_argument("--target_col", default="target")

    # Splits
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)

    # Diffusion training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Synthetic evaluation
    parser.add_argument("--size_multipliers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["lr", "mlp", "xgb", "rfc"],
        choices=["lr", "mlp", "xgb", "rfc"],
    )
    parser.add_argument("--include_real_baseline", action="store_true")

    # Device
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Saving
    parser.add_argument("--save_synthetic", action="store_true")

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_dataset(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> None:
    expected_cols = set(numeric_features + categorical_features + [target_col])
    missing_cols = sorted(expected_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    target_values = sorted(df[target_col].dropna().unique().tolist())
    if target_values != [0, 1]:
        raise ValueError(
            f"Expected binary target values [0, 1], but got {target_values}. "
            "Please convert the target to binary first."
        )


# ---------------------------------------------------------------------------
# Tabular encoder for diffusion
# ---------------------------------------------------------------------------

@dataclass
class CategorySpec:
    column: str
    categories: List
    start: int
    end: int


class TabularEncoder:
    """
    Simple encoder for mixed tabular data.

    Numeric columns are standardized.
    Categorical columns are one-hot encoded manually.

    This is deliberately simple and inspectable.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        self.numeric_mean_: Dict[str, float] = {}
        self.numeric_std_: Dict[str, float] = {}
        self.numeric_min_: Dict[str, float] = {}
        self.numeric_max_: Dict[str, float] = {}
        self.numeric_is_integer_: Dict[str, bool] = {}

        self.category_specs_: List[CategorySpec] = []
        self.output_dim_: int | None = None

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        current = len(self.numeric_features)

        for col in self.numeric_features:
            values = df[col].astype(float)
            std = float(values.std(ddof=0))
            if std == 0 or np.isnan(std):
                std = 1.0

            self.numeric_mean_[col] = float(values.mean())
            self.numeric_std_[col] = std
            self.numeric_min_[col] = float(values.min())
            self.numeric_max_[col] = float(values.max())

            # Useful for decoding plausible integer-valued numeric variables.
            non_na = df[col].dropna()
            self.numeric_is_integer_[col] = bool(np.all(np.isclose(non_na, np.round(non_na))))

        for col in self.categorical_features:
            cats = sorted(df[col].dropna().unique().tolist())
            start = current
            end = current + len(cats)
            self.category_specs_.append(CategorySpec(col, cats, start, end))
            current = end

        self.output_dim_ = current
        return self

    @property
    def output_dim(self) -> int:
        if self.output_dim_ is None:
            raise RuntimeError("Encoder is not fitted.")
        return self.output_dim_

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.output_dim_ is None:
            raise RuntimeError("Encoder is not fitted.")

        n = len(df)
        arr = np.zeros((n, self.output_dim_), dtype=np.float32)

        # Numeric block
        for j, col in enumerate(self.numeric_features):
            values = df[col].astype(float).to_numpy()
            arr[:, j] = (values - self.numeric_mean_[col]) / self.numeric_std_[col]

        # Categorical blocks
        for spec in self.category_specs_:
            values = df[spec.column].to_numpy()
            cat_to_idx = {cat: i for i, cat in enumerate(spec.categories)}

            for row_idx, value in enumerate(values):
                if value in cat_to_idx:
                    arr[row_idx, spec.start + cat_to_idx[value]] = 1.0
                else:
                    # Unknown category: use the first known category as a safe fallback.
                    arr[row_idx, spec.start] = 1.0

        return arr

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        if self.output_dim_ is None:
            raise RuntimeError("Encoder is not fitted.")

        rows: Dict[str, np.ndarray] = {}

        # Numeric decode
        for j, col in enumerate(self.numeric_features):
            values = arr[:, j] * self.numeric_std_[col] + self.numeric_mean_[col]
            values = np.clip(values, self.numeric_min_[col], self.numeric_max_[col])

            if self.numeric_is_integer_[col] and col != "oldpeak":
                values = np.round(values).astype(int)
            else:
                values = np.round(values, 2)

            rows[col] = values

        # Categorical decode by argmax
        for spec in self.category_specs_:
            block = arr[:, spec.start:spec.end]
            idx = np.argmax(block, axis=1)
            cats = np.array(spec.categories, dtype=object)
            values = cats[idx]

            # Preserve integer-like categories where possible.
            if all(isinstance(x, (int, np.integer)) for x in spec.categories):
                values = values.astype(int)

            rows[spec.column] = values

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: integer tensor of shape [batch]
        returns: [batch, dim]
        """
        half_dim = self.dim // 2
        device = t.device

        if half_dim == 0:
            return t.float().unsqueeze(1)

        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(len(t), 1, device=device)], dim=1)

        return emb


class ConditionalDenoiseMLP(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        time_dim: int,
        label_dim: int,
    ):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.label_emb = nn.Embedding(2, label_dim)

        in_dim = x_dim + time_dim + label_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        y_emb = self.label_emb(y.long())
        inp = torch.cat([x_t, t_emb, y_emb], dim=1)
        return self.net(inp)


class DDPM:
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        device: torch.device,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def training_loss(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).unsqueeze(1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_bars[t]).unsqueeze(1)

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        pred_noise = self.model(x_t, t, y)

        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n: int, y: np.ndarray, x_dim: int) -> np.ndarray:
        self.model.eval()

        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        x = torch.randn(n, x_dim, device=self.device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, dtype=torch.long, device=self.device)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]

            pred_noise = self.model(x, t, y_tensor)

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )

            if step > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean

        return x.detach().cpu().numpy()


def train_diffusion_model(
    X_encoded: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[DDPM, List[Dict[str, float]]]:
    set_seed(seed)

    x_tensor = torch.tensor(X_encoded, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = ConditionalDenoiseMLP(
        x_dim=X_encoded.shape[1],
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        label_dim=args.label_dim,
    ).to(device)

    ddpm = DDPM(model=model, timesteps=args.timesteps, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            loss = ddpm.training_loss(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})

        if epoch == 1 or epoch % max(args.epochs // 10, 1) == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:4d}/{args.epochs} | loss={mean_loss:.5f}", flush=True)

    return ddpm, history


# ---------------------------------------------------------------------------
# Classifier evaluation
# ---------------------------------------------------------------------------

def make_onehot_encoder():
    # scikit-learn changed sparse -> sparse_output in newer versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_classifier(name: str, seed: int):
    name = name.lower()

    if name == "lr":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )

    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        )

    if name == "rfc":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed, but classifier 'xgb' was requested."
            ) from exc

        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier: {name}")


def get_score_vector(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    clf_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
    seed: int,
) -> Dict[str, float]:
    feature_cols = numeric_features + categorical_features

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int).to_numpy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(int).to_numpy()

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_features, categorical_features)),
            ("clf", make_classifier(clf_name, seed)),
        ]
    )

    model.fit(X_train, y_train)
    y_score = get_score_vector(model, X_test)

    return evaluate_predictions(y_test, y_score)


# ---------------------------------------------------------------------------
# Synthetic data evaluation helpers
# ---------------------------------------------------------------------------

def generate_target_labels(y_train: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p1 = float(np.mean(y_train == 1))
    return rng.binomial(1, p1, size=n).astype(int)


def make_synthetic_df(
    ddpm: DDPM,
    encoder: TabularEncoder,
    y_train: np.ndarray,
    n: int,
    seed: int,
    target_col: str,
) -> pd.DataFrame:
    set_seed(seed)
    y_syn = generate_target_labels(y_train, n=n, seed=seed)

    x_syn_encoded = ddpm.sample(n=n, y=y_syn, x_dim=encoder.output_dim)
    syn_features = encoder.inverse_transform(x_syn_encoded)
    syn_features[target_col] = y_syn

    # Keep clean column order.
    return syn_features[encoder.numeric_features + encoder.categorical_features + [target_col]]


def pcd_metric(real_encoded: np.ndarray, syn_encoded: np.ndarray) -> float:
    real_corr = np.corrcoef(real_encoded, rowvar=False)
    syn_corr = np.corrcoef(syn_encoded, rowvar=False)

    real_corr = np.nan_to_num(real_corr)
    syn_corr = np.nan_to_num(syn_corr)

    idx = np.triu_indices_from(real_corr, k=1)
    return float(np.mean(np.abs(real_corr[idx] - syn_corr[idx])))


def wasserstein_metric_numeric_scaled(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    numeric_features: List[str],
) -> float:
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return float("nan")

    distances = []
    for col in numeric_features:
        real_values = real_df[col].astype(float).to_numpy()
        syn_values = syn_df[col].astype(float).to_numpy()

        mean = float(real_values.mean())
        std = float(real_values.std(ddof=0)) or 1.0

        real_scaled = (real_values - mean) / std
        syn_scaled = (syn_values - mean) / std

        distances.append(wasserstein_distance(real_scaled, syn_scaled))

    return float(np.mean(distances))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return float(0.5 * (kl_pm + kl_qm))


def js_metric_categorical(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    categorical_features: List[str],
    target_col: str,
) -> float:
    values = []
    for col in categorical_features + [target_col]:
        cats = sorted(set(real_df[col].dropna().unique().tolist()) | set(syn_df[col].dropna().unique().tolist()))

        real_counts = np.array([(real_df[col] == c).sum() for c in cats], dtype=float)
        syn_counts = np.array([(syn_df[col] == c).sum() for c in cats], dtype=float)

        values.append(js_divergence(real_counts, syn_counts))

    return float(np.mean(values))


def dcr_metrics(real_encoded: np.ndarray, syn_encoded: np.ndarray) -> Dict[str, float]:
    # For Cleveland sizes this is small enough for direct pairwise distances.
    distances = pairwise_distances(syn_encoded, real_encoded, metric="euclidean")
    nearest = distances.min(axis=1)

    return {
        "dcr_mean": float(np.mean(nearest)),
        "dcr_min": float(np.min(nearest)),
        "dcr_p05": float(np.quantile(nearest, 0.05)),
    }


def duplicate_rate(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    columns: List[str],
) -> float:
    real_tuples = set(map(tuple, real_df[columns].round(3).to_numpy()))
    syn_tuples = list(map(tuple, syn_df[columns].round(3).to_numpy()))

    if not syn_tuples:
        return float("nan")

    matches = sum(row in real_tuples for row in syn_tuples)
    return float(matches / len(syn_tuples))


def compute_fidelity_privacy(
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    encoder: TabularEncoder,
    numeric_features: List[str],
    categorical_features: List[str],
    target_col: str,
) -> Dict[str, float]:
    feature_cols = numeric_features + categorical_features

    real_encoded = encoder.transform(real_train[feature_cols])
    syn_encoded = encoder.transform(syn_df[feature_cols])

    out = {
        "pcd": pcd_metric(real_encoded, syn_encoded),
        "ws_numeric_scaled": wasserstein_metric_numeric_scaled(real_train, syn_df, numeric_features),
        "js_categorical_target": js_metric_categorical(real_train, syn_df, categorical_features, target_col),
        "duplicate_rate_vs_real_train": duplicate_rate(real_train, syn_df, feature_cols + [target_col]),
    }

    out.update(dcr_metrics(real_encoded, syn_encoded))

    return out


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir = out_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    numeric_features = DEFAULT_NUMERIC_FEATURES
    categorical_features = DEFAULT_CATEGORICAL_FEATURES
    feature_cols = numeric_features + categorical_features

    df = pd.read_csv(args.data_path)
    validate_dataset(df, args.target_col, numeric_features, categorical_features)

    X = df[feature_cols].copy()
    y = df[args.target_col].astype(int).copy()

    device = get_device(args.device)
    print(f"Using device: {device}", flush=True)

    dataset_summary = pd.DataFrame(
        [
            {"key": "n_rows", "value": len(df)},
            {"key": "n_features", "value": len(feature_cols)},
            {"key": "target_0_count", "value": int((y == 0).sum())},
            {"key": "target_1_count", "value": int((y == 1).sum())},
            {"key": "target_1_rate", "value": float((y == 1).mean())},
            {"key": "numeric_features", "value": json.dumps(numeric_features)},
            {"key": "categorical_features", "value": json.dumps(categorical_features)},
        ]
    )
    dataset_summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    result_rows: List[Dict] = []
    history_rows: List[Dict] = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"\n=== Split {split_idx}/{args.n_splits - 1} ===", flush=True)

        real_train = df.iloc[train_idx].copy().reset_index(drop=True)
        real_test = df.iloc[test_idx].copy().reset_index(drop=True)

        y_train = real_train[args.target_col].astype(int).to_numpy()

        split_seed = args.random_seed + split_idx

        encoder = TabularEncoder(numeric_features, categorical_features).fit(real_train[feature_cols])
        X_train_encoded = encoder.transform(real_train[feature_cols])

        print(f"Training conditional diffusion model on {len(real_train)} real rows...", flush=True)
        ddpm, history = train_diffusion_model(
            X_encoded=X_train_encoded,
            y=y_train,
            args=args,
            seed=split_seed,
            device=device,
        )

        for h in history:
            history_rows.append({"split": split_idx, **h})

        # Optional real baseline using the exact same split and classifiers.
        if args.include_real_baseline:
            print("Evaluating real-data baseline for this split...", flush=True)
            for clf_name in args.classifiers:
                metrics = evaluate_classifier(
                    train_df=real_train,
                    test_df=real_test,
                    target_col=args.target_col,
                    clf_name=clf_name,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    seed=split_seed,
                )

                result_rows.append(
                    {
                        "dataset": "cleveland",
                        "generator": "REAL",
                        "split": split_idx,
                        "repeat": 0,
                        "size_multiplier": 1,
                        "n_synthetic_rows": np.nan,
                        "classifier": clf_name,
                        "epochs": args.epochs,
                        "timesteps": args.timesteps,
                        "pcd": np.nan,
                        "ws_numeric_scaled": np.nan,
                        "js_categorical_target": np.nan,
                        "dcr_mean": np.nan,
                        "dcr_min": np.nan,
                        "dcr_p05": np.nan,
                        "duplicate_rate_vs_real_train": np.nan,
                        **metrics,
                    }
                )

        for repeat in range(args.n_repeats):
            for multiplier in args.size_multipliers:
                n_syn = int(len(real_train) * multiplier)
                syn_seed = args.random_seed + 10000 * split_idx + 100 * repeat + multiplier

                print(f"Generating synthetic data: repeat={repeat}, size={multiplier}x ({n_syn} rows)", flush=True)

                syn_df = make_synthetic_df(
                    ddpm=ddpm,
                    encoder=encoder,
                    y_train=y_train,
                    n=n_syn,
                    seed=syn_seed,
                    target_col=args.target_col,
                )

                if args.save_synthetic:
                    split_dir = synthetic_dir / f"split_{split_idx}"
                    split_dir.mkdir(parents=True, exist_ok=True)
                    syn_df.to_csv(split_dir / f"synthetic_repeat{repeat}_{multiplier}x.csv", index=False)

                fp_metrics = compute_fidelity_privacy(
                    real_train=real_train,
                    syn_df=syn_df,
                    encoder=encoder,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    target_col=args.target_col,
                )

                for clf_name in args.classifiers:
                    metrics = evaluate_classifier(
                        train_df=syn_df,
                        test_df=real_test,
                        target_col=args.target_col,
                        clf_name=clf_name,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        seed=syn_seed,
                    )

                    result_rows.append(
                        {
                            "dataset": "cleveland",
                            "generator": "COND_DDPM",
                            "split": split_idx,
                            "repeat": repeat,
                            "size_multiplier": multiplier,
                            "n_synthetic_rows": n_syn,
                            "classifier": clf_name,
                            "epochs": args.epochs,
                            "timesteps": args.timesteps,
                            **fp_metrics,
                            **metrics,
                        }
                    )

        # Save after each split, so partial results are not lost.
        pd.DataFrame(result_rows).to_csv(out_dir / "diffusion_utility.csv", index=False)
        pd.DataFrame(history_rows).to_csv(out_dir / "training_history.csv", index=False)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(out_dir / "diffusion_utility.csv", index=False)

    summary_df = (
        result_df
        .groupby(["generator", "classifier", "size_multiplier"], dropna=False, as_index=False)
        .agg(
            n_rows=("auc", "size"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"),
            brier_mean=("brier", "mean"),
            pcd_mean=("pcd", "mean"),
            ws_mean=("ws_numeric_scaled", "mean"),
            js_mean=("js_categorical_target", "mean"),
            dcr_mean=("dcr_mean", "mean"),
            dcr_min=("dcr_min", "min"),
            duplicate_rate_mean=("duplicate_rate_vs_real_train", "mean"),
        )
        .sort_values(["generator", "classifier", "size_multiplier"])
    )
    summary_df.to_csv(out_dir / "diffusion_utility_summary.csv", index=False)

    print("\nSaved:")
    print(f"  {out_dir / 'dataset_summary.csv'}")
    print(f"  {out_dir / 'training_history.csv'}")
    print(f"  {out_dir / 'diffusion_utility.csv'}")
    print(f"  {out_dir / 'diffusion_utility_summary.csv'}")

    print("\nAUC summary:")
    print(summary_df[["generator", "classifier", "size_multiplier", "n_rows", "auc_mean", "auc_std"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
