#!/usr/bin/env python3
"""
Unified CKD synthetic-data experiment.

Compares:
    REAL
    GAUSSIAN_EMPIRICAL
    EMPIRICAL_BOOTSTRAP
    COND_DDPM

Protocol:
    - fixed stratified real train/test splits
    - all preprocessing fitted on the training split only
    - 1x / 2x / 3x synthetic training sizes
    - repeated synthetic generation
    - Train-Synthetic-Test-Real downstream evaluation
    - utility, fidelity, missingness, and privacy-like sanity checks

The sanitized input must contain 24 predictive variables and the binary target:
    target = 0 -> not CKD
    target = 1 -> CKD

Important:
    DCR, duplicate rate, and nearest-neighbour ratios are privacy-like sanity
    checks. They do not provide a formal privacy guarantee.

Example:
    python run_ckd_synthetic_experiment.py \
      --data_path data/kidney_disease_sanitized.csv \
      --out_dir results/ckd/synthetic_experiment \
      --n_splits 5 \
      --n_repeats 3 \
      --size_multipliers 1 2 3 \
      --generators gaussian empirical_bootstrap ddpm \
      --classifiers lr mlp xgb rfc \
      --epochs 300 \
      --timesteps 100 \
      --include_real_baseline \
      --save_synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


NUMERIC_FEATURES = [
    "age", "bp", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wc", "rc",
]
INTEGER_LIKE_NUMERIC = {
    "age", "bp", "bgr", "bu", "sod", "pcv", "wc",
}
ORDINAL_CATEGORICAL_FEATURES = ["sg", "al", "su"]
NOMINAL_CATEGORICAL_FEATURES = [
    "rbc", "pc", "pcc", "ba", "htn", "dm",
    "cad", "appet", "pe", "ane",
]
CATEGORICAL_FEATURES = (
    ORDINAL_CATEGORICAL_FEATURES
    + NOMINAL_CATEGORICAL_FEATURES
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"
MISSING_TOKEN = "__MISSING__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPOSITORY_ROOT / "data/processed/ckd/kidney_disease_sanitized.csv"))
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/ckd/synthetic_experiment"),
    )
    parser.add_argument("--target_col", default=TARGET)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument(
        "--size_multipliers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=["gaussian", "empirical_bootstrap", "ddpm"],
        default=["gaussian", "empirical_bootstrap", "ddpm"],
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=["lr", "mlp", "xgb", "rfc"],
        default=["lr", "mlp", "xgb", "rfc"],
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--include_real_baseline", action="store_true")
    parser.add_argument("--save_synthetic", action="store_true")

    # Diffusion parameters.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', or a torch device such as 'cuda'.",
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_and_validate_data(path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected = FEATURE_COLUMNS + [target_col]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataset is missing columns: {missing}"
        )

    df = df[expected].copy()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Convert categorical values to stable strings while preserving missingness.
    for col in CATEGORICAL_FEATURES:
        series = df[col]
        df[col] = series.where(series.notna(), np.nan)

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="raise",
    ).astype(int)

    target_values = sorted(df[target_col].unique().tolist())
    if target_values != [0, 1]:
        raise ValueError(
            f"Expected binary target [0, 1], got {target_values}"
        )

    return df


# ---------------------------------------------------------------------
# Downstream classifier pipeline
# ---------------------------------------------------------------------

def make_onehot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_classifier_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        ("onehot", make_onehot_encoder()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def make_classifier(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(
            max_iter=2000,
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
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed, but 'xgb' was requested."
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


def prediction_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(classifier, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))

    return model.predict(X).astype(float)


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "auc": roc_auc_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": (
            tp / (tp + fn) if (tp + fn) else np.nan
        ),
        "specificity": (
            tn / (tn + fp) if (tn + fp) else np.nan
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
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
    classifier_name: str,
    seed: int,
) -> Dict[str, float]:
    model = Pipeline([
        ("preprocess", make_classifier_preprocessor()),
        ("classifier", make_classifier(classifier_name, seed)),
    ])

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_col].astype(int).to_numpy()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_col].astype(int).to_numpy()

    model.fit(X_train, y_train)
    y_score = prediction_scores(model, X_test)

    return evaluate_predictions(y_test, y_score)


# ---------------------------------------------------------------------
# Mixed-type encoder for diffusion and distance calculations
# ---------------------------------------------------------------------

@dataclass
class Block:
    name: str
    start: int
    end: int
    levels: List[str] | None = None


class CKDMixedEncoder:
    """
    Training-only mixed-type encoder.

    Representation:
      - standardized, median-imputed numerical values
      - a 2-dimensional observed/missing one-hot block per numerical feature
      - one-hot categorical blocks including an explicit missing category
    """

    def __init__(self):
        self.numeric_medians: Dict[str, float] = {}
        self.numeric_means: Dict[str, float] = {}
        self.numeric_stds: Dict[str, float] = {}
        self.numeric_mins: Dict[str, float] = {}
        self.numeric_maxs: Dict[str, float] = {}
        self.category_levels: Dict[str, List[str]] = {}
        self.blocks: List[Block] = []
        self.dimension: int = 0

    @staticmethod
    def _categorical_tokens(series: pd.Series) -> pd.Series:
        return series.map(
            lambda value: (
                MISSING_TOKEN
                if pd.isna(value)
                else str(value)
            )
        )

    def fit(self, train_df: pd.DataFrame) -> "CKDMixedEncoder":
        self.blocks = []
        cursor = 0

        # Continuous numeric block.
        self.blocks.append(
            Block("numeric_values", cursor, cursor + len(NUMERIC_FEATURES))
        )
        cursor += len(NUMERIC_FEATURES)

        for col in NUMERIC_FEATURES:
            observed = pd.to_numeric(
                train_df[col],
                errors="coerce",
            )
            median = float(observed.median())
            filled = observed.fillna(median)

            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0

            nonmissing = observed.dropna()
            if len(nonmissing) == 0:
                minimum = median
                maximum = median
            else:
                minimum = float(nonmissing.min())
                maximum = float(nonmissing.max())

            self.numeric_medians[col] = median
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std
            self.numeric_mins[col] = minimum
            self.numeric_maxs[col] = maximum

        # Numerical missingness blocks.
        for col in NUMERIC_FEATURES:
            self.blocks.append(
                Block(
                    f"missing::{col}",
                    cursor,
                    cursor + 2,
                    levels=["observed", "missing"],
                )
            )
            cursor += 2

        # Categorical blocks with explicit missing category.
        for col in CATEGORICAL_FEATURES:
            tokens = self._categorical_tokens(train_df[col])
            levels = sorted(tokens.unique().tolist())
            if MISSING_TOKEN not in levels:
                levels.append(MISSING_TOKEN)

            self.category_levels[col] = levels
            self.blocks.append(
                Block(
                    f"category::{col}",
                    cursor,
                    cursor + len(levels),
                    levels=levels,
                )
            )
            cursor += len(levels)

        self.dimension = cursor
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []

        numeric_matrix = []
        for col in NUMERIC_FEATURES:
            values = pd.to_numeric(
                df[col],
                errors="coerce",
            )
            filled = values.fillna(self.numeric_medians[col])
            standardized = (
                filled.to_numpy(dtype=float)
                - self.numeric_means[col]
            ) / self.numeric_stds[col]
            numeric_matrix.append(standardized)

        parts.append(np.column_stack(numeric_matrix))

        for col in NUMERIC_FEATURES:
            missing = df[col].isna().to_numpy().astype(int)
            one_hot = np.zeros((len(df), 2), dtype=float)
            one_hot[np.arange(len(df)), missing] = 1.0
            parts.append(one_hot)

        for col in CATEGORICAL_FEATURES:
            levels = self.category_levels[col]
            level_to_index = {
                level: idx for idx, level in enumerate(levels)
            }
            tokens = self._categorical_tokens(df[col])
            indices = tokens.map(
                lambda token: level_to_index.get(
                    token,
                    level_to_index[MISSING_TOKEN],
                )
            ).to_numpy()

            one_hot = np.zeros(
                (len(df), len(levels)),
                dtype=float,
            )
            one_hot[np.arange(len(df)), indices] = 1.0
            parts.append(one_hot)

        encoded = np.concatenate(parts, axis=1).astype(np.float32)

        if encoded.shape[1] != self.dimension:
            raise RuntimeError(
                f"Encoded dimension {encoded.shape[1]} "
                f"does not match fitted dimension {self.dimension}"
            )

        return encoded

    def decode(self, encoded: np.ndarray) -> pd.DataFrame:
        encoded = np.asarray(encoded)
        output = pd.DataFrame(index=np.arange(len(encoded)))

        numeric_values = encoded[:, :len(NUMERIC_FEATURES)]

        for idx, col in enumerate(NUMERIC_FEATURES):
            raw = (
                numeric_values[:, idx] * self.numeric_stds[col]
                + self.numeric_means[col]
            )
            raw = np.clip(
                raw,
                self.numeric_mins[col],
                self.numeric_maxs[col],
            )

            if col in INTEGER_LIKE_NUMERIC:
                raw = np.rint(raw)
            else:
                raw = np.round(raw, 3)

            output[col] = raw.astype(float)

        cursor = len(NUMERIC_FEATURES)

        for col in NUMERIC_FEATURES:
            block = encoded[:, cursor:cursor + 2]
            is_missing = np.argmax(block, axis=1) == 1
            output.loc[is_missing, col] = np.nan
            cursor += 2

        for col in CATEGORICAL_FEATURES:
            levels = self.category_levels[col]
            block = encoded[:, cursor:cursor + len(levels)]
            indices = np.argmax(block, axis=1)
            tokens = [levels[index] for index in indices]

            decoded: List[object] = []
            for token in tokens:
                if token == MISSING_TOKEN:
                    decoded.append(np.nan)
                elif col in ORDINAL_CATEGORICAL_FEATURES:
                    decoded.append(float(token))
                else:
                    decoded.append(token)

            output[col] = decoded
            cursor += len(levels)

        return output[FEATURE_COLUMNS]


# ---------------------------------------------------------------------
# Simple synthetic baselines
# ---------------------------------------------------------------------

def sample_target_labels(
    y_train: np.ndarray,
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probability_one = float(np.mean(y_train == 1))
    return rng.binomial(
        1,
        probability_one,
        size=n_rows,
    ).astype(int)


def empirical_bootstrap(
    real_train: pd.DataFrame,
    target_col: str,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_syn = sample_target_labels(
        real_train[target_col].astype(int).to_numpy(),
        n_rows,
        rng,
    )

    parts = []
    for class_value in [0, 1]:
        count = int((y_syn == class_value).sum())
        if count == 0:
            continue

        class_df = real_train[
            real_train[target_col] == class_value
        ]
        selected = rng.choice(
            class_df.index.to_numpy(),
            size=count,
            replace=True,
        )
        parts.append(class_df.loc[selected].copy())

    synthetic = pd.concat(parts, ignore_index=True)
    synthetic = synthetic.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return synthetic[FEATURE_COLUMNS + [target_col]]


def sample_empirical_tokens(
    series: pd.Series,
    n_rows: int,
    rng: np.random.Generator,
) -> List[object]:
    tokens = series.map(
        lambda value: (
            MISSING_TOKEN
            if pd.isna(value)
            else str(value)
        )
    )
    probabilities = tokens.value_counts(
        normalize=True,
        dropna=False,
    )

    levels = probabilities.index.to_list()
    probs = probabilities.to_numpy(dtype=float)
    probs /= probs.sum()

    sampled = rng.choice(
        levels,
        size=n_rows,
        replace=True,
        p=probs,
    )

    return [
        np.nan if token == MISSING_TOKEN else token
        for token in sampled
    ]


def gaussian_empirical(
    real_train: pd.DataFrame,
    target_col: str,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    """
    Class-conditional Gaussian/empirical baseline.

    Numerical values:
      multivariate Gaussian fitted to median-imputed values within each class;
      class-conditional feature-level missingness is sampled separately.

    Categorical values:
      independently sampled from class-conditional empirical distributions,
      including missing values as a valid outcome.
    """
    rng = np.random.default_rng(seed)
    y_syn = sample_target_labels(
        real_train[target_col].astype(int).to_numpy(),
        n_rows,
        rng,
    )

    parts = []

    for class_value in [0, 1]:
        count = int((y_syn == class_value).sum())
        if count == 0:
            continue

        class_df = real_train[
            real_train[target_col] == class_value
        ].copy()

        numeric = class_df[NUMERIC_FEATURES].astype(float)
        medians = numeric.median()
        filled = numeric.fillna(medians)

        mean = filled.mean(axis=0).to_numpy(dtype=float)
        covariance = np.cov(
            filled.to_numpy(dtype=float),
            rowvar=False,
        )
        covariance = np.asarray(covariance, dtype=float)

        if covariance.ndim == 0:
            covariance = np.eye(len(NUMERIC_FEATURES)) * float(
                covariance
            )

        covariance += np.eye(len(NUMERIC_FEATURES)) * 1e-6

        try:
            generated_numeric = rng.multivariate_normal(
                mean,
                covariance,
                size=count,
            )
        except np.linalg.LinAlgError:
            diagonal = np.diag(
                np.maximum(np.diag(covariance), 1e-6)
            )
            generated_numeric = rng.multivariate_normal(
                mean,
                diagonal,
                size=count,
            )

        part = pd.DataFrame(
            generated_numeric,
            columns=NUMERIC_FEATURES,
        )

        for col in NUMERIC_FEATURES:
            observed = numeric[col].dropna()
            if len(observed):
                part[col] = part[col].clip(
                    float(observed.min()),
                    float(observed.max()),
                )

            missing_probability = float(
                numeric[col].isna().mean()
            )
            missing_mask = rng.random(count) < missing_probability

            if col in INTEGER_LIKE_NUMERIC:
                part[col] = np.rint(part[col])
            else:
                part[col] = part[col].round(3)

            part.loc[missing_mask, col] = np.nan

        for col in CATEGORICAL_FEATURES:
            sampled = sample_empirical_tokens(
                class_df[col],
                count,
                rng,
            )

            if col in ORDINAL_CATEGORICAL_FEATURES:
                sampled = [
                    np.nan if pd.isna(value) else float(value)
                    for value in sampled
                ]

            part[col] = sampled

        part[target_col] = int(class_value)
        parts.append(part)

    synthetic = pd.concat(parts, ignore_index=True)
    synthetic = synthetic.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return synthetic[FEATURE_COLUMNS + [target_col]]


# ---------------------------------------------------------------------
# Conditional diffusion model
# ---------------------------------------------------------------------

def resolve_device(device_arg: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the DDPM generator."
        ) from exc

    if device_arg == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device_arg)


def sinusoidal_time_embedding(timesteps, dimension: int):
    import torch

    half = dimension // 2
    exponent = -math.log(10000.0) * torch.arange(
        half,
        device=timesteps.device,
        dtype=torch.float32,
    ) / max(half - 1, 1)

    frequencies = torch.exp(exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

    embedding = torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=1,
    )

    if dimension % 2 == 1:
        embedding = torch.cat(
            [
                embedding,
                torch.zeros(
                    len(timesteps),
                    1,
                    device=timesteps.device,
                ),
            ],
            dim=1,
        )

    return embedding


def build_denoiser(
    input_dim: int,
    hidden_dim: int,
    time_dim: int,
):
    import torch
    import torch.nn as nn

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_embedding = nn.Embedding(2, time_dim)
            self.network = nn.Sequential(
                nn.Linear(input_dim + time_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x, t, y):
            time_embedding = sinusoidal_time_embedding(
                t,
                time_dim,
            )
            label_embedding = self.label_embedding(y)
            conditioned = torch.cat(
                [x, time_embedding, label_embedding],
                dim=1,
            )
            return self.network(conditioned)

    return Denoiser()


@dataclass
class DiffusionSchedule:
    betas: object
    alphas: object
    alpha_bars: object


def make_diffusion_schedule(
    timesteps: int,
    device,
) -> DiffusionSchedule:
    import torch

    betas = torch.linspace(
        1e-4,
        0.02,
        timesteps,
        device=device,
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )


def train_conditional_ddpm(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    timesteps: int,
    batch_size: int,
    hidden_dim: int,
    time_dim: int,
    learning_rate: float,
    device,
):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    set_all_seeds(seed)

    x_tensor = torch.tensor(
        encoded_train,
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=min(batch_size, len(x_tensor)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    model = build_denoiser(
        input_dim=encoded_train.shape[1],
        hidden_dim=hidden_dim,
        time_dim=time_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    schedule = make_diffusion_schedule(
        timesteps=timesteps,
        device=device,
    )

    model.train()

    for epoch in range(epochs):
        epoch_losses = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_n = len(x_batch)
            t = torch.randint(
                low=0,
                high=timesteps,
                size=(batch_n,),
                device=device,
            )

            noise = torch.randn_like(x_batch)
            alpha_bar = schedule.alpha_bars[t].unsqueeze(1)

            noisy = (
                torch.sqrt(alpha_bar) * x_batch
                + torch.sqrt(1.0 - alpha_bar) * noise
            )

            predicted_noise = model(
                noisy,
                t,
                y_batch,
            )
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
            or epoch + 1 == epochs
        ):
            print(
                f"      DDPM epoch {epoch + 1}/{epochs}, "
                f"loss={np.mean(epoch_losses):.6f}",
                flush=True,
            )

    return model, schedule


def sample_conditional_ddpm(
    model,
    schedule: DiffusionSchedule,
    labels: np.ndarray,
    input_dim: int,
    timesteps: int,
    seed: int,
    device,
) -> np.ndarray:
    import torch

    set_all_seeds(seed)
    model.eval()

    y = torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )
    x = torch.randn(
        len(labels),
        input_dim,
        device=device,
    )

    with torch.no_grad():
        for step in reversed(range(timesteps)):
            t = torch.full(
                (len(labels),),
                step,
                device=device,
                dtype=torch.long,
            )

            predicted_noise = model(x, t, y)

            alpha = schedule.alphas[step]
            alpha_bar = schedule.alpha_bars[step]
            beta = schedule.betas[step]

            mean = (
                1.0 / torch.sqrt(alpha)
            ) * (
                x
                - (
                    beta
                    / torch.sqrt(1.0 - alpha_bar)
                ) * predicted_noise
            )

            if step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean

    return x.detach().cpu().numpy()


# ---------------------------------------------------------------------
# Fidelity and privacy-like metrics
# ---------------------------------------------------------------------

def safe_correlation_difference(
    real_encoded: np.ndarray,
    synthetic_encoded: np.ndarray,
) -> float:
    real_corr = np.corrcoef(real_encoded, rowvar=False)
    synthetic_corr = np.corrcoef(
        synthetic_encoded,
        rowvar=False,
    )

    real_corr = np.nan_to_num(real_corr)
    synthetic_corr = np.nan_to_num(synthetic_corr)

    upper = np.triu_indices_from(real_corr, k=1)
    return float(
        np.mean(
            np.abs(
                real_corr[upper]
                - synthetic_corr[upper]
            )
        )
    )


def wasserstein_numeric(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> float:
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return float("nan")

    distances = []

    for col in NUMERIC_FEATURES:
        real_values = pd.to_numeric(
            real_train[col],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        synthetic_values = pd.to_numeric(
            synthetic[col],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        if len(real_values) == 0 or len(synthetic_values) == 0:
            continue

        mean = float(real_values.mean())
        std = float(real_values.std(ddof=0))
        if std < 1e-8:
            std = 1.0

        real_scaled = (real_values - mean) / std
        synthetic_scaled = (
            synthetic_values - mean
        ) / std

        distances.append(
            wasserstein_distance(
                real_scaled,
                synthetic_scaled,
            )
        )

    return (
        float(np.mean(distances))
        if distances
        else float("nan")
    )


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon

    p /= p.sum()
    q /= q.sum()

    midpoint = 0.5 * (p + q)

    return float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )


def categorical_tokens(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: (
            MISSING_TOKEN
            if pd.isna(value)
            else str(value)
        )
    )


def categorical_js(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
) -> float:
    values = []

    for col in CATEGORICAL_FEATURES + [target_col]:
        real_tokens = categorical_tokens(real_train[col])
        synthetic_tokens = categorical_tokens(synthetic[col])

        levels = sorted(
            set(real_tokens.unique())
            | set(synthetic_tokens.unique())
        )

        real_counts = np.array(
            [(real_tokens == level).sum() for level in levels],
            dtype=float,
        )
        synthetic_counts = np.array(
            [
                (synthetic_tokens == level).sum()
                for level in levels
            ],
            dtype=float,
        )

        values.append(
            js_divergence(
                real_counts,
                synthetic_counts,
            )
        )

    return float(np.mean(values))


def missingness_mae(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> float:
    real_rates = real_train[FEATURE_COLUMNS].isna().mean()
    synthetic_rates = synthetic[FEATURE_COLUMNS].isna().mean()

    return float(
        np.mean(
            np.abs(
                real_rates.to_numpy()
                - synthetic_rates.to_numpy()
            )
        )
    )


def missingness_js(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> float:
    divergences = []

    for col in FEATURE_COLUMNS:
        p_missing = float(real_train[col].isna().mean())
        q_missing = float(synthetic[col].isna().mean())

        divergences.append(
            js_divergence(
                np.array([1.0 - p_missing, p_missing]),
                np.array([1.0 - q_missing, q_missing]),
            )
        )

    return float(np.mean(divergences))


def real_to_real_reference(
    real_encoded: np.ndarray,
) -> Dict[str, float]:
    distances = pairwise_distances(
        real_encoded,
        real_encoded,
        metric="euclidean",
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)

    return {
        "real_nn_mean": float(nearest.mean()),
        "real_nn_p05": float(np.quantile(nearest, 0.05)),
        "real_nn_min": float(nearest.min()),
    }


def dcr_metrics(
    real_encoded: np.ndarray,
    synthetic_encoded: np.ndarray,
    reference: Dict[str, float],
) -> Dict[str, float]:
    distances = pairwise_distances(
        synthetic_encoded,
        real_encoded,
        metric="euclidean",
    )
    nearest = distances.min(axis=1)

    real_nn_mean = reference["real_nn_mean"]
    real_nn_p05 = reference["real_nn_p05"]

    return {
        "dcr_mean": float(nearest.mean()),
        "dcr_median": float(np.median(nearest)),
        "dcr_p05": float(np.quantile(nearest, 0.05)),
        "dcr_min": float(nearest.min()),
        "real_nn_mean": real_nn_mean,
        "real_nn_p05": real_nn_p05,
        "real_nn_min": reference["real_nn_min"],
        "dcr_mean_to_real_nn_ratio": (
            float(nearest.mean() / real_nn_mean)
            if real_nn_mean > 0
            else np.nan
        ),
        "fraction_below_real_nn_p05": float(
            np.mean(nearest < real_nn_p05)
        ),
    }


def canonicalize(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    output = df[list(columns)].copy()

    for col in columns:
        output[col] = output[col].map(
            lambda value: (
                MISSING_TOKEN
                if pd.isna(value)
                else (
                    round(float(value), 6)
                    if col in NUMERIC_FEATURES
                    or col in ORDINAL_CATEGORICAL_FEATURES
                    or col == TARGET
                    else str(value)
                )
            )
        )

    return output


def duplicate_rate_against_real(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    real_rows = set(
        map(
            tuple,
            canonicalize(real_train, columns).to_numpy(),
        )
    )
    synthetic_rows = list(
        map(
            tuple,
            canonicalize(synthetic, columns).to_numpy(),
        )
    )

    if not synthetic_rows:
        return np.nan

    return float(
        sum(row in real_rows for row in synthetic_rows)
        / len(synthetic_rows)
    )


def internal_duplicate_rate(
    synthetic: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    if len(synthetic) == 0:
        return np.nan

    canonical = canonicalize(synthetic, columns)
    return float(
        1.0
        - len(canonical.drop_duplicates())
        / len(canonical)
    )


def range_violation_rate(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> float:
    total = 0
    violations = 0

    for col in NUMERIC_FEATURES:
        real_values = pd.to_numeric(
            real_train[col],
            errors="coerce",
        ).dropna()

        synthetic_values = pd.to_numeric(
            synthetic[col],
            errors="coerce",
        )
        observed = synthetic_values.dropna()

        if len(real_values) == 0 or len(observed) == 0:
            continue

        minimum = float(real_values.min())
        maximum = float(real_values.max())

        total += len(observed)
        violations += int(
            ((observed < minimum) | (observed > maximum)).sum()
        )

    return (
        float(violations / total)
        if total > 0
        else np.nan
    )


def quality_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    encoder: CKDMixedEncoder,
    real_encoded: np.ndarray,
    real_reference: Dict[str, float],
) -> Dict[str, float]:
    synthetic_encoded = encoder.transform(synthetic)

    result = {
        "pcd": safe_correlation_difference(
            real_encoded,
            synthetic_encoded,
        ),
        "ws": wasserstein_numeric(
            real_train,
            synthetic,
        ),
        "js": categorical_js(
            real_train,
            synthetic,
            target_col,
        ),
        "missingness_mae": missingness_mae(
            real_train,
            synthetic,
        ),
        "missingness_js": missingness_js(
            real_train,
            synthetic,
        ),
        "exact_duplicate_rate": duplicate_rate_against_real(
            real_train,
            synthetic,
            FEATURE_COLUMNS + [target_col],
        ),
        "feature_duplicate_rate": duplicate_rate_against_real(
            real_train,
            synthetic,
            FEATURE_COLUMNS,
        ),
        "internal_duplicate_rate": internal_duplicate_rate(
            synthetic,
            FEATURE_COLUMNS + [target_col],
        ),
        "range_violation_rate": range_violation_rate(
            real_train,
            synthetic,
        ),
        "real_target_1_rate": float(
            real_train[target_col].mean()
        ),
        "synthetic_target_1_rate": float(
            synthetic[target_col].mean()
        ),
    }

    result["target_1_rate_abs_diff"] = abs(
        result["real_target_1_rate"]
        - result["synthetic_target_1_rate"]
    )

    result.update(
        dcr_metrics(
            real_encoded,
            synthetic_encoded,
            real_reference,
        )
    )

    return result


# ---------------------------------------------------------------------
# Validation and summaries
# ---------------------------------------------------------------------

def validate_synthetic(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    expected_rows: int,
) -> Dict[str, object]:
    missing_columns = [
        col
        for col in FEATURE_COLUMNS + [target_col]
        if col not in synthetic.columns
    ]

    invalid_target = (
        int(
            (~synthetic[target_col].isin([0, 1])).sum()
        )
        if target_col in synthetic.columns
        else np.nan
    )

    invalid_categories = 0
    if not missing_columns:
        for col in CATEGORICAL_FEATURES:
            allowed = set(
                categorical_tokens(real_train[col]).unique()
            )
            generated = set(
                categorical_tokens(synthetic[col]).unique()
            )
            invalid_categories += len(generated - allowed)

    status = (
        "OK"
        if (
            not missing_columns
            and len(synthetic) == expected_rows
            and invalid_target == 0
            and invalid_categories == 0
        )
        else "FAIL"
    )

    return {
        "status": status,
        "n_rows": len(synthetic),
        "expected_rows": expected_rows,
        "missing_columns": "|".join(missing_columns),
        "invalid_target_values": invalid_target,
        "invalid_category_levels": invalid_categories,
        "total_missing_feature_values": int(
            synthetic[FEATURE_COLUMNS].isna().sum().sum()
        ) if not missing_columns else np.nan,
    }


def create_summaries(
    utility_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    utility_summary = (
        utility_df
        .groupby(
            ["generator", "classifier", "size_multiplier"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_evaluations=("auc", "size"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"),
            brier_mean=("brier", "mean"),
        )
        .sort_values(
            ["generator", "classifier", "size_multiplier"]
        )
    )
    utility_summary.to_csv(
        out_dir / "ckd_synthetic_utility_summary.csv",
        index=False,
    )

    quality_summary = (
        quality_df
        .groupby(
            ["generator", "size_multiplier"],
            as_index=False,
        )
        .agg(
            n_evaluations=("split", "size"),
            pcd_mean=("pcd", "mean"),
            ws_mean=("ws", "mean"),
            js_mean=("js", "mean"),
            missingness_mae_mean=("missingness_mae", "mean"),
            missingness_js_mean=("missingness_js", "mean"),
            dcr_mean=("dcr_mean", "mean"),
            dcr_p05=("dcr_p05", "mean"),
            dcr_min=("dcr_min", "min"),
            real_nn_mean=("real_nn_mean", "mean"),
            dcr_ratio_mean=(
                "dcr_mean_to_real_nn_ratio",
                "mean",
            ),
            fraction_below_real_nn_p05_mean=(
                "fraction_below_real_nn_p05",
                "mean",
            ),
            exact_duplicate_rate_mean=(
                "exact_duplicate_rate",
                "mean",
            ),
            exact_duplicate_rate_max=(
                "exact_duplicate_rate",
                "max",
            ),
            feature_duplicate_rate_mean=(
                "feature_duplicate_rate",
                "mean",
            ),
            internal_duplicate_rate_mean=(
                "internal_duplicate_rate",
                "mean",
            ),
            range_violation_rate_mean=(
                "range_violation_rate",
                "mean",
            ),
            target_1_rate_abs_diff_mean=(
                "target_1_rate_abs_diff",
                "mean",
            ),
        )
        .sort_values(
            ["generator", "size_multiplier"]
        )
    )
    quality_summary.to_csv(
        out_dir / "ckd_synthetic_quality_summary.csv",
        index=False,
    )

    # Mean utility over classifiers for a compact generator/size comparison.
    synthetic_utility = utility_summary[
        utility_summary["generator"] != "REAL"
    ]
    generator_size = (
        synthetic_utility
        .groupby(
            ["generator", "size_multiplier"],
            as_index=False,
        )
        .agg(
            auc_mean_over_classifiers=("auc_mean", "mean"),
            brier_mean_over_classifiers=("brier_mean", "mean"),
            f1_mean_over_classifiers=("f1_mean", "mean"),
        )
    )
    generator_size = generator_size.merge(
        quality_summary,
        on=["generator", "size_multiplier"],
        how="left",
    )
    generator_size.to_csv(
        out_dir / "ckd_generator_size_summary.csv",
        index=False,
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for generator, subset in generator_size.groupby(
            "generator"
        ):
            subset = subset.sort_values("size_multiplier")
            plt.plot(
                subset["size_multiplier"],
                subset["auc_mean_over_classifiers"],
                marker="o",
                label=generator,
            )
        plt.xlabel("Synthetic data size multiplier")
        plt.ylabel("Mean AUROC over classifiers")
        plt.title("CKD: utility by generator and synthetic size")
        plt.xticks(
            sorted(generator_size["size_multiplier"].unique())
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / "ckd_auc_by_generator_size.png",
            dpi=200,
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        for generator, subset in quality_summary.groupby(
            "generator"
        ):
            subset = subset.sort_values("size_multiplier")
            plt.plot(
                subset["size_multiplier"],
                subset["missingness_mae_mean"],
                marker="o",
                label=generator,
            )
        plt.xlabel("Synthetic data size multiplier")
        plt.ylabel("Mean absolute missingness-rate difference")
        plt.title("CKD: preservation of feature-level missingness")
        plt.xticks(
            sorted(quality_summary["size_multiplier"].unique())
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / "ckd_missingness_by_generator_size.png",
            dpi=200,
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        for generator, subset in quality_summary.groupby(
            "generator"
        ):
            subset = subset.sort_values("size_multiplier")
            plt.plot(
                subset["size_multiplier"],
                subset["exact_duplicate_rate_mean"],
                marker="o",
                label=generator,
            )
        plt.xlabel("Synthetic data size multiplier")
        plt.ylabel("Exact duplicate rate vs real train")
        plt.title("CKD: duplicate-rate sanity check")
        plt.xticks(
            sorted(quality_summary["size_multiplier"].unique())
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / "ckd_duplicate_rate_by_generator_size.png",
            dpi=200,
        )
        plt.close()

    except ImportError:
        print(
            "[WARN] matplotlib unavailable; plots were not created.",
            flush=True,
        )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_root = out_dir / "synthetic"
    if args.save_synthetic:
        synthetic_root.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_data(
        args.data_path,
        args.target_col,
    )

    metadata = {
        "dataset": "CKD",
        "records": len(df),
        "features": len(FEATURE_COLUMNS),
        "numeric_features": NUMERIC_FEATURES,
        "ordinal_categorical_features": (
            ORDINAL_CATEGORICAL_FEATURES
        ),
        "nominal_categorical_features": (
            NOMINAL_CATEGORICAL_FEATURES
        ),
        "target": args.target_col,
        "target_distribution": (
            df[args.target_col]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "n_repeats": args.n_repeats,
        "size_multipliers": args.size_multipliers,
        "generators": args.generators,
        "classifiers": args.classifiers,
        "random_seed": args.random_seed,
        "ddpm": {
            "epochs": args.epochs,
            "timesteps": args.timesteps,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "time_dim": args.time_dim,
            "learning_rate": args.learning_rate,
            "device": args.device,
        },
    }
    (
        out_dir / "ckd_experiment_metadata.json"
    ).write_text(json.dumps(metadata, indent=2))

    X = df[FEATURE_COLUMNS]
    y = df[args.target_col].astype(int)

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    utility_rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []

    device = (
        resolve_device(args.device)
        if "ddpm" in args.generators
        else None
    )

    if device is not None:
        print(f"[INFO] DDPM device: {device}", flush=True)

    for split, (train_idx, test_idx) in enumerate(
        splitter.split(X, y)
    ):
        real_train = (
            df.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )
        real_test = (
            df.iloc[test_idx]
            .copy()
            .reset_index(drop=True)
        )

        print(
            f"\n=== CKD split {split + 1}/{args.n_splits}: "
            f"train={len(real_train)}, test={len(real_test)} ===",
            flush=True,
        )

        encoder = CKDMixedEncoder().fit(real_train)
        real_encoded = encoder.transform(real_train)
        real_reference = real_to_real_reference(real_encoded)

        if args.include_real_baseline:
            print("  Evaluating REAL baseline...", flush=True)

            for classifier_name in args.classifiers:
                metrics = evaluate_classifier(
                    train_df=real_train,
                    test_df=real_test,
                    target_col=args.target_col,
                    classifier_name=classifier_name,
                    seed=args.random_seed + split,
                )

                utility_rows.append({
                    "dataset": "CKD",
                    "generator": "REAL",
                    "split": split,
                    "repeat": 0,
                    "size_multiplier": 1,
                    "n_training_rows": len(real_train),
                    "classifier": classifier_name,
                    **metrics,
                })

        # Simple baselines.
        simple_generators = [
            generator
            for generator in args.generators
            if generator in {
                "gaussian",
                "empirical_bootstrap",
            }
        ]

        for generator in simple_generators:
            output_name = {
                "gaussian": "GAUSSIAN_EMPIRICAL",
                "empirical_bootstrap": (
                    "EMPIRICAL_BOOTSTRAP"
                ),
            }[generator]

            for repeat in range(args.n_repeats):
                for multiplier in args.size_multipliers:
                    n_rows = len(real_train) * multiplier
                    seed = (
                        args.random_seed
                        + split * 100000
                        + repeat * 1000
                        + multiplier * 10
                        + (
                            777
                            if generator
                            == "empirical_bootstrap"
                            else 0
                        )
                    )

                    print(
                        f"  {output_name}: repeat={repeat}, "
                        f"size={multiplier}x",
                        flush=True,
                    )

                    if generator == "gaussian":
                        synthetic = gaussian_empirical(
                            real_train,
                            args.target_col,
                            n_rows,
                            seed,
                        )
                    else:
                        synthetic = empirical_bootstrap(
                            real_train,
                            args.target_col,
                            n_rows,
                            seed,
                        )

                    validation = validate_synthetic(
                        real_train,
                        synthetic,
                        args.target_col,
                        n_rows,
                    )
                    validation_rows.append({
                        "generator": output_name,
                        "split": split,
                        "repeat": repeat,
                        "size_multiplier": multiplier,
                        **validation,
                    })

                    quality = quality_metrics(
                        real_train,
                        synthetic,
                        args.target_col,
                        encoder,
                        real_encoded,
                        real_reference,
                    )
                    quality_rows.append({
                        "dataset": "CKD",
                        "generator": output_name,
                        "split": split,
                        "repeat": repeat,
                        "size_multiplier": multiplier,
                        "n_synthetic_rows": n_rows,
                        **quality,
                    })

                    if args.save_synthetic:
                        folder = (
                            synthetic_root
                            / output_name
                            / f"split_{split}"
                        )
                        folder.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        synthetic.to_csv(
                            folder
                            / (
                                f"synthetic_repeat{repeat}_"
                                f"{multiplier}x.csv"
                            ),
                            index=False,
                        )

                    for classifier_name in args.classifiers:
                        metrics = evaluate_classifier(
                            train_df=synthetic,
                            test_df=real_test,
                            target_col=args.target_col,
                            classifier_name=classifier_name,
                            seed=seed,
                        )

                        utility_rows.append({
                            "dataset": "CKD",
                            "generator": output_name,
                            "split": split,
                            "repeat": repeat,
                            "size_multiplier": multiplier,
                            "n_training_rows": n_rows,
                            "classifier": classifier_name,
                            **metrics,
                        })

                    # Checkpoint.
                    pd.DataFrame(utility_rows).to_csv(
                        out_dir
                        / "ckd_synthetic_utility.csv",
                        index=False,
                    )
                    pd.DataFrame(quality_rows).to_csv(
                        out_dir
                        / "ckd_synthetic_quality.csv",
                        index=False,
                    )
                    pd.DataFrame(validation_rows).to_csv(
                        out_dir
                        / "ckd_synthetic_validation.csv",
                        index=False,
                    )

        # Conditional DDPM.
        if "ddpm" in args.generators:
            encoded_train = real_encoded
            y_train = real_train[
                args.target_col
            ].astype(int).to_numpy()

            for repeat in range(args.n_repeats):
                model_seed = (
                    args.random_seed
                    + split * 100000
                    + repeat * 1000
                    + 9000000
                )

                print(
                    f"  COND_DDPM: training repeat={repeat}",
                    flush=True,
                )

                model, schedule = train_conditional_ddpm(
                    encoded_train=encoded_train,
                    y_train=y_train,
                    seed=model_seed,
                    epochs=args.epochs,
                    timesteps=args.timesteps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    time_dim=args.time_dim,
                    learning_rate=args.learning_rate,
                    device=device,
                )

                for multiplier in args.size_multipliers:
                    n_rows = len(real_train) * multiplier
                    sample_seed = (
                        model_seed + multiplier * 10
                    )

                    rng = np.random.default_rng(sample_seed)
                    labels = sample_target_labels(
                        y_train,
                        n_rows,
                        rng,
                    )

                    print(
                        f"    sampling size={multiplier}x "
                        f"({n_rows} rows)",
                        flush=True,
                    )

                    generated_encoded = sample_conditional_ddpm(
                        model=model,
                        schedule=schedule,
                        labels=labels,
                        input_dim=encoder.dimension,
                        timesteps=args.timesteps,
                        seed=sample_seed,
                        device=device,
                    )

                    synthetic = encoder.decode(
                        generated_encoded
                    )
                    synthetic[args.target_col] = labels
                    synthetic = synthetic[
                        FEATURE_COLUMNS + [args.target_col]
                    ]

                    validation = validate_synthetic(
                        real_train,
                        synthetic,
                        args.target_col,
                        n_rows,
                    )
                    validation_rows.append({
                        "generator": "COND_DDPM",
                        "split": split,
                        "repeat": repeat,
                        "size_multiplier": multiplier,
                        **validation,
                    })

                    quality = quality_metrics(
                        real_train,
                        synthetic,
                        args.target_col,
                        encoder,
                        real_encoded,
                        real_reference,
                    )
                    quality_rows.append({
                        "dataset": "CKD",
                        "generator": "COND_DDPM",
                        "split": split,
                        "repeat": repeat,
                        "size_multiplier": multiplier,
                        "n_synthetic_rows": n_rows,
                        **quality,
                    })

                    if args.save_synthetic:
                        folder = (
                            synthetic_root
                            / "COND_DDPM"
                            / f"split_{split}"
                        )
                        folder.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        synthetic.to_csv(
                            folder
                            / (
                                f"synthetic_repeat{repeat}_"
                                f"{multiplier}x.csv"
                            ),
                            index=False,
                        )

                    for classifier_name in args.classifiers:
                        metrics = evaluate_classifier(
                            train_df=synthetic,
                            test_df=real_test,
                            target_col=args.target_col,
                            classifier_name=classifier_name,
                            seed=sample_seed,
                        )

                        utility_rows.append({
                            "dataset": "CKD",
                            "generator": "COND_DDPM",
                            "split": split,
                            "repeat": repeat,
                            "size_multiplier": multiplier,
                            "n_training_rows": n_rows,
                            "classifier": classifier_name,
                            **metrics,
                        })

                    # Checkpoint.
                    pd.DataFrame(utility_rows).to_csv(
                        out_dir
                        / "ckd_synthetic_utility.csv",
                        index=False,
                    )
                    pd.DataFrame(quality_rows).to_csv(
                        out_dir
                        / "ckd_synthetic_quality.csv",
                        index=False,
                    )
                    pd.DataFrame(validation_rows).to_csv(
                        out_dir
                        / "ckd_synthetic_validation.csv",
                        index=False,
                    )

    utility_df = pd.DataFrame(utility_rows)
    quality_df = pd.DataFrame(quality_rows)
    validation_df = pd.DataFrame(validation_rows)

    utility_df.to_csv(
        out_dir / "ckd_synthetic_utility.csv",
        index=False,
    )
    quality_df.to_csv(
        out_dir / "ckd_synthetic_quality.csv",
        index=False,
    )
    validation_df.to_csv(
        out_dir / "ckd_synthetic_validation.csv",
        index=False,
    )

    create_summaries(
        utility_df=utility_df,
        quality_df=quality_df,
        out_dir=out_dir,
    )

    print("\nValidation status:")
    print(
        validation_df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    summary = pd.read_csv(
        out_dir / "ckd_synthetic_utility_summary.csv"
    )
    print("\nUtility summary:")
    print(summary.to_string(index=False))

    print("\nSaved results under:")
    print(f"  {out_dir}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
