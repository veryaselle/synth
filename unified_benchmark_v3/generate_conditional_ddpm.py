#!/usr/bin/env python3
"""Generate one 1× synthetic release with a conditional tabular DDPM.

Unified thesis benchmark contract:
- consumes one frozen processed real_train.csv + schema.json
- sees real_train only; held-out real_test is never used
- target is binary and used as the DDPM condition
- release size is exactly 1× len(real_train)
- numeric features: real-train standardization
- categorical features: real-train one-hot blocks
- numeric decoding: inverse standardization, NO min/max clipping
- categorical decoding: argmax over levels observed in real_train
- integer-like numeric columns are rounded back to integers
- writes synthetic_1x.csv, generation_metadata.json, training_history.csv

Default hyperparameters match the earlier standalone conditional diffusion
experiments in the thesis: 300 epochs, 100 timesteps, batch size 64,
hidden dim 128, time embedding 32, label embedding 8, AdamW lr=1e-3,
weight_decay=1e-5.

Use evaluate_release.py for utility/fidelity/privacy evaluation.
"""
from __future__ import annotations

import os
for _var in [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_schema(path: Path) -> Dict[str, object]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    for key in ["target", "numeric", "categorical"]:
        if key not in schema:
            raise ValueError(f"schema.json missing '{key}'")
    return schema


@dataclass
class CategoryBlock:
    column: str
    levels: List[object]
    start: int
    end: int


class FrozenTabularEncoder:
    """Train-split-only mixed-type encoder."""

    def __init__(self, numeric: List[str], categorical: List[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.numeric_mean: Dict[str, float] = {}
        self.numeric_std: Dict[str, float] = {}
        self.numeric_integer_like: Dict[str, bool] = {}
        self.category_blocks: List[CategoryBlock] = []
        self.output_dim: int | None = None

    def fit(self, df: pd.DataFrame) -> "FrozenTabularEncoder":
        if df[self.numeric + self.categorical].isna().any().any():
            raise ValueError(
                "Frozen generator input contains missing feature values; "
                "preprocessing contract was violated"
            )

        for col in self.numeric:
            values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std
            self.numeric_integer_like[col] = bool(
                np.all(np.isclose(values, np.rint(values), atol=1e-8))
            )

        cursor = len(self.numeric)
        for col in self.categorical:
            levels = sorted(df[col].dropna().unique().tolist(), key=lambda x: str(x))
            if not levels:
                raise ValueError(f"Categorical column '{col}' has no observed levels")
            self.category_blocks.append(
                CategoryBlock(col, levels, cursor, cursor + len(levels))
            )
            cursor += len(levels)

        self.output_dim = cursor
        if self.output_dim <= 0:
            raise ValueError("No generator feature columns found")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.output_dim is None:
            raise RuntimeError("Encoder not fitted")

        n = len(df)
        arr = np.zeros((n, self.output_dim), dtype=np.float32)

        for j, col in enumerate(self.numeric):
            values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
            arr[:, j] = (values - self.numeric_mean[col]) / self.numeric_std[col]

        for block in self.category_blocks:
            level_to_index = {str(level): idx for idx, level in enumerate(block.levels)}
            tokens = df[block.column].map(str)
            unknown = sorted(set(tokens.unique()) - set(level_to_index))
            if unknown:
                raise ValueError(
                    f"Column '{block.column}' contains unseen values during transform: "
                    f"{unknown[:10]}"
                )
            indices = tokens.map(level_to_index).to_numpy(dtype=int)
            arr[np.arange(n), block.start + indices] = 1.0

        return arr

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        if self.output_dim is None:
            raise RuntimeError("Encoder not fitted")

        encoded = np.asarray(encoded, dtype=float)
        if encoded.ndim != 2 or encoded.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected encoded shape (*, {self.output_dim}), got {encoded.shape}"
            )

        out = pd.DataFrame(index=np.arange(len(encoded)))

        # Deliberate benchmark choice: no numeric support clipping.
        for j, col in enumerate(self.numeric):
            raw = encoded[:, j] * self.numeric_std[col] + self.numeric_mean[col]
            if self.numeric_integer_like[col]:
                raw = np.rint(raw)
            out[col] = raw

        # Argmax keeps categorical values inside frozen real-train support.
        for block in self.category_blocks:
            block_values = encoded[:, block.start:block.end]
            idx = np.argmax(block_values, axis=1)
            levels = np.asarray(block.levels, dtype=object)
            out[block.column] = levels[idx]

        return out[self.numeric + self.categorical]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return t.float().unsqueeze(1)

        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -scale
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros(len(t), 1, device=t.device)], dim=1
            )
        return emb


class ConditionalDenoiseMLP(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, time_dim: int, label_dim: int):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.label_embedding = nn.Embedding(2, label_dim)

        in_dim = x_dim + time_dim + label_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        y_emb = self.label_embedding(y.long())
        return self.network(torch.cat([x_t, t_emb, y_emb], dim=1))


class ConditionalDDPM:
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
        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=device, dtype=torch.float32
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def training_loss(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(
            0, self.timesteps, (batch,), device=self.device, dtype=torch.long
        )
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).unsqueeze(1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_bars[t]).unsqueeze(1)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        predicted_noise = self.model(x_t, t, y)
        return nn.functional.mse_loss(predicted_noise, noise)

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

            predicted_noise = self.model(x, t, y_tensor)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x
                - (beta_t / torch.sqrt(1.0 - alpha_bar_t))
                * predicted_noise
            )

            if step > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

        return x.detach().cpu().numpy()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def sample_target_labels(y_train: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sample conditioning labels from empirical real-training prevalence."""
    rng = np.random.default_rng(seed)
    p1 = float(np.mean(y_train == 1))
    return rng.binomial(1, p1, size=n).astype(int)


def train_ddpm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    timesteps: int,
    batch_size: int,
    hidden_dim: int,
    time_dim: int,
    label_dim: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    verbose: bool,
):
    set_seed(seed)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = ConditionalDenoiseMLP(
        x_dim=X.shape[1],
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        label_dim=label_dim,
    ).to(device)

    ddpm = ConditionalDDPM(model=model, timesteps=timesteps, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            loss = ddpm.training_loss(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})

        if verbose and (
            epoch == 1
            or epoch == epochs
            or epoch % max(epochs // 10, 1) == 0
        ):
            print(f"epoch {epoch:4d}/{epochs} | loss={mean_loss:.6f}", flush=True)

    return ddpm, pd.DataFrame(history)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real_train", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--time_dim", type=int, default=32)
    p.add_argument("--label_dim", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)

    train = pd.read_csv(args.real_train)
    schema = load_schema(Path(args.schema))

    target = str(schema["target"])
    numeric = list(schema["numeric"])
    categorical = list(schema["categorical"])
    expected = list(schema.get("columns", train.columns))

    if list(train.columns) != expected:
        raise ValueError("real_train columns differ from frozen schema order")
    if train.isna().any().any():
        raise ValueError(
            "Main DDPM generator input contains missing values; "
            "frozen preprocessing contract was violated"
        )

    y = pd.to_numeric(train[target], errors="raise").astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Conditional DDPM requires binary target encoded as 0/1")

    encoder = FrozenTabularEncoder(numeric, categorical).fit(
        train[numeric + categorical]
    )
    X = encoder.transform(train[numeric + categorical])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            f"DRY RUN PASS: Conditional DDPM on {len(train)} rows, "
            f"x_dim={X.shape[1]}, numeric={len(numeric)}, "
            f"categorical={len(categorical)}, device={device}"
        )
        return

    t0 = time.time()

    ddpm, history = train_ddpm(
        X,
        y,
        epochs=args.epochs,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        label_dim=args.label_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        verbose=args.verbose,
    )

    synthetic_target = sample_target_labels(
        y, len(train), args.seed + 100_000
    )
    set_seed(args.seed + 200_000)
    X_syn = ddpm.sample(
        n=len(train), y=synthetic_target, x_dim=X.shape[1]
    )
    syn_features = encoder.inverse_transform(X_syn)

    synthetic = syn_features.copy()
    synthetic[target] = synthetic_target
    synthetic = synthetic[expected]

    elapsed = time.time() - t0

    if len(synthetic) != len(train):
        raise RuntimeError(
            f"Expected exactly 1×={len(train)} rows, got {len(synthetic)}"
        )
    if synthetic[target].nunique() < 2:
        raise RuntimeError("Generated target contains fewer than two classes")
    if synthetic.isna().any().any():
        raise RuntimeError("DDPM generated missing values unexpectedly")

    for col in categorical:
        real_levels = set(train[col].map(str).unique())
        syn_levels = set(synthetic[col].map(str).unique())
        unseen = syn_levels - real_levels
        if unseen:
            raise RuntimeError(
                f"DDPM produced unseen category in {col}: {sorted(unseen)[:10]}"
            )

    synthetic.to_csv(outdir / "synthetic_1x.csv", index=False)
    history.to_csv(outdir / "training_history.csv", index=False)

    support_violations = {}
    for col in numeric:
        r = pd.to_numeric(train[col], errors="raise").to_numpy(float)
        s = pd.to_numeric(synthetic[col], errors="raise").to_numpy(float)
        support_violations[col] = int(
            np.sum((s < np.min(r)) | (s > np.max(r)))
        )

    metadata = {
        "method": "Conditional DDPM",
        "method_key": "conditional_ddpm",
        "seed": args.seed,
        "device_requested": args.device,
        "device_used": str(device),
        "n_real_train": int(len(train)),
        "n_synthetic": int(len(synthetic)),
        "synthetic_size": "1x",
        "x_dim": int(X.shape[1]),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "epochs": args.epochs,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "time_dim": args.time_dim,
        "label_dim": args.label_dim,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "target_sampling": (
            "Bernoulli sample from real-training target prevalence; "
            "seed=generator_seed+100000"
        ),
        "target_prevalence_real": float(np.mean(y)),
        "target_prevalence_synthetic": float(np.mean(synthetic_target)),
        "numeric_decoding": (
            "inverse standardization; integer-like columns rounded; "
            "NO min/max clipping"
        ),
        "categorical_decoding": (
            "argmax over category levels observed in frozen real_train"
        ),
        "numeric_support_violation_counts": support_violations,
        "generation_seconds": elapsed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "note": (
            "Generator consumes frozen real_train only. No post-hoc numeric "
            "support clipping or arbitrary category repair is applied."
        ),
    }

    (outdir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
