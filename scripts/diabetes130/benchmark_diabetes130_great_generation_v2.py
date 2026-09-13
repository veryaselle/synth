#!/usr/bin/env python3
"""
Corrected GReaT generation-speed benchmark.

A configuration is counted as successful only when BOTH target-class calls
return exactly the requested number of rows. Partial output after an internal
CUDA OOM is labelled PARTIAL_FAILURE and is never recommended.

Example:
    python benchmark_diabetes130_great_generation_v2.py \
      --model_dir results/diabetes130/great_smoke_v2/saved_model \
      --real_train_csv results/diabetes130/great_smoke_v2/smoke_training_subset.csv \
      --out_dir results/diabetes130/great_generation_benchmark_v2 \
      --n_synthetic 100 \
      --k_values 10 25 50 \
      --max_length 512 \
      --temperature 0.7
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

TARGET = "readmitted_30d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--real_train_csv", required=True)
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/great_generation_benchmark_v2"),
    )
    parser.add_argument("--n_synthetic", type=int, default=100)
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[10, 25, 50],
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_class(
    model,
    *,
    target_value: int,
    n_samples: int,
    k: int,
    max_length: int,
    temperature: float,
) -> pd.DataFrame:
    if n_samples <= 0:
        return pd.DataFrame(columns=model.columns)

    return model.sample(
        n_samples=n_samples,
        start_col=TARGET,
        start_col_dist={int(target_value): 1.0},
        temperature=temperature,
        k=min(k, n_samples),
        max_length=max_length,
        drop_nan=False,
        device="cuda",
        guided_sampling=False,
        random_feature_order=True,
        conditions=None,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    real_train = pd.read_csv(args.real_train_csv)
    positive_rate = float(
        pd.to_numeric(real_train[TARGET], errors="raise").mean()
    )
    positive_n = int(round(args.n_synthetic * positive_rate))
    positive_n = max(1, min(positive_n, args.n_synthetic - 1))
    negative_n = args.n_synthetic - positive_n

    from be_great import GReaT

    model = GReaT.load_from_dir(args.model_dir)
    model.model.float()

    results: List[Dict[str, object]] = []

    for k in args.k_values:
        set_seeds(args.random_seed)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.time()
        negative = pd.DataFrame()
        positive = pd.DataFrame()
        error_text = ""

        try:
            negative = generate_class(
                model,
                target_value=0,
                n_samples=negative_n,
                k=k,
                max_length=args.max_length,
                temperature=args.temperature,
            )
            positive = generate_class(
                model,
                target_value=1,
                n_samples=positive_n,
                k=k,
                max_length=args.max_length,
                temperature=args.temperature,
            )
        except torch.OutOfMemoryError as exc:
            error_text = str(exc)
            torch.cuda.empty_cache()
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - start
        negative_returned = len(negative)
        positive_returned = len(positive)
        total_returned = negative_returned + positive_returned

        complete = (
            negative_returned == negative_n
            and positive_returned == positive_n
        )

        if complete:
            status = "OK"
        elif total_returned > 0:
            status = "PARTIAL_FAILURE"
        else:
            status = "FAILED"

        frames = [
            frame
            for frame in [negative, positive]
            if not frame.empty
        ]
        output_csv = ""

        if frames:
            synthetic = (
                pd.concat(frames, ignore_index=True)
                .sample(frac=1.0, random_state=args.random_seed)
                .reset_index(drop=True)
            )
            output_path = out_dir / f"synthetic_benchmark_k{k}.csv"
            synthetic.to_csv(output_path, index=False)
            output_csv = str(output_path)

        row: Dict[str, object] = {
            "k": k,
            "status": status,
            "n_synthetic_requested": args.n_synthetic,
            "negative_requested": negative_n,
            "negative_returned": negative_returned,
            "positive_requested": positive_n,
            "positive_returned": positive_returned,
            "n_synthetic_returned": total_returned,
            "generation_seconds": elapsed,
            "seconds_per_returned_row": (
                elapsed / total_returned
                if total_returned
                else np.nan
            ),
            "rows_per_second": (
                total_returned / elapsed
                if elapsed > 0 and total_returned
                else np.nan
            ),
            "peak_gpu_memory_gb": round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            ),
            "error": error_text,
            "output_csv": output_csv,
        }
        results.append(row)
        print(json.dumps(row, indent=2))

        gc.collect()
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        out_dir / "generation_benchmark.csv",
        index=False,
    )

    successful = results_df[
        results_df["status"] == "OK"
    ].copy()
    safe = successful[
        successful["peak_gpu_memory_gb"] < 8.5
    ].copy()

    recommendation: Dict[str, object] = {
        "selection_rule": (
            "Fastest complete configuration below 8.5 GB peak allocated "
            "memory. Partial releases are excluded."
        ),
        "recommended_k": None,
    }

    if not safe.empty:
        best = safe.sort_values(
            "seconds_per_returned_row"
        ).iloc[0]
        recommendation.update(
            {
                "recommended_k": int(best["k"]),
                "seconds_per_returned_row": float(
                    best["seconds_per_returned_row"]
                ),
                "peak_gpu_memory_gb": float(
                    best["peak_gpu_memory_gb"]
                ),
            }
        )

    (
        out_dir / "generation_benchmark_recommendation.json"
    ).write_text(json.dumps(recommendation, indent=2))

    print("\nBenchmark summary:")
    print(results_df.to_string(index=False))
    print("\nRecommendation:")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
