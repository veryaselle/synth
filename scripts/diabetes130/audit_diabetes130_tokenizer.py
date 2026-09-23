#!/usr/bin/env python3
"""
Measure the exact token lengths of serialized Diabetes 130-US rows using a
chosen local or Hugging Face tokenizer.

This script performs no model training.

Example:
    python audit_diabetes130_tokenizer.py \
      --csv_path results/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
      --model_name_or_path distilgpt2 \
      --out_dir results/diabetes130/tokenizer_audit \
      --sample_rows 5000 \
      --max_length 1024

For an offline cluster, --model_name_or_path may point to a local directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from typing import Sequence

import numpy as np
import pandas as pd


TARGET = "readmitted_30d"
MISSING_TOKEN = "__MISSING__"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--out_dir",
        default=str(REPOSITORY_ROOT / "results/diabetes130/tokenizer_audit"),
    )
    parser.add_argument("--sample_rows", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def serialise_row(
    row: pd.Series,
    feature_columns: Sequence[str],
) -> str:
    fields = [f"{TARGET}={int(row[TARGET])}"]

    for col in feature_columns:
        value = row[col]
        displayed = (
            MISSING_TOKEN
            if pd.isna(value)
            else str(value).strip()
        )
        fields.append(f"{col}={displayed}")

    return " | ".join(fields)


def main():
    args = parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install transformers before running this audit: "
            "pip install transformers"
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET}' was not found."
        )

    sample = df.sample(
        n=min(args.sample_rows, len(df)),
        random_state=args.random_seed,
    ).reset_index(drop=True)

    feature_columns = [
        col for col in sample.columns if col != TARGET
    ]

    texts = sample.apply(
        lambda row: serialise_row(row, feature_columns),
        axis=1,
    ).tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )

    token_lengths = []
    truncated_rows = []

    for row_index, text in enumerate(texts):
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        length = len(token_ids)
        token_lengths.append(length)

        if length > args.max_length:
            truncated_rows.append(
                {
                    "sample_row": row_index,
                    "token_length": length,
                    "text": text,
                }
            )

    lengths = np.asarray(token_lengths, dtype=int)

    summary = {
        "csv_path": args.csv_path,
        "model_name_or_path": args.model_name_or_path,
        "sample_rows": len(sample),
        "feature_count": len(feature_columns),
        "max_length_threshold": args.max_length,
        "min_tokens": int(lengths.min()),
        "median_tokens": float(np.median(lengths)),
        "p90_tokens": float(np.quantile(lengths, 0.90)),
        "p95_tokens": float(np.quantile(lengths, 0.95)),
        "p99_tokens": float(np.quantile(lengths, 0.99)),
        "max_tokens": int(lengths.max()),
        "rows_above_threshold": int(
            (lengths > args.max_length).sum()
        ),
        "rows_above_threshold_pct": float(
            (lengths > args.max_length).mean() * 100
        ),
    }

    (
        out_dir / "tokenizer_length_summary.json"
    ).write_text(json.dumps(summary, indent=2))

    pd.DataFrame(
        {
            "sample_row": np.arange(len(lengths)),
            "token_length": lengths,
        }
    ).to_csv(
        out_dir / "tokenizer_row_lengths.csv",
        index=False,
    )

    pd.DataFrame(truncated_rows).to_csv(
        out_dir / "rows_above_max_length.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved under: {out_dir}")


if __name__ == "__main__":
    main()
