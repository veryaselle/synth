#!/usr/bin/env python3
"""Generate one 1× synthetic release with SDV models or arfpy.

Supported methods
-----------------
TVAE, CTGAN, CopulaGAN, Gaussian Copula, ARF

The script follows the benchmark release contract. It reads a frozen processed
`real_train.csv` plus `schema.json`, fits one generator only on that split, and
writes `synthetic_1x.csv` and `generation_metadata.json`.

Deep SDV models use an explicit 300 epochs by default. ARF uses arfpy's native
training/forde/forge pipeline. Categorical columns and the target are temporarily
cast to strings for modeling and restored to the frozen real-training dtypes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

METHOD_MAP = {
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "copulagan": "CopulaGAN",
    "gaussian_copula": "Gaussian Copula",
    "arf": "ARF",
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_schema(path: Path) -> Dict[str, object]:
    d = json.loads(path.read_text())
    for k in ["target", "numeric", "categorical"]:
        if k not in d:
            raise ValueError(f"schema.json missing '{k}'")
    return d


def cast_for_generator(train: pd.DataFrame, schema: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """Force categorical fields to strings while preserving an exact restore map."""
    model = train.copy()
    categorical = list(schema["categorical"]) + [str(schema["target"])]
    restore: Dict[str, Dict[str, object]] = {}
    for col in categorical:
        vals = train[col].dropna().unique().tolist()
        mapping = {str(v): v for v in vals}
        if len(mapping) != len(vals):
            raise ValueError(f"String conversion is not one-to-one for categorical column {col}")
        restore[col] = mapping
        model[col] = model[col].map(lambda x: str(x))
    return model, restore


def restore_types(syn: pd.DataFrame, train: pd.DataFrame,
                  schema: Dict[str, object], restore: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    out = syn.copy()
    expected = list(train.columns)
    missing = set(expected) - set(out.columns)
    if missing:
        raise ValueError(f"Generated table is missing columns: {sorted(missing)}")
    out = out[expected]

    for col, mapping in restore.items():
        raw = out[col].astype(str)
        unseen = sorted(set(raw.unique()) - set(mapping))
        if unseen:
            raise ValueError(f"Generator produced unseen category in {col}: {unseen[:10]}")
        out[col] = raw.map(mapping)

    # Numeric columns must be parseable. Do not clip; out-of-support values are
    # legitimate fidelity findings and should remain visible to the evaluator.
    for col in schema["numeric"]:
        out[col] = pd.to_numeric(out[col], errors="raise")

    # Restore integer-like frozen columns when conversion is safe.
    for col in expected:
        dtype = train[col].dtype
        try:
            if pd.api.types.is_integer_dtype(dtype):
                num = pd.to_numeric(out[col], errors="raise")
                if np.all(np.isclose(num, np.rint(num), atol=1e-8)):
                    out[col] = np.rint(num).astype(dtype)
            elif pd.api.types.is_float_dtype(dtype) and col in schema["numeric"]:
                out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        except Exception:
            # Categorical codes may be restored as Python scalars and do not need
            # forced dtype conversion for the evaluator.
            pass
    return out


def make_sdv_metadata(model_df: pd.DataFrame):
    """Version-adaptive metadata construction for recent and legacy SDV APIs."""
    try:
        from sdv.metadata import Metadata
        if hasattr(Metadata, "detect_from_dataframe"):
            try:
                return Metadata.detect_from_dataframe(data=model_df, table_name="benchmark")
            except TypeError:
                return Metadata.detect_from_dataframe(model_df, table_name="benchmark")
        if hasattr(Metadata, "detect_from_dataframes"):
            try:
                return Metadata.detect_from_dataframes(data={"benchmark": model_df})
            except TypeError:
                return Metadata.detect_from_dataframes({"benchmark": model_df})
    except Exception:
        pass

    from sdv.metadata import SingleTableMetadata
    metadata = SingleTableMetadata()
    try:
        metadata.detect_from_dataframe(data=model_df)
    except TypeError:
        metadata.detect_from_dataframe(model_df)
    return metadata


def sdv_generate(method: str, model_df: pd.DataFrame, n: int, epochs: int, cuda: bool):
    from sdv.single_table import (
        TVAESynthesizer, CTGANSynthesizer, CopulaGANSynthesizer,
        GaussianCopulaSynthesizer,
    )
    metadata = make_sdv_metadata(model_df)
    if method == "tvae":
        synth = TVAESynthesizer(metadata, epochs=epochs, cuda=cuda)
    elif method == "ctgan":
        synth = CTGANSynthesizer(metadata, epochs=epochs, cuda=cuda)
    elif method == "copulagan":
        synth = CopulaGANSynthesizer(metadata, epochs=epochs, cuda=cuda)
    elif method == "gaussian_copula":
        synth = GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError(method)
    synth.fit(model_df)
    return synth.sample(num_rows=n)


def arf_generate(model_df: pd.DataFrame, n: int):
    from arfpy import arf
    model = arf.arf(x=model_df)
    model.forde()
    syn = model.forge(n=n)
    if not isinstance(syn, pd.DataFrame):
        syn = pd.DataFrame(syn, columns=model_df.columns)
    return syn


def pkg_version(name: str):
    try: return importlib.metadata.version(name)
    except Exception: return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=sorted(METHOD_MAP), required=True)
    p.add_argument("--real_train", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Validate contract without importing generator packages")
    args = p.parse_args()

    set_seeds(args.seed)
    train = pd.read_csv(args.real_train)
    schema = load_schema(Path(args.schema))
    target = str(schema["target"])
    expected = list(schema.get("columns", train.columns))
    if list(train.columns) != expected:
        raise ValueError("real_train columns differ from frozen schema order")
    if train.isna().any().any():
        raise ValueError("Main generator input contains missing values; frozen preprocessing contract was violated")
    if train[target].nunique() != 2:
        raise ValueError("Binary target required")

    model_df, restore = cast_for_generator(train, schema)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"DRY RUN PASS: {METHOD_MAP[args.method]} on {len(train)} rows, schema={len(train.columns)} columns")
        return

    t0 = time.time()
    if args.method == "arf":
        syn_model = arf_generate(model_df, len(train))
    else:
        syn_model = sdv_generate(args.method, model_df, len(train), args.epochs, args.cuda)
    elapsed = time.time() - t0
    syn = restore_types(syn_model, train, schema, restore)

    if len(syn) != len(train):
        raise RuntimeError(f"Expected exactly 1×={len(train)} rows, got {len(syn)}")
    if syn[target].nunique() < 2:
        raise RuntimeError("Generated target contains fewer than two classes")

    out_csv = outdir / "synthetic_1x.csv"
    syn.to_csv(out_csv, index=False)
    meta = {
        "method": METHOD_MAP[args.method],
        "method_key": args.method,
        "seed": args.seed,
        "n_real_train": len(train),
        "n_synthetic": len(syn),
        "synthetic_size": "1x",
        "epochs": args.epochs if args.method in {"tvae","ctgan","copulagan"} else None,
        "cuda_requested": bool(args.cuda),
        "generation_seconds": elapsed,
        "sdv_version": pkg_version("sdv"),
        "arfpy_version": pkg_version("arfpy"),
        "target_prevalence_real": float(pd.to_numeric(train[target]).mean()),
        "target_prevalence_synthetic": float(pd.to_numeric(syn[target]).mean()),
        "note": "No post-hoc clipping or arbitrary category repair is applied. Invalid category output causes failure.",
    }
    (outdir / "generation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
