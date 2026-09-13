#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.metadata, json, time
from pathlib import Path
import pandas as pd
import generate_sdv_arf_safe_cpu as BASE

def ver(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_train", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--config_name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--generator_lr", type=float, default=2e-4)
    p.add_argument("--discriminator_lr", type=float, required=True)
    p.add_argument("--discriminator_steps", type=int, default=1)
    p.add_argument("--pac", type=int, default=10)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    if a.batch_size % 2:
        raise ValueError("CTGAN requires an even batch_size")
    if a.batch_size % a.pac:
        raise ValueError("batch_size must be divisible by pac")
    if a.discriminator_steps < 1:
        raise ValueError("discriminator_steps must be >= 1")

    BASE.set_seeds(a.seed)
    train = pd.read_csv(a.real_train)
    schema = BASE.load_schema(Path(a.schema))
    target = str(schema["target"])
    expected = list(schema.get("columns", train.columns))
    if list(train.columns) != expected:
        raise ValueError("real_train columns differ from frozen schema order")
    if train.isna().any().any():
        raise ValueError("Frozen generator input contains missing values")
    if train[target].nunique() != 2:
        raise ValueError("Binary target required")

    model_df, restore = BASE.cast_for_generator(train, schema)
    metadata = BASE.make_sdv_metadata(model_df)
    steps_per_epoch = max(len(model_df) // a.batch_size, 1)
    g_updates = a.epochs * steps_per_epoch
    d_updates = g_updates * a.discriminator_steps

    print(f"config={a.config_name}")
    print(f"n_train={len(train)}, batch={a.batch_size}, steps/epoch={steps_per_epoch}")
    print(f"G lr={a.generator_lr}, D lr={a.discriminator_lr}, D steps={a.discriminator_steps}")
    print(f"expected G updates={g_updates}, D updates={d_updates}")

    if a.dry_run:
        print("DRY RUN PASS")
        return

    from sdv.single_table import CTGANSynthesizer
    synth = CTGANSynthesizer(
        metadata,
        epochs=a.epochs,
        batch_size=a.batch_size,
        generator_lr=a.generator_lr,
        discriminator_lr=a.discriminator_lr,
        discriminator_steps=a.discriminator_steps,
        pac=a.pac,
        verbose=a.verbose,
        enable_gpu=a.cuda,
    )

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    synth.fit(model_df)
    syn_model = synth.sample(num_rows=len(train))
    elapsed = time.time() - t0
    syn = BASE.restore_types(syn_model, train, schema, restore)

    if len(syn) != len(train):
        raise RuntimeError("Wrong synthetic row count")
    if syn[target].nunique() < 2:
        raise RuntimeError("Generated target contains fewer than two classes")
    if syn.isna().any().any():
        raise RuntimeError("Synthetic release contains missing values")

    syn.to_csv(out / "synthetic_1x.csv", index=False)

    try:
        loss = synth.get_loss_values()
    except Exception:
        loss = getattr(getattr(synth, "_model", None), "loss_values", None)

    loss_summary = {}
    if loss is not None:
        loss = pd.DataFrame(loss).copy()
        loss.to_csv(out / "training_loss.csv", index=False)
        gcol = next((c for c in loss.columns if "Generator" in str(c)), None)
        dcol = next((c for c in loss.columns if "Discriminator" in str(c) or "Distriminator" in str(c)), None)
        loss_summary = {
            "generator_loss_first": float(loss.iloc[0][gcol]) if gcol and len(loss) else None,
            "generator_loss_last": float(loss.iloc[-1][gcol]) if gcol and len(loss) else None,
            "discriminator_loss_first": float(loss.iloc[0][dcol]) if dcol and len(loss) else None,
            "discriminator_loss_last": float(loss.iloc[-1][dcol]) if dcol and len(loss) else None,
        }

    meta = {
        "experiment": "ctgan_hyperparameter_sensitivity",
        "config_name": a.config_name,
        "seed": a.seed,
        "n_real_train": len(train),
        "n_synthetic": len(syn),
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "generator_lr": a.generator_lr,
        "discriminator_lr": a.discriminator_lr,
        "discriminator_steps": a.discriminator_steps,
        "pac": a.pac,
        "steps_per_epoch": steps_per_epoch,
        "expected_generator_updates": g_updates,
        "expected_discriminator_updates": d_updates,
        "generation_seconds": elapsed,
        "sdv_version": ver("sdv"),
        "ctgan_version": ver("ctgan"),
        **loss_summary,
    }
    (out / "generation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
