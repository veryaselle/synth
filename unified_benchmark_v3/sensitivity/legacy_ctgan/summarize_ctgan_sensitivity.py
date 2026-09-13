#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]

CONFIGS = [
    "A_baseline_b500_d2e4_s1",
    "B_b100_d2e4_s1",
    "C_b100_d1e4_s1",
    "D_b100_d2e4_s2",
    "E_b100_d2e5_s1",
]
METRICS = [
    "utility_auroc","utility_f1","utility_brier",
    "fidelity_pcd","fidelity_ws","fidelity_js",
    "privacy_dcr_mean","privacy_mia_auc","privacy_aia_risk",
]

def split_id(path):
    for part in path.parts:
        if part.startswith("split_"):
            return int(part.split("_",1)[1])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(BENCHMARK_ROOT / "results/gan_sensitivity/ctgan"))
    p.add_argument("--outdir", default=str(BENCHMARK_ROOT / "results/gan_sensitivity/ctgan/summary"))
    p.add_argument("--datasets", nargs="+", default=["cleveland"])
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    root, out = Path(a.root), Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    for ds in a.datasets:
        for cfg in CONFIGS:
            found = [f for f in (root/ds).rglob("main_results_row.csv") if cfg in f.parts]
            splits = sorted(split_id(f) for f in found)
            print(f"{ds:12s} {cfg:28s}: n={len(found)} splits={splits}")
            if a.strict and (len(found) != 5 or set(splits) != set(range(5))):
                errors.append(f"{ds}/{cfg}: expected 5 splits 0..4; found {splits}")
            for f in found:
                d = pd.read_csv(f)
                if len(d) != 1:
                    raise SystemExit(f"{f}: expected exactly one row")
                r = d.iloc[0].to_dict()
                r["dataset"] = ds
                r["config"] = cfg
                r["split_id"] = split_id(f)
                meta_path = f.parent.parent / "generation_metadata.json"
                if meta_path.exists():
                    m = json.loads(meta_path.read_text())
                    for k in [
                        "batch_size","generator_lr","discriminator_lr","discriminator_steps",
                        "pac","steps_per_epoch","expected_generator_updates",
                        "expected_discriminator_updates","generator_loss_first","generator_loss_last",
                        "discriminator_loss_first","discriminator_loss_last","generation_seconds",
                    ]:
                        r[k] = m.get(k)
                rows.append(r)

    if errors:
        print("\nSTRICT FINALIZATION REFUSED:")
        for e in errors: print(" -", e)
        raise SystemExit(2)
    if not rows:
        raise SystemExit("No sensitivity results found")

    df = pd.DataFrame(rows)
    df.to_csv(out/"ctgan_sensitivity_split_level.csv", index=False)
    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = {}
    for c in METRICS:
        agg[c+"_mean"] = (c,"mean")
        agg[c+"_sd"] = (c,"std")
    for c in ["generator_loss_first","generator_loss_last","discriminator_loss_first",
              "discriminator_loss_last","generation_seconds"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg[c+"_mean"] = (c,"mean")
            agg[c+"_sd"] = (c,"std")

    s = df.groupby(["dataset","config"], as_index=False).agg(**agg)
    hp_cols = ["batch_size","generator_lr","discriminator_lr","discriminator_steps",
               "pac","steps_per_epoch","expected_generator_updates","expected_discriminator_updates"]
    hp = df.groupby(["dataset","config"],as_index=False)[hp_cols].first()
    s = s.merge(hp,on=["dataset","config"],how="left")

    s["rank_auroc"] = s.groupby("dataset")["utility_auroc_mean"].rank(ascending=False,method="min")
    s["rank_f1"] = s.groupby("dataset")["utility_f1_mean"].rank(ascending=False,method="min")
    s["rank_brier"] = s.groupby("dataset")["utility_brier_mean"].rank(ascending=True,method="min")
    s["utility_mean_rank"] = s[["rank_auroc","rank_f1","rank_brier"]].mean(axis=1)
    s = s.sort_values(["dataset","utility_mean_rank","config"])
    s.to_csv(out/"ctgan_sensitivity_summary_numeric.csv", index=False)

    cols = ["dataset","config","utility_auroc_mean","utility_auroc_sd",
            "utility_f1_mean","utility_brier_mean","fidelity_pcd_mean","fidelity_ws_mean",
            "fidelity_js_mean","privacy_mia_auc_mean","privacy_aia_risk_mean",
            "utility_mean_rank","batch_size","discriminator_lr","discriminator_steps",
            "expected_generator_updates","expected_discriminator_updates",
            "generator_loss_last_mean","discriminator_loss_last_mean"]
    cols = [c for c in cols if c in s.columns]
    print("\nCTGAN SENSITIVITY SUMMARY")
    print(s[cols].round(4).to_string(index=False))
    print(f"\nSaved: {out/'ctgan_sensitivity_summary_numeric.csv'}")

if __name__ == "__main__":
    main()
