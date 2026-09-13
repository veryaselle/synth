# IMPORTANT RUN LOCATION

Run the analysis **from the project root**, the directory containing `results/`.

Correct:

```bash
bash tradeoff_analysis_bundle_v3/run_tradeoff_analysis.sh
```

Do **not** `cd tradeoff_analysis_bundle_v3` first.

For Slurm, submit from the project root:

```bash
mkdir -p logs
sbatch tradeoff_analysis_bundle_v3/tradeoff_analysis.sbatch
```

The previous wrapper changed directory into the bundle and therefore searched for
`tradeoff_analysis_bundle/results/...`, which caused every GAN result to appear missing.

---

# Utility–Fidelity–Privacy Trade-off Analysis

This is the analysis stage after the frozen benchmark and GAN sensitivity runs.
It does **not** retrain any model.

## Scientific separation

The script keeps two analyses separate.

### 1. Frozen primary benchmark

Uses only the original `results/main_benchmark` results. Tuned GAN runs never
replace the frozen primary benchmark post hoc.

### 2. Exploratory GAN sensitivity

Compares CTGAN and CopulaGAN configurations:

- frozen baseline: batch 500, discriminator LR 2e-4
- batch 100, discriminator LR 2e-4
- batch 200, discriminator LR 2e-4
- discriminator LR 1e-4 (actual batch is read from `generation_metadata.json`)
- batch 100, discriminator LR 5e-4

An additional **exploratory overlay** places tuned GAN points next to the frozen
methods. It is clearly labelled exploratory and must not be described as a new
primary benchmark.

## Trade-off dimensions

### Utility
- AUROC: higher is better
- F1: higher is better
- Brier: lower is better

### Fidelity
- PCD: lower is better
- normalized Wasserstein: lower is better
- Jensen–Shannon: lower is better

### Empirical privacy
- `abs(MIA AUROC - 0.5)`: lower is better
- AIA risk: lower is better

DCR remains in the output as a descriptive metric but is **excluded from the
privacy composite**, because raw DCR is scale-dependent and does not have a
simple universal direction across datasets.

All rank scores are calculated **within a dataset**. Raw fidelity values are not
ranked across different datasets.

## Paired changes vs GAN baseline

Because every configuration uses the same five frozen splits, the script aligns
each sensitivity result with the corresponding baseline split and calculates:

- AUROC improvement
- F1 improvement
- Brier improvement
- PCD improvement
- WS improvement
- JS improvement
- improvement in MIA closeness to 0.5
- AIA-risk improvement
- raw DCR change (descriptive only)

For the first eight quantities, positive means improvement. The summary also
reports how many of the five splits improved or worsened.

No significance claim is made from five splits.

## Dimension scores and Pareto front

Within each comparison set, metrics are ranked in their correct direction. The
three dimension ranks are converted to descriptive `[0,1]` scores:

- `utility_score`
- `fidelity_score`
- `privacy_score`

These are **descriptive rank scores**, not a validated universal synthetic-data
quality score.

A candidate is Pareto-efficient when no other candidate is at least as good in
all three dimensions and strictly better in at least one.

The script reports four useful strategy labels:

- `utility_first`
- `fidelity_first`
- `privacy_first`
- `balanced_pareto`

`balanced_pareto` uses a maximin rule: among Pareto-efficient candidates it
prefers the one whose weakest of the three dimension scores is strongest.

## Expected paths

The supplied manifest assumes:

```text
results/main_benchmark/
results/gan_sensitivity/
results/gan_sensitivity/b200/
results/gan_sensitivity/lr5e4_b100/
```

If your lower-LR experiment has a different folder name, edit only the `root`
field for `bestbatch_d1e4` in `gan_tradeoff_manifest.csv`.

The script intentionally addresses GAN result files by exact paths such as:

```text
<root>/<dataset>/split_<0..4>/CTGAN/evaluation/main_results_row.csv
<root>/<dataset>/split_<0..4>/CopulaGAN/evaluation/main_results_row.csv
```

so nested sensitivity experiments cannot accidentally contaminate another
configuration.

## Run

From the project root:

```bash
bash tradeoff_analysis_bundle/run_tradeoff_analysis.sh
```

or via Slurm:

```bash
mkdir -p logs
sbatch tradeoff_analysis_bundle/tradeoff_analysis.sbatch
```

If you copy the files directly into the project root, the same commands work
without the `tradeoff_analysis_bundle/` prefix.

## Outputs

```text
results/tradeoff_analysis/
├── gan_tradeoff_split_level.csv
├── gan_tradeoff_config_summary.csv
├── gan_paired_deltas_split_level.csv
├── gan_paired_deltas_summary.csv
├── gan_tradeoff_dimension_scores.csv
├── gan_pareto_front.csv
├── gan_strategy_recommendations.csv
├── frozen_primary_split_level.csv
├── frozen_global_tradeoff_scores.csv
├── frozen_global_pareto_front.csv
├── frozen_global_strategy_recommendations.csv
├── exploratory_overlay_tradeoff_scores.csv
├── exploratory_overlay_pareto_front.csv
└── figures/
```

The figures contain, for each dataset:

- Utility vs Fidelity
- Utility vs Privacy
- Fidelity vs Privacy

for the GAN sensitivity analysis and for the frozen global benchmark.
