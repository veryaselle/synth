# Synthetic Tabular Medical Data Benchmark

Reproducible benchmark accompanying the master's thesis:

**Evaluating the trade-offs between utility, fidelity and privacy in synthetic tabular medical data**

The repository contains the frozen experimental workflow, generator implementations, evaluation code, robustness analyses, GAN configuration-sensitivity analysis, and thesis-reported result snapshots.

## Scope

The benchmark evaluates synthetic tabular medical data across three datasets:

- PIMA Indians Diabetes
- Cleveland Heart Disease
- Chronic Kidney Disease (CKD)

Seven synthetic-data generators are evaluated:

- Gaussian Copula
- Augmented Random Forest (ARF)
- TVAE
- CTGAN
- CopulaGAN
- Conditional DDPM
- GReaT-style LLM generator

`REAL` is included only as a utility reference and is not treated as a synthetic-data generator.

## Evaluation framework

Each synthetic release is evaluated along three dimensions.

### Utility

Synthetic-to-real transfer (TSTR) is evaluated using:

- Logistic Regression
- Multi-Layer Perceptron
- Random Forest
- XGBoost

Reported metrics:

- AUROC
- F1 at threshold 0.5
- Brier score

### Fidelity

Distributional fidelity is assessed using:

- Pairwise Correlation Difference (PCD)
- Normalized Wasserstein distance
- Jensen-Shannon divergence

Lower values indicate better fidelity.

### Privacy-related indicators

The benchmark reports:

- Distance to Closest Record (DCR), descriptive only
- Membership Inference Attack (MIA) AUROC
- Attribute Inference Attack (AIA) risk

For MIA, closeness of AUROC to 0.5 is interpreted as attack indistinguishability in the implemented attack setting. It must not be interpreted as a formal privacy guarantee.

DCR is reported descriptively and is excluded from the composite privacy score.

## Primary benchmark design

The primary benchmark uses five frozen train/test realizations per dataset.

### PIMA

Five independent stratified train/test splits:

- test size: 20%
- random states: 4, 5, 6, 7, 8
- train/test size per split: 614 / 154

### Cleveland

`StratifiedShuffleSplit`:

- 5 splits
- test size: 20%
- random state: 42
- train/test size per split: 242 / 61

### CKD

`StratifiedShuffleSplit`:

- 5 splits
- test size: 20%
- random state: 42
- train/test size per split: 320 / 80

Preprocessing is fitted on the real training partition only.

Each generator produces one fixed 1x synthetic release per split.

Generator seeds follow the schedule `42 + split_id`. Evaluation uses seed 42.

Raw metrics are never pooled across different datasets.

## Supplementary split-realization robustness

Two additional split realizations are evaluated separately from the primary benchmark:

- seed family `2026`
- seed family `3719704`

These analyses test sensitivity to the realization of the train/test split.

Supplementary realizations are not pooled with the primary benchmark.

## GAN configuration-sensitivity analysis

CTGAN and CopulaGAN receive an additional post-hoc configuration-sensitivity analysis.

Evaluated configurations include:

| Configuration | Batch size | Discriminator learning rate |
|---|---:|---:|
| `baseline_b500_d2e4` | 500 | 2e-4 |
| `b200_d2e4` | 200 | 2e-4 |
| `b100_d2e4` | 100 | 2e-4 |
| `b100_d5e4` | 100 | 5e-4 |

The frozen primary GAN configuration participates in the within-GAN comparison.

Trade-off scoring uses three dimensions:

- utility
- fidelity
- privacy-related risk

Pareto efficiency is evaluated within each dataset × GAN combination. Only non-baseline Pareto survivors are subsequently introduced as exploratory candidates into the cross-method analysis.

The cross-method scores are recomputed after candidate selection rather than copied from the within-GAN analysis.

This sensitivity analysis is exploratory and does not replace the frozen primary benchmark.

## Constraint analysis

Cross-method decision analysis evaluates AUROC-retention constraints at:

- 80%
- 85%
- 90%
- 95%
- 98%

The 95% and 98% scenarios are emphasized in the thesis.

These are analytical decision thresholds. They are not clinical, regulatory, or deployment thresholds.

## Repository structure

```text
.
├── README.md
├── scripts/
│   └── maintenance/
│       ├── check_portable_paths.py
│       ├── verify_thesis_repository.py
│       └── verify_clone_ready.py
└── unified_benchmark_v3/
    ├── frozen_data/
    ├── results/
    │   ├── main_benchmark/
    │   ├── gan_sensitivity/
    │   └── final_thesis/
    ├── robustness_split_realization/
    ├── tradeoff_analysis/
    ├── prepare_frozen_datasets.py
    ├── evaluate_release.py
    ├── evaluate_real_reference.py
    ├── generate_sdv_arf_safe_cpu.py
    ├── generate_conditional_ddpm.py
    ├── generate_great_llm_final.py
    └── assemble_main_tables.py
```

## Thesis-reported snapshots

The final thesis values used by the repository verifier are stored under:

```text
unified_benchmark_v3/results/final_thesis/
```

In particular:

```text
reported_primary_means.csv
reported_gan_sensitivity_best_alternatives.csv
```

These files are rounded reporting snapshots derived from the underlying benchmark result tables. The unrounded result tables remain the source for computational analyses.

## Reproducibility checks

From the repository root, run:

```bash
python scripts/maintenance/verify_thesis_repository.py
```

This checks:

- required benchmark files
- frozen split dimensions
- Python syntax
- path portability
- selected thesis-reported numerical values

A successful run ends with:

```text
Repository verification PASS.
Frozen splits: 3 datasets x 5 splits; Python compile PASS; portability PASS.
```

For the stricter clone-readiness test:

```bash
python scripts/maintenance/verify_clone_ready.py
```

This additionally checks:

- Python syntax
- shell and Slurm syntax
- machine-specific path leakage
- current-working-directory dependence
- thesis snapshot consistency

A successful run ends with:

```text
CLONE-READY CHECK PASS
```

The repository was verified successfully with both checks before finalization.

## Python environments

The benchmark contains both conventional tabular models and an LLM-based generator and may therefore require separate Python environments.

Shell and Slurm launchers do not depend on a user-specific absolute Python path. Where applicable, Python executables can be supplied through environment variables such as:

```bash
CORE_PY=/path/to/core/python
LLM_PY=/path/to/llm/python
```

Otherwise the launchers fall back to the Python executable available in the active environment.

## Interpretation notes

The benchmark is intended for comparative methodological evaluation.

In particular:

- high predictive utility does not imply high fidelity;
- high fidelity does not imply privacy;
- MIA performance close to chance is not a formal privacy guarantee;
- DCR is descriptive rather than part of the composite privacy score;
- results are interpreted within datasets before any higher-level comparison;
- supplementary split realizations are robustness analyses, not additional primary benchmark replicates;
- exploratory GAN tuning is reported separately from the frozen primary benchmark.

## Data provenance and redistribution

Dataset acquisition and redistribution rights depend on the original data sources.

Before publishing or redistributing any raw or processed dataset files, the applicable source licenses and terms of use must be checked independently.

The benchmark code is designed so that dataset locations can be supplied without relying on machine-specific absolute paths.

## Scientific provenance

The workflow was informed by prior methodological literature on synthetic tabular data evaluation, including the evaluation framework used as a methodological reference during development.

The thesis benchmark itself is an independently implemented, controlled, and traceable workflow extended to:

- three medical tabular datasets;
- seven generators;
- utility, fidelity, and privacy-related evaluation;
- frozen repeated splits;
- split-realization robustness;
- GAN configuration sensitivity;
- multi-objective and constraint-based decision analysis.
