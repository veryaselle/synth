# Unified benchmark protocol — frozen draft v3

## Main scope

Datasets: **PIMA**, **Cleveland Heart Disease**, **Chronic Kidney Disease (CKD)**.

Methods:
1. TVAE
2. CTGAN
3. CopulaGAN
4. Adversarial Random Forest (ARF)
5. Gaussian Copula
6. Conditional DDPM
7. GReaT-style LLM

The completed Diabetes 130-US Hospitals LLM study is **retained** as an exploratory/scalability result for the Discussion. It is not part of the primary 3-dataset comparison.

## Primary release size

Every method generates **1×** the number of rows in the frozen real training split. The earlier `1× / 2× / 3×` study remains a later sensitivity analysis only.

## Frozen real data

Five frozen stratified train/test splits are used per dataset. The split is made before fitted preprocessing. All fitted imputation parameters are learned from the real training split only.

- **PIMA:** five legacy-compatible split seeds 4–8; zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are treated as missing; train-fitted IterativeImputer (MICE-like) is applied to train/test.
- **Cleveland:** five splits from StratifiedShuffleSplit(n_splits=5, seed 42); no imputation after cleaning.
- **CKD:** five splits from StratifiedShuffleSplit(n_splits=5, seed 42); train medians for numeric variables and train modes for categorical/ordinal variables.

Within each split, the processed `real_train.csv` is the common input to every generator and the processed `real_test.csv` is never shown to a generator. Final tables report mean ± standard deviation across the five frozen splits.

## Utility

**TSTR:** train on synthetic → test on untouched real test.

Metrics:
- AUROC ↑
- F1 at threshold 0.5 ↑
- Brier score ↓

Classifiers: Logistic Regression, MLP, Random Forest, XGBoost. The main table stores the mean across classifiers; classifier-level values are retained.

**Important:** classifier preprocessing is fitted on the synthetic training release, not on real train. Thus TSTR does not borrow real-training distribution statistics.

## Fidelity

Reference: frozen real train vs 1× synthetic release.

- PCD ↓: mean absolute difference between pairwise correlations in a shared real-train-fitted encoded space. The binary target is included, so feature-target relationships contribute.
- WS ↓: mean per-numeric-feature Wasserstein distance normalized by real-train standard deviation.
- JS ↓: mean feature-wise Jensen-Shannon divergence; numerical variables use real-train quantile bins, categorical variables use observed levels, and the target distribution is included.

Absolute fidelity values are interpreted within a dataset, not across datasets with different feature spaces.

## Privacy-related evaluation

These are empirical privacy audits, **not formal privacy guarantees**.

### DCR
Distance from each synthetic record to its closest real training record in a common real-train-fitted distance space. The detailed audit also reports the natural real-to-real nearest-neighbour reference and DCR/reference ratio.

### MIA AUROC
Members = real generator-training rows. Controls/non-members = held-out real test rows. The attack score is negative distance to the nearest synthetic record with the same target class. AUROC ≈ 0.5 indicates no detectable membership signal under this attack.

### AIA risk
Attribute inference hides the diagnosis/target and uses the remaining known attributes to locate the nearest synthetic record (`k=1` primary setting) and infer the hidden target. Because diagnosis may be predictable from population-level correlations, attack success is computed both for real training members and held-out controls. The primary AIA metric is the normalized excess success

`(member success - control success) / (1 - control success)`, clipped at zero for the main risk table.

This follows the train-vs-control logic used in attack-based synthetic-data privacy evaluation: population-level predictability should raise both member and control attack success, whereas train-specific leakage should preferentially improve member inference.

## Main table per dataset

| Method | AUROC ↑ | F1 ↑ | Brier ↓ | PCD ↓ | WS ↓ | JS ↓ | DCR | MIA AUROC | AIA risk ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REAL (utility reference only) | | | | – | – | – | – | – | – |
| TVAE | | | | | | | | | |
| CTGAN | | | | | | | | | |
| CopulaGAN | | | | | | | | | |
| ARF | | | | | | | | | |
| Gaussian Copula | | | | | | | | | |
| Conditional DDPM | | | | | | | | | |
| LLM | | | | | | | | | |

## Execution order

1. Freeze the three datasets and schemas — **done in this package**.
2. Validate real utility references and the evaluator — **done by smoke tests**.
3. Complete PIMA across all seven methods on all five frozen splits.
4. Reuse the same generator/release/evaluator contract for Cleveland.
5. Repeat for CKD.
6. Only after the 21 primary synthetic cells are complete, consider sample-size sensitivity or LLM fine-tuning extensions.

## Standard release contract

```text
results/main_benchmark/<dataset>/split_<0..4>/<method>/
├── synthetic_1x.csv
├── generation_metadata.json
└── evaluation/
    ├── main_results_row.csv
    ├── utility_by_classifier.csv
    ├── fidelity_ws_by_feature.csv
    ├── fidelity_js_by_feature.csv
    └── metrics.json
```
