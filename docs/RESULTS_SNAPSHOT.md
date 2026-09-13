# Verified results snapshot

This document records principal completed outputs already present in the repository. It is not a substitute for raw run-level CSVs.

## PIMA

Real-data mean AUROC was approximately 0.827 across four classifiers. TVAE at 3× reached approximately 0.827 mean AUROC, CTGAN remained preliminary because source artifacts were incomplete, and Gaussian Copula reached approximately 0.764.

## Cleveland

Real mean AUROC was approximately 0.908. Conditional DDPM at 3× reached approximately 0.907, while the Gaussian–empirical baseline reached approximately 0.921.

## CKD

The task showed a ceiling effect. Real mean AUROC was approximately 0.998. Conditional DDPM and Gaussian–empirical releases remained close to the real baseline. Bootstrap achieved strong utility but exact-copy rate 1 and high internal duplication.

## Cross-dataset membership inference

Non-bootstrap generators were generally near chance. Bootstrap controls produced attack AUROC around 0.97 on PIMA, Cleveland and CKD, confirming that the attack detects direct memorization.

## Diabetes 130-US

### Frozen GReaT release

- raw rows: 20,000;
- final valid rows: 19,999;
- target prevalence: 0.113906;
- exact duplicate rate: 0;
- internal duplicate rate: 0;
- final schema pass: true.

### Mean utility across four classifiers

| Source | Mean AUROC | Mean AUPRC | Mean Brier |
|---|---:|---:|---:|
| Real full reference | 0.6760 | 0.2404 | 0.0957 |
| Real 20K | 0.6534 | 0.2181 | 0.0973 |
| Gaussian–empirical 20K | 0.6413 | 0.2038 | 0.1081 |
| Empirical bootstrap 20K | 0.6216 | 0.1926 | 0.1051 |
| GReaT 20K | 0.5850 | 0.1626 | 0.1032 |

Classifier-level values are in `results/diabetes130/evaluation_snapshot/utility_summary_console_snapshot.csv`.

### Fidelity

GReaT had normalized mean numerical Wasserstein distance 0.0323, overall JS divergence 0.0163 and PCD 0.00418. Gaussian releases had better overall JS, missingness preservation and categorical support. GReaT retained approximately 82% of categorical support and 68% of rare-category support.

### Privacy-related audit

GReaT:

- pooled proximity-MIA AUROC: 0.4991;
- 95% bootstrap interval: 0.4889–0.5084;
- DCR / real-NN mean ratio: 0.8690;
- fraction below real-NN fifth percentile: 0.1452;
- exact duplicates: 0;
- internal duplicates: 0.

Gaussian releases were also near chance under MIA and farther from real training records. Bootstrap controls had DCR 0, exact-copy rate 1 and MIA AUROC around 0.81.

## Interpretation boundary

The GReaT result is an exploratory one-epoch configuration using `tabularisai/Qwen3-0.3B-distil`. It does not establish that LLM tabular synthesis fails generally. It establishes that this controlled configuration did not outperform the simpler Gaussian baseline and required explicit schema validation.
