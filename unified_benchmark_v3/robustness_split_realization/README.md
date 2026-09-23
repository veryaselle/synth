# Independent split-realization robustness check

This directory contains a **supplementary robustness experiment** for the thesis benchmark.
It does **not** replace or modify the frozen primary benchmark.

## Scientific question

The primary benchmark uses five frozen train/test partitions per dataset. This check asks:

> Are the main generator rankings and dimension-level conclusions preserved under a second, independently shuffled realization of the train/test partitions?

The only planned intervention is the **partition realization**.

- Second-realization seed: **2026**.
- PIMA: five stratified `train_test_split` calls with seeds **2026–2030**, mirroring the primary PIMA convention of five explicit split seeds.
- Cleveland / CKD: `StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2026)`, mirroring the primary splitter type.
- Generator seed schedule remains **42 + split_id**.
- Evaluator seed remains **42**.
- Release size remains **1x**.
- Generator hyperparameters and evaluation code remain the same as in the frozen primary benchmark.

The predeclared design is also stored in `experiment_config.json`.

## Why this is separate from generator stochasticity

Changing train/test partitions and changing generator seeds in one intervention would make the source of any change ambiguous. Therefore this experiment changes the partition realization while preserving the primary generator-seed schedule. Retraining stochastic models can still contribute residual numerical/run variation, so this check should be interpreted as **split-realization robustness**, not as a complete multi-seed generator-stochasticity study.

## One-command HPC run

From `unified_benchmark_v3`:

```bash
bash robustness_split_realization/submit_all.sh
```

Optional explicit interpreters:

```bash
export CORE_PY=/path/to/core/env/bin/python
export LLM_PY=/path/to/llm/env/bin/python
bash robustness_split_realization/submit_all.sh
```

`submit_all.sh` first runs a preflight and prepares the second realization, then submits:

- 15 REAL-reference tasks;
- 75 SDV/ARF tasks (3 datasets × 5 methods × 5 splits);
- 15 Conditional-DDPM tasks;
- 15 GReaT-style LLM tasks;
- one strict finalization/comparison job.

Total expected final inventory:

```text
3 datasets × 8 training sources × 5 splits = 120 split-level rows
```

## Prepared data

```text
robustness_split_realization/data/seed_2026/
  pima/split_0 ... split_4
  cleveland/split_0 ... split_4
  ckd/split_0 ... split_4
  realization_manifest.json
  split_overlap_with_primary.csv
```

The preparation script checks train/test sizes, absence of processed missing values, train/test index disjointness and verifies that no second-realization test partition exactly duplicates a primary frozen test partition.

Regenerate deterministically with:

```bash
python robustness_split_realization/prepare_second_realization.py --seed 2026
```

## Results and comparison

After a successful HPC run:

```text
robustness_split_realization/results/seed_2026/tables/
```

contains the same mean/SD tables as the primary benchmark plus:

```text
primary_comparison/
  metric_method_comparison.csv
  metric_rank_agreement.csv
  metric_leader_stability.csv
  dimension_leader_stability.csv
  robust_auroc_retention.csv
  comparison_summary.json
  SUMMARY.md
```

`compare_with_primary.py` compares the second realization with the unrounded primary numeric summary if it exists locally. Otherwise it falls back to the thesis-reported rounded means in `results/final_thesis/reported_primary_means.csv`; in that fallback mode, exact rank ties can reflect three-decimal reporting precision.

### What is compared

At metric level, the script reports primary vs second-realization means, deltas, ranks and Spearman rank agreement for:

- utility: AUROC, F1, Brier;
- fidelity: PCD, WS, JS;
- empirical privacy-related risk: `|MIA AUROC - 0.5|` and AIA risk.

DCR remains descriptive and is not given a universal favourable direction.

At dimension level, it also recomputes the thesis-style within-dataset **min-max** and **rank** scores for utility, fidelity and empirical privacy-related risk and reports whether the leading candidate is unchanged.

## Interpretation

A stable ordering strengthens the claim that a finding is not specific to the original partition realization. A changed ordering is also scientifically useful: it identifies conclusions that are sensitive to how a small medical dataset is partitioned.

The primary frozen benchmark remains the thesis reference. The second realization is supplementary robustness evidence and should be reported separately rather than pooled into a new ten-split primary benchmark.
