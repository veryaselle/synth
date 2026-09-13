# Independent split-realization robustness checks

This directory contains **supplementary robustness analyses** for the thesis benchmark. They do **not** replace, modify, or pool with the frozen primary benchmark.

## Scientific question

The primary benchmark uses five frozen train/test partitions per dataset. These checks ask:

> Are the main generator rankings and dimension-level conclusions preserved under independently shuffled realizations of the train/test partitions?

The planned intervention in each robustness run is the **partition realization**. Everything else in the benchmark contract is held fixed.

The thesis robustness evidence currently uses two supplementary realization seeds:

- `2026` — initial supplementary split-realization check;
- `3719704` — additional independent split-realization sensitivity check.

The second seed was added as further sensitivity evidence; neither supplementary realization replaces the frozen primary benchmark.

For an arbitrary realization seed `S`:

- PIMA uses five stratified `train_test_split` calls with seeds `S, S+1, ..., S+4`, mirroring the primary PIMA convention of five explicit split seeds;
- Cleveland and CKD use `StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=S)`, mirroring the primary splitter type;
- generator seed schedule remains `42 + split_id`;
- evaluator seed remains `42`;
- release size remains `1x`;
- frozen generator hyperparameters and evaluation code remain unchanged.

The recorded design is stored in `experiment_config.json`.

## Why generator seeds are not changed here

Changing the train/test partitions and generator training seeds in the same intervention would confound two sources of variation. Therefore these checks vary the partition realization while preserving the primary generator-seed schedule.

Retraining stochastic generators can still contribute residual run or numerical variation. These experiments should therefore be interpreted as **split-realization robustness checks**, not as a complete repeated-generator-seed study.

## Environment setup

For the full HPC run, explicitly resolve the core and LLM Python interpreters whenever the benchmark uses separate environments:

```bash
export CORE_PY="$(conda run -n synthetic-medical-core python -c 'import sys; print(sys.executable)')"
export LLM_PY="$(conda run -n synthetic-medical-llm311 python -c 'import sys; print(sys.executable)')"
```

`preflight.sh` validates the required imports before any Slurm arrays are submitted. `submit_all.sh` also resolves empty interpreter variables to a real fallback, preventing an empty `CORE_PY` or `LLM_PY` from being propagated to jobs.

## Run one supplementary realization

From `unified_benchmark_v3`:

```bash
export ROBUSTNESS_SEED=2026
bash robustness_split_realization/submit_all.sh
```

For the additional realization used in the thesis robustness analysis:

```bash
export ROBUSTNESS_SEED=3719704
bash robustness_split_realization/submit_all.sh
```

`submit_all.sh` runs the preflight and then submits:

- 15 REAL-reference tasks;
- 75 SDV/ARF tasks (`3 datasets × 5 methods × 5 splits`);
- 15 Conditional-DDPM tasks;
- 15 GReaT-style LLM tasks;
- one strict finalization/comparison job with an `afterok` dependency on all four groups.

Total expected final inventory for **each** realization:

```text
3 datasets × 8 training sources × 5 splits = 120 split-level rows
```

## Prepared data

Each realization has its own deterministic data directory:

```text
robustness_split_realization/data/seed_<SEED>/
  pima/split_0 ... split_4
  cleveland/split_0 ... split_4
  ckd/split_0 ... split_4
  realization_manifest.json
  split_overlap_with_primary.csv
```

Regenerate one realization explicitly with:

```bash
python robustness_split_realization/prepare_split_realization.py --seed 2026
python robustness_split_realization/prepare_split_realization.py --seed 3719704
```

The preparation script checks expected train/test sizes, absence of processed missing values, train/test index disjointness, and whether a supplementary test partition exactly duplicates any primary frozen test partition.

`prepare_second_realization.py` is retained only as a backward-compatible wrapper for older commands.

## Per-realization finalization and comparison

After a successful HPC run, outputs are stored separately under:

```text
robustness_split_realization/results/seed_<SEED>/tables/
```

The finalizer validates an exact `120`-row split-level inventory before aggregation. The corresponding primary-comparison directory contains:

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

`compare_with_primary.py` compares each supplementary realization with the unrounded primary numeric summary when available. If the unrounded primary summary is unavailable, it falls back to the thesis-reported rounded means and records that source explicitly.

At metric level it reports primary-vs-supplementary means, deltas, ranks, rank changes, and Spearman rank agreement for:

- utility: AUROC, F1, Brier;
- fidelity: PCD, normalized Wasserstein distance, JS divergence;
- empirical privacy-related risk: `|MIA AUROC - 0.5|` and AIA risk.

DCR remains descriptive and is not assigned a universal favourable direction.

At dimension level the script recomputes the thesis-style within-dataset min-max and rank scores for utility, fidelity, and empirical privacy-related risk, then records whether the dimension leader is unchanged.

## Cross-realization summary

Once both supplementary realizations have been finalized, summarize them without pooling their split-level rows:

```bash
python robustness_split_realization/aggregate_realizations.py \
  --seeds 2026 3719704
```

The script validates that both runs contain the expected metric, leader, and AUROC-retention inventories and that they use a consistent primary reference. It writes:

```text
robustness_split_realization/results/across_realizations/
  realization_overview.csv
  rank_agreement_by_dataset_dimension.csv
  rank_agreement_by_dimension.csv
  dimension_leader_cross_realization.csv
  metric_leader_cross_realization.csv
  auroc_retention_all_realizations.csv
  auroc_retention_summary.csv
  cross_realization_summary.json
  SUMMARY.md
```

The cross-realization analysis distinguishes two questions:

1. **Rank-structure stability:** whether the broader ordering of methods remains similar to the frozen primary benchmark, summarized with Spearman agreement.
2. **Winner stability:** whether the exact metric- or dimension-level leader remains the same. A changed leader is informative and should not be treated as an experimental failure.

## Slurm path handling

Slurm executes a copied batch script from its spool directory (for example, `/var/spool/slurmd/job...`). Therefore `.sbatch` files do not derive the repository path from `BASH_SOURCE[0]`.

`submit_all.sh` exports absolute `BENCHMARK_ROOT` and `ROBUST_ROOT` values and submits every job with `--chdir="$BENCHMARK_ROOT"`. Batch scripts use these exported paths and fall back to `SLURM_SUBMIT_DIR` only if needed.

## Interpretation boundary

The frozen primary benchmark remains the confirmatory reference. Supplementary realizations are reported separately and are **not pooled into a 10-split or 15-split primary benchmark**.

Stable rankings strengthen the claim that a finding is not specific to one partition realization. Changed rankings or leaders identify partition-sensitive conclusions. Because generator training seeds are not independently repeated here, generator stochasticity remains a separate limitation.
