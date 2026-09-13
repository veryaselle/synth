# Evaluating trade-offs in synthetic tabular medical data

This repository contains the executable code and frozen benchmark artifacts for the master's thesis **“Evaluating the trade-offs between utility, fidelity and privacy in synthetic tabular medical data.”**

The primary benchmark compares seven synthetic generators spanning six modelling families on PIMA, Cleveland Heart Disease and Chronic Kidney Disease. Each method is evaluated on five frozen train/test splits with one 1× synthetic release per split. Utility is assessed with TSTR (AUROC, F1@0.5, Brier), fidelity with PCD, normalized Wasserstein distance and Jensen–Shannon divergence, and privacy-related behaviour with empirical DCR, MIA and AIA audits.

## Clone and verify

No checkout path needs to be edited into the source code.

```bash
git clone https://github.com/veryaselle/synth.git
cd synth
python scripts/maintenance/verify_clone_ready.py
```

For the top-level benchmark scripts, the repository root is resolved as `Path(__file__).resolve().parents[1]`; deeper historical scripts use the structurally correct parent depth. See `docs/PORTABILITY.md` for the exact policy and `TESTING.md` for the tested clean-clone sequence.

## Primary benchmark

The final benchmark is under `unified_benchmark_v3/`. The final reported numerical snapshots are under `unified_benchmark_v3/results/final_thesis/`.

The frozen primary comparison remains separate from the post-hoc GAN configuration-sensitivity analysis. Tuned GAN results are exploratory and do not overwrite the frozen benchmark.

## Methodological reference

The framework described by Gamisch et al. served as a methodological reference for relating statistical fidelity to downstream predictive utility. The primary experiments reported for this thesis are produced by the independently implemented three-dataset benchmark under `unified_benchmark_v3/`. Historical development artifacts are retained only for provenance and do not define the primary experimental pipeline.

## Supplementary split-realization robustness

The frozen primary benchmark remains the confirmatory reference. Supplementary robustness runs under `unified_benchmark_v3/robustness_split_realization/` repeat the same benchmark contract under independently shuffled train/test partition realizations while retaining the primary generator-seed schedule. The thesis robustness evidence uses realization seeds `2026` and `3719704`; their split-level observations are not pooled into a larger primary benchmark.

After both realizations have been finalized, their rank- and leader-level stability can be summarized with:

```bash
cd unified_benchmark_v3
python robustness_split_realization/aggregate_realizations.py --seeds 2026 3719704
```

## Environments

Create the repository-supplied environments from the clone:

```bash
conda env create -f environment/environment-core.yml
conda env create -f environment/environment-llm.yml
```

The core benchmark and the GReaT-style LLM branch were validated under different Python environments. Slurm wrappers accept optional `CORE_PY` and `LLM_PY` interpreter overrides; otherwise they use the active `python`. Interpreter locations can be obtained without writing machine-specific paths into the repository:

```bash
export CORE_PY="$(conda run -n synthetic-medical-core python -c 'import sys; print(sys.executable)')"
export LLM_PY="$(conda run -n synthetic-medical-llm311 python -c 'import sys; print(sys.executable)')"
```

The LLM environment requires a PyTorch build appropriate for the target CUDA runtime; see `environment/README.md`.

## PIMA LLM exact versus fresh rerun

`submit_pima_all.sh` preserves the exact final-release logic and requires the archived 10-epoch split-specific checkpoint. The large checkpoint is not distributed here. `submit_pima_all_fresh.sh` is the clone-only rerun path: it trains the PIMA LLM from scratch with the same declared configuration, so the stochastic realization is not byte-identical to the checkpoint used for the thesis release.
