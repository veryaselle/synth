# Changelog

## post-thesis-v1.0 — split-realization robustness update

- added seed-parameterized supplementary split-realization robustness support and documented realized seeds `2026` and `3719704`;
- added `aggregate_realizations.py` for cross-realization rank, leader and AUROC-retention summaries without pooling supplementary splits into the primary benchmark;
- generalized preparation/finalization entry points while keeping backward-compatible wrappers for the original `*_second_realization.py` commands;
- fixed empty `CORE_PY`/`LLM_PY` handling and strengthened interpreter/dependency checks before Slurm submission;
- retained explicit Slurm repository-root export/chdir handling so batch scripts do not resolve paths from the Slurm spool copy;
- reframed repository documentation to describe Gamisch et al. as a methodological reference rather than as the primary implementation pipeline.

## thesis-v1.0 — clone-ready thesis release

- made repository/benchmark path resolution independent of checkout location and current working directory;
- standardized top-level benchmark scripts on `Path(__file__).resolve().parents[1]` and nested scripts on the structurally correct parent depth;
- removed machine-specific interpreter/repository paths from executable code and documentation examples;
- added `verify_clone_ready.py` covering Python syntax, shell/Slurm syntax, path policy and thesis snapshot checks;
- revalidated frozen preprocessing, all three dataset preflights and the evaluator memorization control from outside the repository;
- verified that tracked experimental data/results/frozen artifacts are byte-identical before and after the path refactor;
- preserved the distinction between the frozen primary benchmark and exploratory GAN tuning;
- documented the exact-PIMA-checkpoint versus fresh-clone rerun distinction.

## v0.1.0 — thesis repository freeze

- organized validated scripts by dataset;
- included private-reproduction raw inputs;
- included PIMA, Cleveland, CKD and privacy summaries;
- froze raw and final Diabetes 130-US GReaT releases with audit logs;
- included corrected resumable utility/fidelity evaluation;
- added environment definitions, supervisor protocol, known issues and CI;
- excluded checkpoints and temporary chunks.
