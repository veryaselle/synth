# Portability audit — thesis-v1.0 clone-ready candidate

Audit date: 2026-09-03.

## Scope

This refactor is intentionally conservative. It changes path resolution, CLI default locations, launch wrappers/documentation and verification only. Frozen split definitions, seeds, generator hyperparameters, metric definitions, frozen data and reported thesis values were not changed.

Top-level scripts under `unified_benchmark_v3/` now resolve the repository root with `Path(__file__).resolve().parents[1]`. Scripts nested under `scripts/<area>/` use `parents[2]`, and deeper analysis scripts use the corresponding structural depth. This avoids both personal absolute paths and dependence on the shell's current working directory.

## Passed checks

- machine-specific absolute-path scan over executable `.py`, `.sh` and `.sbatch` files: pass;
- cwd-sensitive Python default scan (`data/...`, `results/...`, `Path('.')`): pass;
- Python `compileall` over `scripts/` and `unified_benchmark_v3/`: pass;
- `bash -n` over every `.sh` and `.sbatch`: pass;
- thesis repository structural/numerical snapshot verification: pass;
- frozen preprocessing regeneration launched from outside the repository: pass; all preserved frozen artifacts match after ignoring path-only provenance fields;
- repository copied to a directory containing spaces: clone-ready verification and frozen-data regeneration pass;
- PIMA, Cleveland and CKD preflight dry-runs launched from outside the repository: pass;
- evaluator memorization control launched from outside the repository: pass, with DCR = 0, MIA AUROC = 1 and AIA risk = 1 as expected for exact training-row reuse;
- checksum comparison of the tracked data/results/frozen artifacts before versus after the path refactor: identical;
- local Git init → commit → fresh clone simulation: all 362 package files staged, clone-ready verification passed, and all tracked file bytes matched the source package. CSV line-ending normalization is disabled via `.gitattributes` so artifact bytes are preserved.

## Deliberate caveat: PIMA LLM

The final PIMA LLM release reused a preserved 10-epoch checkpoint after changing only the release-validity rule. The exact final job therefore refuses to silently retrain when `LLM_unconstrained_epoch10/saved_model` is absent. The checkpoint itself is not included because it is a large model artifact. For a clone-only PIMA rerun, `submit_pima_all_fresh.sh` retrains the LLM with the same declared configuration; this is a fresh stochastic rerun, not a byte-identical recreation of the preserved checkpoint. Cleveland and CKD LLM jobs train from scratch.

## Slurm portability

Dataset submit wrappers derive the benchmark directory from `BASH_SOURCE` and export `BENCHMARK_ROOT` to jobs. Interpreters are supplied optionally with `CORE_PY` and `LLM_PY`; otherwise the active `python` is used. No user-specific interpreter path is stored in the repository.
