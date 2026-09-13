# Reproducibility status

## Structurally verified

- all tracked Python scripts compile;
- required raw data files are present;
- frozen GReaT release artifacts are present;
- the final GReaT audit reports a complete schema pass;
- result snapshots are present for all experiment groups;
- GitHub CI verifies syntax and required artifacts.

## Completed runs represented by tracked outputs

- Cleveland real baseline, conditional DDPM and simple baselines;
- CKD real baseline, conditional DDPM and simple baselines;
- cross-dataset proximity-based membership-inference audit;
- Diabetes 130-US smoke test, benchmark, main generation, finalization, utility/fidelity and privacy audit.

## Not fully self-contained

PIMA clean Exp3 requires correlation-stage synthetic files from the adapted baseline repository. Final PIMA summaries are tracked, but those intermediate generator outputs are not included.

## Deliberately excluded

- Hugging Face caches;
- model checkpoints and trainer state;
- temporary generation chunks;
- prediction folders;
- large regenerable intermediate files.

## Required before final thesis tag

Export and commit exact Conda lock files from the completed local environments. The readable environment files record the validated top-level versions but not every transitive build.
