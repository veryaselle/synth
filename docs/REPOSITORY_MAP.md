# Repository map

## `data/`

### `data/raw/pima/pima.csv`
Raw PIMA table used by the adapted baseline scripts.

### `data/raw/cleveland/Heart_disease_cleveland_new.csv`
Clean Cleveland table with 303 records, 13 predictors and a binary target.

### `data/raw/ckd/kidney_disease.csv`
Original CKD input. The deterministic sanitation script produces the tracked processed copy.

### `data/raw/diabetes130/`
The 101,766-encounter Diabetes 130-US table and ID mapping file.

### `data/processed/`
Contains the tracked CKD sanitized table. Diabetes 130-US split files are generated here and must not be hand-edited.

## `scripts/pima/`

- `clean_exp3_from_correlation.py`: reconstructs clean downstream utility evaluation from correlation-stage synthetic files.
- `evaluate_pima_privacy.py`: DCR, nearest-neighbour and duplicate audit for PIMA releases.

## `scripts/cleveland/`

- `run_cleveland_baseline.py`: five-split real-data baseline.
- `run_cleveland_diffusion.py`: conditional diffusion generator and evaluation.
- `run_cleveland_simple_baselines.py`: Gaussian–empirical and bootstrap baselines.
- `check_cleveland_diffusion_outputs.py`: synthetic-release validation.
- `summarize_cleveland_results.py`: thesis-ready summaries.
- `make_cleveland_final_comparison.py`: cross-generator final comparison.

## `scripts/ckd/`

- `prepare_ckd_dataset.py`: deterministic cleaning and sanitation.
- `run_ckd_baseline.py`: real-data baseline.
- `run_ckd_synthetic_experiment.py`: real, Gaussian, bootstrap and conditional DDPM experiment.
- `repair_ckd_ddpm_categories_v2.py`: deterministic repair of the unused missing-category decoder placeholder.

## `scripts/privacy/`

- `run_membership_inference_privacy.py`: common class-conditional proximity attack across PIMA, Cleveland and CKD.

## `scripts/diabetes130/`

- `audit_diabetes130.py`: raw cohort audit.
- `prepare_diabetes130_llm_pilot.py`: exclusions, patient-grouped split and reduced representation.
- `audit_diabetes130_tokenizer.py`: serialization/token-length audit.
- `smoke_test_diabetes130_great_v2.py`: 2K training and 100-row generation validation.
- `benchmark_diabetes130_great_generation_v2.py`: sampling benchmark with partial-failure detection.
- `run_diabetes130_great_main_pilot.py`: resumable 20K GReaT training and generation.
- `validate_diabetes130_smoke_against_full_train.py`: full-schema smoke validation.
- `constrain_diabetes130_great_schema_v4.py`: diagnostic schema audit.
- `finalize_diabetes130_great_release.py`: deterministic final release construction.
- `run_diabetes130_utility_fidelity_v2.py`: matched-size utility, fidelity and simple baselines.
- `run_diabetes130_privacy.py`: DCR and membership-inference audit.

## `results/`

Contains validated summaries and final artifacts, not every temporary checkpoint or generation chunk.

## `environment/`

Readable environment definitions and instructions for capturing exact lock files from the completed local environments.

## `scripts/maintenance/`

- `verify_repository.py`: syntax, structure and frozen-release validation.
- `create_checksums.py`: SHA-256 manifest generation.
