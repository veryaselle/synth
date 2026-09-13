# Dataset handling

This package includes the raw input files so that a supervisor can reproduce the experiments in a private academic repository without reconstructing data locations manually.

Before public release, verify the current redistribution terms for every dataset. If redistribution is not permitted, remove raw tables and replace them with acquisition instructions plus checksums.

## PIMA Indians Diabetes

Path: `data/raw/pima/pima.csv`

The adapted experiments use target column `Outcome`. Clean Exp3 additionally requires correlation-stage synthetic outputs produced by the adapted baseline implementation.

## Cleveland Heart Disease

Path: `data/raw/cleveland/Heart_disease_cleveland_new.csv`

Expected shape: 303 rows and 14 columns including target.

## Chronic Kidney Disease

Raw path: `data/raw/ckd/kidney_disease.csv`

Sanitized path: `data/processed/ckd/kidney_disease_sanitized.csv`

Sanitation strips whitespace/tab artifacts, converts `?` to missing, normalizes the target, removes the ID and treats three isolated electrolyte values as missing. No global imputation is performed.

## Diabetes 130-US Hospitals

Raw paths:

- `data/raw/diabetes130/diabetic_data.csv`
- `data/raw/diabetes130/IDS_mapping.csv`

The raw table contains 101,766 encounters and 71,518 unique patients. The target is early readmission (`<30`) versus `NO` or `>30`. Death/hospice discharges are excluded. `patient_nbr` is used only for grouping and removed from features; `encounter_id` is removed.

Tracked cohort facts:

- eligible encounters: 99,343;
- train: 79,473;
- validation: 9,935;
- test: 9,935;
- GReaT source subset: 20,000;
- positive prevalence in the 20K source: 11.39%.

Do not recreate the patient split with another seed after downstream results exist.
