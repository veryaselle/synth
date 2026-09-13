# Supervisor reproduction protocol

Run every command from the repository root.

## 1. Clone and verify

```bash
git clone <PRIVATE_REPOSITORY_URL>
cd synthetic-medical-tabular-data-evaluation
python scripts/maintenance/verify_repository.py
```

## 2. Create environments

```bash
conda env create -f environment/environment-core.yml
conda env create -f environment/environment-llm.yml
```

Install a PyTorch build compatible with the local CUDA driver before running GReaT. The validated LLM run used PyTorch 2.11.0+cu130 on an RTX 2080 Ti.

---

# PIMA

The final PIMA tables are tracked. Full regeneration additionally requires correlation-stage synthetic files from the adapted baseline implementation under `results/pima/correlation/`.

```bash
conda activate synthetic-medical-core

python scripts/pima/clean_exp3_from_correlation.py \
  --data_path data/raw/pima/pima.csv \
  --correlation_dir results/pima/correlation \
  --out_path results/pima/clean_exp3/clf_performance.csv \
  --splits 0 1 2 \
  --generators TVAE CTGAN COPULA \
  --classifiers lr mlp xgb rfc \
  --size_multipliers 1 2 3 \
  --n_files_per_model 20 \
  --include_real_baseline

python scripts/pima/evaluate_pima_privacy.py \
  --data_path data/raw/pima/pima.csv \
  --correlation_dir results/pima/correlation \
  --out_dir results/pima/privacy \
  --splits 0 1 2 \
  --n_files_per_model 20 \
  --size_multipliers 1 2 3
```

---

# Cleveland Heart Disease

```bash
conda activate synthetic-medical-core
```

## Real baseline

```bash
python scripts/cleveland/run_cleveland_baseline.py \
  --data_path data/raw/cleveland/Heart_disease_cleveland_new.csv \
  --out_dir results/cleveland/baseline_real \
  --target_col target \
  --n_splits 5 \
  --test_size 0.2 \
  --random_seed 42
```

## Conditional diffusion

```bash
python scripts/cleveland/run_cleveland_diffusion.py \
  --data_path data/raw/cleveland/Heart_disease_cleveland_new.csv \
  --out_dir results/cleveland/diffusion_repeats \
  --target_col target \
  --n_splits 5 \
  --n_repeats 3 \
  --size_multipliers 1 2 3 \
  --epochs 300 \
  --timesteps 100 \
  --batch_size 64 \
  --device auto \
  --include_real_baseline \
  --save_synthetic
```

## Validate diffusion outputs

```bash
python scripts/cleveland/check_cleveland_diffusion_outputs.py \
  --data_path data/raw/cleveland/Heart_disease_cleveland_new.csv \
  --diffusion_dir results/cleveland/diffusion_repeats \
  --target_col target \
  --n_splits 5 \
  --test_size 0.2 \
  --random_seed 42 \
  --fail_on_error
```

## Simple baselines

```bash
python scripts/cleveland/run_cleveland_simple_baselines.py \
  --data_path data/raw/cleveland/Heart_disease_cleveland_new.csv \
  --out_dir results/cleveland/simple_baselines \
  --target_col target \
  --n_splits 5 \
  --n_repeats 3 \
  --size_multipliers 1 2 3 \
  --include_real_baseline \
  --save_synthetic
```

---

# Chronic Kidney Disease

```bash
conda activate synthetic-medical-core
```

## Sanitize input

```bash
python scripts/ckd/prepare_ckd_dataset.py \
  --data_path data/raw/ckd/kidney_disease.csv \
  --out_path data/processed/ckd/kidney_disease_sanitized.csv
```

## Real baseline

```bash
python scripts/ckd/run_ckd_baseline.py \
  --data_path data/processed/ckd/kidney_disease_sanitized.csv \
  --out_dir results/ckd/baseline_real \
  --n_splits 5 \
  --test_size 0.2 \
  --random_seed 42
```

## Unified synthetic experiment

```bash
python scripts/ckd/run_ckd_synthetic_experiment.py \
  --data_path data/processed/ckd/kidney_disease_sanitized.csv \
  --out_dir results/ckd/synthetic_experiment \
  --n_splits 5 \
  --n_repeats 3 \
  --size_multipliers 1 2 3 \
  --generators gaussian empirical_bootstrap ddpm \
  --epochs 300 \
  --timesteps 100 \
  --batch_size 128 \
  --include_real_baseline \
  --save_synthetic
```

## Deterministic DDPM categorical repair

Run only if validation detects the unused missing-category placeholder:

```bash
python scripts/ckd/repair_ckd_ddpm_categories_v2.py \
  --data_path data/processed/ckd/kidney_disease_sanitized.csv \
  --experiment_dir results/ckd/synthetic_experiment \
  --source_script scripts/ckd/run_ckd_synthetic_experiment.py \
  --affected_splits 2 4 \
  --random_seed 42 \
  --n_splits 5 \
  --test_size 0.2 \
  --apply
```

---

# Cross-dataset membership-inference audit

Run after PIMA, Cleveland and CKD releases exist:

```bash
python scripts/privacy/run_membership_inference_privacy.py \
  --datasets pima cleveland ckd \
  --out_dir results/privacy/membership_attack \
  --size_multipliers 1 2 3 \
  --pima_data_path data/raw/pima/pima.csv \
  --pima_correlation_dir results/pima/correlation \
  --include_pima_bootstrap_control \
  --cleveland_data_path data/raw/cleveland/Heart_disease_cleveland_new.csv \
  --cleveland_ddpm_dir results/cleveland/diffusion_repeats \
  --cleveland_simple_dir results/cleveland/simple_baselines \
  --ckd_data_path data/processed/ckd/kidney_disease_sanitized.csv \
  --ckd_experiment_dir results/ckd/synthetic_experiment
```

---

# Diabetes 130-US LLM case study

## 1. Audit raw data

```bash
conda activate synthetic-medical-core

python scripts/diabetes130/audit_diabetes130.py \
  --data_path data/raw/diabetes130/diabetic_data.csv \
  --mapping_path data/raw/diabetes130/IDS_mapping.csv \
  --out_dir results/diabetes130/audit
```

## 2. Prepare frozen grouped split and 20K source subset

```bash
python scripts/diabetes130/prepare_diabetes130_llm_pilot.py \
  --data_path data/raw/diabetes130/diabetic_data.csv \
  --out_dir data/processed/diabetes130/llm_pilot_data \
  --pilot_rows 20000 \
  --rare_min_count 20 \
  --random_seed 42 \
  --n_group_folds 10
```

Do not rerun with another seed after downstream results have been produced.

## 3. Tokenizer audit

```bash
conda activate synthetic-medical-llm311

python scripts/diabetes130/audit_diabetes130_tokenizer.py \
  --csv_path data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --model_name_or_path tabularisai/Qwen3-0.3B-distil \
  --out_dir results/diabetes130/tokenizer_audit \
  --sample_rows 5000 \
  --max_length 1024
```

## 4. Smoke test

```bash
python scripts/diabetes130/smoke_test_diabetes130_great_v2.py \
  --train_csv data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --out_dir results/diabetes130/great_smoke_v2 \
  --train_rows 2000 \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --n_synthetic 100 \
  --sampling_batch_size 10 \
  --max_length 512 \
  --temperature 0.7 \
  --fp16
```

## 5. Sampling benchmark

```bash
python scripts/diabetes130/benchmark_diabetes130_great_generation_v2.py \
  --model_dir results/diabetes130/great_smoke_v2/saved_model \
  --real_train_csv results/diabetes130/great_smoke_v2/smoke_training_subset.csv \
  --out_dir results/diabetes130/great_generation_benchmark \
  --n_synthetic 100 \
  --k_values 10 25 50 \
  --max_length 512 \
  --temperature 0.7
```

The validated RTX 2080 Ti setting selected `k=50`. `k=100` caused CUDA OOM.

## 6. Main one-epoch GReaT pilot

```bash
python scripts/diabetes130/run_diabetes130_great_main_pilot.py \
  --train_csv data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --out_dir results/diabetes130/great_main_20k \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --n_synthetic 20000 \
  --sampling_batch_size 50 \
  --chunk_size 500 \
  --max_length 512 \
  --temperature 0.7 \
  --fp16
```

Resume without retraining:

```bash
python scripts/diabetes130/run_diabetes130_great_main_pilot.py \
  --train_csv data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --out_dir results/diabetes130/great_main_20k \
  --n_synthetic 20000 \
  --sampling_batch_size 50 \
  --chunk_size 500 \
  --max_length 512 \
  --temperature 0.7 \
  --fp16 \
  --reuse_saved_model
```

## 7. Finalize the release

```bash
python scripts/diabetes130/finalize_diabetes130_great_release.py \
  --real_train_csv data/processed/diabetes130/llm_pilot_data/reduced/train.csv \
  --synthetic_csv results/diabetes130/great_main_20k/synthetic_raw_20000.csv \
  --out_dir results/diabetes130/great_main_20k/final_release
```

Expected: 19,999 rows and `overall_schema_pass = true`.

## 8. Utility, fidelity and simple baselines

```bash
conda activate synthetic-medical-core

python scripts/diabetes130/run_diabetes130_utility_fidelity_v2.py \
  --real_source_csv data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --real_full_train_csv data/processed/diabetes130/llm_pilot_data/reduced/train.csv \
  --validation_csv data/processed/diabetes130/llm_pilot_data/reduced/validation.csv \
  --test_csv data/processed/diabetes130/llm_pilot_data/reduced/test.csv \
  --great_csv results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv \
  --great_audit_json results/diabetes130/great_main_20k/final_release/final_release_audit.json \
  --out_dir results/diabetes130/final_evaluation/utility_fidelity \
  --seeds 41 42 43
```

After a summary-only failure, append `--resume_existing` to reuse completed classifier runs and generated baselines.

## 9. Privacy-related evaluation

```bash
python scripts/diabetes130/run_diabetes130_privacy.py \
  --real_source_csv data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
  --test_csv data/processed/diabetes130/llm_pilot_data/reduced/test.csv \
  --great_csv results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv \
  --generated_releases_dir results/diabetes130/final_evaluation/utility_fidelity/generated_releases \
  --out_dir results/diabetes130/final_evaluation/privacy \
  --mia_max_per_group 5000 \
  --bootstrap_iterations 500
```

## 10. Compare outputs

Compare regenerated results with:

```text
results/diabetes130/evaluation_snapshot/
```

The snapshots preserve exact headline values. Raw `utility_runs.csv`, predictions and full generated-release directories should be retained locally for paired follow-up analyses.
