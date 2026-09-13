.PHONY: verify clone-check portable frozen-check checksums diabetes130-audit diabetes130-prepare diabetes130-utility diabetes130-privacy

verify:
	python scripts/maintenance/verify_thesis_repository.py

clone-check:
	python scripts/maintenance/verify_clone_ready.py

portable:
	python scripts/maintenance/check_portable_paths.py

frozen-check:
	python unified_benchmark_v3/check_cleveland_frozen.py --allow_existing_results
	python unified_benchmark_v3/check_ckd_frozen.py --allow_existing_results

checksums:
	python scripts/maintenance/create_checksums.py

diabetes130-audit:
	python scripts/diabetes130/audit_diabetes130.py \
	  --data_path data/raw/diabetes130/diabetic_data.csv \
	  --mapping_path data/raw/diabetes130/IDS_mapping.csv \
	  --out_dir results/diabetes130/audit

diabetes130-prepare:
	python scripts/diabetes130/prepare_diabetes130_llm_pilot.py \
	  --data_path data/raw/diabetes130/diabetic_data.csv \
	  --out_dir data/processed/diabetes130/llm_pilot_data \
	  --pilot_rows 20000 \
	  --rare_min_count 20 \
	  --random_seed 42

diabetes130-utility:
	python scripts/diabetes130/run_diabetes130_utility_fidelity_v2.py \
	  --real_source_csv data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
	  --real_full_train_csv data/processed/diabetes130/llm_pilot_data/reduced/train.csv \
	  --validation_csv data/processed/diabetes130/llm_pilot_data/reduced/validation.csv \
	  --test_csv data/processed/diabetes130/llm_pilot_data/reduced/test.csv \
	  --great_csv results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv \
	  --great_audit_json results/diabetes130/great_main_20k/final_release/final_release_audit.json \
	  --out_dir results/diabetes130/final_evaluation/utility_fidelity \
	  --seeds 41 42 43

diabetes130-privacy:
	python scripts/diabetes130/run_diabetes130_privacy.py \
	  --real_source_csv data/processed/diabetes130/llm_pilot_data/reduced/train_pilot_20000.csv \
	  --test_csv data/processed/diabetes130/llm_pilot_data/reduced/test.csv \
	  --great_csv results/diabetes130/great_main_20k/final_release/synthetic_final_valid.csv \
	  --generated_releases_dir results/diabetes130/final_evaluation/utility_fidelity/generated_releases \
	  --out_dir results/diabetes130/final_evaluation/privacy \
	  --mia_max_per_group 5000 \
	  --bootstrap_iterations 500
