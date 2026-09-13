# Data directory

Raw datasets are included for private supervisor reproduction. See `docs/DATASETS.md` before public release.

Generate the Diabetes 130-US grouped splits with:

```bash
python scripts/diabetes130/prepare_diabetes130_llm_pilot.py   --data_path data/raw/diabetes130/diabetic_data.csv   --out_dir data/processed/diabetes130/llm_pilot_data   --pilot_rows 20000   --rare_min_count 20   --random_seed 42
```
