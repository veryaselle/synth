# Cleveland unified benchmark

This bundle runs the frozen Cleveland Heart Disease benchmark under the same primary contract used for PIMA.

## Frozen data
- 5 StratifiedShuffleSplit splits (seed 42, test size 0.2)
- each split: 242 real-train rows, 61 held-out real-test rows
- target: `target`
- numeric: `age, trestbps, chol, thalach, oldpeak`
- categorical: `sex, cp, fbs, restecg, exang, slope, ca, thal`
- no imputation after cleaning
- 1x synthetic release per method

## Methods
REAL, TVAE, CTGAN, CopulaGAN, ARF, Gaussian Copula, Conditional DDPM, LLM.

## Main run

Copy these files into the unified benchmark working directory, alongside:
- `evaluate_release.py`
- `assemble_main_tables.py`
- `generate_sdv_arf_safe_cpu.py`
- `generate_conditional_ddpm.py`
- `generate_great_llm_final.py`
- `frozen_data/cleveland/...`

Then:

```bash
chmod +x cleveland_preflight.sh submit_cleveland_all.sh
bash submit_cleveland_all.sh
```

The submission script runs preflight first, submits CPU and GPU jobs, and submits a final aggregation job with an `afterok` dependency.

## LLM
The Cleveland LLM uses fresh 10-epoch training on every split. It does not use `--reuse_saved_model`. The final GReaT adapter rejects/resamples structural/type-invalid rows and catastrophic numeric outputs beyond the predeclared 100x loose scale guard. Moderate numeric support extrapolation is retained; no clipping is applied.

## Important
`finalize_cleveland.py` refuses to aggregate unless it sees exactly one result for every `(method, split)` combination: 8 methods x 5 splits = 40 rows. This prevents archived or duplicated experiments from silently contaminating the table.
