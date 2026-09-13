# CKD unified benchmark

Primary contract mirrors the completed Cleveland benchmark.

- 5 stratified splits, seed 42, test size 0.2
- 320 train / 80 held-out test rows per split
- target: `target`
- numeric: `age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc`
- categorical/ordinal: `sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane`
- train-only preprocessing: numeric median + categorical mode imputation
- processed frozen tables contain zero missing values
- primary synthetic size: 1x
- methods: REAL, TVAE, CTGAN, CopulaGAN, ARF, Gaussian Copula, Conditional DDPM, LLM
- strict target inventory: 8 methods x 5 splits = 40 primary rows

## Required shared files

Place these CKD files in the unified benchmark working directory alongside:

- `evaluate_release.py`
- `evaluate_real_reference.py`
- `generate_sdv_arf_safe_cpu.py`
- `generate_conditional_ddpm.py`
- `generate_great_llm_final.py`
- `assemble_main_tables.py`
- `frozen_data/ckd/...`

Use the **current fixed** `generate_great_llm_final.py` that successfully completed all five Cleveland LLM splits. Do not replace it with an older copy.

## Run

```bash
chmod +x ckd_preflight.sh submit_ckd_all.sh
bash submit_ckd_all.sh
```

The finalizer is submitted with `afterany` so it still reports a complete inventory if one upstream branch fails. It remains strict and refuses anything other than 40/40.

## Monitor

```bash
squeue -u $USER
```

```bash
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

Successful finalization must show five splits for every method and save:

```text
results/main_benchmark/tables/ckd_main_table_mean_sd.csv
```
