# GAN batch=200 sensitivity run

Purpose:
- compare the already completed batch=500 baseline and batch=100 sensitivity results
  against one intermediate point: batch=200.
- CTGAN and CopulaGAN only.
- PIMA, Cleveland, CKD.
- 5 frozen splits each.
- 300 epochs.
- same evaluator and seeds as the unified benchmark.

No Python changes are required.

Run:
```bash
mkdir -p logs
bash submit_all_gan_b200.sh
```

Outputs:
```text
results/gan_sensitivity/b200/<dataset>/split_<0..4>/<method>/
```

When all 30 rows are present:
```bash
python finalize_gan_exceptions.py   --results_root results/gan_sensitivity/b200   --tables_out results/gan_sensitivity/b200/tables   --assembler assemble_main_tables.py
```

The resulting tables can then be compared against:
- frozen baseline: batch=500
- previous sensitivity: batch=100
- new sensitivity: batch=200
