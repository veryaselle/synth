# PIMA — first main benchmark runs

The five frozen PIMA splits are under `frozen_data/pima/split_0` ... `split_4`.

For each split, run the real reference once, then generate and evaluate each method. Example for split 0 and TVAE:

```bash
python evaluate_real_reference.py \
  --dataset pima \
  --real_train frozen_data/pima/split_0/real_train.csv \
  --real_test frozen_data/pima/split_0/real_test.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_0/REAL/evaluation

python generate_sdv_arf.py \
  --method tvae \
  --real_train frozen_data/pima/split_0/real_train.csv \
  --schema frozen_data/pima/split_0/schema.json \
  --outdir results/main_benchmark/pima/split_0/TVAE \
  --seed 42 \
  --epochs 300 \
  --cuda

python evaluate_release.py \
  --dataset pima \
  --method TVAE \
  --real_train frozen_data/pima/split_0/real_train.csv \
  --real_test frozen_data/pima/split_0/real_test.csv \
  --synthetic results/main_benchmark/pima/split_0/TVAE/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_0/TVAE/evaluation
```

Change `--method` to `ctgan`, `copulagan`, `gaussian_copula`, or `arf` for the other generators implemented by the same runner.

After all five splits have been evaluated:

```bash
python assemble_main_tables.py \
  --results_root results/main_benchmark \
  --outdir results/main_benchmark/tables
```

The final dataset table reports **mean ± SD across five frozen splits**.
