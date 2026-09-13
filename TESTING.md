# Clean-clone verification

The repository is designed so that the code does not need the checkout path to be edited into any script. Clone it, enter the repository, and run the verifier.

```bash
git clone https://github.com/veryaselle/synth.git
cd synth
python scripts/maintenance/verify_clone_ready.py
```

Expected final line:

```text
CLONE-READY CHECK PASS
```

## 1. Verify from a different working directory

The Python entry points derive the repository root from `__file__`, not from the current working directory. A direct check is:

```bash
REPO="$(pwd -P)"
cd "${TMPDIR:-/tmp}"
python "$REPO/scripts/maintenance/verify_thesis_repository.py"
```

## 2. Regenerate the frozen datasets outside the repository

Start this block from anywhere inside the clone. The repository location is obtained from Git before changing directory:

```bash
REPO="$(git rev-parse --show-toplevel)"
OUT="${TMPDIR:-/tmp}/synth_frozen_regenerated"
rm -rf "$OUT"
cd "${TMPDIR:-/tmp}"
python "$REPO/unified_benchmark_v3/prepare_frozen_datasets.py" --out_root "$OUT"
python "$REPO/scripts/maintenance/compare_frozen_regeneration.py" "$OUT"
```

Expected final line: `Frozen regeneration PASS`.

## 3. Dataset preflights

With compatible Python environments available, the preflight wrappers are location-independent. Start this block anywhere inside the clone:

```bash
REPO="$(git rev-parse --show-toplevel)"
cd "${TMPDIR:-/tmp}"
bash "$REPO/unified_benchmark_v3/pima_preflight.sh"
bash "$REPO/unified_benchmark_v3/cleveland_preflight.sh"
bash "$REPO/unified_benchmark_v3/ckd_preflight.sh"
```

`CORE_PY` and `LLM_PY` are optional interpreter overrides. No interpreter path is stored in the repository. If both Conda environments from `environment/` have been created, their interpreter locations can be resolved automatically:

```bash
export CORE_PY="$(conda run -n synthetic-medical-core python -c 'import sys; print(sys.executable)')"
export LLM_PY="$(conda run -n synthetic-medical-llm311 python -c 'import sys; print(sys.executable)')"
```

## 4. Slurm submission

Run the supplied submit wrappers rather than editing `.sbatch` files. The wrappers derive and export `BENCHMARK_ROOT` automatically.

```bash
cd "$REPO/unified_benchmark_v3"
bash submit_pima_all_fresh.sh
bash submit_cleveland_all.sh
bash submit_ckd_all.sh
```

For the exact final PIMA LLM path, `submit_pima_all.sh` expects the archived split-specific 10-epoch checkpoints. Those large checkpoints are deliberately not distributed in this repository. `submit_pima_all_fresh.sh` retrains the same declared configuration from scratch; it is therefore a fresh stochastic rerun, not a byte-identical recreation of the preserved checkpoint.

## 5. Evaluator memorization control

This is a diagnostic control, not a benchmark result. It deliberately reuses a frozen real training split as the synthetic release. Expected privacy outputs are DCR = 0, MIA AUROC = 1 and AIA risk = 1.

```bash
B="$REPO/unified_benchmark_v3"
OUT="${TMPDIR:-/tmp}/pima_memorization_smoke"
rm -rf "$OUT"
python "$B/evaluate_release.py" \
  --dataset pima \
  --method MEMORIZATION_SMOKE \
  --real_train "$B/frozen_data/pima/split_0/real_train.csv" \
  --real_test "$B/frozen_data/pima/split_0/real_test.csv" \
  --synthetic "$B/frozen_data/pima/split_0/real_train.csv" \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir "$OUT"
```


## 6. Supplementary split-realization robustness

The robustness wrapper is seed-parameterized. For a full HPC run, resolve both interpreters before submission so the preflight can validate the SDV/ARF and LLM dependencies.

```bash
export CORE_PY="$(conda run -n synthetic-medical-core python -c 'import sys; print(sys.executable)')"
export LLM_PY="$(conda run -n synthetic-medical-llm311 python -c 'import sys; print(sys.executable)')"
cd "$REPO/unified_benchmark_v3"

export ROBUSTNESS_SEED=2026
bash robustness_split_realization/preflight.sh

export ROBUSTNESS_SEED=3719704
bash robustness_split_realization/preflight.sh
```

Once both finalized result trees are available, the cross-realization summary is generated without pooling split-level observations:

```bash
python robustness_split_realization/aggregate_realizations.py --seeds 2026 3719704
```
