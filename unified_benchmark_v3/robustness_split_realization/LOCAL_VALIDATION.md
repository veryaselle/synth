# Local validation record

The second-realization pipeline was checked before packaging.

## Passed checks

- `prepare_second_realization.py --seed 2026` completed for PIMA, Cleveland and CKD.
- Expected processed split sizes were verified:
  - PIMA: 614 / 154;
  - Cleveland: 242 / 61;
  - CKD: 320 / 80.
- All processed split tables contain zero missing values.
- Every train/test index pair is disjoint.
- No second-realization test partition exactly duplicates any primary frozen test partition.
- Mean Jaccard overlap of second-realization vs primary test sets:
  - PIMA: 0.1143;
  - Cleveland: 0.1148;
  - CKD: 0.1117.
- REAL-reference evaluator smoke test passed on the new PIMA split 0.
- SDV/ARF generator contract dry-run passed on PIMA, Cleveland and CKD.
- Conditional-DDPM contract dry-run passed on PIMA, Cleveland and CKD.
- GReaT-style LLM contract dry-run passed on CKD (widest schema).
- Strict 120-row finalization and primary-comparison logic passed using a mock complete inventory.
- Repository-wide clone-ready verification passed after adding this experiment:
  - Python compile;
  - shell/Slurm syntax;
  - portable-path checks;
  - primary frozen repository snapshot checks.

## Not executed locally

Full SDV/ARF generation was not executed because the local validation environment does not include `sdv` or `arfpy`. Full LLM training was not executed because the validated GPU/LLM environment is an HPC requirement. These stages are therefore covered locally by input-contract dry runs and are intended to be executed through the supplied Slurm scripts.

The primary benchmark files were not modified by this experiment; all new data and outputs live under `robustness_split_realization/`.
