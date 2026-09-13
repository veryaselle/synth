# CTGAN hyperparameter sensitivity

This is exploratory and does NOT replace the frozen primary benchmark.

Recommended first step: Cleveland only.

```bash
chmod +x submit_ctgan_sensitivity_cleveland.sh
bash submit_ctgan_sensitivity_cleveland.sh
```

Configs:
- A: batch 500, G lr 2e-4, D lr 2e-4, D steps 1 (baseline)
- B: batch 100, G lr 2e-4, D lr 2e-4, D steps 1
- C: batch 100, G lr 2e-4, D lr 1e-4, D steps 1
- D: batch 100, G lr 2e-4, D lr 2e-4, D steps 2
- E: batch 100, G lr 2e-4, D lr 2e-5, D steps 1

Cleveland pilot = 25 runs. Results:
`results/gan_sensitivity/ctgan/cleveland/...`

Final summary:
`results/gan_sensitivity/ctgan/summary/ctgan_sensitivity_summary_numeric.csv`

Compare configs using AUROC/F1/Brier, PCD/WS/JS, MIA/AIA and split stability.
Do not choose the winner from GAN loss alone.

The optional `ctgan_sensitivity_all.sbatch` runs the same predeclared grid on
PIMA, Cleveland and CKD (75 tasks). Use it only after the Cleveland pilot if
you want the full confirmation grid.
