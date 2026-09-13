#!/usr/bin/env python3
# Re-run the historical Cleveland result aggregation without relying on cwd.
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--diffusion_summary",
    default=str(REPOSITORY_ROOT / "results/cleveland/diffusion_repeats/diffusion_utility_summary_repeats.csv"),
)
parser.add_argument(
    "--simple_summary",
    default=str(REPOSITORY_ROOT / "results/cleveland/simple_baselines/simple_baseline_utility_summary.csv"),
)
parser.add_argument(
    "--out_dir",
    default=str(REPOSITORY_ROOT / "results/cleveland/final_outputs"),
)
args = parser.parse_args()

diff_path = Path(args.diffusion_summary)
simple_path = Path(args.simple_summary)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

diff = pd.read_csv(diff_path)
simple = pd.read_csv(simple_path)

common_cols = [
    'generator', 'classifier', 'size_multiplier', 'n_rows',
    'auc_mean', 'auc_std', 'accuracy_mean', 'sensitivity_mean',
    'specificity_mean', 'precision_mean', 'f1_mean', 'brier_mean',
    'pcd_mean', 'ws_mean', 'js_mean', 'dcr_mean', 'dcr_min',
    'duplicate_rate_mean'
]
for df in [diff, simple]:
    for c in common_cols:
        if c not in df.columns:
            df[c] = np.nan

combined = pd.concat(
    [simple[common_cols], diff[diff['generator'] != 'REAL'][common_cols]],
    ignore_index=True
)
combined.to_csv(out_dir / 'cleveland_all_generators_summary.csv', index=False)

synthetic = combined[combined['generator'] != 'REAL'].copy()
size_summary = (
    synthetic.groupby(['generator', 'size_multiplier'], as_index=False)
    .agg(
        auc_mean_over_classifiers=('auc_mean', 'mean'),
        brier_mean_over_classifiers=('brier_mean', 'mean'),
        pcd_mean=('pcd_mean', 'mean'),
        ws_mean=('ws_mean', 'mean'),
        js_mean=('js_mean', 'mean'),
        dcr_mean=('dcr_mean', 'mean'),
        dcr_min=('dcr_min', 'min'),
        duplicate_rate_mean=('duplicate_rate_mean', 'mean'),
    )
)
size_summary.to_csv(out_dir / 'cleveland_generator_size_summary.csv', index=False)

real = combined[combined['generator'] == 'REAL'][['classifier', 'auc_mean', 'auc_std', 'brier_mean']]
real = real.rename(columns={'auc_mean': 'real_auc_mean', 'auc_std': 'real_auc_std', 'brier_mean': 'real_brier_mean'})
best_synthetic = synthetic.loc[synthetic.groupby('classifier')['auc_mean'].idxmax()].copy()
best_synthetic = best_synthetic.rename(columns={
    'generator': 'best_synthetic_generator',
    'size_multiplier': 'best_synthetic_size',
    'auc_mean': 'best_synthetic_auc_mean',
    'auc_std': 'best_synthetic_auc_std',
    'brier_mean': 'best_synthetic_brier_mean',
})
real_vs_best = best_synthetic.merge(real, on='classifier', how='left')
real_vs_best['auc_gap_synthetic_minus_real'] = real_vs_best['best_synthetic_auc_mean'] - real_vs_best['real_auc_mean']
real_vs_best.to_csv(out_dir / 'cleveland_real_vs_best_synthetic_overall.csv', index=False)

plt.figure(figsize=(8, 5))
for generator, sub in size_summary.groupby('generator'):
    sub = sub.sort_values('size_multiplier')
    plt.plot(sub['size_multiplier'], sub['auc_mean_over_classifiers'], marker='o', label=generator)
plt.xlabel('Synthetic data size multiplier')
plt.ylabel('Mean AUC over classifiers')
plt.title('Cleveland: utility by generator and synthetic size')
plt.xticks(sorted(size_summary['size_multiplier'].unique()))
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / 'cleveland_auc_by_generator_size.png', dpi=200)
plt.close()
