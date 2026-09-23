# Cross-method trade-off analysis

Place BOTH files directly in:

/home/sc.uni-leipzig.de/ab20zawy/naster/synthetic_data_evaluation/unified_benchmark_v3/tradeoff_analysis

Required previous outputs:

tradeoff_analysis/output/gan_tradeoff_split_level.csv
tradeoff_analysis/output/gan_pareto_front.csv

Required frozen primary results:

unified_benchmark_v3/results/main_benchmark/<dataset>/split_N/.../main_results_row.csv

The script does NOT assume method folder names inside main_benchmark. It searches each frozen
split directory for main_results_row.csv and validates the exact 8-method inventory.

Run ONE Slurm job:

cd /home/sc.uni-leipzig.de/ab20zawy/naster/synthetic_data_evaluation/unified_benchmark_v3/tradeoff_analysis
sbatch cross_method_tradeoff.sbatch

Output:

tradeoff_analysis/cross_method_output/

Key files:
- cross_method_raw_summary.csv
- cross_method_scores.csv
- cross_method_dimension_pareto_minmax.csv
- cross_method_dimension_pareto_rank.csv
- primary_only_tradeoff_scores.csv
- primary_only_dimension_pareto_minmax.csv
- primary_only_dimension_pareto_rank.csv
- primary_only_strategies.csv
- primary_only_strategy_robustness.csv
- cross_method_strategies.csv
- cross_method_strategy_robustness.csv
- cross_method_constraint_scenarios.csv
- real_reference_summary.csv
- figures/

Scientific safeguards:
- tuned GANs are exploratory overlay only
- frozen CTGAN/CopulaGAN remain in the primary benchmark
- raw metrics are retained and should be the primary evidence
- DCR is excluded from the privacy composite
- privacy uses |MIA AUROC - 0.5| + AIA risk
- both min-max and rank scoring are produced as a robustness check
- if strategy winners differ between scoring schemes, report the choice as score-sensitive
- all score normalization is performed within dataset only

Important: primary-only scores are recomputed using ONLY frozen candidates; exploratory tuned GANs cannot alter the normalized primary benchmark.
