Place both executable files directly in:

/home/sc.uni-leipzig.de/ab20zawy/naster/synthetic_data_evaluation/unified_benchmark_v3/tradeoff_analysis

Then run:

cd /home/sc.uni-leipzig.de/ab20zawy/naster/synthetic_data_evaluation/unified_benchmark_v3/tradeoff_analysis
sbatch tradeoff_analysis.sbatch

Important b100 layout handled by this version:
results/gan_sensitivity/pima/split_0/CTGAN_b100/evaluation/main_results_row.csv
results/gan_sensitivity/pima/split_0/CopulaGAN_b100/evaluation/main_results_row.csv
(and equivalently for all splits, Cleveland and CKD)

Other configs remain:
results/main_benchmark/<dataset>/split_N/<method>/evaluation/main_results_row.csv
results/gan_sensitivity/b200/<dataset>/split_N/<method>/evaluation/main_results_row.csv
results/gan_sensitivity/lr5e4_b100/<dataset>/split_N/<method>/evaluation/main_results_row.csv
