GAN LR sensitivity: discriminator_lr=0.0005, batch_size=100

Requires generate_sdv_arf_safe_cpu.py to already support:
  --discriminator_lr

Run all:
  mkdir -p logs
  bash submit_all_lr5e4_b100.sh

Outputs:
  results/gan_sensitivity/lr5e4_b100/<dataset>/split_<0..4>/<method>/

Finalize:
  python finalize_gan_exceptions.py \
    --results_root results/gan_sensitivity/lr5e4_b100 \
    --tables_out results/gan_sensitivity/lr5e4_b100/tables \
    --assembler assemble_main_tables.py
