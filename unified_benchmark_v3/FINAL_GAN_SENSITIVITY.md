# Final GAN configuration-sensitivity interpretation

The formal sensitivity analysis contains four preserved configurations per GAN/dataset pair: the frozen primary baseline plus three non-baseline alternatives. The primary benchmark remains frozen.

For the thesis appendix/figure, “best alternative” means the best tested **non-baseline** alternative. In particular, Cleveland CTGAN is `0.651 -> 0.607`, with the paired unrounded change reported as `-0.045`, at batch size 100 and discriminator learning rate `5e-4`. The frozen 0.651 remains the best tested CTGAN configuration overall on Cleveland, but is not itself an alternative.

See `results/final_thesis/reported_gan_sensitivity_best_alternatives.csv`.
