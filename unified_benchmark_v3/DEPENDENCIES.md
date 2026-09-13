# Generator dependencies

The shared evaluator needs the scientific Python stack already used in the thesis (`pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`).

For the five generators implemented by `generate_sdv_arf.py`:

```bash
pip install sdv arfpy
```

The current SDV documentation provides single-table synthesizers for TVAE, CTGAN, CopulaGAN and Gaussian Copula. ARF is run through the authors' `arfpy` implementation using the documented `arf → forde → forge` workflow.

Do not silently upgrade the experiment environment after runs have started. Export the exact environment used for the final benchmark.
