# Independent split-realization robustness summary

- Realization seed: `2026`
- Primary source: unrounded primary numeric summary
- Mean metric-level Spearman agreement: `0.807`
- Median metric-level Spearman agreement: `0.875`
- Dimension-leader agreement across dataset × dimension × scoring scheme: `38.9%`

## Dimension leaders

- pima / utility / minmax: LLM -> TVAE (changed)
- pima / utility / rank: LLM / TVAE -> TVAE (changed)
- pima / fidelity / minmax: ARF -> ARF (stable)
- pima / fidelity / rank: ARF / Conditional DDPM -> ARF (changed)
- pima / privacy / minmax: CTGAN -> Gaussian Copula (changed)
- pima / privacy / rank: CTGAN -> Gaussian Copula (changed)
- cleveland / utility / minmax: Conditional DDPM -> Conditional DDPM (stable)
- cleveland / utility / rank: Conditional DDPM -> Conditional DDPM (stable)
- cleveland / fidelity / minmax: ARF -> ARF (stable)
- cleveland / fidelity / rank: ARF -> ARF (stable)
- cleveland / privacy / minmax: CopulaGAN -> CTGAN (changed)
- cleveland / privacy / rank: CopulaGAN -> CTGAN (changed)
- ckd / utility / minmax: ARF -> LLM (changed)
- ckd / utility / rank: ARF -> LLM (changed)
- ckd / fidelity / minmax: ARF -> ARF (stable)
- ckd / fidelity / rank: ARF -> ARF (stable)
- ckd / privacy / minmax: Gaussian Copula -> CopulaGAN (changed)
- ckd / privacy / rank: Gaussian Copula -> CopulaGAN (changed)

## Interpretation boundary

This is supplementary robustness evidence. The primary frozen benchmark remains the confirmatory reference. A changed leader is informative rather than a failure: it identifies conclusions that are sensitive to the train/test partition realization.
