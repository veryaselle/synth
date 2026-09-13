# Methodological conventions

## Utility

Synthetic releases use Train-Synthetic-Test-Real. Real baselines use the corresponding real training partition and the same held-out real test partition. For Diabetes 130-US, the primary comparison uses the same 20,000 real source encounters used to fine-tune GReaT. The 79,473-row real training partition is a separate full-data reference.

Classifiers are Logistic Regression, Random Forest, XGBoost and MLP. AUROC and AUPRC are the main ranking metrics. Brier score measures probability quality. F1, sensitivity, specificity and accuracy use a fixed threshold of 0.5 and are interpreted cautiously under class imbalance.

## Fidelity

Complementary metrics are reported:

- normalized Wasserstein distance;
- Jensen–Shannon divergence;
- pairwise correlation difference;
- missingness-rate MAE;
- categorical support coverage;
- rare-category coverage;
- numerical range coverage.

Absolute values are compared within a dataset, not across differently encoded datasets.

## Privacy-related indicators

DCR, nearest-neighbour references, duplicates and proximity-based membership inference are empirical audits, not formal privacy guarantees.

The attack compares real generator-training rows with held-out real rows. The score is the negative distance to the nearest synthetic row in the same target class. Bootstrap releases serve as a positive memorization control.

## Sample-size analysis

Where supported, releases are evaluated at 1×, 2× and 3× the real training size.

## Simple baselines

Gaussian–empirical synthesis combines a regularized multivariate Gaussian for numerical variables with empirical categorical marginals. Empirical bootstrap samples complete rows with replacement and is treated as a memorization control.

## Deterministic post-processing

Rules are declared before downstream evaluation, applied deterministically and logged. Raw releases remain available separately. Post-processing is never described as improving privacy.
