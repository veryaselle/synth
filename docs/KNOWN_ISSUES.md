# Known issues and implementation decisions

## PIMA baseline artifact

The released baseline repository did not reproduce every paper-level experiment without manual fixes. Package discovery, configuration assumptions and the original Exp3 path required intervention. The clean reconstruction script evaluates available correlation-stage synthetic files directly.

## GReaT BF16 failure

The checkpoint loaded with BF16 parameters while FP16 gradient scaling attempted an unsupported BF16 unscale operation on the RTX 2080 Ti. The validated solution uses FP32 master parameters, FP16 autocast and `bf16=False`.

## Target-conditioned sampling

Exact numerical endpoint conditioning failed inside the installed `be-great` version. The validated method generates target 0 and target 1 separately with `start_col` and `start_col_dist`, then combines classes at the frozen real prevalence.

## Sampling batch size

`k=50` was the fastest complete configuration on the RTX 2080 Ti. `k=100` caused CUDA OOM and produced a partial release. The corrected benchmark never marks partial output as successful.

## GReaT serialization spillover

The generator occasionally continued a valid categorical token with fragments from later serialized fields. The finalizer recovers a value only when a unique longest training-valid prefix exists in a predefined affected column. It does not use arbitrary mode imputation.

## CKD missing-category placeholder

The diffusion decoder could select an unused missing-category placeholder even when absent from the relevant training split. The deterministic repair maps it to the training-split mode and logs every affected cell without retraining.

## Utility aggregation failure

The first Diabetes 130-US utility script completed all classifier runs but failed while merging summary tables because repeated `index` columns caused a Pandas `MergeError`. Version 2 uses named aggregation and supports `--resume_existing`.

## PCD constant dimensions

Constant one-hot dimensions create undefined correlations under `numpy.corrcoef`. Version 2 uses a safe standardized correlation implementation that assigns zero correlation to constant dimensions while preserving the common encoded feature space.
