# Environments

Two environments are used because the classical/diffusion experiments and the GReaT case study were validated under different Python versions.

## Core environment

```bash
conda env create -f environment/environment-core.yml
conda activate synthetic-medical-core
```

This environment covers PIMA, Cleveland, CKD, the cross-dataset membership-inference audit, and the non-LLM portions of the Diabetes 130-US evaluation.

## LLM environment

The completed GReaT run used:

- Python 3.11.15
- PyTorch 2.11.0+cu130
- Transformers 5.14.1
- Accelerate 1.14.0
- be-great 0.0.14
- NVIDIA GeForce RTX 2080 Ti with 10.567 GB VRAM

Install the PyTorch build appropriate for the machine's CUDA runtime before the remaining LLM dependencies. On the validated RTX 2080 Ti setup, GReaT was trained with FP32 master parameters and FP16 autocast. BF16 was disabled.

## Exact local lock files

Before the final GitHub push, capture the exact completed environments:

```bash
conda activate synthetic_data_evaluation
conda env export --no-builds > environment/environment-core-lock.yml

conda activate synthetic_data_llm311
conda env export --no-builds > environment/environment-llm-lock.yml
```

Review exported files for local paths and credentials before committing.


## Final benchmark generator dependencies

The core benchmark additionally requires `sdv` and `arfpy`; the portable environment files include them. The final generator wrapper is deliberately version-adaptive across the SDV metadata APIs used during the thesis. Exact package versions recorded by completed runs remain part of each `generation_metadata.json` artifact when available.
