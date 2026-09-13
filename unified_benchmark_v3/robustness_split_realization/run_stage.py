#!/usr/bin/env python3
"""Run one unit of the independent split-realization robustness experiment.

This wrapper deliberately reuses the *same* generator and evaluator programs as the
primary benchmark. Only the real train/test partitions are changed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent
DEFAULT_REALIZATION_SEED = 2026

METHODS: Dict[str, tuple[str, str]] = {
    "tvae": ("TVAE", "TVAE"),
    "ctgan": ("CTGAN", "CTGAN"),
    "copulagan": ("CopulaGAN", "CopulaGAN"),
    "gaussian_copula": ("Gaussian Copula", "GaussianCopula"),
    "arf": ("ARF", "ARF"),
}


def _python(external_name: str, fallback: str) -> str:
    return os.environ.get(external_name) or fallback


def _csv_arg(values: List[str]) -> str:
    return ",".join(values)


def _run(cmd: List[str], dry_run: bool) -> None:
    print("$ " + shlex.join([str(x) for x in cmd]), flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True)


def _schema_args(schema: Dict[str, object]) -> List[str]:
    args: List[str] = ["--target", str(schema["target"])]
    numeric = list(schema.get("numeric", []))
    categorical = list(schema.get("categorical", []))
    if numeric:
        args += ["--numeric_cols", _csv_arg([str(x) for x in numeric])]
    if categorical:
        args += ["--categorical_cols", _csv_arg([str(x) for x in categorical])]
    return args


def _paths(seed: int, dataset: str, split: int, results_root: Path | None):
    data_root = THIS_DIR / "data" / f"seed_{seed}"
    result_root = (
        results_root
        if results_root is not None
        else THIS_DIR / "results" / f"seed_{seed}"
    )
    split_data = data_root / dataset / f"split_{split}"
    split_result = result_root / dataset / f"split_{split}"
    return data_root, result_root, split_data, split_result


def _write_wrapper_metadata(
    out_dir: Path,
    *,
    stage: str,
    dataset: str,
    split: int,
    realization_seed: int,
    generator_seed: int,
    evaluator_seed: int,
    method: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "independent split-realization robustness check",
        "stage": stage,
        "dataset": dataset,
        "split_id": split,
        "split_realization_seed": realization_seed,
        "generator_seed": generator_seed,
        "evaluator_seed": evaluator_seed,
        "method": method,
        "primary_policy_held_constant": True,
    }
    (out_dir / "split_realization_run_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["real", "sdv_arf", "ddpm", "llm"], required=True)
    p.add_argument("--dataset", choices=["pima", "cleveland", "ckd"], required=True)
    p.add_argument("--split", type=int, choices=range(5), required=True)
    p.add_argument("--method", choices=sorted(METHODS), default=None)
    p.add_argument("--realization_seed", type=int, default=DEFAULT_REALIZATION_SEED)
    p.add_argument("--generator_seed_base", type=int, default=42)
    p.add_argument("--evaluator_seed", type=int, default=42)
    p.add_argument("--results_root", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.stage == "sdv_arf" and args.method is None:
        p.error("--method is required for --stage sdv_arf")
    if args.stage != "sdv_arf" and args.method is not None:
        p.error("--method is only valid for --stage sdv_arf")

    results_root = Path(args.results_root) if args.results_root else None
    _, result_root, split_data, split_result = _paths(
        args.realization_seed, args.dataset, args.split, results_root
    )

    real_train = split_data / "real_train.csv"
    real_test = split_data / "real_test.csv"
    schema_path = split_data / "schema.json"
    for path in [real_train, real_test, schema_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split-realization input {path}. Run prepare_split_realization.py first."
            )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    generator_seed = args.generator_seed_base + args.split
    evaluator_seed = args.evaluator_seed
    core_py = _python("CORE_PY", sys.executable)
    llm_py = _python("LLM_PY", "python")

    evaluator_schema_args = _schema_args(schema)

    if args.stage == "real":
        out = split_result / "REAL" / "evaluation"
        cmd = [
            core_py,
            str(BENCHMARK_ROOT / "evaluate_real_reference.py"),
            "--dataset", args.dataset,
            "--real_train", str(real_train),
            "--real_test", str(real_test),
            *evaluator_schema_args,
            "--seed", str(evaluator_seed),
            "--outdir", str(out),
        ]
        _run(cmd, args.dry_run)
        _write_wrapper_metadata(
            out,
            stage=args.stage,
            dataset=args.dataset,
            split=args.split,
            realization_seed=args.realization_seed,
            generator_seed=generator_seed,
            evaluator_seed=evaluator_seed,
            method="REAL",
            dry_run=args.dry_run,
        )
        return

    if args.stage == "sdv_arf":
        method_name, method_dir = METHODS[args.method]
        out = split_result / method_dir
        gen_cmd = [
            core_py,
            str(BENCHMARK_ROOT / "generate_sdv_arf_safe_cpu.py"),
            "--method", args.method,
            "--real_train", str(real_train),
            "--schema", str(schema_path),
            "--outdir", str(out),
            "--seed", str(generator_seed),
            "--epochs", "300",
            "--batch_size", "500",
            "--joblib_n_jobs", "1",
            "--verbose",
        ]
        if args.dry_run:
            gen_cmd.append("--dry_run")
        _run(gen_cmd, False)
        # In dry-run mode the generator validates the input contract but does not
        # produce synthetic_1x.csv, so evaluator execution is intentionally skipped.
        if args.dry_run:
            print("DRY RUN: evaluator skipped because no synthetic release is created.")
            return
        eval_cmd = [
            core_py,
            str(BENCHMARK_ROOT / "evaluate_release.py"),
            "--dataset", args.dataset,
            "--method", method_name,
            "--real_train", str(real_train),
            "--real_test", str(real_test),
            "--synthetic", str(out / "synthetic_1x.csv"),
            *evaluator_schema_args,
            "--aia_k", "1",
            "--seed", str(evaluator_seed),
            "--outdir", str(out / "evaluation"),
        ]
        _run(eval_cmd, False)
        _write_wrapper_metadata(
            out,
            stage=args.stage,
            dataset=args.dataset,
            split=args.split,
            realization_seed=args.realization_seed,
            generator_seed=generator_seed,
            evaluator_seed=evaluator_seed,
            method=method_name,
            dry_run=False,
        )
        return

    if args.stage == "ddpm":
        out = split_result / "ConditionalDDPM"
        gen_cmd = [
            core_py,
            str(BENCHMARK_ROOT / "generate_conditional_ddpm.py"),
            "--real_train", str(real_train),
            "--schema", str(schema_path),
            "--outdir", str(out),
            "--seed", str(generator_seed),
            "--device", "cpu",
            "--epochs", "300",
            "--timesteps", "100",
            "--batch_size", "64",
            "--hidden_dim", "128",
            "--time_dim", "32",
            "--label_dim", "8",
            "--lr", "0.001",
            "--weight_decay", "0.00001",
            "--verbose",
        ]
        if args.dry_run:
            gen_cmd.append("--dry_run")
        _run(gen_cmd, False)
        if args.dry_run:
            print("DRY RUN: evaluator skipped because no synthetic release is created.")
            return
        eval_cmd = [
            core_py,
            str(BENCHMARK_ROOT / "evaluate_release.py"),
            "--dataset", args.dataset,
            "--method", "Conditional DDPM",
            "--real_train", str(real_train),
            "--real_test", str(real_test),
            "--synthetic", str(out / "synthetic_1x.csv"),
            *evaluator_schema_args,
            "--aia_k", "1",
            "--seed", str(evaluator_seed),
            "--outdir", str(out / "evaluation"),
        ]
        _run(eval_cmd, False)
        _write_wrapper_metadata(
            out,
            stage=args.stage,
            dataset=args.dataset,
            split=args.split,
            realization_seed=args.realization_seed,
            generator_seed=generator_seed,
            evaluator_seed=evaluator_seed,
            method="Conditional DDPM",
            dry_run=False,
        )
        return

    if args.stage == "llm":
        out = split_result / "LLM"
        gen_cmd = [
            llm_py,
            str(BENCHMARK_ROOT / "generate_great_llm_final.py"),
            "--real_train", str(real_train),
            "--schema", str(schema_path),
            "--outdir", str(out),
            "--seed", str(generator_seed),
            "--epochs", "10",
            "--batch_size", "1",
            "--gradient_accumulation_steps", "8",
            "--dataloader_num_workers", "2",
            "--sampling_batch_size", "50",
            "--chunk_size", "250",
            "--max_length", "512",
            "--temperature", "0.7",
            "--catastrophic_numeric_factor", "100",
            "--max_resample_rounds", "20",
            "--fp16",
        ]
        if args.dry_run:
            gen_cmd.append("--dry_run")
        _run(gen_cmd, False)
        if args.dry_run:
            print("DRY RUN: evaluator skipped because no synthetic release is created.")
            return
        eval_cmd = [
            core_py,
            str(BENCHMARK_ROOT / "evaluate_release.py"),
            "--dataset", args.dataset,
            "--method", "LLM",
            "--real_train", str(real_train),
            "--real_test", str(real_test),
            "--synthetic", str(out / "synthetic_1x.csv"),
            *evaluator_schema_args,
            "--aia_k", "1",
            "--seed", str(evaluator_seed),
            "--outdir", str(out / "evaluation"),
        ]
        _run(eval_cmd, False)
        _write_wrapper_metadata(
            out,
            stage=args.stage,
            dataset=args.dataset,
            split=args.split,
            realization_seed=args.realization_seed,
            generator_seed=generator_seed,
            evaluator_seed=evaluator_seed,
            method="LLM",
            dry_run=False,
        )
        return


if __name__ == "__main__":
    main()
