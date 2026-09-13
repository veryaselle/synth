# Portability and path policy

The physical checkout location is not part of the experimental configuration. No personal workstation/HPC path is required in executable code.

## Python path rule

Paths are resolved from the script file rather than from the process working directory.

For Python files directly inside `unified_benchmark_v3/`, the repository root is exactly:

```python
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "unified_benchmark_v3"
```

For historical scripts under `scripts/<area>/`, the repository root is one level farther away and therefore correctly uses `parents[2]`:

```python
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
```

Nested analysis scripts use the corresponding structural depth. This is intentional: using `parents[1]` blindly in every directory would point to the wrong location. CLI arguments still override defaults whenever an alternative input or output location is wanted.

## Shell and Slurm

Shell submit/preflight wrappers derive their own location with `BASH_SOURCE` and export `BENCHMARK_ROOT`. Slurm jobs consume that exported value, with `SLURM_SUBMIT_DIR` only as a fallback. `CORE_PY` and `LLM_PY` are optional interpreter overrides; otherwise `python` from the active environment is used.

## Verification

From the repository root:

```bash
python scripts/maintenance/verify_clone_ready.py
```

The verifier checks Python syntax, all shell/Slurm syntax, machine-specific absolute paths, cwd-sensitive Python defaults, frozen split inventory and the final thesis numerical snapshots.

## Provenance artifacts

Historical logs or metadata may legitimately record the location of the machine on which the original experiment was executed. These files are preserved as provenance and are not executable configuration. The portability audit applies to executable `.py`, `.sh` and `.sbatch` files.
