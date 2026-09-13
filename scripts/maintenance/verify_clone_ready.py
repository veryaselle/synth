#!/usr/bin/env python3
"""Run the static checks expected to pass in a clean clone.

The repository root is derived from this file's location, so this command may be
launched from any current working directory.
"""
from __future__ import annotations

import compileall
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPOSITORY_ROOT / "unified_benchmark_v3"


def fail(message: str) -> None:
    raise SystemExit(message)


def check_python() -> None:
    for directory in [REPOSITORY_ROOT / "scripts", BENCHMARK_ROOT]:
        if not compileall.compile_dir(directory, quiet=1):
            fail(f"Python syntax verification failed under {directory}")


def check_shell() -> None:
    shell_files = sorted(BENCHMARK_ROOT.rglob("*.sh")) + sorted(BENCHMARK_ROOT.rglob("*.sbatch"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)


def check_no_cwd_dependent_python_defaults() -> None:
    checks = [
        (re.compile(r"default\s*=\s*['\"](?:data|results)/"), "cwd-relative argparse default"),
        (re.compile(r"Path\(\s*['\"]\.['\"]\s*\)"), "Path('.') dependency"),
    ]
    violations: list[str] = []
    for path in sorted(REPOSITORY_ROOT.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern, label in checks:
                if pattern.search(line):
                    violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{line_no}: {label}: {line.strip()}")
    if violations:
        fail("CWD-DEPENDENCE CHECK FAILED\n" + "\n".join(violations))


def main() -> None:
    subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts/maintenance/check_portable_paths.py")],
        check=True,
    )
    check_python()
    check_shell()
    check_no_cwd_dependent_python_defaults()
    subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts/maintenance/verify_thesis_repository.py")],
        check=True,
    )
    print("CLONE-READY CHECK PASS")
    print(f"Repository root resolved from script location: {REPOSITORY_ROOT}")
    print("Python syntax: PASS; shell/Slurm syntax: PASS; path policy: PASS; thesis snapshot checks: PASS")


if __name__ == "__main__":
    main()
