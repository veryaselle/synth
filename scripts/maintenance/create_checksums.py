#!/usr/bin/env python3
"""Create SHA-256 checksums for tracked data, scripts, and result artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--out", default="artifact_checksums.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    files = []
    for base in ["data", "scripts", "results", "unified_benchmark_v3"]:
        for path in sorted((root / base).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append({
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest(path),
                })
    out = root / args.out
    out.write_text(json.dumps({"files": files}, indent=2))
    print(f"Wrote {out} with {len(files)} entries")


if __name__ == "__main__":
    main()
