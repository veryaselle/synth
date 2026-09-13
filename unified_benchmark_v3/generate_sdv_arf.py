#!/usr/bin/env python3
"""Compatibility entry point for the validated SDV/ARF generator.

The maintained implementation is ``generate_sdv_arf_safe_cpu.py``. This
wrapper preserves older command lines without keeping a second generator
implementation that can drift from the thesis benchmark.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("generate_sdv_arf_safe_cpu.py")), run_name="__main__")
