#!/usr/bin/env python3
from pathlib import Path
import re, sys
ROOT = Path(__file__).resolve().parents[2]
EXEC_SUFFIXES = {'.py','.sh','.sbatch'}
patterns = [
    (re.compile(r'/home/'), 'absolute /home path'),
    (re.compile(r'/mnt/data/'), 'absolute /mnt/data path'),
    (re.compile(r'ab20zawy'), 'personal HPC account'),
    (re.compile(r'\.conda/envs/'), 'hard-coded conda environment'),
]
violations=[]
SELF = Path(__file__).resolve()
for p in ROOT.rglob('*'):
    if p.resolve() == SELF: continue
    if not p.is_file() or p.suffix not in EXEC_SUFFIXES: continue
    try: txt=p.read_text(errors='replace')
    except Exception: continue
    for lineno,line in enumerate(txt.splitlines(),1):
        for rx,label in patterns:
            if rx.search(line): violations.append((p.relative_to(ROOT),lineno,label,line.strip()))
if violations:
    print('PORTABILITY CHECK FAILED')
    for v in violations: print(f'{v[0]}:{v[1]}: {v[2]}: {v[3]}')
    sys.exit(2)
print('PORTABILITY CHECK PASS: no machine-specific absolute paths in executable files.')
