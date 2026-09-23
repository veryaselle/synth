#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'unified_benchmark_v3/frozen_data'
def digest(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('regenerated_root', type=Path); a=ap.parse_args()
    diffs=[]
    for orig in FROZEN.rglob('*'):
        if not orig.is_file(): continue
        rel=orig.relative_to(FROZEN); new=a.regenerated_root/rel
        if not new.exists(): diffs.append((str(rel),'missing')); continue
        if orig.name=='preparation_metadata.json':
            x=json.loads(orig.read_text()); y=json.loads(new.read_text())
            for obj in (x,y):
                for k in list(obj):
                    if 'path' in k.lower() or 'root' in k.lower(): obj.pop(k,None)
            if x!=y: diffs.append((str(rel),'metadata differs'))
        elif digest(orig)!=digest(new): diffs.append((str(rel),'content differs'))
    if diffs:
        print('Frozen regeneration differs:'); [print(' -',*d) for d in diffs]; raise SystemExit(2)
    print('Frozen regeneration PASS: all preserved artifacts match (path-only provenance fields ignored).')
if __name__=='__main__': main()
