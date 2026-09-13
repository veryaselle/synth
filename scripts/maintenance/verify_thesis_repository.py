#!/usr/bin/env python3
from pathlib import Path
import compileall, csv, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
BENCH=ROOT/'unified_benchmark_v3'
required=[
    BENCH/'evaluate_release.py', BENCH/'evaluate_real_reference.py',
    BENCH/'generate_sdv_arf_safe_cpu.py', BENCH/'generate_conditional_ddpm.py',
    BENCH/'generate_great_llm_final.py', BENCH/'prepare_frozen_datasets.py',
    BENCH/'assemble_main_tables.py',
    BENCH/'results/final_thesis/reported_primary_means.csv',
    BENCH/'results/final_thesis/reported_gan_sensitivity_best_alternatives.csv',
]
for ds in ['pima','cleveland','ckd']:
    for split in range(5):
        for name in ['real_train.csv','real_test.csv','schema.json','train_indices.csv','test_indices.csv']:
            required.append(BENCH/f'frozen_data/{ds}/split_{split}/{name}')
missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    print('MISSING REQUIRED FILES:')
    print('\n'.join(' - ' + x for x in missing))
    raise SystemExit(2)
if not compileall.compile_dir(str(BENCH), quiet=1): raise SystemExit('Python syntax check failed')
subprocess.run([sys.executable, str(ROOT/'scripts/maintenance/check_portable_paths.py')], check=True)
# Frozen shape checks
expected={'pima':(614,154),'cleveland':(242,61),'ckd':(320,80)}
import pandas as pd
for ds,(ntr,nte) in expected.items():
    for s in range(5):
        d=BENCH/f'frozen_data/{ds}/split_{s}'
        tr=pd.read_csv(d/'real_train.csv'); te=pd.read_csv(d/'real_test.csv')
        assert len(tr)==ntr and len(te)==nte, (ds,s,len(tr),len(te))

# Final reported snapshot sanity checks (rounded means used in the thesis tables).
reported=pd.read_csv(BENCH/'results/final_thesis/reported_primary_means.csv')
def value(ds, method, col):
    x=reported[(reported['dataset']==ds)&(reported['method']==method)]
    assert len(x)==1, (ds,method)
    return float(x.iloc[0][col])
assert abs(value('pima','TVAE','AUROC')-0.789) < 5e-4
assert abs(value('cleveland','Conditional DDPM','AUROC')-0.880) < 5e-4
assert abs(value('ckd','ARF','AUROC')-0.996) < 5e-4
sens=pd.read_csv(BENCH/'results/final_thesis/reported_gan_sensitivity_best_alternatives.csv')
row=sens[(sens.dataset=='cleveland')&(sens.method=='CTGAN')].iloc[0]
assert abs(float(row.frozen_AUROC)-0.651) < 5e-4
assert abs(float(row.best_nonbaseline_AUROC)-0.607) < 5e-4
assert abs(float(row.delta_AUROC)-(-0.045)) < 5e-4

print('Repository verification PASS.')
print('Frozen splits: 3 datasets x 5 splits; Python compile PASS; portability PASS.')
