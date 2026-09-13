#!/usr/bin/env python3
"""Aggregate split-level results into final mean ± SD tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

METHOD_ORDER=['REAL','TVAE','CTGAN','CopulaGAN','ARF','Gaussian Copula','Conditional DDPM','LLM']
DATASET_ORDER=['pima','cleveland','ckd']
METRICS={
'utility_auroc':'AUROC ↑','utility_f1':'F1 ↑','utility_brier':'Brier ↓',
'fidelity_pcd':'PCD ↓','fidelity_ws':'WS ↓','fidelity_js':'JS ↓',
'privacy_dcr_mean':'DCR','privacy_mia_auc':'MIA AUROC','privacy_aia_risk':'AIA risk ↓'
}

def fmt(mean, sd):
    if pd.isna(mean): return '–'
    if pd.isna(sd): return f'{mean:.3f}'
    return f'{mean:.3f} ± {sd:.3f}'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--results_root',required=True); p.add_argument('--outdir',required=True); a=p.parse_args()
    root,out=Path(a.results_root),Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    files=list(root.rglob('main_results_row.csv'))
    if not files: raise SystemExit('No main_results_row.csv files found')
    frames=[]
    for f in files:
        d=pd.read_csv(f)
        # Derive split id from any path component named split_N.
        split=np.nan
        for part in f.parts:
            if part.startswith('split_'):
                try: split=int(part.split('_',1)[1])
                except: pass
        d['split_id']=split
        frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    df.to_csv(out/'all_main_results_split_level.csv',index=False)
    for c in METRICS:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    agg_spec={}
    for c in METRICS:
        agg_spec[c+'_mean']=(c,'mean'); agg_spec[c+'_sd']=(c,'std')
    agg=df.groupby(['dataset','method'],as_index=False).agg(**agg_spec)
    agg.to_csv(out/'main_results_summary_numeric.csv',index=False)
    for ds in DATASET_ORDER:
        d=agg[agg.dataset.astype(str).str.lower()==ds].copy()
        if d.empty: continue
        d['order']=d.method.apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999)
        d=d.sort_values('order')
        rows=[]
        for _,r in d.iterrows():
            row={'Method':r.method}
            for key,label in METRICS.items(): row[label]=fmt(r[key+'_mean'],r[key+'_sd'])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/f'{ds}_main_table_mean_sd.csv',index=False)
    print(f'Aggregated {len(df)} split-level rows into {len(agg)} dataset-method summaries')
if __name__=='__main__': main()
