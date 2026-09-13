#!/usr/bin/env python3
"""Evaluate the real-train → real-test utility reference with the same classifiers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import evaluate_release as E


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--real_train', required=True)
    p.add_argument('--real_test', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--numeric_cols', default=None)
    p.add_argument('--categorical_cols', default=None)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tr=pd.read_csv(a.real_train); te=pd.read_csv(a.real_test)[tr.columns]
    schema=E.infer_schema(tr,a.target,E.parse_columns(a.numeric_cols),E.parse_columns(a.categorical_cols))
    yr,yt=E.encode_binary_target(tr[a.target],te[a.target])
    # Real reference is TRTR: fit preprocessing on real train and transform real test.
    prep=E.build_preprocessor(schema)
    Xr=prep.fit_transform(tr.drop(columns=[a.target])); Xt=prep.transform(te.drop(columns=[a.target]))
    u=E.utility_tstr(Xr,yr,Xt,yt,a.seed)
    u.to_csv(out/'utility_by_classifier.csv',index=False)
    row={
      'dataset':a.dataset,'method':'REAL','synthetic_size_label':'NA',
      'n_real_train':len(tr),'n_real_test':len(te),'n_synthetic':'',
      'utility_auroc':float(u.auroc.mean()),'utility_f1':float(u.f1_at_0_5.mean()),
      'utility_brier':float(u.brier.mean()),
      'fidelity_pcd':'','fidelity_ws':'','fidelity_js':'',
      'privacy_dcr_mean':'','privacy_dcr_ratio':'','privacy_mia_auc':'','privacy_aia_risk':''
    }
    pd.DataFrame([row]).to_csv(out/'main_results_row.csv',index=False)
    (out/'metrics.json').write_text(json.dumps({'utility':row},indent=2),encoding='utf-8')
    print(pd.DataFrame([row]).to_string(index=False))
if __name__=='__main__': main()
