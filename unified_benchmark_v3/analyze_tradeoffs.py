#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASETS=['pima','cleveland','ckd']
GAN_METHODS=['CTGAN','CopulaGAN']
RAW_METRICS=['utility_auroc','utility_f1','utility_brier','fidelity_pcd','fidelity_ws','fidelity_js','privacy_dcr_mean','privacy_mia_auc','privacy_aia_risk']
UTILITY={'utility_auroc':'high','utility_f1':'high','utility_brier':'low'}
FIDELITY={'fidelity_pcd':'low','fidelity_ws':'low','fidelity_js':'low'}
PRIVACY={'privacy_mia_distance':'low','privacy_aia_risk':'low'}
DELTA_SPECS={
 'delta_auroc_improvement':('utility_auroc','high'),
 'delta_f1_improvement':('utility_f1','high'),
 'delta_brier_improvement':('utility_brier','low'),
 'delta_pcd_improvement':('fidelity_pcd','low'),
 'delta_ws_improvement':('fidelity_ws','low'),
 'delta_js_improvement':('fidelity_js','low'),
 'delta_mia_closeness_improvement':('privacy_mia_auc','mia'),
 'delta_aia_improvement':('privacy_aia_risk','low'),
}

def one(path):
 d=pd.read_csv(path)
 if len(d)!=1: raise ValueError(f'{path}: expected 1 row, found {len(d)}')
 return d.iloc[0].to_dict()

def meta(path):
 p=path.parent.parent/'generation_metadata.json'
 if not p.exists(): return {}
 try: return json.loads(p.read_text())
 except Exception as e:
  print(f'WARNING: cannot read {p}: {e}'); return {}

def numeric(df):
 d=df.copy()
 for c in RAW_METRICS:
  d[c]=pd.to_numeric(d[c],errors='coerce')
 d['privacy_mia_distance']=(d['privacy_mia_auc']-0.5).abs()
 return d

def manifest(path):
 m=pd.read_csv(path)
 req={'config_id','root','discriminator_lr','is_baseline'}
 miss=req-set(m.columns)
 if miss: raise ValueError(f'Manifest missing {sorted(miss)}')
 if m.config_id.duplicated().any(): raise ValueError('config_id must be unique')
 m['is_baseline']=m['is_baseline'].astype(int)
 if m.is_baseline.sum()!=1: raise ValueError('Exactly one baseline required')
 return m

def collect_gan(m):
 rows=[]; missing=[]
 for _,cfg in m.iterrows():
  root=Path(str(cfg.root)); cid=str(cfg.config_id)
  for ds in DATASETS:
   for method in GAN_METHODS:
    for split in range(5):
     f=root/ds/f'split_{split}'/method/'evaluation'/'main_results_row.csv'
     if not f.exists(): missing.append(str(f)); continue
     r=one(f)
     if str(r.get('dataset','')).lower()!=ds: raise ValueError(f'{f}: dataset mismatch')
     if str(r.get('method',''))!=method: raise ValueError(f'{f}: method mismatch')
     md=meta(f)
     r.update(dataset=ds,method=method,split_id=split,config_id=cid,
              config_root=str(root),is_baseline=bool(int(cfg.is_baseline)),
              declared_discriminator_lr=float(cfg.discriminator_lr),
              batch_size=md.get('batch_size',np.nan),epochs=md.get('epochs',np.nan),
              seed=md.get('seed',np.nan),notes=str(cfg.get('notes','')),_path=str(f))
     rows.append(r)
 if missing:
  print('MISSING GAN FILES:'); [print(' -',x) for x in missing]
  raise SystemExit(f'Refusing analysis: {len(missing)} GAN result files missing')
 d=pd.DataFrame(rows)
 exp=len(m)*3*2*5
 if len(d)!=exp: raise SystemExit(f'Expected {exp} GAN rows, found {len(d)}')
 key=['config_id','dataset','method','split_id']
 if d.duplicated(key).any(): raise SystemExit('Duplicate GAN config rows detected')
 return numeric(d)

def collect_frozen(root):
 methods=['REAL','TVAE','CTGAN','CopulaGAN','ARF','Gaussian Copula','Conditional DDPM','LLM']
 rows=[]; errors=[]
 for ds in DATASETS:
  files=list((root/ds).rglob('main_results_row.csv'))
  for f in files:
   r=one(f)
   method=str(r.get('method','')); dataset=str(r.get('dataset','')).lower()
   if dataset!=ds:
    errors.append(f'{f}: dataset={dataset}, expected {ds}'); continue
   if method not in methods:
    errors.append(f'{f}: unexpected method={method}'); continue
   split=None
   for part in f.parts:
    if part.startswith('split_'):
     try: split=int(part.split('_',1)[1])
     except ValueError: pass
   if split is None:
    errors.append(f'{f}: no split_N component'); continue
   r.update(dataset=ds,method=method,split_id=split,_path=str(f)); rows.append(r)
 d=pd.DataFrame(rows)
 for ds in DATASETS:
  for method in methods:
   sub=d[(d.dataset==ds)&(d.method==method)] if len(d) else pd.DataFrame()
   splits=sorted(sub.split_id.tolist()) if len(sub) else []
   if len(sub)!=5 or set(splits)!=set(range(5)):
    errors.append(f'{ds}/{method}: expected 5 splits 0..4, found n={len(sub)} splits={splits}')
 if errors:
  print('FROZEN INVENTORY ERRORS:'); [print(' -',x) for x in errors]
  raise SystemExit('Refusing frozen analysis because the primary inventory is incomplete or inconsistent')
 if len(d)!=120: raise SystemExit(f'Expected 120 frozen rows, found {len(d)}')
 return numeric(d)

def aggregate(df, groups):
 metrics=RAW_METRICS+['privacy_mia_distance']; spec={}
 for c in metrics:
  spec[c+'_mean']=(c,'mean'); spec[c+'_sd']=(c,'std')
 return df.groupby(groups,as_index=False).agg(**spec)

def add_scores(summary, groups):
 d=summary.copy()
 d['_n']=d.groupby(groups)[groups[0]].transform('size')
 defs={**UTILITY,**FIDELITY,**PRIVACY}
 for metric,direction in defs.items():
  col=metric+'_mean'; rank=metric+'_rank'; asc=direction=='low'
  d[rank]=d.groupby(groups)[col].rank(method='average',ascending=asc)
 util=[x+'_rank' for x in UTILITY]; fid=[x+'_rank' for x in FIDELITY]; priv=[x+'_rank' for x in PRIVACY]
 d['utility_rank']=d[util].mean(axis=1); d['fidelity_rank']=d[fid].mean(axis=1); d['privacy_rank']=d[priv].mean(axis=1)
 den=(d['_n'].astype(float)-1).replace(0,np.nan)
 for dim in ['utility','fidelity','privacy']:
  d[dim+'_score']=(1-(d[dim+'_rank']-1)/den).fillna(1.0)
 d['balanced_mean_score']=d[['utility_score','fidelity_score','privacy_score']].mean(axis=1)
 d['balanced_min_score']=d[['utility_score','fidelity_score','privacy_score']].min(axis=1)
 return d.drop(columns='_n')

def pareto_mask(df):
 cols=['utility_score','fidelity_score','privacy_score']; x=df[cols].to_numpy(float); keep=np.ones(len(df),bool)
 for i in range(len(df)):
  if not np.all(np.isfinite(x[i])): keep[i]=False; continue
  for j in range(len(df)):
   if i==j or not np.all(np.isfinite(x[j])): continue
   if np.all(x[j]>=x[i]) and np.any(x[j]>x[i]): keep[i]=False; break
 return keep

def add_pareto(scored,groups):
 parts=[]
 for _,s in scored.groupby(groups,sort=False):
  z=s.copy(); z['pareto_efficient']=pareto_mask(z); parts.append(z)
 return pd.concat(parts,ignore_index=True)

def deltas(gan,baseline):
 base=gan[gan.config_id==baseline][['dataset','method','split_id']+RAW_METRICS].copy()
 base=base.rename(columns={c:c+'_baseline' for c in RAW_METRICS})
 m=gan.merge(base,on=['dataset','method','split_id'],how='left',validate='many_to_one')
 for new,(metric,mode) in DELTA_SPECS.items():
  b=m[metric+'_baseline']; x=m[metric]
  if mode=='high': m[new]=x-b
  elif mode=='low': m[new]=b-x
  else: m[new]=(b-0.5).abs()-(x-0.5).abs()
 m['delta_dcr_raw']=m['privacy_dcr_mean']-m['privacy_dcr_mean_baseline']
 nb=m[m.config_id!=baseline].copy(); dcols=list(DELTA_SPECS)+['delta_dcr_raw']; spec={}
 for c in dcols: spec[c+'_mean']=(c,'mean'); spec[c+'_sd']=(c,'std')
 s=nb.groupby(['dataset','method','config_id'],as_index=False).agg(**spec)
 cnt=[]
 for keys,g in nb.groupby(['dataset','method','config_id']):
  r=dict(zip(['dataset','method','config_id'],keys)); r['n_splits']=len(g)
  for c in DELTA_SPECS:
   r[c+'_n_improved']=int((g[c]>0).sum()); r[c+'_n_worse']=int((g[c]<0).sum())
  cnt.append(r)
 return m,s.merge(pd.DataFrame(cnt),on=['dataset','method','config_id'],validate='one_to_one')

def strategies(scored,groups,label):
 rows=[]
 for keys,s in scored.groupby(groups,sort=False):
  if not isinstance(keys,tuple): keys=(keys,)
  base=dict(zip(groups,keys))
  picks={}
  for name,col in [('utility_first','utility_score'),('fidelity_first','fidelity_score'),('privacy_first','privacy_score')]:
   picks[name]=s.sort_values([col,'balanced_mean_score'],ascending=[False,False]).iloc[0]
  p=s[s.pareto_efficient].copy()
  if p.empty: p=s.copy()
  picks['balanced_pareto']=p.sort_values(['balanced_min_score','balanced_mean_score'],ascending=[False,False]).iloc[0]
  for name,r in picks.items():
   q=dict(base); q.update(strategy=name,**{label:r[label]},utility_score=r.utility_score,fidelity_score=r.fidelity_score,
                         privacy_score=r.privacy_score,balanced_min_score=r.balanced_min_score,
                         balanced_mean_score=r.balanced_mean_score,pareto_efficient=bool(r.pareto_efficient)); rows.append(q)
 return pd.DataFrame(rows)

def pairplots(scored,ds,label,prefix):
 s=scored[scored.dataset==ds].copy(); pairs=[('utility_score','fidelity_score'),('utility_score','privacy_score'),('fidelity_score','privacy_score')]
 for x,y in pairs:
  fig,ax=plt.subplots(figsize=(7.2,5.2))
  for method,g in s.groupby('method',sort=False):
   ax.scatter(g[x],g[y],s=75,label=method)
   for _,r in g.iterrows(): ax.annotate(str(r[label]),(r[x],r[y]),xytext=(4,4),textcoords='offset points',fontsize=7)
  p=s[s.pareto_efficient]
  if len(p): ax.scatter(p[x],p[y],s=170,facecolors='none',linewidths=1.5,label='Pareto-efficient')
  ax.set_xlabel(x.replace('_',' ').title()+' (higher = better)'); ax.set_ylabel(y.replace('_',' ').title()+' (higher = better)')
  ax.set_title(f'{ds.upper()}: {x.replace("_score","")} vs {y.replace("_score","")}'); ax.set_xlim(-.08,1.08); ax.set_ylim(-.08,1.08); ax.legend(fontsize=8); fig.tight_layout()
  fig.savefig(prefix.parent/f'{prefix.name}_{x.replace("_score","")}_vs_{y.replace("_score","")}.png',dpi=200); plt.close(fig)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default=str(Path(__file__).resolve().parent / 'tradeoff_analysis' / 'gan_tradeoff_manifest.csv')); ap.add_argument('--frozen_root',default=str(Path(__file__).resolve().parent / 'results' / 'main_benchmark')); ap.add_argument('--outdir',default=str(Path(__file__).resolve().parent / 'results' / 'tradeoff_analysis')); a=ap.parse_args()
 out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); figdir=out/'figures'; figdir.mkdir(exist_ok=True)
 man=manifest(Path(a.manifest)); baseline=str(man.loc[man.is_baseline==1,'config_id'].iloc[0])

 gan=collect_gan(man); gan.to_csv(out/'gan_tradeoff_split_level.csv',index=False)
 gs=aggregate(gan,['dataset','method','config_id'])
 md=gan.groupby(['dataset','method','config_id'],as_index=False).agg(batch_size=('batch_size','first'),declared_discriminator_lr=('declared_discriminator_lr','first'),epochs=('epochs','first'),is_baseline=('is_baseline','first'),notes=('notes','first'))
 gs=gs.merge(md,on=['dataset','method','config_id'],validate='one_to_one'); gs.to_csv(out/'gan_tradeoff_config_summary.csv',index=False)
 gscore=add_pareto(add_scores(gs,['dataset','method']),['dataset','method']); gscore.to_csv(out/'gan_tradeoff_dimension_scores.csv',index=False); gscore[gscore.pareto_efficient].to_csv(out/'gan_pareto_front.csv',index=False)
 strategies(gscore,['dataset','method'],'config_id').to_csv(out/'gan_strategy_recommendations.csv',index=False)
 ps,pm=deltas(gan,baseline); ps.to_csv(out/'gan_paired_deltas_split_level.csv',index=False); pm.to_csv(out/'gan_paired_deltas_summary.csv',index=False)

 frozen=collect_frozen(Path(a.frozen_root)); frozen.to_csv(out/'frozen_primary_split_level.csv',index=False); fs=aggregate(frozen,['dataset','method'])
 real=fs[fs.method=='REAL'][['dataset','utility_auroc_mean','utility_f1_mean','utility_brier_mean']].rename(columns={'utility_auroc_mean':'real_auroc_mean','utility_f1_mean':'real_f1_mean','utility_brier_mean':'real_brier_mean'})
 syn=fs[fs.method!='REAL'].merge(real,on='dataset',validate='many_to_one'); syn['delta_auroc_vs_real']=syn.utility_auroc_mean-syn.real_auroc_mean; syn['delta_f1_vs_real']=syn.utility_f1_mean-syn.real_f1_mean; syn['brier_excess_vs_real']=syn.utility_brier_mean-syn.real_brier_mean
 fscore=add_pareto(add_scores(syn,['dataset']),['dataset']); fscore.to_csv(out/'frozen_global_tradeoff_scores.csv',index=False); fscore[fscore.pareto_efficient].to_csv(out/'frozen_global_pareto_front.csv',index=False); strategies(fscore,['dataset'],'method').to_csv(out/'frozen_global_strategy_recommendations.csv',index=False)

 # Exploratory overlay: rank frozen methods + tuned GAN configs together, but keep explicit type label.
 fo=fscore.copy(); fo['candidate']=fo.method; fo['candidate_type']='frozen_primary'
 tuned=gs[~gs.is_baseline].copy(); tuned['candidate']=tuned.method+'::'+tuned.config_id; tuned['candidate_type']='exploratory_gan_tuning'
 metriccols=[c for c in gs.columns if c.endswith('_mean') or c.endswith('_sd')]
 cols=['dataset','method','candidate','candidate_type']+metriccols
 overlay=pd.concat([fo[[c for c in cols if c in fo.columns]],tuned[[c for c in cols if c in tuned.columns]]],ignore_index=True)
 oscore=add_pareto(add_scores(overlay,['dataset']),['dataset']); oscore.to_csv(out/'exploratory_overlay_tradeoff_scores.csv',index=False); oscore[oscore.pareto_efficient].to_csv(out/'exploratory_overlay_pareto_front.csv',index=False)

 for ds in DATASETS:
  pairplots(gscore,ds,'config_id',figdir/f'gan_{ds}'); pairplots(fscore,ds,'method',figdir/f'frozen_{ds}')

 print('\nGAN STRATEGIES'); print(strategies(gscore,['dataset','method'],'config_id').sort_values(['dataset','method','strategy']).round(3).to_string(index=False))
 print('\nFROZEN BENCHMARK STRATEGIES'); print(strategies(fscore,['dataset'],'method').sort_values(['dataset','strategy']).round(3).to_string(index=False))
 print('\nSaved:',out)
 print('NOTE: scores are descriptive within-dataset rank scores, not universal validated composite metrics.')

if __name__=='__main__': main()
