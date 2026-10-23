#!/usr/bin/env python3
"""Public entrypoint: audited run, source discovery, or application of frozen scores."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from metric_core import transform,bounded,combine,METRICS
ROOT=Path(__file__).resolve().parents[1]

def pick(root,name,preferred=None):
 if preferred:
  for p in preferred:
   if p.is_file():return p
 found=sorted([p for p in root.rglob(name) if ROOT not in p.parents and 'backup' not in str(p).lower() and 'legacy' not in str(p).lower()],key=lambda p:(p.stat().st_mtime_ns,str(p)),reverse=True)
 if not found:raise SystemExit(f'No {name} found below {root}; pass its exact path explicitly')
 # Do not guess among different live exports. Identical duplicate copies are harmless.
 hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in found}
 if len(hashes)>1:raise SystemExit('Multiple distinct '+name+' files; use an explicit argument:\n'+'\n'.join(map(str,found)))
 return found[0]

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 run=sub.add_parser('run');run.add_argument('--project-root',type=Path);run.add_argument('--master',type=Path);run.add_argument('--selected',type=Path);run.add_argument('--pairs',type=Path);run.add_argument('--reference',type=Path);run.add_argument('--output',type=Path,required=True);run.add_argument('--bootstrap',type=int,default=300);run.add_argument('--cv-repeats',type=int,default=3);run.add_argument('--audit-only',action='store_true');run.add_argument('--skip-plots',action='store_true');run.add_argument('--dry-run',action='store_true')
 app=sub.add_parser('apply-frozen');app.add_argument('--input',type=Path,required=True);app.add_argument('--calibration',type=Path,default=ROOT/'results/v1');app.add_argument('--output',type=Path,required=True)
 qa=sub.add_parser('complete');qa.add_argument('--source',type=Path,required=True);qa.add_argument('--output',type=Path,required=True);qa.add_argument('--bootstrap',type=int,default=300)
 val=sub.add_parser('validate');val.add_argument('--output',type=Path,required=True)
 plots=sub.add_parser('plots');plots.add_argument('--output',type=Path,required=True);plots.add_argument('--repair-empty',action='store_true')
 for command in ['migrate-figures','validate-figures']:
  op=sub.add_parser(command);op.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.command in ['migrate-figures','validate-figures']:
  from figure_layout import migrate_layout,validate_layout
  print(json.dumps((migrate_layout if a.command=='migrate-figures' else validate_layout)(a.output),indent=2));return
 if a.command=='plots':
  from plot_metric_analysis import plot_all
  plot_all(a.output,a.repair_empty);return
 if a.command=='validate':
  from complete_metric_analysis import validate_completion
  validate_completion(a.output);return
 if a.command=='complete':
  from complete_metric_analysis import complete
  complete(a.source,a.output,a.bootstrap);return
 if a.command=='run':
  if a.project_root:
   base=a.project_root/'analysis/image_classification';flat=base/'flatness_physics_analysis';morph=base/'morph_prod';old=base/'morphology_production'
   a.selected=a.selected or pick(flat,'analysis_dataset_selected.csv');a.pairs=a.pairs or pick(flat,'primary_matched_pairs.csv')
   a.reference=a.reference or pick(morph,'baseline_core.csv',[morph/'results/baselines/prod_20261001/baseline_core.csv',morph/'results/reference_baselines/prod_gro_final_20261001/baseline_core.csv'])
   a.master=a.master or pick(morph if morph.exists() else old,'master_with_morphology.csv')
  if any(getattr(a,k) is None for k in ['master','selected','pairs','reference']):p.error('Supply --project-root or all four explicit source files')
  command=[sys.executable,str(ROOT/'scripts/build_metric_analysis.py')]
  for k in ['master','selected','pairs','reference','output','bootstrap','cv_repeats']:command.extend(['--'+k.replace('_','-'),str(getattr(a,k))])
  for k in ['audit_only','skip_plots']:
   if getattr(a,k):command.append('--'+k.replace('_','-'))
  print(json.dumps({'command':command,'source_paths':{k:str(getattr(a,k).resolve()) for k in ['master','selected','pairs','reference']}},indent=2),flush=True)
  if not a.dry_run:subprocess.run(command,check=True)
 else:
  if a.output.exists():raise SystemExit('Output exists; choose a new file')
  modelpath=a.calibration/'data/frozen_parameters.json';model=json.loads(modelpath.read_text());u=pd.read_csv(a.input)
  if u.pair_id.duplicated().any():raise SystemExit('Duplicate pair identity')
  if not {'qc_B','qc_M'}.issubset(u.columns):raise SystemExit('QC status columns are required')
  cal=pd.read_csv(a.calibration/'data/unified_metrics.csv',nrows=1)
  for side in ['B','M']:
   for k in METRICS:
    col=f'{side}__{k}__pristine_reference'
    if col not in u or not np.allclose(u[col],cal[col].iloc[0],rtol=0,atol=1e-10):raise SystemExit('Pristine reference mismatch: '+col)
  values={}
  for key,state in model.items():
   if not isinstance(state,dict):continue
   view,method=key.split('__')
   for side in ['B','M']:
    X=u[[f'{side}__{k}__{view}' for k in state['names']]].to_numpy(float)
    if not np.isfinite(X).all():raise SystemExit('Missing required physical feature')
    for family,v in transform(X,state).items():values[f'Effect_{side}__{view}__{method}__{family}']=bounded(v)
  for side in ['B','M']:values[f'Effect_{side}__endpoint_vs_start__q95__single_dAPL']=bounded(abs(u[f'{side}__APL__endpoint_vs_start'].to_numpy())/model['dAPL_scale'])
  for b in list(values):
   if b.startswith('Effect_B__'):
    family=b.split('__',1)[1]
    for name,v in combine(values[b],values[b.replace('Effect_B__','Effect_M__')]).items():values[name+'__'+family]=v
  identity=u[['pair_id','molecule','qc_B','qc_M']].reset_index(drop=True).copy();identity['qc_eligible']=identity.qc_B.eq('PASS')&identity.qc_M.eq('PASS')
  result=pd.concat([identity,pd.DataFrame(values)],axis=1);a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False,float_format='%.12g')
  a.output.with_suffix('.provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'frozen_model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'rows':len(u),'refit':False,'no_jobs_submitted':True},indent=2))
  print('Applied frozen parameters without refitting:',a.output)
if __name__=='__main__':main()
