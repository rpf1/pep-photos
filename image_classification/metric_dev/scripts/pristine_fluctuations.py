#!/usr/bin/env python3
"""Read two pristine production trajectories using the existing Fourier kernel.
Inventory is default. --execute consumes an inspected JSON plan. No simulation,
source modification, conversion, submission or overwriting is performed.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CONTROL_ROOT=Path('/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall')
SPECS={'B':dict(folder='POPE-3_POPG-1(123Axy-206Az)/6_production',resnames=['POPE','POPG'],markers=['POPE','POPG'],total=512,po4=512), 'M':dict(folder='POPC-6_CHOL-4(127Axy-197Az)/6_production',resnames=['POPC','CHOL'],markers=['POPC'],total=720,po4=432)}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def stats_series(x,t,block_ns):
 """ACF initial-positive-sequence ESS plus contiguous physical-time blocks.
Short nonstationary records make ESS provisional; no replicate CI is claimed.
"""
 x=np.asarray(x,float);t=np.asarray(t,float);n=len(x)
 if n<2 or not np.isfinite(x).all() or not np.isfinite(t).all():raise ValueError('Finite time/metric series with >=2 observations required')
 if not np.allclose(np.diff(t),np.median(np.diff(t)),rtol=1e-3,atol=1e-6):raise ValueError('ACF ESS requires regular frame spacing; inspect trajectory sampling')
 z=x-x.mean();var=np.dot(z,z)
 if var==0:ess=float(n);tau=1.
 else:
  acf=np.correlate(z,z,'full')[n-1:]/var;positive=[]
  for k in range(1,n-1,2):
   pair=acf[k]+acf[k+1]
   if pair<=0:break
   positive.append(pair)
  tau=max(1.,1+2*sum(positive));ess=max(1.,min(n,n/tau))
 block=np.floor((t-t.min())/block_ns).astype(int);blocks=[x[block==b].mean() for b in sorted(set(block)) if np.sum(block==b)>=2]
 return dict(n_frames=n,mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)),MAD=float(np.median(abs(x-np.median(x)))),q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975)),effective_n_ACF=ess,tau_frames=tau,n_blocks=len(blocks),block_mean_sd=float(np.std(blocks,ddof=1)) if len(blocks)>1 else None,block_ns=block_ns,interpretation='within-trajectory fluctuation envelope; provisional ESS; no independent-replicate confidence interval')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-root',type=Path,default=CONTROL_ROOT);p.add_argument('--plan',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--stride-ps',type=float,default=100);p.add_argument('--block-ns',type=float,default=1);p.add_argument('--geometry-script',type=Path,default=ROOT.parent/'morph_prod/scripts/morphology_geometry.py');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Output exists; choose a new --out directory')
 if a.stride_ps<=0 or a.block_ns<=0:raise ValueError('Positive stride and block duration required')
 a.out.mkdir(parents=True)
 if not a.execute:
  plan={};inv=[]
  for side,s in SPECS.items():
   base=(a.control_root/s['folder']).resolve();gro=sorted(base.glob('*.gro'));xtc=sorted(base.glob('*.xtc'));tpr=sorted(base.glob('*.tpr'))
   inv.append(dict(side=side,production_directory=str(base),exists=base.is_dir(),gro=[str(x) for x in gro],xtc=[str(x) for x in xtc],tpr=[str(x) for x in tpr]))
   plan[side]=dict(gro=str(gro[0]) if len(gro)==1 else 'SELECT_EXACT_GRO_FROM_INVENTORY',xtc=str(xtc[0]) if len(xtc)==1 else 'SELECT_EXACT_PRODUCTION_XTC_FROM_INVENTORY')
  (a.out/'inventory.json').write_text(json.dumps(inv,indent=2)+'\n');(a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('Inventory written. Inspect exact GRO/XTC pairing before --execute.');return
 if not a.plan:raise ValueError('--execute requires an inspected --plan JSON')
 import MDAnalysis as mda,importlib.util
 spec=importlib.util.spec_from_file_location('pristine_geometry',a.geometry_script);G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
 plan=json.loads(a.plan.read_text());assert set(plan)=={'B','M'};summaries=[];inputs=[]
 for side,s in SPECS.items():
  paths={k:Path(plan[side][k]).resolve() for k in ['gro','xtc']};expected=(a.control_root/s['folder']).resolve()
  for path in paths.values():
   if not path.is_file() or path.parent!=expected:raise ValueError('Input must be a file directly in declared 6_production: '+str(path))
  hashes={k:digest(v) for k,v in paths.items()};inputs.append(dict(side=side,paths={k:str(v) for k,v in paths.items()},sha256=hashes))
  u=mda.Universe(str(paths['gro']),str(paths['xtc']),to_guess=());counts={n:len(u.select_atoms('resname '+n).residues) for n in s['resnames']}
  if sum(counts.values())!=s['total']:raise ValueError(f'{side}: composition/count differs from matched reference: {counts}')
  allowed_resnames=set(s['resnames'])|{'W','WF','NA','CL','NA+','CL-','ION'}
  unexpected=set(map(str,u.residues.resnames))-allowed_resnames
  if unexpected:raise ValueError('Unexpected non-pristine residue names; inspect topology: '+str(sorted(unexpected)))
  po4=u.select_atoms('name PO4 and resname '+' '.join(s['markers']))
  if len(po4)!=s['po4'] or len(np.unique(po4.resindices))!=len(po4):raise ValueError('Unexpected PO4 marker count')
  rows=[];last=-np.inf
  for ts in u.trajectory:
   if ts.time-last<a.stride_ps-1e-5:continue
   last=float(ts.time);geom=G.build_surface_geometry(po4.positions.copy(),u.dimensions.copy(),min_wavelength_nm=4.,grid_n=32);metrics=G.geometry_metrics(geom,s['total']);rows.append(dict(side=side,time_ns=last/1000,frame_index=int(ts.frame),**metrics))
  f=pd.DataFrame(rows)
  if len(f)<10 or np.any(np.diff(f.time_ns)<=0):raise ValueError('Too few frames or nonmonotonic times')
  f.to_csv(a.out/f'pristine_frames_{side}.csv',index=False)
  names=['projected_APL_A2','projected_area_nm2','upper_surface_area_nm2','lower_surface_area_nm2','midplane_surface_area_nm2','height_rms_nm','thickness_median_nm','H_rms_nm_inv','height_rms_upper_minus_lower_nm','surface_area_upper_minus_lower_nm2','H_rms_upper_minus_lower_nm_inv']
  for name in names:summaries.append(dict(side=side,variable=name,duration_ns=float(f.time_ns.max()-f.time_ns.min()),**stats_series(f[name],f.time_ns,a.block_ns)))
  if any(digest(paths[k])!=v for k,v in hashes.items()):raise RuntimeError('Source bytes changed during extraction')
 pd.DataFrame(summaries).to_csv(a.out/'pristine_fluctuation_summary.csv',index=False)
 (a.out/'provenance.json').write_text(json.dumps(dict(inputs=inputs,geometry_sha256=digest(a.geometry_script),kernel='existing production Fourier',grid_n=32,min_wavelength_nm=4,stride_ps=a.stride_ps,block_ns=a.block_ns,units='MDAnalysis Angstrom/ps converted by existing kernel to nm',limitations='one short control per membrane; no universal null calibration; no noise-normalized cohort score fitted'),indent=2)+'\n')
 print('Pristine extraction complete; return output directory for separate noise-score feasibility review.')
if __name__=='__main__':main()
