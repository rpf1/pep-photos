#!/usr/bin/env python3
"""Consolidate existing exports; preserve frozen transforms; no trajectory fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(k,'1')
import argparse,json,hashlib,sys,warnings,itertools,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix,silhouette_score
from metric_core import transform,bounded,bh,MEMBRANES,combine
from review_saved_pca import frozen_projection,resolve_palette
warnings.filterwarnings('ignore',category=stats.ConstantInputWarning)
SEED=20261006
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,Path):return str(x)
 return x

def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2)+'\n')
def table(out,name,d):
 d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)
 p=out/'tables'/name;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False);return d

def corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<4 or np.std(x[ok])<1e-15 or np.std(y[ok])<1e-15:return ok.sum(),np.nan,np.nan
 r,p=stats.spearmanr(x[ok],y[ok]);return int(ok.sum()),float(r),float(p)

def bootstrap_corr(x,y,nboot=499):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<8:return np.nan,np.nan
 rng=np.random.default_rng(SEED);rs=[]
 for _ in range(nboot):
  ix=rng.integers(0,len(x),len(x));rs.append(corr(x[ix],y[ix])[1])
 return np.nanquantile(rs,[.025,.975])

def consolidate(project,result,out,u):
 prod=project/'analysis/image_classification/morph_prod/slurm/production_20260930_120836';mf=pd.read_csv(prod/'manifest.csv');assert not mf.physical_system_key.duplicated().any()
 mf=mf.set_index('physical_system_key');rows=[];missing=[];couplings=[];bins=[];qc=[]
 for _,r in u.iterrows():
  for side,mem in MEMBRANES.items():
   key=r['physical_key_'+side];assert key==f'{r.molecule}|{r.nmol}|{mem}|{r.physical_run}'
   if key not in mf.index:missing.append(dict(pair_id=r.pair_id,side=side,reason='missing morphology manifest'));continue
   m=mf.loc[key];assert m.molecule==r.molecule and m.physical_run==r.physical_run and m.bilayer==mem and m.manual_label==r['label_'+side]
   task=prod/'tasks'/m.manifest_row_id;f=pd.read_csv(task/'frames.csv');b=pd.read_csv(task/'baseline.csv').iloc[0];res=json.loads((task/'result.json').read_text());pr=json.loads((task/'provenance.json').read_text());assert res['physical_system_key']==key and pr['physical_system_key']==key
   assert set(f.stage)=={'Relax'} and f.time_ns.min()==0 and f.time_ns.max()==100 and not f.time_ns.duplicated().any()
   late=f[f.time_ns.between(80,100)];assert len(late)==21 and set(late.time_ns)==set(range(80,101))
   q=dict(pair_id=r.pair_id,molecule=r.molecule,side=side,membrane=mem,physical_run=r.physical_run,phenotype=r['label_'+side],physical_key=key,task_id=m.manifest_row_id,n_late_frames=len(late),late_start_ns=80,late_end_ns=100,denominator_total=int(m.expected_membrane_total),surface_source=str(task.relative_to(project)/'frames.csv'),baseline_source=str(task.relative_to(project)/'baseline.csv'),qc_status=res['qc'].get('qc_status',str(res['qc'])))
   q['pAPL_endpoint_A2']=r[f'{side}__APL__endpoint'];q['pAPL_start_A2']=r[f'{side}__APL__system_start'];q['pAPL_reference_A2']=r[f'{side}__APL__pristine_reference']
   q['delta_pAPL_endpoint_start_A2']=q['pAPL_endpoint_A2']-q['pAPL_start_A2'];q['delta_pAPL_endpoint_ref_A2']=q['pAPL_endpoint_A2']-q['pAPL_reference_A2']
   q['pAPL_late_A2']=late.projected_APL_A2.mean();q['pAPL_late_median_A2']=late.projected_APL_A2.median();q['delta_pAPL_late_ref_A2']=q['pAPL_late_A2']-q['pAPL_reference_A2'];q['delta_pAPL_late_start_A2']=q['pAPL_late_A2']-q['pAPL_start_A2'];q['pAPL_late_std_1ns_A2']=late.projected_APL_A2.std()
   q['frame_pAPL_endpoint_error_A2']=f.iloc[-1].projected_APL_A2-q['pAPL_endpoint_A2']
   # Means for additive area decomposition; medians separately retained for historical morphology.
   q['A_projected_late_nm2']=late.projected_area_nm2.mean();q['A_projected_baseline_nm2']=b.projected_area_nm2
   for s in ['upper','lower','midplane']:
    c=s+'_surface_area_nm2';q[f'A_surface_{s}_late_nm2']=late[c].mean();q[f'A_surface_{s}_baseline_nm2']=b[c];q[f'A_surface_{s}_endpoint_nm2']=f.iloc[-1][c]
    q[f'delta_A_surface_{s}_late_baseline_nm2']=late[c].mean()-b[c]
    q[f'geometry_{s}_late_nm2']=(late[c]-late.projected_area_nm2).mean();q[f'delta_geometry_{s}_late_baseline_nm2']=q[f'geometry_{s}_late_nm2']-(b[c]-b.projected_area_nm2)
   q['A_surface_mean_late_nm2']=(q['A_surface_upper_late_nm2']+q['A_surface_lower_late_nm2'])/2
   q['delta_A_surface_mean_late_baseline_nm2']=(q['delta_A_surface_upper_late_baseline_nm2']+q['delta_A_surface_lower_late_baseline_nm2'])/2
   q['geometry_mean_late_nm2']=q['A_surface_mean_late_nm2']-q['A_projected_late_nm2'];q['delta_geometry_mean_late_baseline_nm2']=(q['delta_geometry_upper_late_baseline_nm2']+q['delta_geometry_lower_late_baseline_nm2'])/2
   q['surface_asymmetry_late_nm2']=q['A_surface_upper_late_nm2']-q['A_surface_lower_late_nm2'];q['surface_abs_asymmetry_late_nm2']=abs(q['surface_asymmetry_late_nm2'])
   q['upper_geometry_share']=q['geometry_upper_late_nm2']/(q['geometry_upper_late_nm2']+q['geometry_lower_late_nm2'])
   q['sAPL_proxy_late_A2']=200*q['A_surface_mean_late_nm2']/q['denominator_total'];q['sAPL_proxy_baseline_A2']=100*(b.upper_surface_area_nm2+b.lower_surface_area_nm2)/q['denominator_total'];q['delta_sAPL_proxy_late_baseline_A2']=q['sAPL_proxy_late_A2']-q['sAPL_proxy_baseline_A2']
   q['geometry_area_contribution_A2']=200*q['geometry_mean_late_nm2']/q['denominator_total']
   q['delta_geometry_area_contribution_A2']=200*q['delta_geometry_mean_late_baseline_nm2']/q['denominator_total']
   for short in ['height','thickness','curvature','height_asym','excess','curvature_asym']:
    q[f'delta_{short}_late_ref']=r[f'{side}__{short}__late_vs_ref']
   q['delta_height_asym_signed_late_ref']=r[f'{side}__height_asym_signed__late_vs_ref']
   # Historical transformations from supplied original features; no calibration fit.
   for view in ['late_vs_ref','endpoint_vs_ref']:
    for scale in ['q95','null_mad','iqr']:
     st=json.loads((result/'data/frozen_parameters.json').read_text())[view+'__'+scale]
     X=np.array([[r[f'{side}__{feat}__{view}'] for feat in st['names']]])
     D=transform(X,st)['euclidean_reduced'][0];q[f'physical_deviation_{view}_{scale}']=D/(1+D)
     if scale=='q95':
      q[f'physical_distance_{view}_q95']=D
      for feat,z in zip(st['names'],(X/np.asarray(st['scale']))[0]):q[f'q95_z_{view}_{feat}']=z
   # Strict physical-run mapping. Never substitute run_0 for a cycle run.
   ap=project/'analysis/output'/(r.molecule+'_80')/mem/r.physical_run/'binned_APL.csv';q['binned_pAPL_available']=ap.exists()
   if ap.exists():
    a=pd.read_csv(ap);sp=ap.with_name('binned_APL_slope.csv');sl=pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    assert (a.n_frames==a.pull_frames+a.relax_frames).all()
    mixed=a[(a.pull_frames>0)&(a.relax_frames>0)]
    offset=float(mixed.bin_start_ns.iloc[0]) if len(mixed)==1 else float(a[a.relax_frames>0].bin_start_ns.min())
    q['pAPL_relax_offset_ns']=offset;q['pAPL_binned_source']=str(ap.relative_to(project));q['pAPL_slope_source']=str(sp.relative_to(project)) if sp.exists() else ''
    a['local_start_ns']=a.bin_start_ns-offset;a['local_end_ns']=a.bin_end_ns-offset
    la=a[(a.local_start_ns>=80)&(a.local_end_ns<=100)&(a.pull_frames==0)]
    assert len(la)==4 and la.n_frames.sum()==200
    q['pAPL_late_binned_A2']=np.average(la.apl_mean,weights=la.n_frames);q['pAPL_binned_minus_1ns_late_A2']=q['pAPL_late_binned_A2']-q['pAPL_late_A2']
    ar=a[(a.pull_frames==0)&(a.local_start_ns>=0)&(a.local_end_ns<=100)&(a.n_frames>1)].copy()
    early=ar[ar.local_end_ns<=20];q['pAPL_early_binned_A2']=np.average(early.apl_mean,weights=early.n_frames);q['pAPL_early_late_change_A2']=q['pAPL_late_binned_A2']-q['pAPL_early_binned_A2']
    q['pAPL_binmean_range_A2']=ar.apl_mean.max()-ar.apl_mean.min();q['pAPL_binmean_std_A2']=ar.apl_mean.std();q['pAPL_max_excursion_ref_A2']=np.max(abs(ar.apl_mean-q['pAPL_reference_A2']));q['pAPL_positive_bin_fraction']=np.mean(ar.apl_mean>q['pAPL_reference_A2'])
    if not sl.empty:
     assert np.allclose(sl.delta_apl,np.diff(a.apl_mean))
     valid=sl[(sl.segment_start_ns>=offset+5)&(sl.segment_end_ns<=offset+100)]
     q['pAPL_max_abs_slope_A2_per_ns']=valid.slope_apl_per_ns.abs().max();q['pAPL_major_slope_local_ns']=valid.loc[valid.slope_apl_per_ns.abs().idxmax(),'x_ns']-offset
    for _,ab in ar.iterrows():
     ff=f[(f.time_ns>=ab.local_start_ns)&(f.time_ns<ab.local_end_ns)]
     row=dict(pair_id=r.pair_id,side=side,local_start_ns=ab.local_start_ns,pAPL_A2=ab.apl_mean,geometry_nm2=((ff.upper_surface_area_nm2+ff.lower_surface_area_nm2)/2-ff.projected_area_nm2).mean(),height_nm=ff.height_rms_nm.mean(),thickness_nm=ff.thickness_median_nm.mean(),surface_mean_nm2=((ff.upper_surface_area_nm2+ff.lower_surface_area_nm2)/2).mean());bins.append(row)
   else:missing.append(dict(pair_id=r.pair_id,side=side,reason='no exact-run binned pAPL; matched 1 ns pAPL available from morphology frames',expected_path=str(ap.relative_to(project))))
   rows.append(q);qc.append(dict(pair_id=r.pair_id,side=side,**res['qc']))
 d=pd.DataFrame(rows);assert len(d)==312 and not d.duplicated(['pair_id','side']).any();table(out,'endpoint_pAPL_reconciliation.csv',d[['pair_id','side','pAPL_endpoint_A2','frame_pAPL_endpoint_error_A2']]);print('Endpoint pAPL max difference',np.max(abs(d.frame_pAPL_endpoint_error_A2)),flush=True)
 table(out,'matched_late_dataset.csv',d);table(out,'consolidated_curved_surface.csv',d[[c for c in d if c in ['pair_id','molecule','side','phenotype','physical_run','denominator_total','surface_source','baseline_source','n_late_frames'] or 'surface' in c or 'geometry' in c or c.startswith('A_')]])
 table(out,'matched_late_pAPL.csv',d[[c for c in d if c in ['pair_id','molecule','side','phenotype','physical_run','n_late_frames'] or 'pAPL' in c]])
 table(out,'missing_inputs.csv',missing);table(out,'production_qc.csv',qc);bb=table(out,'temporal_bin_coupling_source.csv',bins)
 for (pid,s),g in bb.groupby(['pair_id','side']):
  for y in ['height_nm','thickness_nm','surface_mean_nm2','geometry_nm2']:
   n,rho,p=corr(g.pAPL_A2,g[y]);couplings.append(dict(pair_id=pid,side=s,variable=y,n_bins=n,rho=rho,inference='descriptive within trajectory; no frame-level p-value'))
 table(out,'temporal_coupling.csv',couplings)
 return d

def analyses(out,d,u,result):
 variables=['pAPL_late_A2','delta_pAPL_late_ref_A2','delta_pAPL_late_start_A2','A_surface_mean_late_nm2','delta_A_surface_mean_late_baseline_nm2','geometry_mean_late_nm2','delta_geometry_mean_late_baseline_nm2','delta_height_late_ref','delta_thickness_late_ref','delta_curvature_late_ref','surface_abs_asymmetry_late_nm2','delta_height_asym_signed_late_ref','physical_deviation_late_vs_ref_q95']
 rows=[]
 for side,g in d.groupby('side'):
  for label,h in [('ALL',g)]+list(g.groupby('phenotype')):
   for a,b in itertools.combinations(variables,2):
    n,r,p=corr(h[a],h[b]);slope=np.nan
    if n>=8 and np.ptp(h[a])>0:slope=stats.theilslopes(h[b],h[a])[0]
    lo,hi=bootstrap_corr(h[a],h[b],199) if label=='ALL' else (np.nan,np.nan)
    rows.append(dict(side=side,phenotype=label,x=a,y=b,n=n,rho=r,p=p,CI95_low=lo,CI95_high=hi,theil_slope=slope,status='inferential exploratory' if n>=8 else 'descriptive small group'))
 rr=pd.DataFrame(rows);rr['q_BH']=np.nan
 for _,ix in rr.groupby(['side','phenotype']).groups.items():rr.loc[ix,'q_BH']=bh(rr.loc[ix,'p'])
 table(out,'phenotype_relationships.csv',rr)
 # OLS phenotype x projected response, restricted to >=8 in each membrane; HC3 robust covariance.
 ir=[]
 for side,g in d.groupby('side'):
  valid=g.phenotype.value_counts();g=g[g.phenotype.isin(valid[valid>=8].index)].copy()
  for y in ['delta_thickness_late_ref','geometry_mean_late_nm2','delta_A_surface_mean_late_baseline_nm2']:
   x=g.delta_pAPL_late_ref_A2.to_numpy();lab=pd.get_dummies(g.phenotype,drop_first=True,dtype=float);D=lab.to_numpy();X=np.c_[np.ones(len(g)),x,D,D*x[:,None]];yy=g[y].to_numpy();inverse=np.linalg.pinv(X.T@X);beta=inverse@X.T@yy;residual=yy-X@beta;hat=np.einsum('ij,jk,ik->i',X,inverse,X);sandwich=inverse@(X.T@((residual/(1-hat))[:,None]**2*X))@inverse;terms=np.arange(2+D.shape[1],X.shape[1]);W=float(beta[terms]@np.linalg.pinv(sandwich[np.ix_(terms,terms)])@beta[terms]);p=float(stats.chi2.sf(W,len(terms)));R2=1-np.sum(residual**2)/np.sum((yy-yy.mean())**2)
   ir.append(dict(side=side,y=y,n=len(g),classes=';'.join(sorted(g.phenotype.unique())),interaction_wald=W,p=p,R2=R2,method='OLS with HC3 joint interaction Wald (asymptotic chi-square); exploratory',coefficients=json.dumps(beta.tolist())))
 ir=pd.DataFrame(ir);ir['q_BH']=bh(ir.p);table(out,'phenotype_interaction_tests.csv',ir)
 rankvars=['pAPL_endpoint_A2','delta_pAPL_endpoint_start_A2','pAPL_late_A2','delta_pAPL_late_ref_A2','A_surface_upper_late_nm2','A_surface_lower_late_nm2','A_surface_midplane_late_nm2','A_surface_mean_late_nm2','delta_A_surface_mean_late_baseline_nm2','geometry_mean_late_nm2','delta_geometry_mean_late_baseline_nm2','delta_height_late_ref','physical_deviation_late_vs_ref_q95','physical_deviation_endpoint_vs_ref_q95']
 ranks=[]
 for side,g in d.groupby('side'):
  q=g[['pair_id','molecule','phenotype','physical_run']].copy()
  for c in rankvars:
   q[c]=g[c];q[c+'__rank_low_to_high']=g[c].rank(method='average',ascending=True);q[c+'__rank_high_to_low']=g[c].rank(method='average',ascending=False)
  q=q.sort_values(['physical_deviation_late_vs_ref_q95__rank_high_to_low','pair_id']);table(out,f'complete_rankings_{side}.csv',q)
  q['side']=side;ranks.append(q)
 ranks=pd.concat(ranks);wide=ranks.pivot(index=['pair_id','molecule'],columns='side',values=[c+'__rank_low_to_high' for c in rankvars]);wide.columns=[f'{s}__{c}' for c,s in wide.columns];wide=wide.reset_index();table(out,'BM_rank_side_by_side.csv',wide)
 disagree=[]
 for side,g in ranks.groupby('side'):
  for _,r in g.iterrows():
   p=r['delta_pAPL_late_ref_A2__rank_low_to_high'];s=r['delta_A_surface_mean_late_baseline_nm2__rank_low_to_high'];e=r['physical_deviation_late_vs_ref_q95__rank_low_to_high'];geo=r['geometry_mean_late_nm2__rank_low_to_high']
   disagree.append(dict(pair_id=r.pair_id,molecule=r.molecule,side=side,phenotype=r.phenotype,pAPL_rank=p,surface_rank=s,q95_rank=e,geometry_rank=geo,surface_minus_pAPL_rank=s-p,q95_minus_pAPL_rank=e-p,geometry_minus_pAPL_rank=geo-p,max_rank_gap=max(p,s,e,geo)-min(p,s,e,geo)))
 table(out,'rank_disagreement.csv',pd.DataFrame(disagree).sort_values('max_rank_gap',ascending=False))
 agree=[]
 for side,g in d.groupby('side'):
  for a,b in [('pAPL_endpoint_A2','pAPL_late_A2'),('physical_deviation_endpoint_vs_ref_q95','physical_deviation_late_vs_ref_q95'),('physical_deviation_late_vs_ref_null_mad','physical_deviation_late_vs_ref_q95'),('delta_pAPL_late_ref_A2','delta_A_surface_mean_late_baseline_nm2'),('pAPL_late_binned_A2','pAPL_late_A2')]:
   n,r,p=corr(g[a],g[b]);agree.append(dict(side=side,a=a,b=b,n=n,rho=r,p=p,median_absolute_difference=np.nanmedian(abs(g[a]-g[b]))))
 for v in rankvars:
  b=d[d.side=='B'].set_index('pair_id');m=d[d.side=='M'].set_index('pair_id').loc[b.index];n,r,p=corr(b[v],m[v]);agree.append(dict(side='B versus M',a=v,b=v,n=n,rho=r,p=p,median_absolute_difference=np.nan))
 table(out,'rank_and_endpoint_agreement.csv',agree)
 # Counterexamples selected by normalized near matching, report all eligible unique pairs.
 cs=[]
 for side,g in d.groupby('side'):
  for a,b,tol in [('delta_pAPL_late_ref_A2','geometry_mean_late_nm2',.15),('delta_height_late_ref','delta_pAPL_late_ref_A2',.02)]:
   gg=g.reset_index(drop=True)
   for i,j in itertools.combinations(range(len(gg)),2):
    x,y=gg.iloc[i],gg.iloc[j]
    if x.phenotype!=y.phenotype and abs(x[a]-y[a])<=tol:cs.append(dict(side=side,match_variable=a,match_tolerance=tol,pair_1=x.pair_id,pair_2=y.pair_id,phenotype_1=x.phenotype,phenotype_2=y.phenotype,match_1=x[a],match_2=y[a],discordant_variable=b,discordant_1=x[b],discordant_2=y[b],discordant_gap=abs(x[b]-y[b])))
 table(out,'near_matched_counterexamples.csv',pd.DataFrame(cs).sort_values('discordant_gap',ascending=False))
 paired=d.pivot(index='pair_id',columns='side',values=['delta_pAPL_late_ref_A2','delta_sAPL_proxy_late_baseline_A2','physical_deviation_late_vs_ref_q95']);paired.columns=[f'{s}_{c}' for c,s in paired.columns]
 paired['delta_delta_pAPL_late_ref_A2']=paired.B_delta_pAPL_late_ref_A2-paired.M_delta_pAPL_late_ref_A2;paired['delta_delta_sAPL_proxy_late_baseline_A2']=paired.B_delta_sAPL_proxy_late_baseline_A2-paired.M_delta_sAPL_proxy_late_baseline_A2
 B=paired.B_physical_deviation_late_vs_ref_q95;M=paired.M_physical_deviation_late_vs_ref_q95;paired['physical_magnitude']=np.hypot(B,M);paired['physical_contrast']=(B-M)/(B+M)
 table(out,'paired_quantities.csv',paired.reset_index())
 # Verify frozen PC1/2 before PC3; no estimator instantiated in frozen branch.
 params=json.loads((result/'data/pca_parameters.json').read_text());scores,validation=frozen_projection(pd.read_csv(result/'data/pca_scores.csv'),u,params,pd.read_csv(result/'data/pca_loadings.csv'))
 table(out,'frozen_PCA_projection_validation.csv',validation);table(out,'frozen_PCA_scores.csv',scores)
 assoc=[]
 for model,g in scores.groupby('model'):
  side=model.split('__')[1]
  h=d[d.side==side].set_index('pair_id').loc[g.pair_id] if side!='paired' else paired.loc[g.pair_id]
  vv=variables if side!='paired' else ['delta_delta_pAPL_late_ref_A2','delta_delta_sAPL_proxy_late_baseline_A2','physical_magnitude','physical_contrast']
  for pc in ['PC1','PC2','PC3']:
   for v in vv:
    n,r,p=corr(g[pc],h[v]);assoc.append(dict(model=model,PC=pc,variable=v,phenotype='ALL',n=n,rho=r,p=p))
    if side!='paired':
     for label in sorted(h.phenotype.unique()):
      mask=(h.phenotype==label).to_numpy();n,r,p=corr(g[pc].to_numpy()[mask],h[v].to_numpy()[mask]);assoc.append(dict(model=model,PC=pc,variable=v,phenotype=label,n=n,rho=r,p=p))
 assoc=pd.DataFrame(assoc);assoc['q_BH']=np.nan
 for _,ix in assoc.groupby(['model','phenotype']).groups.items():assoc.loc[ix,'q_BH']=bh(assoc.loc[ix,'p'])
 table(out,'frozen_PCA_associations.csv',assoc)
 return scores,paired

def classifiers(out,d,repeats):
 features={'height':['delta_height_late_ref'],'pAPL':['delta_pAPL_late_ref_A2'],'height_pAPL':['delta_height_late_ref','delta_pAPL_late_ref_A2'],'height_geometry':['delta_height_late_ref','geometry_mean_late_nm2'],'height_thickness_asymmetry':['delta_height_late_ref','delta_thickness_late_ref','delta_height_asym_signed_late_ref'],'surface_geometry':['geometry_mean_late_nm2','surface_asymmetry_late_nm2','delta_curvature_late_ref'],'pAPL_geometry':['delta_pAPL_late_ref_A2','geometry_mean_late_nm2'],'full_morphology':['delta_height_late_ref','delta_thickness_late_ref','delta_height_asym_signed_late_ref','delta_curvature_late_ref','geometry_mean_late_nm2','surface_asymmetry_late_nm2','delta_pAPL_late_ref_A2']}
 rows=[];pred=[];folds=[]
 for side,gg in d.groupby('side'):
  vc=gg.phenotype.value_counts();g=gg[gg.phenotype.isin(vc[vc>=8].index)].reset_index(drop=True);y=g.phenotype.to_numpy();groups=g.molecule.to_numpy()
  for repeat in range(repeats):
   outer=StratifiedGroupKFold(4,shuffle=True,random_state=SEED+repeat)
   for fold,(tr,te) in enumerate(outer.split(g,y,groups)):
    assert not set(groups[tr])&set(groups[te]);folds.append(dict(side=side,repeat=repeat,fold=fold,train_ids=';'.join(g.pair_id.iloc[tr]),test_ids=';'.join(g.pair_id.iloc[te])))
    for name,ff in features.items():
     estimator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',solver='lbfgs'))
     inner=StratifiedGroupKFold(3,shuffle=True,random_state=SEED+repeat*10+fold)
     search=GridSearchCV(estimator,{'logisticregression__C':[.01,.1,1,10]},cv=inner,scoring='balanced_accuracy',n_jobs=1)
     search.fit(g[ff].iloc[tr],y[tr],groups=groups[tr]);yp=search.predict(g[ff].iloc[te])
     rows.append(dict(side=side,repeat=repeat,fold=fold,feature_set=name,features=';'.join(ff),n_train=len(tr),n_test=len(te),balanced_accuracy=balanced_accuracy_score(y[te],yp),macro_F1=f1_score(y[te],yp,average='macro'),C=search.best_params_['logisticregression__C']))
     for ix,pp in zip(te,yp):pred.append(dict(pair_id=g.pair_id.iloc[ix],molecule=g.molecule.iloc[ix],side=side,repeat=repeat,fold=fold,feature_set=name,truth=y[ix],prediction=pp))
 r=table(out,'type_discrimination_folds.csv',rows);p=table(out,'type_discrimination_predictions.csv',pred);table(out,'CV_splits.csv',folds)
 summary=r.groupby(['side','feature_set']).agg(n_folds=('fold','size'),balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_F1_mean=('macro_F1','mean')).reset_index();table(out,'type_discrimination_summary.csv',summary)
 differences=[]
 for side,g in r.groupby('side'):
  pv=g.pivot(index=['repeat','fold'],columns='feature_set',values='balanced_accuracy')
  for a,b in [('height_pAPL','height'),('height_geometry','height'),('full_morphology','height'),('full_morphology','pAPL_geometry')]:
   v=pv[a]-pv[b];differences.append(dict(side=side,comparison=f'{a} minus {b}',mean_delta=v.mean(),min_delta=v.min(),max_delta=v.max(),positive_folds=int((v>0).sum()),n_folds=len(v),inference='paired descriptive fold differences; folds/repeats are dependent'))
 table(out,'type_discrimination_added_information.csv',differences);js(out/'classifier_protocol.json',dict(seed=SEED,repeats=repeats,outer_folds=4,inner_folds=3,C=[.01,.1,1,10],preprocessing='training-only StandardScaler',estimator='balanced multinomial logistic regression',groups='molecule, separately per membrane',excluded='classes with <8 systems: descriptive only',limitations='chemistry-family generalization and biological endpoints unvalidated; CV scores are label recovery'))
 return summary

def exploratory(out,d,u):
 defs={'projected':['delta_pAPL_late_ref_A2','delta_height_late_ref','delta_thickness_late_ref','delta_height_asym_signed_late_ref','delta_curvature_late_ref'],'surface':['delta_sAPL_proxy_late_baseline_A2','delta_height_late_ref','delta_thickness_late_ref','delta_height_asym_signed_late_ref','delta_curvature_late_ref'],'decomposed':['delta_pAPL_late_start_A2','delta_geometry_area_contribution_A2','delta_height_late_ref','delta_thickness_late_ref','delta_height_asym_signed_late_ref','delta_curvature_late_ref']}
 pars={};ss=[];ll=[];vv=[];overlap=[]
 for rep,ff in defs.items():
  for side in ['B','M','paired']:
   if side=='paired':
    b=d[d.side=='B'].set_index('pair_id').loc[u.pair_id];m=d[d.side=='M'].set_index('pair_id').loc[u.pair_id];X=np.c_[b[ff],m[ff]];names=['B_'+x for x in ff]+['M_'+x for x in ff];ids=u.pair_id.to_numpy();labels=u[['label_B','label_M']].copy()
   else:
    g=d[d.side==side].set_index('pair_id').loc[u.pair_id];X=g[ff].to_numpy();names=ff;ids=u.pair_id.to_numpy();labels=u[['label_B','label_M']].copy()
   assert np.isfinite(X).all();sc=StandardScaler().fit(X);p=PCA(svd_solver='full').fit(sc.transform(X));sign=np.where(p.components_[np.arange(len(p.components_)),np.argmax(abs(p.components_),axis=1)]>=0,1,-1);components=p.components_*sign[:,None];coords=sc.transform(X)@components.T;model=f'exploratory_late__{side}__{rep}'
   pars[model]=dict(features=names,centre=sc.mean_,scale=sc.scale_,components=components,variance_ratio=p.explained_variance_ratio_,orientation_multiplier=sign,cohort_n=len(X),preprocessing='cohort mean/std; exploratory descriptive fit',rationale='Area representations kept separate to avoid algebraic double-counting; independent-role morphology candidates remain empirically correlated, audit loadings.')
   for i,pid in enumerate(ids):ss.append(dict(model=model,pair_id=pid,label_B=labels.label_B.iloc[i],label_M=labels.label_M.iloc[i],PC1=coords[i,0],PC2=coords[i,1],PC3=coords[i,2]))
   for j,variance in enumerate(p.explained_variance_ratio_):
    vv.append(dict(model=model,component=j+1,explained_variance=variance,cumulative_variance=p.explained_variance_ratio_[:j+1].sum(),n=len(X)))
    for k,f in enumerate(names):ll.append(dict(model=model,component=j+1,feature=f,loading=components[j,k]))
   for labside in ['B','M'] if side=='paired' else [side]:
    lab=labels['label_'+labside].to_numpy();mask=np.array([np.sum(lab==v)>=8 for v in lab]);ov=silhouette_score(coords[mask,:3],lab[mask]);overlap.append(dict(model=model,colour_side=labside,n=int(mask.sum()),silhouette_PC1_PC2_PC3=ov,interpretation='negative/near zero indicates label overlap; cohort-fit descriptive only'))
 js(out/'exploratory_PCA/parameters.json',pars);table(out,'exploratory_PCA_scores.csv',ss);table(out,'exploratory_PCA_loadings.csv',ll);table(out,'exploratory_PCA_variance.csv',vv);table(out,'exploratory_PCA_overlap.csv',overlap)
 return pd.DataFrame(ss),pars

def q95_validation(out,d,project):
 old=project/'analysis/image_classification/figure_gui_analysis/metric_dev/results/archie_8193723_complete/figure_guide.json';g=json.loads(old.read_text());rows=[]
 for f in g['figures']:
  for e in f['numeric_evidence']:
   if e['source_table']!='data/effect_scores.csv':continue
   for record in e['records']:
    name=record.get('variable','')
    if 'euclidean_reduced' not in name or '__q95__' not in name:continue
    effect_side,view,scale,_=name.split('__');side=effect_side[-1];v=d[d.side==side][f'physical_deviation_{view}_{scale}']
    for summary,fn in [('median',lambda v:v.median()),('q25',lambda v:v.quantile(.25)),('q75',lambda v:v.quantile(.75)),('min',lambda v:v.min()),('max',lambda v:v.max())]:
     saved=record[summary];got=fn(v);rows.append(dict(side=side,view=view,summary=summary,saved=saved,reproduced=got,error=abs(saved-got),tolerance=1e-9,pass_summary=abs(saved-got)<1e-9,source=f['figure_id'],verification_level='saved distribution summary only; historical row-level effect table omitted from attachment'))
 v=table(out,'pooled_q95_reproduction_validation.csv',rows);assert v.pass_summary.all()
 return v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=ROOT.parents[2]);ap.add_argument('--result',type=Path,default=ROOT/'results/archie_8193723_complete');ap.add_argument('--out',type=Path);ap.add_argument('--cv-repeats',type=int,default=3);args=ap.parse_args();args.project=args.project.resolve();args.result=args.result.resolve();out=args.out or args.result/'surface_papl_analysis';out.mkdir(parents=True,exist_ok=True)
 frozen={str(p.relative_to(args.project)):sha(p) for p in args.result.rglob('*') if p.is_file() and 'surface_papl_analysis' not in p.parts};u=pd.read_csv(args.result/'data/unified_metrics.csv');u=u[u.selected_pair].copy();assert len(u)==156 and u.molecule.nunique()==156 and u.qc_pass_pair.all()
 print('Consolidating 312 systems',flush=True);d=consolidate(args.project,args.result,out,u);q=q95_validation(out,d,args.project);print('Scores verified against saved summaries; analysing',flush=True);scores,paired=analyses(out,d,u,args.result);print('Nested CV',flush=True);summary=classifiers(out,d,args.cv_repeats);print('Separated exploratory PCA',flush=True);xs,pars=exploratory(out,d,u)
 assert all(sha(args.project/p)==h for p,h in frozen.items());js(out/'analysis_manifest.json',dict(seed=SEED,cohort_pairs=156,systems=312,binned_coverage=int(d.binned_pAPL_available.sum()),primary_window='Relax local 80–100 ns inclusive at 1 ns sampling: 21 frames',late_area_statistic='arithmetic mean to preserve additive identities',historical_morphology_statistic='80–100 ns median',surface_change_reference='per-system Pull initial geometry',pAPL_change_reference='pristine pAPL and system Pull start explicitly separated',frozen_files=frozen,frozen_unchanged=True,q95_summary_tolerance=1e-9,q95_row_level_validation='unavailable: no historical effect_scores.csv',pristine_trajectories='not supplied; extraction script required',versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)))
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
