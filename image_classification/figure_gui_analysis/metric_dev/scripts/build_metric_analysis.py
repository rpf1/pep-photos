#!/usr/bin/env python3
"""One entrypoint for read-only source audit, metric analysis, validation and plots."""
import os
for _n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(_n,'1')
import argparse,hashlib,json,sys,warnings,time,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,roc_curve,log_loss
import sklearn,scipy
from metric_core import *
warnings.filterwarnings('ignore',category=stats.ConstantInputWarning)
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def save(out,name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);p=out/name;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False,float_format='%.12g');return d

def read_unified(args,out):
 m=pd.read_csv(args.master);s=pd.read_csv(args.selected).rename(columns={'bilayer':'bilayer_type'});pp=pd.read_csv(args.pairs);ref=pd.read_csv(args.reference)
 keys=['molecule','nmol','bilayer_type','physical_run']
 assert len(ref)==2 and set(ref.membrane)==set(MEMBRANES.values()),'Exactly two canonical pristine references required'
 assert not m.duplicated(keys).any() and not s.duplicated(keys).any() and not pp.duplicated(['molecule','physical_run']).any(),'Identity duplicates'
 j=s[keys+['manual_label','APL_change','morph__qc_status']].merge(m,on=keys,suffixes=('_selected',''),validate='one_to_one',indicator=True)
 assert (j._merge=='both').all(),'Unreconciled selected identities'
 assert (j.manual_label_selected.fillna('')==j.manual_label.fillna('')).all(),'Conflicting labels'
 assert np.allclose(j.APL_change_selected,j.APL_change_BOX,equal_nan=True),'Canonical BOX dAPL disagreement'
 assert (j['morph__qc_status']==j['production_morph__qc_status']).all(),'QC disagreement'
 # The authoritative strict paired inclusion file is never silently expanded.
 tri=j[(j.molecule_type=='peptoid')&(j.length==3)&j.bilayer_type.isin(MEMBRANES.values())]
 b=tri[tri.bilayer_type==MEMBRANES['B']];mm=tri[tri.bilayer_type==MEMBRANES['M']]
 p=b.merge(mm,on=['molecule','nmol','physical_run'],suffixes=('_B','_M'),validate='one_to_one')
 p['pair_id']=p.molecule+'|'+p.nmol.astype(int).astype(str)+'|'+p.physical_run
 p['qc_pass_pair']=(p.production_morph__qc_status_B=='PASS')&(p.production_morph__qc_status_M=='PASS')
 p['selected_pair']=pd.MultiIndex.from_frame(p[['molecule','physical_run']]).isin(pd.MultiIndex.from_frame(pp[['molecule','physical_run']]))
 assert p.selected_pair.sum()==len(pp) and (p.loc[p.selected_pair,'qc_pass_pair']).all(),'Strict pair identity/QC mismatch'
 r=ref.set_index('membrane');result={k:p[k] for k in ['pair_id','molecule','nmol','physical_run','qc_pass_pair','selected_pair']};audit=[]
 for side,mem in MEMBRANES.items():
  result['label_'+side]=p['manual_label_'+side];result['physical_key_'+side]=p['physical_key_'+side]
  result['qc_'+side]=p['production_morph__qc_status_'+side]
  for short,(source,refcol,unit) in METRICS.items():
   pristine=float(r.loc[mem,refcol])
   if short=='APL':
    start=p['start_APL_BOX_'+side];end=p['finish_APL_BOX_100ns_'+side];delta=p['APL_change_BOX_'+side];late=np.full(len(p),np.nan)
    result[f'{side}__APL__late_90_100_mean']=p['APL_mean_last10ns_BOX_'+side]
   else:
    start=p[f'production_morph__baseline__{source}_{side}'];end=p[f'production_morph__endpoint__{source}_{side}'];delta=p[f'production_morph__delta__{source}_{side}'];late=p[f'production_morph__relax_80_100__{source}__median_{side}']
   assert np.allclose(end-start,delta,equal_nan=True,atol=1e-8),f'Delta mismatch {short}'
   values={'pristine_reference':np.repeat(pristine,len(p)),'system_start':start,'endpoint':end,'late_80_100_median':late,'endpoint_vs_ref':end-pristine,'late_vs_ref':late-pristine,'endpoint_vs_start':delta,'start_vs_ref':start-pristine}
   for kind,v in values.items():
    name=f'{side}__{short}__{kind}';result[name]=v
    if kind.endswith(('vs_ref','vs_start')):result[name+'__abs']=abs(v)
    audit.append(dict(membrane=side,metric=short,quantity=kind,unit=unit,missing=int(pd.isna(v).sum())))
  for base,upper,lower,unit in [('height','height_rms','nm','nm'),('excess','excess_area','','fraction'),('curvature','H_rms','nm_inv','nm^-1')]:
   src={'height':'height_rms_upper_minus_lower_nm','excess':'excess_area_upper_minus_lower','curvature':'H_rms_upper_minus_lower_nm_inv'}[base]
   rf={'height':float(r.loc[mem,'upper_height_rms_nm']-r.loc[mem,'lower_height_rms_nm']),'excess':float(r.loc[mem,'upper_excess_area_fraction']-r.loc[mem,'lower_excess_area_fraction']),'curvature':float(r.loc[mem,'upper_H_rms_nm_inv']-r.loc[mem,'lower_H_rms_nm_inv'])}[base]
   for phase,old in [('system_start','baseline'),('endpoint','endpoint')]:result[f'{side}__{base}_asym_signed__{phase}']=p[f'production_morph__{old}__{src}_{side}']
   result[f'{side}__{base}_asym_signed__pristine_reference']=np.repeat(rf,len(p))
   result[f'{side}__{base}_asym_signed__late_80_100_median']=p[f'production_morph__relax_80_100__{src}__median_{side}']
   for name,phase in [('endpoint_vs_ref','endpoint'),('late_vs_ref','late_80_100_median'),('start_vs_ref','system_start')]:
    v=result[f'{side}__{base}_asym_signed__{phase}']-rf;result[f'{side}__{base}_asym_signed__{name}']=v;result[f'{side}__{base}_asym_signed__{name}__abs']=abs(v)
   result[f'{side}__{base}_asym_signed__endpoint_vs_start']=result[f'{side}__{base}_asym_signed__endpoint']-result[f'{side}__{base}_asym_signed__system_start']
 u=pd.DataFrame(result);paired={}
 for c in [c for c in u if c.startswith('B__')]:
  other='M__'+c[3:]
  if other in u:
   paired['B_minus_M__'+c[3:]]=u[c]-u[other];paired['abs_B_minus_M__'+c[3:]]=abs(u[c]-u[other])
 u=pd.concat([u,pd.DataFrame(paired)],axis=1)
 u['ddAPL']=u['B__APL__endpoint_vs_start']-u['M__APL__endpoint_vs_start']
 check=u[u.selected_pair].merge(pp[['molecule','physical_run','ddAPL']],on=['molecule','physical_run'],suffixes=('','_authoritative'),validate='one_to_one')
 assert np.allclose(check.ddAPL,check.ddAPL_authoritative),'ddAPL pair-file mismatch'
 save(out,'data/unified_metrics.csv',u);save(out,'tables/missingness.csv',audit)
 save(out,'tables/label_counts.csv',j.groupby(['membrane_role','manual_label','production_morph__qc_status']).size().rename('n').reset_index())
 save(out,'tables/pair_exclusions.csv',u[~u.selected_pair][['pair_id','molecule','physical_run','qc_B','qc_M']])
 excluded=m[m.production_morph__qc_status.notna()].merge(s[keys],on=keys,how='left',indicator=True).query('_merge=="left_only"')
 save(out,'tables/strict_identity_exclusions.csv',excluded[keys+['manual_label']])
 featurecols=[f'{side}__{k}__{view}' for side in MEMBRANES for k in METRICS for view in ['endpoint_vs_ref','late_vs_ref'] if not(k=='APL' and view=='late_vs_ref')]
 assert np.isfinite(u.loc[u.selected_pair,featurecols].to_numpy(float)).all(),'Missing primary features; cannot silently drop/impute'
 info={'master_rows':len(m),'strict_system_rows':len(j),'strict_pass_systems':int((j.production_morph__qc_status=='PASS').sum()),'reference_rows':len(ref),'paired_including_review':len(u),'primary_pairs':int(u.selected_pair.sum()),'unique_primary_molecules':u.loc[u.selected_pair,'molecule'].nunique(),'label_counts_primary':{side:u.loc[u.selected_pair,'label_'+side].value_counts().to_dict() for side in MEMBRANES},'unresolved_identity_systems':len(excluded),'primary_run_counts':u.loc[u.selected_pair,'physical_run'].value_counts().to_dict()}
 return u.reset_index(drop=True),info

def getX(u,side,view):
 names=[k for k in ORDER if not(view=='late_vs_ref' and k=='APL')]
 return u[[f'{side}__{k}__{view}' for k in names]].to_numpy(float),names

def effect_engine(train,test,save_states=False):
 data={};models={};meta={}
 for view in ['endpoint_vs_ref','late_vs_ref']:
  tb,names=getX(train,'B',view);tm,_=getX(train,'M',view);pooled=np.vstack([tb,tm]);null=np.r_[train.label_B.eq('flat_no_interaction'),train.label_M.eq('flat_no_interaction')]
  for method in SCALERS:
   failure=None
   try:state=fit_state(pooled,names,method,null)
   except ValueError as exc:
    if method!='null_mad':raise
    state=fit_state(pooled,names,'q95',null);failure=str(exc)
   models[f'{view}__{method}']=state if failure is None else {'unavailable':failure}
   for side in MEMBRANES:
    x,_=getX(test,side,view)
    for family,v in transform(x,state).items():
     col=f'Effect_{side}__{view}__{method}__{family}';data[col]=bounded(v) if failure is None else np.full(len(test),np.nan);meta[col]=dict(kind='effect',side=side,view=view,scaler=method,family=family,range='[0,1)',direction='larger physical deviation',formula='d/(1+d)',limitation='Exploratory calibration-relative geometry effect, not biological harm')
 # dAPL control uses simulation change, normalised by pooled q95 |dAPL|.
 scale=np.quantile(abs(np.r_[train['B__APL__endpoint_vs_start'],train['M__APL__endpoint_vs_start']]),.95)
 models['dAPL_scale']=float(scale)
 for side in MEMBRANES:
  col=f'Effect_{side}__endpoint_vs_start__q95__single_dAPL';data[col]=bounded(abs(test[f'{side}__APL__endpoint_vs_start'])/scale);meta[col]=dict(kind='effect',side=side,view='endpoint_vs_start',scaler='q95',family='single_dAPL',range='[0,1)',direction='magnitude of projected response',formula='|dAPL|/(q95+|dAPL|)',limitation='Does not measure bending')
 return pd.DataFrame(data,index=test.index),models,meta

def all_scores(effects,meta):
 data={c:effects[c].to_numpy() for c in effects};details=dict(meta);properties={r['metric_name']:r for r in metric_metadata()}
 for b in [c for c in effects if c.startswith('Effect_B__')]:
  m=b.replace('Effect_B__','Effect_M__');family=b.split('__',1)[1]
  for metric,v in combine(effects[b].to_numpy(),effects[m].to_numpy()).items():
   name=metric+'__'+family;data[name]=v;details[name]=dict(kind='joint',side='B/M',view=meta[b]['view'],scaler=meta[b]['scaler'],family=meta[b]['family'],range=properties[metric]['range'],direction='smaller favours bacteria' if metric=='ratio_MB' else ('total magnitude' if metric=='magnitude' else 'larger favours bacteria'),formula=metric,limitation='Ranking utility or direction only; requires Effect_B and Effect_M')
 return pd.DataFrame(data,index=effects.index),details

def mathematical_tests(out):
 axis=np.unique(np.r_[0,1e-12,1e-9,1e-6,.001,np.linspace(.01,1,101)])
 B,M=np.meshgrid(axis,axis,indexing='ij');v=combine(B.ravel(),M.ravel());rows=metric_metadata();surfaces=pd.DataFrame({'B':B.ravel(),'M':M.ravel(),**v});save(out,'data/synthetic_metric_grid.csv',surfaces)
 for r in rows:
  arr=v[r['metric_name']].reshape(B.shape);r.update(grid_points=arr.size,nonfinite_grid_points=int((~np.isfinite(arr)).sum()),grid_B_monotonic_violations=int((np.diff(arr,axis=0)<-1e-10).sum()),grid_M_penalty_violations=int((np.diff(arr,axis=1)>1e-10).sum()),both_high=float(arr[-1,-1]),weak_B_zero_M=float(arr[np.argmin(abs(axis-.001)),0]),B_high_M_zero=float(arr[-1,0]))
 save(out,'tables/candidate_metric_properties.csv',rows)
 # Exact mathematical equivalence: angle is a strictly increasing transform of contrast off origin.
 assert np.max(abs(v['angle']-4/np.pi*np.arctan(v['contrast'])))<1e-12
 edge=pd.DataFrame({'B':[0,1,1e-9,1e-3,1,.5],'M':[0,1,0,1e-12,0,.5]});save(out,'tables/synthetic_edge_cases.csv',edge.assign(**combine(edge.B.to_numpy(),edge.M.to_numpy())))


def pca_analysis(u,out):
 score_rows=[];load=[];summary=[];associations=[];models={};redund=[]
 for view in ['endpoint_vs_ref','late_vs_ref']:
  for side in ['B','M','paired']:
   xb,names=getX(u,'B',view);xm,_=getX(u,'M',view)
   X=np.hstack([xb,xm]) if side=='paired' else (xb if side=='B' else xm)
   labels=['B_'+k for k in names]+['M_'+k for k in names] if side=='paired' else names
   scale=np.quantile(abs(X),.95,axis=0)
   keep,_=reduction(np.vstack([xb,xm]),names)
   for scope in ['full','reduced']:
    ix=list(range(len(labels))) if scope=='full' else (keep+[len(names)+i for i in keep] if side=='paired' else keep)
    Z=X[:,ix]/scale[ix];pca=PCA(svd_solver='full').fit(Z);scores=pca.transform(Z);vectors=pca.components_.copy();signs=[]
    for k in range(len(vectors)):
     sign=1 if vectors[k,np.argmax(abs(vectors[k]))]>=0 else -1;vectors[k]*=sign;scores[:,k]*=sign;signs.append(sign)
    model=f'{view}__{side}__{scope}';models[model]={'features':[labels[i] for i in ix],'scale':scale[ix].tolist(),'centre':pca.mean_.tolist(),'components':vectors.tolist(),'orientation_multipliers':signs}
    for k,ev in enumerate(pca.explained_variance_ratio_):
     summary.append(dict(model=model,view=view,side=side,feature_set=scope,component=k+1,explained_variance=ev,cumulative_variance=sum(pca.explained_variance_ratio_[:k+1]),n=len(u),features=len(ix),orientation_multiplier=signs[k]))
     for n,c in zip(models[model]['features'],vectors[k]):load.append(dict(model=model,component=k+1,feature=n,loading=c,squared_loading_contribution=c*c))
    for i,row in u.iterrows():score_rows.append(dict(model=model,pair_id=row.pair_id,molecule=row.molecule,label_B=row.label_B,label_M=row.label_M,PC1=scores[i,0],PC2=scores[i,1]))
    for target in ['ddAPL','B__APL__endpoint_vs_start','M__APL__endpoint_vs_start']:
     associations.append(dict(model=model,target=target,rho_PC1=rho(scores[:,0],u[target])))
    for side2 in ([side] if side!='paired' else ['B','M']):
     labels2=u['label_'+side2];groups=[scores[labels2==lab,0] for lab in LABELS if (labels2==lab).sum()>=3]
     associations.append(dict(model=model,target='visual_classes_'+side2,kw_H=stats.kruskal(*groups).statistic,kw_p=stats.kruskal(*groups).pvalue))
  for side in ['B','M']:
   X,names=getX(u,side,view);cor=stats.spearmanr(X).statistic
   for i in range(len(names)):
    for j in range(i+1,len(names)):redund.append(dict(view=view,side=side,metric1=names[i],metric2=names[j],rho=cor[i,j]))
 save(out,'data/pca_scores.csv',score_rows);save(out,'data/pca_loadings.csv',load);save(out,'tables/pca_summary.csv',summary);save(out,'tables/pca_associations.csv',associations);save(out,'tables/physical_redundancy.csv',redund);write_json(out/'data/pca_parameters.json',models)


def corr_columns(X,y):
 """Spearman correlation per column, no pairwise omission needed for finite main candidates."""
 X=np.asarray(X);y=np.asarray(y);a=stats.rankdata(X,axis=0);b=stats.rankdata(y);a-=a.mean(axis=0);b-=b.mean();den=np.sqrt((a*a).sum(axis=0)*(b*b).sum());return np.divide(b@a,den,out=np.full(X.shape[1],np.nan),where=den>0)

def associations(scores,u,out,nboot,rng):
 labels=[];pairwise=[];kw=[];cor=[];classif=[]
 # Component correlations with molecule-resampled percentile uncertainty for all scores.
 targets=['ddAPL']+[f'{side}__{k}__{view}' for side in ['B','M'] for k in METRICS for view in ['endpoint_vs_ref','late_vs_ref'] if not(k=='APL' and view=='late_vs_ref')]
 X=scores.to_numpy();good=np.isfinite(X).all(axis=0);finite=X[:,good];names=scores.columns[good]
 def corr_matrix(x,y):
  a=stats.rankdata(x,axis=0);b=stats.rankdata(y,axis=0);a-=a.mean(axis=0);b-=b.mean(axis=0);den=np.sqrt((a*a).sum(axis=0)[:,None]*(b*b).sum(axis=0)[None,:]);return np.divide(a.T@b,den,out=np.full(den.shape,np.nan),where=den>0)
 Y=u[targets].to_numpy();point=corr_matrix(finite,Y);boot=np.empty((nboot,len(names),len(targets)),dtype=np.float32)
 for k in range(nboot):
  ix=cluster_draw(u.molecule,rng);boot[k]=corr_matrix(finite[ix],Y[ix])
 lo,hi=np.nanquantile(boot,[.025,.975],axis=0)
 for i,c in enumerate(names):
  for j,target in enumerate(targets):cor.append(dict(score=c,target=target,rho=point[i,j],ci_low=lo[i,j],ci_high=hi[i,j],n=len(u),bootstrap=nboot))
 del boot
 # Explicitly report any singular empirical raw ratio instead of dropping it silently.
 save(out,'tables/nonfinite_score_counts.csv',[dict(score=c,n_nonfinite=int((~np.isfinite(scores[c])).sum())) for c in scores])
 for side in ['B','M']:
  labs=u['label_'+side].to_numpy()
  for c in scores:
   if c.startswith('Effect_') and not c.startswith('Effect_'+side):continue
   v=scores[c].to_numpy();groups=[]
   for lab in LABELS:
    z=v[(labs==lab)&np.isfinite(v)]
    if len(z):labels.append(dict(score=c,side=side,label=lab,n=len(z),median=np.median(z),q25=np.quantile(z,.25),q75=np.quantile(z,.75)))
    if len(z)>=3:groups.append(z)
    y=(labs==lab).astype(int);ok=np.isfinite(v)
    if y.sum()>=3 and (1-y).sum()>=3 and np.std(v[ok])>0:classif.append(dict(score=c,side=side,label=lab,n_positive=int(y[ok].sum()),n_negative=int((1-y[ok]).sum()),AUROC=roc_auc_score(y[ok],v[ok]),status='descriptive direction fixed; not held-out prediction'))
   if len(groups)>1 and np.std(np.concatenate(groups))>0:
    a=stats.kruskal(*groups);kw.append(dict(score=c,side=side,H=a.statistic,p=a.pvalue,epsilon_squared=max(0,(a.statistic-len(groups)+1)/(sum(map(len,groups))-len(groups)))))
   for a,b in [('curved','flat_interaction'),('bulge_flat','flat_interaction'),('bulge_curved','curved'),('flat_interaction','flat_no_interaction')]:
    va=v[(labs==a)&np.isfinite(v)];vb=v[(labs==b)&np.isfinite(v)]
    if min(len(va),len(vb))<3:continue
    t=stats.mannwhitneyu(va,vb,alternative='two-sided');pairwise.append(dict(score=c,side=side,class_a=a,class_b=b,n_a=len(va),n_b=len(vb),p=t.pvalue,rank_biserial=2*t.statistic/(len(va)*len(vb))-1))
 for rows,name in [(kw,'class_omnibus'),(pairwise,'class_pairwise')]:
  df=pd.DataFrame(rows);df['q_BH_all_tests_in_table']=bh(df.p);save(out,'tables/'+name+'.csv',df)
 save(out,'tables/class_descriptives.csv',labels);save(out,'tables/descriptive_OVR_AUROC.csv',classif);save(out,'tables/score_component_correlations.csv',cor)
 # Full symmetric score matrix retained as a table; no opaque aggregate.
 corr=pd.DataFrame(stats.spearmanr(finite).statistic,index=names,columns=names);corr.index.name='score';corr.to_csv(out/'tables/score_rank_correlation_matrix.csv',float_format='%.8g')


def bootstrap_calibration(u,effects,scores,out,nboot,rng):
 base=scores.to_numpy();names=scores.columns;ids=u.pair_id.tolist();boot=np.empty((nboot,len(u),len(names)),dtype=np.float32);rank_agree=[];draws=[]
 for k in range(nboot):
  ix=cluster_draw(u.molecule,rng);sample=u.iloc[ix].reset_index(drop=True)
  try:e,_,meta=effect_engine(sample,u);bs,_=all_scores(e,meta)
  except ValueError:
   boot[k]=np.nan;draws.append(dict(iteration=k,status='degenerate null scale'));continue
  boot[k]=bs[names].to_numpy(dtype=np.float32);draws.append(dict(iteration=k,status='PARTIAL_NULL_UNAVAILABLE' if bs.isna().any().any() else 'PASS'))
  if (k+1)%50==0:print('calibration bootstrap',k+1,'/',nboot,flush=True)
 lo,hi=np.nanquantile(boot,[.025,.975],axis=0);mean=np.nanmean(boot,axis=0)
 rows=[];stability=[]
 for j,c in enumerate(names):
  ranks=stats.rankdata(-boot[:,:,j],axis=1,nan_policy='omit');original=stats.rankdata(-base[:,j],nan_policy='omit');top=original<=10
  agreement=[rho(original,r) for r in ranks];overlap=np.nanmean((ranks<=10)[:,top].sum(axis=1)[np.isfinite(ranks).all(axis=1)]/max(1,top.sum()))
  stability.append(dict(score=c,median_rank_rho=np.nanmedian(agreement),rank_rho_p025=np.nanquantile(agreement,.025),top10_overlap_mean=overlap,mean_score_interval_width=np.nanmean(hi[:,j]-lo[:,j]),successful_bootstraps=int(np.isfinite(boot[:,:,j]).all(axis=1).sum())))
  rlo,rhi=np.nanquantile(ranks,[.025,.975],axis=0);valid=np.isfinite(ranks);topprob=np.divide(((ranks<=10)&valid).sum(axis=0),valid.sum(axis=0),out=np.full(len(u),np.nan),where=valid.sum(axis=0)>0)
  rows.extend(dict(pair_id=ids[i],score=c,point=base[i,j],bootstrap_mean=mean[i,j],ci_low=lo[i,j],ci_high=hi[i,j],top10_probability=topprob[i],rank_p025=rlo[i],rank_p975=rhi[i]) for i in range(len(u)))
 save(out,'tables/score_bootstrap_intervals.csv',rows);save(out,'tables/ranking_stability.csv',stability);save(out,'tables/bootstrap_status.csv',draws)
 return pd.DataFrame(stability)


def cv_evaluation(u,out,repeats,nboot,rng):
 """Outer molecule groups; every scale/feature choice/threshold refitted inside training folds."""
 predictions=[];threshold_rows=[];supervised_weights=[];oof={};score_names=None
 for rep in range(repeats):
  strata=(u.label_B.isin(['curved','bulge_curved','bulge_flat']).astype(int)*2+u.label_M.isin(['curved','bulge_curved','bulge_flat']).astype(int)).to_numpy()
  split=StratifiedGroupKFold(4,shuffle=True,random_state=SEED+rep)
  for fold,(tr,te) in enumerate(split.split(u,strata,u.molecule)):
   train=u.iloc[tr].reset_index(drop=True);test=u.iloc[te].reset_index(drop=True)
   assert not set(train.molecule)&set(test.molecule)
   et,_,mt=effect_engine(train,train);ev,_,_=effect_engine(train,test);st,_=all_scores(et,mt);sv,_=all_scores(ev,mt)
   # Vectorised training-only ROC/Youden threshold fitting, including tie boundaries.
   score_names=st.columns;A=st.to_numpy();scoreV=sv.to_numpy();ranks=stats.rankdata(A,axis=0)
   for side in ['B','M']:
    for lab in LABELS:
     yt=train['label_'+side].eq(lab).to_numpy()
     if min(yt.sum(),(~yt).sum())<2:continue
     n1=yt.sum();n0=len(yt)-n1;auc=(ranks[yt].sum(axis=0)-n1*(n1+1)/2)/(n1*n0);sign=np.where(auc>=.5,1.,-1.);Z=A*sign;order=np.argsort(Z,axis=0,kind='stable');ordered=np.take_along_axis(Z,order,axis=0);yy=yt[order];prev=np.vstack([np.zeros((1,Z.shape[1])),np.cumsum(yy,axis=0)[:-1]]);before=np.arange(len(yt))[:,None];J=(n1-prev)/n1-(n0-before+prev)/n0
     allowed=np.vstack([np.ones((1,Z.shape[1]),bool),np.diff(ordered,axis=0)>0]);J[~allowed]=-np.inf;best=np.argmax(J,axis=0);cut=ordered[best,np.arange(Z.shape[1])];value=scoreV*sign;prediction=value>=cut
     key=(side,lab)
     if key not in oof:oof[key]={'value':np.zeros((len(u),len(score_names))), 'prediction':np.zeros((len(u),len(score_names))), 'count':np.zeros((len(u),len(score_names)))}
     validcols=np.isfinite(A).all(axis=0)&np.isfinite(scoreV).all(axis=0);oof[key]['value'][te]+=np.where(validcols,value,0);oof[key]['prediction'][te]+=np.where(validcols,prediction,0);oof[key]['count'][te]+=validcols
    y=train['label_'+side].isin(['curved','bulge_curved','bulge_flat']).astype(int).to_numpy();yv=test['label_'+side].isin(['curved','bulge_curved','bulge_flat']).astype(int).to_numpy()
    X,names=getX(train,side,'late_vs_ref');V,_=getX(test,side,'late_vs_ref')
    for feat in ['dAPL_only','height_only','morphology_full','dAPL_plus_morphology']:
     xa=train[f'{side}__APL__endpoint_vs_start'].to_numpy()[:,None];va=test[f'{side}__APL__endpoint_vs_start'].to_numpy()[:,None]
     xx,vv=(xa,va) if feat=='dAPL_only' else ((X[:,[0]],V[:,[0]]) if feat=='height_only' else ((X,V) if feat=='morphology_full' else (np.hstack([xa,X]),np.hstack([va,V]))))
     inner=StratifiedGroupKFold(3,shuffle=True,random_state=SEED+rep);best=None
     for C in [.1,1.,10.]:
      auc=[]
      for it,iv in inner.split(xx,y,train.molecule):
       if len(np.unique(y[it]))<2 or len(np.unique(y[iv]))<2:continue
       mdl=make_pipeline(RobustScaler(),LogisticRegression(C=C,penalty='l2',solver='liblinear',class_weight='balanced',random_state=SEED,max_iter=2000));mdl.fit(xx[it],y[it]);auc.append(roc_auc_score(y[iv],mdl.predict_proba(xx[iv])[:,1]))
      val=np.mean(auc) if auc else .5
      if best is None or val>best[0]:best=(val,C)
     mdl=make_pipeline(RobustScaler(),LogisticRegression(C=best[1],penalty='l2',solver='liblinear',class_weight='balanced',random_state=SEED,max_iter=2000));mdl.fit(xx,y);prob=mdl.predict_proba(vv)[:,1]
     for ii,v,yy in zip(te,prob,yv):predictions.append(dict(row=int(ii),pair_id=u.iloc[ii].pair_id,molecule=u.iloc[ii].molecule,repeat=rep,fold=fold,side=side,features=feat,y=int(yy),probability=float(v),C=best[1]))
     supervised_weights.append(dict(repeat=rep,fold=fold,side=side,features=feat,C=best[1],intercept=float(mdl[-1].intercept_[0]),coefficients=json.dumps(mdl[-1].coef_[0].tolist()),scaler_centre=json.dumps(mdl[0].center_.tolist()),scaler_scale=json.dumps(mdl[0].scale_.tolist())))
  print('grouped CV repeat',rep+1,'/',repeats,flush=True)
 pred=pd.DataFrame(predictions);save(out,'data/supervised_oof_predictions.csv',pred);save(out,'data/supervised_linear_parameters.csv',supervised_weights)
 rows=[];wide={}
 for (side,feat),g in pred.groupby(['side','features']):
  a=g.groupby('row').agg(probability=('probability','mean'),y=('y','first'));yy=a.y.to_numpy();pr=a.probability.to_numpy();ci=[]
  for _ in range(nboot):
   ix=cluster_draw(u.iloc[a.index].molecule,rng)
   if len(np.unique(yy[ix]))==2:ci.append(roc_auc_score(yy[ix],pr[ix]))
  rows.append(dict(side=side,features=feat,n=len(a),AUROC=roc_auc_score(yy,pr),auc_ci_low=np.quantile(ci,.025),auc_ci_high=np.quantile(ci,.975),balanced_accuracy=balanced_accuracy_score(yy,pr>=.5),log_loss=log_loss(yy,pr),status='nested grouped OOF phenotype comparator; not chemistry prediction'))
  wide[(side,feat)]=(yy,pr)
 save(out,'tables/supervised_nested_CV.csv',rows)
 added=[]
 for side in ['B','M']:
  y,p0=wide[(side,'dAPL_only')];_,p1=wide[(side,'dAPL_plus_morphology')];d=[]
  for _ in range(nboot):
   ix=cluster_draw(u.molecule,rng)
   if len(np.unique(y[ix]))==2:d.append(roc_auc_score(y[ix],p1[ix])-roc_auc_score(y[ix],p0[ix]))
  added.append(dict(side=side,delta_AUROC=roc_auc_score(y,p1)-roc_auc_score(y,p0),ci_low=np.quantile(d,.025),ci_high=np.quantile(d,.975)))
 save(out,'tables/added_information_vs_APL.csv',added)
 summary=[]
 for (side,lab),v in oof.items():
  for j,c in enumerate(score_names):
   if c.startswith('Effect_') and not c.startswith('Effect_'+side):continue
   valid=v['count'][:,j]>0;y=u.loc[valid,'label_'+side].eq(lab).astype(int).to_numpy()
   if len(y)<3 or len(np.unique(y))<2:continue
   values=v['value'][valid,j]/v['count'][valid,j];pr=v['prediction'][valid,j]/v['count'][valid,j]>=.5
   summary.append(dict(side=side,label=lab,score=c,n=len(y),n_positive=int(y.sum()),AUROC=roc_auc_score(y,values),balanced_accuracy=balanced_accuracy_score(y,pr),min_repeats_available=int(v['count'][valid,j].min()),max_repeats_available=int(v['count'][valid,j].max()),status='OOF train-only transformations; unavailable null scales excluded and coverage reported'))
 save(out,'tables/all_score_grouped_CV.csv',summary)
 # Probability is a supervised comparator, not an origin-zero effect.
 sup=u[['pair_id','molecule']].copy()
 for side in ['B','M']:sup[f'Effect_{side}_supervised_linear_comparator']=wide[(side,'morphology_full')][1]
 save(out,'data/supervised_effect_comparator.csv',sup)
 return pd.DataFrame(rows)


def sensitivity(u,all_u,effects,models,out,rng,nboot):
 rows=[];starts=[]
 for side in ['B','M']:
  for k in METRICS:
   a=u[f'{side}__{k}__endpoint_vs_start'].to_numpy();b=u[f'{side}__{k}__endpoint_vs_ref'].to_numpy();off=u[f'{side}__{k}__start_vs_ref'];boot=[rho(a[ix],b[ix]) for ix in [cluster_draw(u.molecule,rng) for _ in range(nboot)]]
   rows.append(dict(side=side,metric=k,rho=rho(a,b),ci_low=np.nanquantile(boot,.025),ci_high=np.nanquantile(boot,.975),median_start_offset=off.median(),start_offset_p05=off.quantile(.05),start_offset_p95=off.quantile(.95),median_abs_endpoint_late=abs(u[f'{side}__{k}__endpoint_vs_ref']-u[f'{side}__{k}__late_vs_ref']).median() if k!='APL' else np.nan))
 save(out,'tables/reference_vs_start.csv',rows)
 # Per-membrane scaling sensitivity; main models use shared B+M denominators.
 for view in ['endpoint_vs_ref','late_vs_ref']:
  for method in SCALERS:
   for side in ['B','M']:
    X,names=getX(u,side,view);local=fit_state(X,names,method,u['label_'+side].eq('flat_no_interaction'));vals=transform(X,local)
    for fam,v in vals.items():starts.append(dict(view=view,scaler=method,side=side,family=fam,within_vs_shared_rank_rho=rho(bounded(v),effects[f'Effect_{side}__{view}__{method}__{fam}'])))
 save(out,'tables/membrane_scaling_sensitivity.csv',starts)
 # Review-QC refit sensitivity scored on the same strict primary pairs.
 e,_,_=effect_engine(all_u,u);save(out,'tables/QC_sensitivity.csv',[dict(score=c,clean_vs_review_calibration_rho=rho(e[c],effects[c]),mean_abs_difference=float(abs(e[c]-effects[c]).mean()),clean_n=len(u),with_review_n=len(all_u)) for c in effects])
 # Feature ablation, separately preserving all remaining physical axes.
 ab=[]
 for view in ['endpoint_vs_ref','late_vs_ref']:
  xb,names=getX(u,'B',view);xm,_=getX(u,'M',view)
  for drop in names:
   ix=[i for i,n in enumerate(names) if n!=drop];state=fit_state(np.vstack([xb[:,ix],xm[:,ix]]),[names[i] for i in ix],'q95')
   for side,X in [('B',xb),('M',xm)]:ab.append(dict(view=view,side=side,removed=drop,rho_with_full=rho(bounded(transform(X[:,ix],state)['euclidean_full']),effects[f'Effect_{side}__{view}__q95__euclidean_full'])))
 save(out,'tables/feature_ablation.csv',ab)


def decision_and_pareto(u,effects,scores,meta,stability,out):
 rows=[];fronts=[];sing=[]
 for b in [c for c in effects if c.startswith('Effect_B__')]:
  m=b.replace('Effect_B__','Effect_M__');fam=b.split('__',1)[1];B=effects[b].to_numpy();M=effects[m].to_numpy();dom=dominance(B,M);front=~dom.any(axis=0)
  fronts.extend(dict(pair_id=u.iloc[i].pair_id,family=fam,Effect_B=B[i],Effect_M=M[i],pareto_nondominated=bool(front[i])) for i in range(len(u)))
  sing.append(dict(family=fam,n_M_zero=int((M==0).sum()),n_B_zero=int((B==0).sum()),n_M_below_001=int((M<.01).sum()),n_B_below_001=int((B<.01).sum()),min_M=float(M.min()),min_B=float(B.min())))
  for metric in combine(B,M):
   c=metric+'__'+fam;v=scores[c].to_numpy();viol=int((dom&(v[:,None]+1e-12<v[None,:])).sum());stab=stability[stability.score==c].iloc[0]
   mathematical=next(r for r in metric_metadata() if r['metric_name']==metric)
   rows.append(dict(score=c,family=fam,metric=metric,simplicity='two effects plus explicit formula',interpretability=mathematical['interpretation'],class_separation='See class/OOF tables; labels are not biological efficacy',bootstrap_stability=stab.median_rank_rho,top10_overlap=stab.top10_overlap_mean,pareto_violations=viol,pareto_comparisons=int(dom.sum()),denominator_instability=mathematical['singularity'],arbitrary_constants=mathematical['arbitrary_constant_required'],rho_ddAPL=rho(v,u.ddAPL),added_information_beyond_ddAPL='Retains structural response; incremental phenotype CV in added_information_vs_APL.csv',recommendation_status='rejected as stand-alone target' if metric in ['ratio_BM','ratio_MB','magnitude','pc1_absolute'] else ('shortlisted with magnitude, not alone' if metric in ['contrast','angle'] else 'exploratory utility')))
 save(out,'data/pareto_fronts.csv',fronts);save(out,'tables/denominator_instability.csv',sing);save(out,'tables/decision_table.csv',rows)
 # Explicit illustrative screens on the q95 reduced late candidate, never decision thresholds.
 fam='late_vs_ref__q95__euclidean_reduced';B=effects['Effect_B__'+fam];M=effects['Effect_M__'+fam];s=combine(B.to_numpy(),M.to_numpy());d=u[['pair_id','molecule','physical_run','label_B','label_M','ddAPL']].assign(Effect_B=B,Effect_M=M,contrast=s['contrast'],magnitude=s['magnitude'],desirability=s['desirability_w0.5'])
 cases=[]
 screens={'near_zero_ddAPL_high_effect':d[abs(d.ddAPL)<=.5].sort_values('magnitude',ascending=False).head(5),'large_ddAPL_low_effect':d[abs(d.ddAPL)>=abs(d.ddAPL).quantile(.75)].sort_values('magnitude').head(5),'high_selectivity_low_magnitude':d[d.contrast>=d.contrast.quantile(.8)].sort_values('magnitude').head(5),'high_magnitude_low_selectivity':d[abs(d.contrast)<=abs(d.contrast).quantile(.25)].sort_values('magnitude',ascending=False).head(5)}
 alt=scores['desirability_w0.5__late_vs_ref__iqr__mahalanobis_reduced'];d['candidate_rank_gap']=abs(stats.rankdata(-d.desirability)-stats.rankdata(-alt));screens['candidate_disagreement']=d.sort_values('candidate_rank_gap',ascending=False).head(5)
 for name,z in screens.items():cases.extend(z.assign(screen=name).to_dict('records'))
 save(out,'tables/disagreement_cases.csv',cases)
 return pd.DataFrame(rows)


def main():
 global SEED
 ap=argparse.ArgumentParser();ap.add_argument('--master',required=True,type=Path);ap.add_argument('--selected',required=True,type=Path);ap.add_argument('--pairs',required=True,type=Path);ap.add_argument('--reference',required=True,type=Path);ap.add_argument('--output',type=Path,default=ROOT/'results/v1');ap.add_argument('--bootstrap',type=int,default=300);ap.add_argument('--cv-repeats',type=int,default=3);ap.add_argument('--seed',type=int,default=SEED);ap.add_argument('--audit-only',action='store_true');ap.add_argument('--skip-plots',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 SEED=args.seed
 out=args.output
 if (out/'analysis_manifest.json').exists():raise SystemExit('Output already has a completed run; choose a new --output to preserve previous analysis')
 for d in ['data','tables','figures']:(out/d).mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(args.seed);start=time.monotonic();sources={str(getattr(args,k).resolve()):sha(getattr(args,k)) for k in ['master','selected','pairs','reference']}
 print('Audit and unify',flush=True);all_u,info=read_unified(args,out);u=all_u[all_u.selected_pair].reset_index(drop=True);write_json(out/'data/population.json',info)
 if args.audit_only:return
 print('Mathematical stress tests and PCA',flush=True);mathematical_tests(out);pca_analysis(u,out)
 effects,models,meta=effect_engine(u,u);scores,details=all_scores(effects,meta);write_json(out/'data/frozen_parameters.json',models)
 save(out,'data/effect_scores.csv',pd.concat([u[['pair_id','molecule']],effects],axis=1));save(out,'data/candidate_scores.csv',pd.concat([u[['pair_id','molecule']],scores],axis=1))
 scaler=[];removed=[]
 for model,state in models.items():
  if not isinstance(state,dict):continue
  for n,v,w in zip(state['names'],state['scale'],state['weights']):scaler.append(dict(model=model,feature=n,scale=v,centre=0,weight_inverse_redundancy=w,n_fit=state['n_fit'],n_null=state['n_null'],scope='shared pooled B+M',method=state['method']))
  removed.extend(dict(model=model,**r) for r in state['removed'])
 save(out,'data/scaler_parameters.csv',scaler);save(out,'tables/reduced_feature_decisions.csv',removed)
 definitions=[]
 for c,v in details.items():definitions.append(dict(score_name=c,formula=(v['formula']+'; raw d: '+({'euclidean_full':'sqrt(mean(z_j^2))','euclidean_reduced':'sqrt(mean(z_j^2)) on retained indices','mahalanobis_full':'sqrt(z.T precision z / p)','mahalanobis_reduced':'sqrt(z_retained.T precision z_retained / p_retained)','linear_redundancy':'sum(weight_j*abs(z_j)); weights proportional to 1/sum(abs(rho_jk))','pc1_absolute':'abs(dot(z,oriented_PC1))','directional_reduced':'RMS of positive roughness/asymmetry deviation and positive thinning; APL absolute','single_dAPL':'abs(dAPL)/q95(abs(dAPL))'}.get(v['family'],'abs(deviation_j)/scale_j'))) if v['kind']=='effect' else next(r['formula'] for r in metric_metadata() if r['metric_name']==v['formula']),required_inputs='B/M physical deviations, scales, indices and weights in frozen_parameters.json',scaling=v['scaler']+' with origin zero; d/(1+d) for each effect',direction=v['direction'],range=v['range'],interpretation=v['family']+' '+v['view'],limitations=v['limitation'],status='exploratory'))
 save(out,'score_definitions.csv',definitions)
 print('Sensitivity and reference checks',flush=True);sensitivity(u,all_u,effects,models,out,np.random.default_rng(args.seed),args.bootstrap)
 print('Component associations and all phenotype comparisons',len(scores.columns),'scores',flush=True)
 if not args.resume or not (out/'tables/score_component_correlations.csv').exists():associations(scores,u,out,args.bootstrap,np.random.default_rng(args.seed+1))
 print('Refitting molecule-bootstrap calibrations',flush=True);stability=bootstrap_calibration(u,effects,scores,out,args.bootstrap,np.random.default_rng(args.seed+2))
 print('Leakage-safe grouped validation',flush=True);cv=cv_evaluation(u,out,args.cv_repeats,args.bootstrap,np.random.default_rng(args.seed+3))
 decisions=decision_and_pareto(u,effects,scores,details,stability,out)
 for p,h in sources.items():assert sha(p)==h,'Source modified during analysis'
 scripts={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*.py')};manifest=dict(timestamp=datetime.now(timezone.utc).isoformat(),source_hashes=sources,script_hashes=scripts,parameters={'seed':args.seed,'rng_stream_offsets':{'sensitivity':0,'associations':1,'calibration':2,'CV':3},'bootstrap':args.bootstrap,'cv_repeats':args.cv_repeats,'outer_folds':4,'inner_folds':3,'redundancy_rho_threshold':.9,'late_window':'80-100 ns median, morphology only','pseudocounts':[.01,.1,.5],'utility_weights':[.25,.5,.75],'scale_origin':0,'score_squashing':'d/(1+d)','reference_noise_floor':'unavailable: one pristine structure per membrane'},versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},population=info,effect_candidates=len(effects.columns),joint_and_effect_candidates=len(scores.columns),elapsed_seconds=time.monotonic()-start,no_jobs_submitted=True,source_hashes_unchanged=True,exclusions='tables/pair_exclusions.csv and strict_identity_exclusions.csv')
 write_json(out/'analysis_manifest.json',manifest)
 from plot_metric_analysis import plot_all
 if not args.skip_plots:plot_all(out)
 from report_metric_analysis import write_reports
 write_reports(out,ROOT)
 print('COMPLETE',out,'seconds',round(time.monotonic()-start),flush=True)
if __name__=='__main__':main()
