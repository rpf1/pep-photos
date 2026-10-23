"""Pure, serialisable metric transformations. No canonical state or job operations."""
import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

SEED=20261002
MEMBRANES={'B':'POPE75POPG25_W_WF','M':'POPC60CHOL40_W_WF'}
METRICS={
 'APL':('projected_APL_A2','projected_APL_A2','angstrom^2'),
 'height':('height_rms_nm','midplane_height_rms_nm','nm'),
 'excess':('midplane_excess_area_fraction','midplane_excess_area_fraction','fraction'),
 'curvature':('H_rms_nm_inv','midplane_H_rms_nm_inv','nm^-1'),
 'thickness':('thickness_median_nm','thickness_median_nm','nm'),
 'height_asym':('height_rms_abs_upper_minus_lower_nm','height_rms_abs_upper_minus_lower_nm','nm'),
 'excess_asym':('excess_area_abs_upper_minus_lower','excess_area_abs_upper_minus_lower','fraction'),
 'curvature_asym':('H_rms_abs_upper_minus_lower_nm_inv','H_rms_abs_upper_minus_lower_nm_inv','nm^-1')}
ORDER=['height','thickness','height_asym','excess','curvature','excess_asym','curvature_asym','APL']
SCALERS=['iqr','null_mad','q95']
LABELS=['flat_no_interaction','flat_interaction','bulge_flat','curved','bulge_curved','pore_formed']

def bh(p):
 p=np.asarray(p,float);out=np.full(len(p),np.nan);ok=np.isfinite(p);v=p[ok];ix=np.argsort(v);q=np.empty(len(v));q[ix]=np.minimum(1,np.minimum.accumulate((v[ix]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1]);out[ok]=q;return out

def rho(x,y):
 ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<3 or np.std(np.asarray(x)[ok])==0 or np.std(np.asarray(y)[ok])==0:return np.nan
 return float(stats.spearmanr(np.asarray(x)[ok],np.asarray(y)[ok]).statistic)

def cluster_draw(groups,rng):
 groups=np.asarray(groups);u=np.unique(groups)
 return np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(u,len(u),replace=True)])

def reduction(X,names,threshold=.9):
 """Greedy absolute Spearman redundancy, with declared physical priority order."""
 corr=np.nan_to_num(stats.spearmanr(X,axis=0).statistic,nan=0.)
 if np.ndim(corr)==0:corr=np.eye(X.shape[1])
 keep=[];removed=[]
 for name in ORDER:
  if name not in names:continue
  j=names.index(name);matches=[k for k in keep if abs(corr[j,k])>=threshold]
  if matches:removed.append({'removed':name,'retained':names[matches[0]],'absolute_rho':float(abs(corr[j,matches[0]]))})
  else:keep.append(j)
 return keep,removed

def fit_state(X,names,method='q95',null_mask=None):
 """All learned parameters belong to the training population only. Origin stays pristine."""
 X=np.asarray(X,float)
 if not np.isfinite(X).all():raise ValueError('Nonfinite input; explicit complete-case selection is required')
 if method=='iqr':scale=np.subtract(*np.quantile(X,[.75,.25],axis=0))
 elif method=='q95':scale=np.quantile(abs(X),.95,axis=0)
 elif method=='null_mad':
  null=X[np.asarray(null_mask,bool)]
  if len(null)<3:raise ValueError('Null-MAD needs >=3 no-interaction training observations')
  scale=1.4826*np.median(abs(null-np.median(null,axis=0)),axis=0)
 else:raise ValueError(method)
 if np.any(scale<=1e-12):raise ValueError('Degenerate scale; no silent epsilon or fallback')
 Z=X/scale;keep,removed=reduction(X,names)
 corr=np.nan_to_num(np.abs(stats.spearmanr(X,axis=0).statistic),nan=0.)
 weight=1/np.maximum(corr.sum(axis=1),1);weight/=weight.sum()
 pca=PCA(svd_solver='full').fit(Z);v=pca.components_[0].copy();anchor=np.argmax(abs(v));sign=1 if v[anchor]>=0 else -1;v*=sign
 precision={};condition={}
 for scope,inds in [('full',list(range(len(names)))),('reduced',keep)]:
  zz=Z[:,inds];lo,hi=np.quantile(zz,[.01,.99],axis=0);cov=LedoitWolf().fit(np.clip(zz,lo,hi))
  precision[scope]=cov.precision_.tolist();condition[scope]=float(np.linalg.cond(cov.covariance_))
 return {'names':names,'method':method,'scale':scale.tolist(),'origin':[0.]*len(names),'reduced_indices':keep,'removed':removed,'weights':weight.tolist(),'pc1_vector':v.tolist(),'pc1_orientation_multiplier':int(sign),'pc1_anchor':names[anchor],'pc1_explained':float(pca.explained_variance_ratio_[0]),'precision':precision,'condition':condition,'n_fit':len(X),'n_null':int(np.sum(null_mask)) if null_mask is not None else 0}

def transform(X,state):
 names=state['names'];Z=np.asarray(X)/np.asarray(state['scale']);out={}
 for i,n in enumerate(names):out['single_'+n]=abs(Z[:,i])
 for scope,inds in [('full',list(range(len(names)))),('reduced',state['reduced_indices'])]:
  zz=Z[:,inds];out['euclidean_'+scope]=np.sqrt(np.mean(zz**2,axis=1));out['mahalanobis_'+scope]=np.sqrt(np.maximum(0,np.einsum('ni,ij,nj->n',zz,np.asarray(state['precision'][scope]),zz))/len(inds))
 out['linear_redundancy']=abs(Z)@np.asarray(state['weights'])
 out['pc1_absolute']=abs(Z@np.asarray(state['pc1_vector']))
 # Directional comparator explicitly measures increasing roughness/asymmetry and thinning.
 directional=Z.copy()
 for j,n in enumerate(names):directional[:,j]=abs(Z[:,j]) if n=='APL' else np.maximum(0,-Z[:,j] if n=='thickness' else Z[:,j])
 out['directional_reduced']=np.sqrt(np.mean(directional[:,state['reduced_indices']]**2,axis=1))
 return out

def bounded(x):return np.asarray(x)/(1+np.asarray(x))

def combine(B,M,constants=(.01,.1,.5),weights=(.25,.5,.75)):
 """B/M in [0,1]. NaN explicitly marks raw-ratio singularities. Angle origin convention=0."""
 B=np.asarray(B,float);M=np.asarray(M,float);den=B+M
 with np.errstate(divide='ignore',invalid='ignore'):
  out={'difference':B-M,'ratio_BM':np.divide(B,M,out=np.full_like(B,np.nan),where=M!=0),'ratio_MB':np.divide(M,B,out=np.full_like(B,np.nan),where=B!=0),'contrast':np.divide(B-M,den,out=np.zeros_like(B),where=den!=0),'angle':np.where(den==0,0,4/np.pi*np.arctan2(B,M)-1),'magnitude':np.hypot(B,M)}
 for c in constants:out[f'logratio_c{c:g}']=np.log((B+c)/(M+c))
 for w in weights:
  out[f'desirability_w{w:g}']=B**w*(1-M)**(1-w)
  out[f'linear_utility_w{w:g}']=w*B+(1-w)*(1-M)
  out[f'ideal_w{w:g}']=1-np.sqrt(w*(1-B)**2+(1-w)*M**2)
 return out

def dominance(B,M):
 B=np.asarray(B);M=np.asarray(M)
 return (B[:,None]>=B[None,:])&(M[:,None]<=M[None,:])&((B[:,None]>B[None,:])|(M[:,None]<M[None,:]))

def metric_metadata():
 rows=[]
 def add(name,formula,ran,bounded_,singular,zero,monoB,monoM,pareto,const,meaning,failure):
  rows.append(dict(metric_name=name,formula=formula,range=ran,bounded=bounded_,singularity=singular,zero_zero_behavior=zero,monotonic_in_B=monoB,monotonic_penalty_in_M=monoM,Pareto_consistent=pareto,arbitrary_constant_required=const,interpretation=meaning,principal_failure_mode=failure))
 add('difference','B-M','[-1,1]',True,False,0,True,True,True,False,'Differential effect','Both low and both high share zero')
 add('ratio_BM','B/M','[0,infinity)',False,True,'NaN',True,True,'finite domain only',False,'Relative response','Diverges as M approaches zero; weak B can rank highest')
 add('ratio_MB','M/B','[0,infinity)',False,True,'NaN',False,False,False,False,'Inverse relative response; smaller favours B','Wrong direction for better-is-larger; singular at B=0')
 add('contrast','(B-M)/(B+M); origin=0','[-1,1]',True,'directional discontinuity at origin',0,True,True,True,False,'Bounded directional contrast','Weak B with zero M scores +1; needs magnitude')
 add('angle','4/pi*atan2(B,M)-1; origin=0','[-1,1]',True,'directional discontinuity at origin',0,True,True,True,False,'Angular direction','Monotonic transform of contrast; origin path-dependent; needs magnitude')
 add('magnitude','sqrt(B^2+M^2)','[0,sqrt(2)]',True,False,0,True,False,False,False,'Total response','Not selectivity; rewards mammalian effect too')
 for c in [.01,.1,.5]:add(f'logratio_c{c:g}',f'log((B+{c})/(M+{c}))',f'[-{np.log((1+c)/c):.6g},+{np.log((1+c)/c):.6g}]',True,False,0,True,True,True,True,'Regularised relative response','Ranking and attenuation depend on pseudocount')
 for w in [.25,.5,.75]:
  add(f'desirability_w{w:g}',f'B^{w}*(1-M)^{1-w}','[0,1]',True,False,0,True,True,True,True,'Weighted bacterial response and mammalian low-effect utility','Trade-off weights; zero if B=0 or M=1; strict dominance can tie')
  add(f'linear_utility_w{w:g}',f'{w}*B+{1-w}*(1-M)','[0,1]',True,False,1-w,True,True,True,True,'Additive utility','No-effect origin receives nonzero utility')
  add(f'ideal_w{w:g}',f'1-sqrt({w}*(1-B)^2+{1-w}*M^2)','[0,1]',True,False,1-np.sqrt(w),True,True,True,True,'Closeness to ideal B=1,M=0','Ideal and weight chosen, no-effect origin can score well')
 return rows
