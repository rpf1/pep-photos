#!/usr/bin/env python3
"""Targeted continuation diagnostics from saved tables; no nested CV or frozen refit."""
import argparse,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from surface_papl_analysis import js,table,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args();r=a.result.resolve();o=r/'surface_papl_analysis';d=pd.read_csv(o/'tables/matched_late_dataset.csv');assert len(d)==312
 rows=[]
 checks=[('pAPL_endpoint_A2','pAPL_late_A2'),('physical_deviation_endpoint_vs_ref_q95','physical_deviation_late_vs_ref_q95'),('delta_pAPL_late_ref_A2','delta_height_late_ref'),('delta_pAPL_late_ref_A2','delta_thickness_late_ref'),('delta_pAPL_late_ref_A2','delta_A_surface_mean_late_baseline_nm2'),('delta_pAPL_late_ref_A2','geometry_mean_late_nm2')]
 for s,g in d.groupby('side'):
  for x,y in checks:
   h=g[g.molecule!='Ns-NmO-Nab'];full=spearmanr(g[x],g[y]).statistic;excluded=spearmanr(h[x],h[y]).statistic
   rows.append(dict(side=s,x=x,y=y,n_full=len(g),n_excluded=len(h),rho_full=full,rho_excluded=excluded,delta_rho=excluded-full,excluded_molecule='Ns-NmO-Nab',scope='association sensitivity only; classifier refit not performed'))
 table(o,'endpoint_exclusion_sensitivity.csv',rows)
 # Unit-consistent decomposition, reference comparison and rank summaries.
 summary=[];rankrows=[];rr=pd.read_csv(o/'tables/rank_disagreement.csv')
 for s,g in d.groupby('side'):
  assert np.allclose(g.pAPL_start_A2,g.pAPL_reference_A2,atol=1e-8,rtol=0)
  assert np.allclose(g.delta_sAPL_proxy_late_baseline_A2,g.delta_pAPL_late_start_A2+g.delta_geometry_area_contribution_A2,atol=1e-8,rtol=0)
  for lab,h in [('ALL',g)]+list(g.groupby('phenotype')):
   for v in ['geometry_mean_late_nm2','delta_geometry_mean_late_baseline_nm2','upper_geometry_share','surface_asymmetry_late_nm2','geometry_area_contribution_A2','delta_pAPL_late_ref_A2','delta_sAPL_proxy_late_baseline_A2']:
    summary.append(dict(side=s,phenotype=lab,variable=v,n=len(h),median=h[v].median(),q25=h[v].quantile(.25),q75=h[v].quantile(.75),minimum=h[v].min(),maximum=h[v].max()))
   rg=rr[(rr.side==s)&rr.pair_id.isin(h.pair_id)]
   for c in ['surface','q95','geometry']:
    rankrows.append(dict(side=s,phenotype=lab,comparison=c+' minus pAPL',n=len(h),median_abs_rank_gap=np.median(abs(rg[c+'_rank']-rg.pAPL_rank)),max_abs_rank_gap=np.max(abs(rg[c+'_rank']-rg.pAPL_rank)),rho=spearmanr(rg[c+'_rank'],rg.pAPL_rank).statistic))
 table(o,'leaflet_geometry_summary.csv',summary);table(o,'phenotype_rank_disagreement_summary.csv',rankrows)
 # Comparisons across scaling choices use correlation only when units differ.
 ag=pd.read_csv(o/'tables/rank_and_endpoint_agreement.csv');units={'pAPL_endpoint_A2':'A2','pAPL_late_A2':'A2','pAPL_late_binned_A2':'A2','delta_pAPL_late_ref_A2':'A2','delta_A_surface_mean_late_baseline_nm2':'nm2'}
 ag['difference_interpretation']=['not applicable (B/M reference states)' if row.side=='B versus M' else 'not applicable (different units)' if units.get(row.a,'unitless')!=units.get(row.b,'unitless') else 'same units; different definitions/scales' for row in ag.itertuples()]
 ag.loc[ag.difference_interpretation.str.startswith('not applicable'),'median_absolute_difference']=np.nan
 table(o,'rank_and_endpoint_agreement_reviewed.csv',ag)
 pars=json.loads((o/'exploratory_PCA/parameters.json').read_text());audit=[]
 for model,st in pars.items():
  ff=st['features'];side=model.split('__')[1]
  if side=='paired':
   b=d[d.side=='B'].set_index('pair_id');m=d[d.side=='M'].set_index('pair_id').loc[b.index];X=np.c_[b[[f[2:] for f in ff if f.startswith('B_')]],m[[f[2:] for f in ff if f.startswith('M_')]]]
  else:X=d[d.side==side][ff].to_numpy()
  for i,j in itertools.combinations(range(len(ff)),2):
   rho=spearmanr(X[:,i],X[:,j]).statistic
   audit.append(dict(model=model,feature_1=ff[i],feature_2=ff[j],rho=rho,redundancy_flag=abs(rho)>=.9,n=156))
 table(o,'exploratory_PCA_feature_redundancy.csv',audit)
 # Explicit eligible-class coverage; votes summarize repeated OOF predictions without new fitting.
 pred=pd.read_csv(o/'tables/type_discrimination_predictions.csv');coverage=[]
 for (s,lab),g in d.groupby(['side','phenotype']):coverage.append(dict(side=s,phenotype=lab,n=len(g),included=len(g)>=8,reason='eligible' if len(g)>=8 else 'n<8; descriptive only'))
 table(o,'classifier_class_coverage.csv',coverage)
 voted=[]
 for key,g in pred.groupby(['side','feature_set','pair_id']):
  vc=g.prediction.value_counts();maximum=vc.max();ties=sorted(vc[vc==maximum].index);voted.append(dict(side=key[0],feature_set=key[1],pair_id=key[2],truth=g.truth.iloc[0],prediction=ties[0],n_repeats=len(g),vote_tie=len(ties)>1,correct=ties[0]==g.truth.iloc[0]))
 table(o,'classifier_repeated_OOF_votes.csv',voted)
 js(o/'diagnostics_manifest.json',dict(cohort_systems=312,endpoint_discrepancy_threshold_A2=1e-6,flagged=d.loc[abs(d.frame_pAPL_endpoint_error_A2)>1e-6,['pair_id','side','frame_pAPL_endpoint_error_A2']].to_dict('records'),decomposition_identity='delta_sAPL_proxy = delta_pAPL_start + delta_geometry_contribution',identity_tolerance_A2=1e-8,reference_start_equal_A2=True,cv_reused=True,pristine_results='unavailable',frozen_refit=False))
 print('Targeted diagnostics complete; saved CV reused; no frozen fit.')
if __name__=='__main__':main()
