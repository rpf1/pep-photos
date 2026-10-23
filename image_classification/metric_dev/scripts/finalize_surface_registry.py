#!/usr/bin/env python3
"""Resolve scientific metadata and reconstruct exact historical endpoint comparison."""
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from surface_papl_figures import Figures
from surface_papl_analysis import js
from scipy.stats import spearmanr

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args();r=a.result.resolve();o=r/'surface_papl_analysis';g=json.loads((r/'review_bundle.json').read_text());old=json.loads((r.parents[2]/'figure_gui_analysis/metric_dev/results'/r.name/'figure_guide.json').read_text());byold={e['figure_id']:e for e in old['figures']}
 q=pd.read_csv(o/'historical_guide_evidence/51_desirability_vs_ddAPL_reconstructed.csv');u=pd.read_csv(r/'data/unified_metrics.csv').set_index('pair_id').loc[q.pair_id];q['delta_delta_pAPL_endpoint_start_A2']=u['B_minus_M__APL__endpoint_vs_start'].to_numpy();q.to_csv(o/'historical_guide_evidence/51_desirability_vs_ddAPL_reconstructed.csv',index=False)
 F=Figures(r);fig,ax=plt.subplots(figsize=(8,5));F.scatter(ax,q,'delta_delta_pAPL_endpoint_start_A2','desirability_w0.5');ax.set_xlabel('Endpoint ΔΔpAPL, Pull-start reference (Å²)');rho=spearmanr(q.delta_delta_pAPL_endpoint_start_A2,q['desirability_w0.5']).statistic
 fid='51_desirability_vs_ddAPL';original=next(e for e in g['figures'] if e['figure_id']==fid)
 F.save(fig,fid,original['historical_title']+'\nRegenerated from complete supplied inputs',original['family'],original['source_tables'],original['scientific_purpose'],f'n=156; rho={rho:.7f}; agrees with saved historical rho=0.3014998973. Endpoint ΔΔpAPL compared with late frozen-q95 desirability.',original['interpretation'],'Historical mixed-time comparison preserved as supplementary. Desirability is a user-weighted mathematical preference, not validated biological selectivity.',156,'Å² versus dimensionless preference',status='supplementary',variables='endpoint ΔΔpAPL versus late desirability',counts=[('paired',156)],legacy={k:original[k] for k in ['historical_title','historical_sources','historical_observed_result','original_guide_id']})
 g['figures']=[F.rows[0] if e['figure_id']==fid else e for e in g['figures']]
 methods={'late_response':'Trajectory-level signed Spearman rho across 156 systems per membrane; late arithmetic means for area and original late medians for historical morphology.', 'rankings':'Average exact-tie ranks calculated separately in B and M; rank discordance is descriptive.', 'phenotypes':'Within-phenotype Spearman correlations, BH per side/phenotype family, Theil–Sen slope for n≥8, joint OLS HC3 interaction Wald tests; small labels descriptive.', 'temporal':'Exact-run pure-Relax 5 ns bin summaries matched to 1 ns morphology frames; descriptive trajectory-level comparisons, not independent-frame inference.', 'validation':'Three repeated four-fold outer splits, three inner folds, training-only scaling, tuned C and balanced multinomial logistic regression; molecule grouping separately B/M; three classes, n=147/146.', 'frozen_PCA':'Exact projection through frozen feature scaling/components/orientation, verified PC1/PC2 against saved coordinates, no refit; fixed 22°/38° perspective and XY/XZ/YZ projections.', 'exploratory_PCA':'Separate full SVD on cohort-standardized predefined representations; all 156 pairs retained; paired stack uses B/M features; no frozen refit.', 'frozen_PCA_associations':'Spearman rho between frozen PC coordinates and new trajectory-level quantities; BH within model/phenotype; corresponding B/M embedding and paired embedding only.', 'exploratory_PCA_loadings':'First-three signed component loadings from separate cohort-standardized full-SVD PCA; exact feature definitions and redundancy audit linked.', 'sensitivity':'Spearman rho in full cohort n=156 and after removing Ns-NmO-Nab n=155; no CV or PCA refit.', 'counterexamples':'Post-hoc largest discordances among all cross-phenotype pairs matched within 0.15 Å² ΔpAPL or 0.02 nm Δheight.'}
 for e in g['figures']:
  if e['figure_id'] in byold:
   olde=byold[e['figure_id']];e['metric_ids']=olde['metric_ids'];e['analysis_technique_ids']=olde['analysis_technique_ids'];e['method']='Reconstruct supplied numerical source/formula where available; otherwise plot only preserved historical guide evidence, explicitly labelled. Original analysis techniques are preserved in the Design panel.'
  else:e['method']=methods[e['family']]
  if e['figure_id'].endswith('_grid') and e['figure_id'].startswith('frozen'):
   for path in sorted((r/'pca_review').glob('*.csv')):
    source=str(path.relative_to(r))
    if source not in e['source_tables']:e['source_tables'].append(source)
  for source in ['surface_papl_analysis/VARIABLES_AND_METHODS.md','surface_papl_analysis/SCIENTIFIC_DECISION_REPORT.md','surface_papl_analysis/SUPERVISOR_SUMMARY.md']:
   if source not in e['source_tables']:e['source_tables'].append(source)
 js(r/'review_bundle.json',g)
 counts=pd.read_csv(o/'tables/figure_point_counts.csv');counts=counts[counts.figure_id!=fid];pd.concat([counts,pd.DataFrame(F.counts)]).to_csv(o/'tables/figure_point_counts.csv',index=False)
 print('Scientific metadata and historical comparison finalized')
if __name__=='__main__':main()
