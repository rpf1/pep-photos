#!/usr/bin/env python3
"""Add association, loading, temporal and sensitivity figures to the same registry."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from surface_papl_figures import Figures,display_label
from surface_papl_analysis import js

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args();r=a.result.resolve();o=r/'surface_papl_analysis';bundle=json.loads((r/'review_bundle.json').read_text());F=Figures(r);d=pd.read_csv(o/'tables/matched_late_dataset.csv');assoc=pd.read_csv(o/'tables/frozen_PCA_associations.csv')
 for model,g in assoc[assoc.phenotype=='ALL'].groupby('model'):
  piv=g.pivot(index='variable',columns='PC',values='rho').reindex(columns=['PC1','PC2','PC3']);fig,ax=plt.subplots(figsize=(8,max(4,len(piv)*.4)));im=ax.imshow(piv,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set_xticks(range(3),['PC1','PC2','PC3']);ax.set_yticks(range(len(piv)),[display_label(v) for v in piv.index]);fig.colorbar(im,ax=ax,label='Trajectory-level Spearman rho')
  for i in range(len(piv)):
   for j in range(3):ax.text(j,i,f'{piv.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8)
  F.save(fig,'association_'+model,model+': new quantities versus frozen coordinates','frozen_PCA_associations',['surface_papl_analysis/tables/frozen_PCA_associations.csv','surface_papl_analysis/tables/matched_late_dataset.csv','surface_papl_analysis/tables/paired_quantities.csv'],'Which new quantities associate with frozen PCs?',f'n=156; signed rho annotated; maximum |rho|={g.rho.abs().max():.4f}.','B/M quantities use their corresponding embedding; paired differences use paired PCs. Shared feature content makes some associations mathematical.','BH per model/phenotype family; no causal interpretation; endpoint PCA versus late quantities is a sensitivity comparison.',156,'rho (unitless)',status='sensitivity' if model.startswith('endpoint') else 'primary',variables='new physical variables versus PC1/PC2/PC3')
 loads=pd.read_csv(o/'tables/exploratory_PCA_loadings.csv');red=pd.read_csv(o/'tables/exploratory_PCA_feature_redundancy.csv')
 for model,g in loads.groupby('model'):
  piv=g[g.component<=3].pivot(index='feature',columns='component',values='loading');fig,ax=plt.subplots(figsize=(9,max(5,len(piv)*.4)));im=ax.imshow(piv,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set_xticks(range(3),['PC1','PC2','PC3']);ax.set_yticks(range(len(piv)),piv.index);fig.colorbar(im,ax=ax,label='Loading')
  for i in range(len(piv)):
   for j in range(3):ax.text(j,i,f'{piv.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8)
  nred=int(red[(red.model==model)&red.redundancy_flag].shape[0])
  F.save(fig,'loadings_'+model,model+': feature design and loadings','exploratory_PCA_loadings',['surface_papl_analysis/tables/exploratory_PCA_loadings.csv','surface_papl_analysis/tables/exploratory_PCA_feature_redundancy.csv','surface_papl_analysis/exploratory_PCA/parameters.json'],'What variables dominate the exploratory representation?',f'{nred} feature pairs have |Spearman rho|≥0.9; first-three loadings shown.','Projected, surface and decomposed area inputs are separated; remaining morphology variables still share empirical information.','Independent physical roles do not imply statistical independence. All feature scaling and fitting uses this cohort; no prospective validation.',156,'dimensionless loading',status='exploratory',variables='exact feature rows by principal component')
 se=pd.read_csv(o/'tables/endpoint_exclusion_sensitivity.csv');fig,axs=plt.subplots(1,2,figsize=(12,6))
 for ax,(s,g) in zip(axs,se.groupby('side')):
  ax.plot(g.rho_full,range(len(g)),'o',label='All 156');ax.plot(g.rho_excluded,range(len(g)),'x',label='Excluded: n=155');ax.set_yticks(range(len(g)),[display_label(v) for v in g.y]);ax.set(xlim=(-1,1),xlabel='Spearman rho',title=s);ax.legend()
 F.save(fig,'endpoint_discrepancy_exclusion','Sensitivity to excluding Ns-NmO-Nab','sensitivity',['surface_papl_analysis/tables/endpoint_exclusion_sensitivity.csv','surface_papl_analysis/tables/endpoint_pAPL_reconciliation.csv'],'Does the known endpoint mismatch drive the selected associations?',f'Maximum |rho change|={se.delta_rho.abs().max():.5f}.','Selected aggregate relationships are qualitatively stable under this exclusion.','Only association sensitivity was performed; CV models were not refitted and individual rankings may move.',dict(full_per_side=156,excluded_per_side=155),'rho',status='sensitivity',variables='full versus excluded cohort correlation')
 cs=pd.read_csv(o/'tables/near_matched_counterexamples.csv');chosen=cs.groupby(['side','match_variable'],sort=True).head(1).copy();chosen.to_csv(o/'tables/highlighted_counterexamples.csv',index=False);fig,axs=plt.subplots(2,2,figsize=(12,10));counts=[]
 for ax,row in zip(axs.flat,chosen.itertuples()):
  h=d[d.side==row.side];F.scatter(ax,h,row.match_variable,row.discordant_variable)
  for pid in [row.pair_1,row.pair_2]:
   v=h[h.pair_id==pid].iloc[0];ax.scatter(v[row.match_variable],v[row.discordant_variable],s=130,facecolors='none',edgecolors='black');ax.annotate(v.molecule,(v[row.match_variable],v[row.discordant_variable]),xytext=(8,7),textcoords='offset points',fontsize=8)
  ax.set_title(f'{row.side}; matching tolerance {row.match_tolerance:g}');counts.append((row.side+'_'+row.match_variable,len(h)))
 F.save(fig,'matched_counterexample_maps','Near-matched responses with discordant geometry','counterexamples',['surface_papl_analysis/tables/highlighted_counterexamples.csv','surface_papl_analysis/tables/near_matched_counterexamples.csv','surface_papl_analysis/tables/matched_late_dataset.csv'],'Can similar projected response or height hide different geometry or projected change?','Largest eligible cross-phenotype gaps are highlighted in each membrane and match direction.','pAPL, height and geometry carry distinguishable information.','Post-hoc extremes; thresholds 0.15 Å² and 0.02 nm are descriptive matching rules, not measurement equivalence tests.',312,'Å², nm, nm²',variables='ΔpAPL/Δheight versus discordant response',counts=counts)
 bb=pd.read_csv(o/'tables/temporal_bin_coupling_source.csv');fig,axs=plt.subplots(2,2,figsize=(12,8));selected=[]
 for (side,h),axes in zip(d[d.binned_pAPL_available].groupby('side'),axs):
  h=h.assign(abschange=h.pAPL_early_late_change_A2.abs()).sort_values(['abschange','pair_id'],ascending=[False,True]).head(3)
  for row in h.itertuples():
   q=bb[(bb.side==side)&(bb.pair_id==row.pair_id)].sort_values('local_start_ns');time=q.local_start_ns+2.5;axes[0].plot(time,q.pAPL_A2,label=row.molecule);axes[1].plot(time,q.geometry_nm2,label=row.molecule);selected.extend(q.to_dict('records'))
  for ax in axes:ax.axvspan(80,100,color='gray',alpha=.12);ax.legend(fontsize=7);ax.set(xlabel='Relax-local bin centre (ns)',title=side)
  axes[0].set_ylabel('pAPL bin mean (Å²)');axes[1].set_ylabel('Curved minus projected area (nm²)')
 pd.DataFrame(selected).to_csv(o/'tables/highlighted_temporal_trajectories.csv',index=False)
 F.save(fig,'temporal_selected_trajectories','Largest early-to-late projected changes in the available subset','temporal',['surface_papl_analysis/tables/highlighted_temporal_trajectories.csv','surface_papl_analysis/tables/temporal_coupling.csv','surface_papl_analysis/tables/missing_inputs.csv'],'How do projected and geometric changes evolve within selected trajectories?','Three systems per membrane chosen by |early-to-late pAPL mean change|; actual saved bin means shown.','Time evolution qualifies endpoint interpretations; matched averaging windows make geometric comparison interpretable.','Selected extremes from 24 systems per side; first mixed Pull/Relax bin excluded; trajectories and bins are not independent samples.',6,'ns, Å², nm²',status='supplementary',variables='pAPL and geometry versus time')
 # Every central numeric table is discoverable through existing Sources selector.
 for entry in F.rows:entry['generator']='scripts/surface_papl_supplement.py';entry['provenance']=[dict(source_document='../../scripts/surface_papl_supplement.py',section_or_heading=entry['figure_id'],provenance_type='method_definition')]
 newids={e['figure_id'] for e in F.rows};bundle['figures']=[e for e in bundle['figures'] if e['figure_id'] not in newids]+F.rows
 for entry in bundle['figures']:
  for source in ['surface_papl_analysis/VARIABLES_AND_METHODS.md','surface_papl_analysis/SCIENTIFIC_DECISION_REPORT.md']:
   if source not in entry['source_tables']:entry['source_tables'].append(source)
 for family in {e['family'] for e in F.rows}:bundle['figure_families'][family]=dict(name=family.replace('_',' '))
 js(r/'review_bundle.json',bundle)
 counts=pd.read_csv(o/'tables/figure_point_counts.csv');counts=counts[~counts.figure_id.isin(newids)];pd.concat([counts,pd.DataFrame(F.counts)]).to_csv(o/'tables/figure_point_counts.csv',index=False)
 js(o/'figure_generation.json',dict(figures=len(bundle['figures']),PNG=len(bundle['figures']),SVG=len(bundle['figures']),PCA_grids=sum(e['figure_id'].endswith('_grid') for e in bundle['figures']),canonical_colours=F.colours))
 print('Added',len(F.rows),'figures; total',len(bundle['figures']))
if __name__=='__main__':main()
