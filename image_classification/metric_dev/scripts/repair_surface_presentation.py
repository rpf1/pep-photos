#!/usr/bin/env python3
"""Targeted rendered-presentation repairs, retaining guide IDs and provenance."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from surface_papl_analysis import js
from surface_papl_figures import Figures

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args();r=a.result.resolve();o=r/'surface_papl_analysis';g=json.loads((r/'review_bundle.json').read_text());byid={e['figure_id']:e for e in g['figures']};F=Figures(r);rel=pd.read_csv(o/'tables/phenotype_relationships.csv');images={}
 for side,h in rel.groupby('side'):
  h=h[h.x=='delta_pAPL_late_ref_A2'];ys=['delta_height_late_ref','delta_thickness_late_ref','geometry_mean_late_nm2','delta_A_surface_mean_late_baseline_nm2','physical_deviation_late_vs_ref_q95'];piv=h[h.y.isin(ys)].pivot(index='phenotype',columns='y',values='rho').reindex(columns=ys);fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(piv,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set_xticks(range(5),['Δheight','Δthickness','Geometry','Δsurface','q95 response']);ax.set_yticks(range(len(piv)),[lab+f' (n={int(h[h.phenotype==lab].n.iloc[0])})' for lab in piv.index]);fig.colorbar(im,ax=ax,label='Spearman rho')
  for i in range(len(piv)):
   for j in range(5):
    if np.isfinite(piv.iloc[i,j]):ax.text(j,i,f'{piv.iloc[i,j]:+.2f}',ha='center',va='center')
  images['phenotype_correlations_'+side]=fig
 fid='52_nested_CV_physical_phenotype';e=byid[fid];saved=pd.read_csv(r/'surface_papl_analysis/historical_guide_evidence/52_nested_CV_physical_phenotype_0.csv');fig,axs=plt.subplots(1,2,figsize=(12,5))
 for ax,(side,h) in zip(axs,saved.groupby('side')):
  h=h.reset_index(drop=True);ax.errorbar(h.AUROC,range(len(h)),xerr=[h.AUROC-h.auc_ci_low,h.auc_ci_high-h.AUROC],fmt='o',capsize=3);ax.set_yticks(range(len(h)),[f'{row.features} (n={row.n})' for row in h.itertuples()]);ax.set(xlim=(0,1),xlabel='Historical binary-task AUROC (saved intervals)',title=side);ax.axvline(.5,ls='--',color='gray')
 images[fid]=fig
 for fid,fig in images.items():
  e=byid[fid];F.save(fig,fid,e['title'],e['family'],e['source_tables'],e['scientific_purpose'],e['observed_result'],e['interpretation'],e['limitations'],e['n'],str(e['axes']['units']),status=e['status'],variables=str(e['axes']))
  new=F.rows[-1]
  for key in ['png_path','svg_path','png_sha256','svg_sha256']:e[key]=new[key]
  e['axes']['saved_axis_limits']=new['axes']['saved_axis_limits'];e['presentation_repair']='Readable heatmap columns / explicit B/M historical binary AUROC panels with saved intervals'
  e['provenance'].append(dict(source_document='../../scripts/repair_surface_presentation.py',section_or_heading=fid,provenance_type='method_definition'))
 js(r/'review_bundle.json',g);print('Repaired three rendered figures; historical task distinguished from new nominal validation')
if __name__=='__main__':main()
