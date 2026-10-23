#!/usr/bin/env python3
"""Generate figures and extend the registry consumed by the existing reviewer."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from surface_papl_analysis import js,sha,clean
from review_saved_pca import resolve_palette
from metric_core import combine
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def display_label(name):
 return {'delta_pAPL_late_ref_A2':'Late ΔpAPL, pristine reference (Å²)', 'delta_pAPL_late_start_A2':'Late ΔpAPL, Pull start (Å²)', 'delta_height_late_ref':'Late Δheight RMS (nm)', 'delta_thickness_late_ref':'Late Δthickness (nm)', 'delta_A_surface_mean_late_baseline_nm2':'Late Δmean leaflet surface area (nm²)', 'geometry_mean_late_nm2':'Mean curved minus projected area (nm²)', 'geometry_upper_late_nm2':'Upper curved minus projected area (nm²)', 'geometry_lower_late_nm2':'Lower curved minus projected area (nm²)', 'physical_deviation_late_vs_ref_q95':'Late physical deviation, frozen q95 (unitless)', 'physical_deviation_endpoint_vs_ref_q95':'Endpoint physical deviation, frozen q95', 'physical_deviation_late_vs_ref_null_mad':'Late physical deviation, frozen null-MAD', 'pAPL_late_A2':'Late pAPL (Å²)', 'pAPL_endpoint_A2':'Endpoint pAPL (Å²)', 'pAPL_late_binned_A2':'Binned late pAPL (Å²)', 'pAPL_early_late_change_A2':'Early-to-late pAPL change (Å²)'}.get(name,name.replace('_',' '))

class Figures:
 def __init__(self,result):
  self.r=result;self.out=result/'surface_papl_analysis';self.rows=[];self.counts=[];self.colours,_=resolve_palette(['flat_no_interaction','flat_interaction','bulge_flat','curved','bulge_curved']);self.markers=dict(zip(self.colours,['s','o','^','D','P']))
 def save(self,fig,fid,title,family,sources,question,result,interpretation,limits,n,units,status='primary',variables='',counts=None,legacy=None):
  fig.suptitle(title,fontsize=12);fig.tight_layout(rect=(0,0,1,.94),pad=2);paths={}
  for ext in ['png','svg']:
   p=self.out/'figures'/ext/family/(fid+'.'+ext);p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=160,bbox_inches='tight');paths[ext]=str(p.relative_to(self.r))
  axes=[dict(xlim=list(a.get_xlim()),ylim=list(a.get_ylim()),xlabel=a.get_xlabel(),ylabel=a.get_ylabel()) for a in fig.axes];plt.close(fig)
  entry=dict(figure_id=fid,title=title,family=family,png_path=paths['png'],svg_path=paths['svg'],scientific_purpose=question,original_question=question,original_hypothesis='Exploratory comparison specified by the continuation task; no directional result assumed.',hypothesis_status='exploratory_no_directional_hypothesis',hypothesis_provenance=[dict(source_document='surface_papl_analysis/TASK_SPECIFICATION.md',section_or_heading=family,provenance_type='method_definition')],expected_pattern={'support':'Consistent trajectory-level associations or validated label-recovery differences.','against':'Within-class counterexamples, overlap and validation reversals.'},axes={'variables':variables,'units':units,'saved_axis_limits':axes},colour_grouping_semantics='Canonical nominal phenotype colours and redundant markers for sample plots; B/M panel titles define membrane.',figure_specific_context=[f'{status}; cohort/subset n={n}; '+variables],what_to_look_for='Inspect effect sizes and overlap, exact n, source tables and counterexamples before inferring a relationship.',observed_result=result,interpretation=interpretation,limitations=limits,common_misreadings='Label recovery, membrane response, contact/burial and biological efficacy are different constructs.',source_tables=sources,provenance=[dict(source_document='../../scripts/surface_papl_figures.py',section_or_heading=fid,provenance_type='method_definition')],result_provenance=[dict(source_document=sources[0],section_or_heading=variables or fid,provenance_type='completed_result')],numeric_evidence=[dict(source_table=sources[0],row_filter='See figure-specific variables and panel definitions; source is previewable',columns=[variables or fid],records=[dict(observed_result=result,n=n)])],metric_ids=['continuation'],analysis_technique_ids=['continuation'],related_figures=[],status=status,n=n,generator='scripts/surface_papl_figures.py',method='Trajectory-level Spearman correlations / grouped nested logistic classification / exact frozen PCA projection / separate standardized exploratory PCA, as specified by analysis family; exact definitions in VARIABLES_AND_METHODS.md.',png_sha256=sha(self.r/paths['png']),svg_sha256=sha(self.r/paths['svg']))
  if legacy:entry.update(legacy)
  self.rows.append(entry)
  if counts:
   for panel,nn in counts:self.counts.append(dict(figure_id=fid,panel=panel,n_plotted=nn))
 def scatter(self,ax,g,x,y):
  for lab,h in g.groupby('phenotype'):
   ax.scatter(h[x],h[y],s=25,c=self.colours[lab],marker=self.markers[lab],label=f'{lab} (n={len(h)})',alpha=.75,linewidths=.3,edgecolors='white')
  ax.grid(alpha=.15);ax.set(xlabel=display_label(x),ylabel=display_label(y));ax.legend(fontsize=7)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,default=ROOT/'results/archie_8193723_complete');a=ap.parse_args();r=a.result.resolve();o=r/'surface_papl_analysis';F=Figures(r);d=pd.read_csv(o/'tables/matched_late_dataset.csv');rel=pd.read_csv(o/'tables/phenotype_relationships.csv')
 # Full matched late evidence; additive decomposition uses frame means.
 for x,y,name,title in [
 ('delta_pAPL_late_ref_A2','delta_height_late_ref','papl_height','Projected response versus deformation'),
 ('delta_pAPL_late_ref_A2','delta_thickness_late_ref','papl_thickness','Projected response versus thickness'),
 ('delta_pAPL_late_start_A2','delta_A_surface_mean_late_baseline_nm2','papl_surface','Projected response versus curved-surface response'),
 ('delta_pAPL_late_ref_A2','geometry_mean_late_nm2','papl_geometry','Projected response versus geometry correction'),
 ('delta_height_late_ref','physical_deviation_late_vs_ref_q95','height_q95','Deformation versus multivariate physical deviation'),
 ('physical_deviation_late_vs_ref_q95','geometry_mean_late_nm2','magnitude_geometry','Physical response versus geometry'),
 ('geometry_upper_late_nm2','geometry_lower_late_nm2','leaflet_geometry','Upper versus lower geometry contribution'),
 ('pAPL_endpoint_A2','pAPL_late_A2','endpoint_late','Endpoint versus matched late pAPL'),
 ('physical_deviation_endpoint_vs_ref_q95','physical_deviation_late_vs_ref_q95','q95_endpoint_late','Frozen q95 endpoint versus late'),
 ('physical_deviation_late_vs_ref_null_mad','physical_deviation_late_vs_ref_q95','nullmad_q95','Historical null-MAD versus q95 scaling')]:
  fig,axs=plt.subplots(1,2,figsize=(12,5));obs=[];counts=[]
  for ax,(side,g) in zip(axs,d.groupby('side')):
   F.scatter(ax,g,x,y);ax.set_title(f'{side}; n={len(g)}');counts.append((side,len(g)));rho=g[[x,y]].corr(method='spearman').iloc[0,1];obs.append(f'{side}: Spearman rho={rho:.3f}, n={len(g)}')
  F.save(fig,name,title,'late_response',['surface_papl_analysis/tables/matched_late_dataset.csv','surface_papl_analysis/tables/phenotype_relationships.csv','surface_papl_analysis/tables/near_matched_counterexamples.csv'],f'What does {x} capture about {y}?','; '.join(obs),'Associations depend on the membrane and phenotype; linked within-class and matched-case tables test counterexamples.','One physical run per molecule; summaries are autocorrelated-frame aggregates. Endpoint BOX discrepancy for Ns-NmO-Nab is recorded; score correlations with constituent metrics are partly mathematical.',312,'x/y suffixes declare Å², nm², nm or unitless',status='sensitivity' if name in ['endpoint_late','q95_endpoint_late','nullmad_q95'] else 'primary',variables=x+' versus '+y,counts=counts)
 # Complete ranking discordances, source tables accessible through GUI.
 rd=pd.read_csv(o/'tables/rank_disagreement.csv');fig,axs=plt.subplots(1,2,figsize=(11,5))
 for ax,(side,g) in zip(axs,rd.groupby('side')):
  F.scatter(ax,g,'pAPL_rank','geometry_rank');ax.plot([1,156],[1,156],'k--',lw=.6);ax.set_title(f'{side}; average ties; low to high')
 F.save(fig,'rank_projected_geometry','Projected-response and geometry ranks','rankings',['surface_papl_analysis/tables/rank_disagreement.csv','surface_papl_analysis/tables/complete_rankings_B.csv','surface_papl_analysis/tables/complete_rankings_M.csv','surface_papl_analysis/tables/BM_rank_side_by_side.csv'],'Do projected response and geometry rank the same systems?',f'Maximum within-system rank gaps: B={rd[rd.side=="B"].max_rank_gap.max():g}; M={rd[rd.side=="M"].max_rank_gap.max():g}.','Rank disagreement motivates separate projected, geometry and multivariate-response constructs.','Average ranks resolve exact ties; absolute B/M areas must be ranked separately.',312,'rank',variables='pAPL response rank versus geometry rank',counts=[('B',156),('M',156)])
 # Phenotype-specific correlation heatmaps.
 for side,g in rel.groupby('side'):
  g=g[g.x=='delta_pAPL_late_ref_A2'];ys=['delta_height_late_ref','delta_thickness_late_ref','geometry_mean_late_nm2','delta_A_surface_mean_late_baseline_nm2','physical_deviation_late_vs_ref_q95'];p=g[g.y.isin(ys)].pivot(index='phenotype',columns='y',values='rho').reindex(columns=ys);fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(p.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(ys)),['Δheight','Δthickness','Geometry','Δsurface','q95 response']);ax.set_yticks(range(len(p)),[lab+' (n='+str(int(g[g.phenotype==lab].n.iloc[0]))+')' for lab in p.index]);fig.colorbar(im,ax=ax,label='Spearman rho')
  for i in range(len(p)):
   for j in range(len(ys)):
    if np.isfinite(p.iloc[i,j]):ax.text(j,i,f'{p.iloc[i,j]:+.2f}',ha='center',va='center')
  F.save(fig,f'phenotype_correlations_{side}',f'{side}: projected-response relationships within phenotype','phenotypes',['surface_papl_analysis/tables/phenotype_relationships.csv','surface_papl_analysis/tables/phenotype_interaction_tests.csv'],'Does phenotype change the relationship between projected response and morphology?','Exact correlations, group n, robust slopes and joint interaction tests are in the linked tables.','Pooled trends can conceal within-class reversals; nominal labels are not ordered strength levels.','Groups n<8 are descriptive; BH adjustment is per side/phenotype correlation family. No equivalence follows from non-significance.',156,'rho, dimensionless',variables='within-phenotype correlations')
 # Temporal subset: exact mapping and no frame-level inference.
 t=d[d.binned_pAPL_available];fig,axs=plt.subplots(1,2,figsize=(11,5));obs=[]
 for ax,(side,g) in zip(axs,t.groupby('side')):
  F.scatter(ax,g,'pAPL_late_binned_A2','pAPL_late_A2');lo=min(g.pAPL_late_A2.min(),g.pAPL_late_binned_A2.min());hi=max(g.pAPL_late_A2.max(),g.pAPL_late_binned_A2.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.6);ax.set_title(f'{side}; n={len(g)}');obs.append(f'{side}: median absolute sampling difference={np.median(abs(g.pAPL_binned_minus_1ns_late_A2)):.4f} Å²')
 F.save(fig,'binned_frame_reconciliation','Late pAPL: 0.1 ns bins versus 1 ns surface sampling','temporal',['surface_papl_analysis/tables/matched_late_pAPL.csv','surface_papl_analysis/tables/temporal_coupling.csv','surface_papl_analysis/tables/temporal_bin_coupling_source.csv','surface_papl_analysis/tables/missing_inputs.csv'],'Do the available binned outputs support the matched late frame representation?','; '.join(obs),'Time windows match after subtracting the 100 ns Pull offset; sampling densities differ.','Only 48 of 312 primary systems have exact-run binned files. Mixed Pull/Relax bin and singleton endpoint are excluded from bin means.',len(t),'Å²',variables='late mean at 0.1 ns versus 1 ns sampling',counts=[(s,len(g)) for s,g in t.groupby('side')])
 fig,axs=plt.subplots(1,2,figsize=(11,5))
 for ax,(side,g) in zip(axs,t.groupby('side')):F.scatter(ax,g,'pAPL_early_late_change_A2','delta_height_late_ref');ax.set_title(f'{side}; n={len(g)}')
 F.save(fig,'temporal_papl_deformation','Early-to-late projected response versus late deformation','temporal',['surface_papl_analysis/tables/matched_late_pAPL.csv','surface_papl_analysis/tables/temporal_coupling.csv'],'Do persistent or transient projected changes relate to deformation?','Subset values and bin-level ranges, excursions, persistence fractions and slopes are supplied.','Temporal descriptors distinguish relaxation behaviour from one endpoint.','Small selected subset; sparse morphology bins; no mechanistic causality or lag inference.',len(t),'Å² versus nm',status='exploratory',variables='early-to-late pAPL mean change versus Δheight',counts=[(s,len(g)) for s,g in t.groupby('side')])
 cv=pd.read_csv(o/'tables/type_discrimination_summary.csv');fig,axs=plt.subplots(1,2,figsize=(12,5))
 for ax,(side,g) in zip(axs,cv.groupby('side')):
  ax.barh(g.feature_set,g.balanced_accuracy_mean,xerr=g.balanced_accuracy_sd,color='#3E7CB1' if side=='B' else '#2A9D8F');ax.axvline(1/3,ls='--',color='gray');ax.set(xlim=(0,1),xlabel='Mean outer-fold balanced accuracy',title=f'{side}; {147 if side=="B" else 146} systems / three classes')
 F.save(fig,'nominal_type_nested_CV','Nominal phenotype recovery under nested validation','validation',['surface_papl_analysis/tables/type_discrimination_summary.csv','surface_papl_analysis/tables/type_discrimination_added_information.csv','surface_papl_analysis/tables/type_discrimination_folds.csv','surface_papl_analysis/tables/type_discrimination_predictions.csv','surface_papl_analysis/classifier_protocol.json'],'Does pAPL add information after deformation amplitude is controlled?',cv.to_json(orient='records'),'Compare height+pAPL with height on the same held-out splits; compare simple geometry sets with full morphology.','Both flat_no_interaction (B/M n=4) and bulge_flat (B n=5, M n=6) excluded by n<8 rule; error bars are dependent fold SD, not confidence intervals. Nested validation tunes C only; feature sets are prespecified. Chemical-family transfer is untested.',293,'balanced accuracy',status='exploratory',variables='feature sets versus balanced accuracy')
 # Every retained PC1/2/3 sample view gets a 2x2 grid, fixed angle and consistent within-model limits.
 frozen=pd.read_csv(o/'tables/frozen_PCA_scores.csv');explore=pd.read_csv(o/'tables/exploratory_PCA_scores.csv');pv=pd.read_csv(r/'tables/pca_summary.csv');ep=json.loads((o/'exploratory_PCA/parameters.json').read_text())
 for state,scores in [('frozen',frozen),('exploratory',explore)]:
  for model,g in scores.groupby('model'):
   side=model.split('__')[1];variance=pv[pv.model==model].sort_values('component').explained_variance.to_numpy()[:3] if state=='frozen' else np.array(ep[model]['variance_ratio'][:3]);coords=g[['PC1','PC2','PC3']].to_numpy();lim=[]
   for j in range(3):
    lo,hi=coords[:,j].min(),coords[:,j].max();pad=max(.05,(hi-lo)*.08);lim.append((lo-pad,hi+pad))
   for labside in ['B','M'] if side=='paired' else [side]:
    fig=plt.figure(figsize=(11,9));axs=[fig.add_subplot(221,projection='3d'),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224)];ax=axs[0];ax.view_init(elev=22,azim=38)
    labels=[f'PC{j+1} ({variance[j]*100:.1f}%)' for j in range(3)]
    for lab,h in g.groupby('label_'+labside):
     v=h[['PC1','PC2','PC3']].to_numpy();kwargs=dict(color=F.colours[lab],marker=F.markers[lab],s=26,alpha=.8,label=f'{lab} (n={len(h)})');ax.scatter(*v.T,**kwargs)
     for aa,(i,j) in zip(axs[1:],[(0,1),(0,2),(1,2)]):aa.scatter(v[:,i],v[:,j],**kwargs)
    ax.set(xlabel=labels[0],ylabel=labels[1],zlabel=labels[2],xlim=lim[0],ylim=lim[1],zlim=lim[2],title='Fixed perspective (22°, 38°)')
    for aa,(i,j) in zip(axs[1:],[(0,1),(0,2),(1,2)]):aa.set(xlabel=labels[i],ylabel=labels[j],xlim=lim[i],ylim=lim[j]);aa.grid(alpha=.15)
    axs[1].legend(fontsize=7);fid=f'{state}_{model}_{labside}_grid';source=f'surface_papl_analysis/tables/{state if state=="frozen" else "exploratory"}_PCA_scores.csv'
    assoc='surface_papl_analysis/tables/frozen_PCA_associations.csv' if state=='frozen' else 'surface_papl_analysis/tables/exploratory_PCA_overlap.csv'
    F.save(fig,fid,f'{model}; {labside} phenotype colours',state+'_PCA',[source,assoc,'surface_papl_analysis/tables/frozen_PCA_projection_validation.csv' if state=='frozen' else 'surface_papl_analysis/exploratory_PCA/parameters.json','surface_papl_analysis/tables/exploratory_PCA_loadings.csv' if state=='exploratory' else 'data/pca_loadings.csv'],'How do physical response and nominal phenotype overlap in the first three components?',f'n=156; first-three variance={np.sum(variance)*100:.2f}%; each panel contains the identical cohort.','Orthographic views expose overlaps hidden by perspective. Paired coordinates are shown independently with B and M colouring.','PCA is a covariance summary; phenotype separation is descriptive. Independently fitted B and M coordinates cannot be subtracted. Cohort fitting does not establish prospective performance.',156,'scaled PCA coordinates',status='exploratory' if state=='exploratory' else ('sensitivity' if model.startswith('endpoint') else 'primary'),variables='PC1,PC2,PC3',counts=[(p,156) for p in ['3D','XY','XZ','YZ']])
 # Recover the existing 55 guide figures. Missing historical tables are represented by saved
 # numerical evidence, explicitly labelled as reconstructions, never as recovered original images.
 old=r.parents[3]/'figure_gui_analysis/metric_dev/results/archie_8193723_complete/figure_guide.json'
 # r.parents[3] is image_classification.
 old=r.parents[2]/'figure_gui_analysis/metric_dev/results/archie_8193723_complete/figure_guide.json'
 guide=json.loads(old.read_text());u=pd.read_csv(r/'data/unified_metrics.csv');u=u[u.selected_pair];arch=[]
 for entry in guide['figures']:
  fid=entry['figure_id'];tables=[];frames=[]
  for k,e in enumerate(entry['numeric_evidence']):
   df=pd.DataFrame(e['records']);p=o/'historical_guide_evidence'/f'{fid}_{k}.csv';p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False);tables.append(str(p.relative_to(r)));frames.append(df)
  exact=False
  if fid.endswith('_by_class'):
   metric=fid.split('_',1)[1].split('_endpoint')[0].split('_late')[0];view='endpoint_vs_ref' if 'endpoint' in fid else 'late_vs_ref';fig,axs=plt.subplots(1,2,figsize=(12,5))
   for ax,s in zip(axs,['B','M']):
    col=f'{s}__{metric}__{view}';labs=sorted(u['label_'+s].unique());values=[u[u['label_'+s]==lab][col] for lab in labs];bp=ax.boxplot(values,tick_labels=[lab.replace('_','\n') for lab in labs],patch_artist=True)
    for patch,lab in zip(bp['boxes'],labs):patch.set_facecolor(F.colours[lab])
    ax.set(ylabel=col,title=f'{s}; n=156');ax.axhline(0,color='gray',ls='--')
   exact=True;tables.insert(0,'data/unified_metrics.csv')
  elif fid=='40_B_M_effect_plane':
   fig,axs=plt.subplots(1,2,figsize=(11,5));b=d[d.side=='B'].set_index('pair_id');m=d[d.side=='M'].set_index('pair_id').loc[b.index]
   for ax,s in zip(axs,['B','M']):
    q=u.set_index('pair_id').loc[b.index].copy();q['phenotype']=q['label_'+s];q['B_q95']=b.physical_deviation_late_vs_ref_q95;q['M_q95']=m.physical_deviation_late_vs_ref_q95;F.scatter(ax,q,'M_q95','B_q95');ax.set(xlim=(0,1),ylim=(0,1),title=s+' labels')
   exact=True;tables.insert(0,'surface_papl_analysis/tables/matched_late_dataset.csv')
  elif fid.startswith(('41_surface','42_surface','43_surface','44_surface','45_surface','46_surface','47_surface','48_surface','49_surface')):
   name=fid.split('_surface_',1)[1];v=np.linspace(0,1,100);mm,bb=np.meshgrid(v,v);values=combine(bb,mm)[name];fig,ax=plt.subplots(figsize=(7,5));im=ax.pcolormesh(mm,bb,values,shading='auto',cmap='viridis');ax.set(xlabel='Effect M',ylabel='Effect B');fig.colorbar(im,ax=ax,label=name);exact=True
  elif fid.startswith('31_PCA'):
   view='endpoint_vs_ref' if 'endpoint' in fid else 'late_vs_ref';scope=fid.rsplit('_',1)[1];fig,axs=plt.subplots(1,2,figsize=(11,5))
   for ax,s in zip(axs,['B','M']):
    q=frozen[frozen.model==f'{view}__{s}__{scope}'].copy();q['phenotype']=q['label_'+s];F.scatter(ax,q,'PC1','PC2');ax.set_title(s)
   exact=True;tables.insert(0,'surface_papl_analysis/tables/frozen_PCA_scores.csv')
  elif fid.startswith('30_scree'):
   view=fid.removeprefix('30_scree_');fig,ax=plt.subplots(figsize=(9,5));pv0=pv[pv.view==view]
   for model,h in pv0.groupby('model'):ax.plot(h.component,h.explained_variance,'o-',label=model)
   ax.set(xlabel='Component',ylabel='Explained variance fraction');ax.legend(fontsize=7);exact=True;tables.insert(0,'tables/pca_summary.csv')
  elif fid.startswith('32_loadings'):
   view='endpoint_vs_ref' if 'endpoint' in fid else 'late_vs_ref';scope=fid.rsplit('_',1)[1];lo=pd.read_csv(r/'data/pca_loadings.csv');fig,axs=plt.subplots(1,2,figsize=(12,5))
   for ax,side0 in zip(axs,['B','M']):
    h=lo[(lo.model==f'{view}__{side0}__{scope}')&(lo.component==1)];ax.barh(h.feature,h.loading);ax.set(xlabel='PC1 loading',title=side0);ax.axvline(0,color='gray',lw=.5)
   exact=True;tables.insert(0,'data/pca_loadings.csv')
  elif fid.startswith(('21_','22_','23_','24_','25_','26_','27_','28_')):
   metric=fid.split('_',1)[1].split('_reference')[0];fig,axs=plt.subplots(1,2,figsize=(11,5))
   for ax,side0 in zip(axs,['B','M']):
    h=u.copy();h['phenotype']=h['label_'+side0];xx=f'{side0}__{metric}__endpoint_vs_ref';yy=f'{side0}__{metric}__endpoint_vs_start';F.scatter(ax,h,xx,yy);ax.set_title(side0)
   exact=True;tables.insert(0,'data/unified_metrics.csv')
  elif fid=='20_system_start_QC':
   fig,axs=plt.subplots(1,2,figsize=(12,5))
   for ax,side0 in zip(axs,['B','M']):
    h=frames[0];h=h[h.side==side0];ax.errorbar(h.median_start_offset,range(len(h)),xerr=[h.median_start_offset-h.start_offset_p05,h.start_offset_p95-h.median_start_offset],fmt='o');ax.set_yticks(range(len(h)),h.metric);ax.set(xlabel='Start-reference offset (metric-specific units)',title=side0);ax.axvline(0,color='gray',lw=.5)
  elif fid.startswith('33_redundancy'):
   view=fid.removeprefix('33_redundancy_');fig,axs=plt.subplots(1,2,figsize=(12,5))
   for ax,side0 in zip(axs,['B','M']):
    names=['height','thickness','height_asym','excess','curvature','excess_asym','curvature_asym'];c=[f'{side0}__{v}__{view}' for v in names];h=u[c].corr(method='spearman');im=ax.imshow(h,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(7),names,rotation=60,ha='right');ax.set_yticks(range(7),names);ax.set_title(side0);fig.colorbar(im,ax=ax)
   exact=True;tables.insert(0,'data/unified_metrics.csv')
  elif fid in ['50_selectivity_and_magnitude','51_desirability_vs_ddAPL','57_candidate_rank_agreement']:
   b=d[d.side=='B'].set_index('pair_id');m=d[d.side=='M'].set_index('pair_id').loc[b.index];cs=combine(b.physical_deviation_late_vs_ref_q95,m.physical_deviation_late_vs_ref_q95);q=u.set_index('pair_id').loc[b.index].copy();q['phenotype']=q.label_B
   for name0,val in cs.items():q[name0]=val
   if fid=='50_selectivity_and_magnitude':
    fig,ax=plt.subplots(figsize=(8,5));F.scatter(ax,q,'contrast','magnitude')
   elif fid=='51_desirability_vs_ddAPL':
    q['delta_delta_pAPL_endpoint_start_A2']=q['B_minus_M__APL__endpoint_vs_start'];fig,ax=plt.subplots(figsize=(8,5));F.scatter(ax,q,'delta_delta_pAPL_endpoint_start_A2','desirability_w0.5')
   else:
    cc=['difference','contrast','angle','desirability_w0.5','linear_utility_w0.5','ideal_w0.5','magnitude'];fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(q[cc].corr(method='spearman'),vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(7),cc,rotation=60,ha='right');ax.set_yticks(range(7),cc);fig.colorbar(im,ax=ax)
   p0=o/'historical_guide_evidence'/f'{fid}_reconstructed.csv';q.reset_index().to_csv(p0,index=False);tables.insert(0,str(p0.relative_to(r)));exact=True
  else:
   fig,axs=plt.subplots(1,len(frames),figsize=(max(7,6*len(frames)),5),squeeze=False)
   for ax,df in zip(axs[0],frames):
    cols=[c for c in df if pd.api.types.is_numeric_dtype(df[c]) and c not in ['n','component','features','orientation_multiplier']]
    priorities=['rho','AUROC','balanced_accuracy','top10_overlap','median_rank_rho','epsilon2','contrast','median','loading','explained_variance','q','p','score']
    col=next((c for c in priorities if c in cols),cols[0] if cols else None)
    labels=[c for c in ['molecule','metric','score','features','side','model','variable','group','screen','removed'] if c in df and not pd.api.types.is_numeric_dtype(df[c])]
    if col:
     labels=df[labels[0]].astype(str).str.slice(0,28) if labels else pd.Series([str(i+1) for i in range(len(df))]);n=len(df);ax.scatter(df[col],range(n),s=15);ax.set_yticks(range(n),labels,fontsize=max(4,9-n/10));ax.set(xlabel=col,title='Saved numerical evidence view');ax.grid(alpha=.12)
    else:ax.text(.02,.98,df.to_string(index=False),transform=ax.transAxes,va='top',fontsize=7);ax.axis('off')
  status='supplementary';tag='Regenerated from complete supplied inputs' if exact else 'Saved-evidence reconstruction: original figure/table absent'
  legacy=dict(metric_ids=entry['metric_ids'],analysis_technique_ids=entry['analysis_technique_ids'],historical_title=entry['title'],historical_sources=entry['source_tables'],historical_observed_result=entry['observed_result'],original_guide_id=fid,reconstruction_status=tag)
  F.save(fig,fid,entry['title']+'\n'+tag,'historical_'+entry['family'],tables,entry['scientific_purpose'],entry['observed_result'],entry['interpretation'],str(entry['limitations'])+(' Historical row tables/images are absent; this displays preserved guide evidence, not the original plot.' if not exact else ' Original image absent; data/formula regenerated.'),156 if exact and not '_surface_' in fid else 'see source evidence',str(entry['axes'].get('units','see original description')),status=status,variables=str(entry['axes']),legacy=legacy)
  arch.append(dict(figure_id=fid,status=tag,original_sources=';'.join(entry['source_tables']),evidence_files=';'.join(tables)))
 js(r/'review_bundle.json',dict(schema_version=2,figures=F.rows,project_context=guide['project_context'],terminology=dict(**guide['terminology'],pAPL='Projected XY box area per membrane molecule per nominal leaflet',sAPL_proxy='Mean leaflet surface area divided by nominal total/2; cholesterol counts not measured per leaflet'),metrics=guide['metrics'],analysis_techniques=guide['analysis_techniques'],figure_families={family:dict(name=family.replace('_',' ')) for family in sorted(set(x['family'] for x in F.rows))},provenance_sources=guide['provenance_sources'],result_set=dict(primary='80–100 ns Relax-local matched representation',generator='scripts/surface_papl_figures.py',frozen_source='preserved; no refit',historical_guide=str(old.relative_to(ROOT.parents[2])))))
 pd.DataFrame(F.counts).to_csv(o/'tables/figure_point_counts.csv',index=False);pd.DataFrame(arch).to_csv(o/'tables/historical_figure_recovery.csv',index=False);js(o/'figure_generation.json',dict(figures=len(F.rows),PNG=len(F.rows),SVG=len(F.rows),PCA_grids=sum(x['figure_id'].endswith('_grid') for x in F.rows),canonical_colours=F.colours))
 print(f'Generated {len(F.rows)} scientific figure pairs; {sum(x["figure_id"].endswith("_grid") for x in F.rows)} PCA grids',flush=True)
if __name__=='__main__':main()
