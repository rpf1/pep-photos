"""One figure configuration and one manifest for every metric-development plot."""
import json
from figure_layout import sha, SCHEMA
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from metric_core import METRICS,LABELS,MEMBRANES,combine,rho
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,'figure.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.2,'lines.markersize':4,'savefig.dpi':300,'svg.fonttype':'none','pdf.fonttype':42}
COLORS={'B':'#3E7CB1','M':'#2A9D8F'};CL_COLORS=dict(zip(LABELS,['#657685','#4796BA','#DD9A35','#427943','#963966','#111111']))
SIZE=(10,4.4);MARKER_SIZE=18;ALPHA=.7

class Plotter:
 def __init__(self,out,repair_empty=False):
  self.repair_empty=repair_empty;self.out=Path(out);self.rows=[];plt.rcParams.update(STYLE)
  legacy=[p for p in (self.out/'figures').glob('*/*') if p.suffix.lower() in ('.png','.svg')]
  if legacy:raise ValueError('Legacy figure layout detected; run workflow.py migrate-figures --output for this result first')
 def save(self,fig,family,name,title,question,x,y,group,units,source,interpretation,axis_rule='metric-specific'):
  fig.suptitle(title);fig.set_layout_engine('constrained');self.out=Path(self.out)
  axes=[{'xlim':list(a.get_xlim()),'ylim':list(a.get_ylim()),'xlabel':a.get_xlabel(),'ylabel':a.get_ylabel()} for a in fig.axes]
  row=dict(figure_id=name,family=family,title=title,question=question,x_variable=x,y_variable=y,grouping_color_variable=group,units=units,axis_limits=json.dumps(axes),axis_rule=axis_rule,source_table=source,interpretation=interpretation,layout_schema_version=SCHEMA)
  for ext in ['png','svg']:
   folder=self.out/'figures'/ext/family;folder.mkdir(parents=True,exist_ok=True)
   path=folder/(name+'.'+ext)
   valid=path.exists() and path.stat().st_size>0
   if valid and ext=='png':
    from PIL import Image
    try:
     with Image.open(path) as im:im.load()
    except Exception:valid=False
   if not self.repair_empty or not valid:fig.savefig(path,dpi=300,bbox_inches='tight')
   row[ext+'_filename']=str(path.relative_to(self.out));row[ext+'_sha256']=sha(path)
  self.rows.append(row)
  plt.close(fig)
 def finish(self):pd.DataFrame(self.rows).to_csv(self.out/'plot_manifest.csv',index=False);(self.out/'plot_style.json').write_text(json.dumps(STYLE,indent=2))

def limits(v):
 v=np.asarray(v);v=v[np.isfinite(v)];lo=min(0,v.min());hi=max(0,v.max());p=max(1e-9,(hi-lo)*.08);return lo-p,hi+p

def plot_all(out,repair_empty=False):
 out=Path(out);P=Plotter(out,repair_empty);u=pd.read_csv(out/'data/unified_metrics.csv');u=u[u.selected_pair].reset_index(drop=True);eff=pd.read_csv(out/'data/effect_scores.csv');scores=pd.read_csv(out/'data/candidate_scores.csv');pca=pd.read_csv(out/'data/pca_scores.csv');loads=pd.read_csv(out/'data/pca_loadings.csv');scree=pd.read_csv(out/'tables/pca_summary.csv')
 # A. Exact visual classes, no ordinal interpretation; common B/M y-scale.
 for i,(metric,(_,__,unit)) in enumerate(METRICS.items(),1):
  for view in ['endpoint_vs_ref','late_vs_ref']:
   if metric=='APL' and view=='late_vs_ref':continue
   fig,axs=plt.subplots(1,2,figsize=SIZE,layout='constrained',sharey=True);lim=limits(np.r_[u[f'B__{metric}__{view}'],u[f'M__{metric}__{view}']])
   for ax,side in zip(axs,['B','M']):
    groups=[u.loc[u['label_'+side]==lab,f'{side}__{metric}__{view}'].to_numpy() for lab in LABELS if (u['label_'+side]==lab).any()];labs=[lab for lab in LABELS if (u['label_'+side]==lab).any()]
    ax.boxplot(groups,tick_labels=[lab.replace('_','\n') for lab in labs],patch_artist=True,boxprops={'facecolor':COLORS[side],'alpha':.35},medianprops={'color':'black'},flierprops={'markersize':3});ax.axhline(0,color='black',ls='--',lw=.8,label='Pristine reference');ax.set(ylim=lim,ylabel=f'{metric} deviation ({unit})',title=f'{side}, N = {len(u)}')
    for k,g in enumerate(groups,1):ax.text(k,.97,f'n={len(g)}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
   P.save(fig,'01_reference_deviation',f'{i:02d}_{metric}_{view}_by_class',f'{metric}: B and M {view.replace("_"," ")} ({unit})','Does reference deviation correspond to visual phenotype?','visual class',metric,'membrane and visual class',unit,'data/unified_metrics.csv','Pristine reference is zero; class distributions do not imply an ordered biological response.','shared_BM_y')
 # B. Preparation QC in physical units.
 fig,axs=plt.subplots(2,4,figsize=(13,6),layout='constrained')
 for ax,(metric,(_,__,unit)) in zip(axs.flat,METRICS.items()):
  groups=[u[f'{s}__{metric}__start_vs_ref'] for s in ['B','M']];ax.boxplot(groups,tick_labels=['B','M']);ax.axhline(0,color='black',ls='--',lw=.8);ax.set(title=metric,ylabel=unit)
 P.save(fig,'01_reference_deviation','20_system_start_QC','B and M system start minus pristine reference (metric-specific units)','Are preparation offsets negligible?','membrane','system_start - pristine_reference','membrane','per-axis units','data/unified_metrics.csv','Start offsets are preparation context, not induced membrane effects.')
 # C. Reference versus simulation change, matched axes and identity line.
 for i,(metric,(_,__,unit)) in enumerate(METRICS.items(),21):
  fig,axs=plt.subplots(1,2,figsize=SIZE,layout='constrained');lim=limits(np.concatenate([u[f'{s}__{metric}__{v}'] for s in ['B','M'] for v in ['endpoint_vs_start','endpoint_vs_ref']]))
  for ax,side in zip(axs,['B','M']):
   x=u[f'{side}__{metric}__endpoint_vs_start'];y=u[f'{side}__{metric}__endpoint_vs_ref'];ax.scatter(x,y,s=MARKER_SIZE,alpha=ALPHA,color=COLORS[side]);ax.plot(lim,lim,color='gray',ls='--');ax.set(xlim=lim,ylim=lim,xlabel=f'Endpoint − system start ({unit})',ylabel=f'Endpoint − pristine ({unit})',title=f'{side}, Spearman ρ = {rho(x,y):.3f}');ax.set_aspect('equal',adjustable='box')
  P.save(fig,'01_reference_deviation',f'{i:02d}_{metric}_reference_vs_start',f'{metric}: B and M endpoint deviations ({unit})','Does changing reference change ranking?','endpoint_vs_start','endpoint_vs_ref','membrane',unit,'data/unified_metrics.csv','Identity line distinguishes similar ranking from identical values.','shared_BM_xy_equal')
 # D. PCA: shared B/M limits only within the same view and feature scope.
 for view in ['endpoint_vs_ref','late_vs_ref']:
  fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
  for ax,side in zip(axs,['B','M','paired']):
   for scope,style in [('full','-'),('reduced','--')]:
    a=scree[(scree.view==view)&(scree.side==side)&(scree.feature_set==scope)];ax.plot(a.component,a.cumulative_variance,style,marker='o',label=scope)
   ax.set(ylim=(0,1.02),xlabel='Principal component',ylabel='Cumulative explained variance',title=side);ax.legend()
  P.save(fig,'04_pca',f'30_scree_{view}',f'PCA cumulative variance: B, M and paired {view} (dimensionless)','How many directions carry variance?','PC','explained variance','full/reduced','dimensionless','tables/pca_summary.csv','Variance is not predictive value or biological effect.')
  for scope in ['full','reduced']:
   fig,axs=plt.subplots(1,3,figsize=(12,3.9),layout='constrained');a=pca[pca.model.isin([f'{view}__B__{scope}',f'{view}__M__{scope}'])];xl=limits(a.PC1);yl=limits(a.PC2)
   for ax,side in zip(axs,['B','M','paired']):
    a=pca[pca.model==f'{view}__{side}__{scope}']
    for lab in LABELS:
     mask=a['label_'+('B' if side=='paired' else side)]==lab
     if mask.any():ax.scatter(a.loc[mask,'PC1'],a.loc[mask,'PC2'],s=MARKER_SIZE,alpha=ALPHA,label=lab,color=CL_COLORS[lab])
    ax.set(xlabel='PC1 (scaled)',ylabel='PC2 (scaled)',title=side)
    if side!='paired':ax.set(xlim=xl,ylim=yl)
   axs[-1].legend(fontsize=6,loc='best')
   P.save(fig,'04_pca',f'31_PCA_{view}_{scope}',f'PCA {scope}: B/M and paired {view} (scaled units)','Does dimensional structure align with phenotype?','PC1','PC2','visual class; paired coloured by B','dimensionless','data/pca_scores.csv','Separate B/M PCA bases have different loadings; equal axis limits do not align their latent meanings.','PCA_BM_common_limits')
   fig,axs=plt.subplots(1,2,figsize=SIZE,layout='constrained')
   for ax,side in zip(axs,['B','M']):
    a=loads[(loads.model==f'{view}__{side}__{scope}')&(loads.component<=2)];pivot=a.pivot(index='feature',columns='component',values='loading');im=ax.imshow(pivot,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set(xticks=[0,1],xticklabels=['PC1','PC2'],yticks=range(len(pivot)),yticklabels=pivot.index,title=side)
   fig.colorbar(im,ax=axs.tolist(),label='Loading',fraction=.02)
   P.save(fig,'04_pca',f'32_loadings_{view}_{scope}',f'PCA loadings: B/M {view}, {scope} (dimensionless)','Which physical metrics drive the PCs?','PC','physical metric','membrane','dimensionless','data/pca_loadings.csv','Signs use the saved largest-loading-positive convention; they are not biological polarity.')
 # Redundancy endpoint and late, common rho colour range.
 r=pd.read_csv(out/'tables/physical_redundancy.csv')
 for view in ['endpoint_vs_ref','late_vs_ref']:
  fig,axs=plt.subplots(1,2,figsize=SIZE,layout='constrained')
  for ax,side in zip(axs,['B','M']):
   a=r[(r.view==view)&(r.side==side)];names=list(dict.fromkeys(list(a.metric1)+list(a.metric2)));mat=np.eye(len(names))
   for row in a.itertuples():i=names.index(row.metric1);j=names.index(row.metric2);mat[i,j]=mat[j,i]=row.rho
   im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,title=side);ax.tick_params(axis='x',rotation=70)
  fig.colorbar(im,ax=axs.tolist(),label='Spearman ρ',fraction=.025)
  P.save(fig,'03_redundancy',f'33_redundancy_{view}',f'B/M {view} physical-metric redundancy (Spearman ρ)','Which measurements overlap?','physical metric','physical metric','membrane','dimensionless','tables/physical_redundancy.csv','Strong correlation motivates reduced candidates, not silent removal of source measurements.')
 fam='late_vs_ref__q95__euclidean_reduced';B=eff['Effect_B__'+fam].to_numpy();M=eff['Effect_M__'+fam].to_numpy();joint=combine(B,M)
 fig,axs=plt.subplots(1,2,figsize=SIZE,layout='constrained')
 for ax,side in zip(axs,['B','M']):
  for lab in LABELS:
   mask=u['label_'+side]==lab
   if mask.any():ax.scatter(M[mask],B[mask],s=MARKER_SIZE,alpha=ALPHA,label=lab,color=CL_COLORS[lab])
  ax.plot([0,1],[0,1],color='gray',ls='--');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Effect M (0–1)',ylabel='Effect B (0–1)',title='Colour: '+side+' class');ax.set_aspect('equal');ax.legend(fontsize=6)
 P.save(fig,'06_effect_scores','40_B_M_effect_plane', 'B/M effect plane: late vs pristine, pooled q95 reduced Euclidean (0–1)','Are effects selective or shared?','Effect_M','Effect_B','visual class','dimensionless','data/effect_scores.csv','Both effects remain visible; this is a physical deviation map, not a biological efficacy assay.','effect_plane_0_1_equal')
 # F. Surfaces use an identical 0–1 grid. Ratio capped explicitly, singular axis masked.
 grid=np.linspace(0,1,151);bb,mm=np.meshgrid(grid,grid,indexing='ij');surfaces=combine(bb.ravel(),mm.ravel())
 for i,metric in enumerate(['difference','ratio_BM','logratio_c0.1','contrast','angle','magnitude','desirability_w0.5','linear_utility_w0.5','ideal_w0.5'],41):
  fig,ax=plt.subplots(figsize=(5.4,4.6),layout='constrained');v=surfaces[metric].reshape(bb.shape);kwargs={}
  if metric in ['difference','contrast','angle']:kwargs={'vmin':-1,'vmax':1,'cmap':'coolwarm'}
  elif metric=='ratio_BM':v=np.ma.masked_invalid(np.minimum(v,10));kwargs={'vmin':0,'vmax':10,'cmap':'viridis'}
  elif metric.startswith(('desirability','linear_utility','ideal')):kwargs={'vmin':0,'vmax':1,'cmap':'viridis'}
  im=ax.pcolormesh(mm,bb,v,shading='auto',**kwargs);fig.colorbar(im,ax=ax,label=metric+(' (capped at 10)' if metric=='ratio_BM' else ''));ax.set(xlim=(0,1),ylim=(0,1),xlabel='Effect M (0–1)',ylabel='Effect B (0–1)');ax.set_aspect('equal')
  if metric=='ratio_BM':ax.axvline(0,color='red',lw=2,label='M=0 singular');ax.legend(loc='lower right')
  P.save(fig,'05_metric_surfaces',f'{i:02d}_surface_{metric}',f'{metric}: synthetic B/M effect grid (dimensionless)','Does this formula reward weak response or hide magnitude?','Effect_M','Effect_B','score colour','dimensionless','tables/candidate_metric_properties.csv','Same 0–1 domain; ratio singularities are masked and values above 10 are explicitly capped.' if metric=='ratio_BM' else 'Directional scores require a separate magnitude.','synthetic_0_1_equal')
 fig,ax=plt.subplots(figsize=(6.5,4.6),layout='constrained');sc=ax.scatter(joint['contrast'],joint['magnitude'],c=u.ddAPL,cmap='coolwarm',s=MARKER_SIZE,alpha=ALPHA);fig.colorbar(sc,ax=ax,label='ddAPL (Å²)');ax.set(xlim=(-1,1),ylim=(0,np.sqrt(2)),xlabel='Normalised contrast (−1 to +1)',ylabel='Total effect magnitude (0 to √2)')
 P.save(fig,'07_selectivity','50_selectivity_and_magnitude','B/M selectivity and magnitude: late vs pristine q95 reduced effect (dimensionless)','Can equal selectivity conceal weak and strong effects?','contrast','magnitude','ddAPL','dimensionless / Å²','data/candidate_scores.csv','Magnitude retains the information lost by directional selectivity.','selectivity_minus1_plus1')
 fig,ax=plt.subplots(figsize=(6.5,4.6),layout='constrained');ax.scatter(u.ddAPL,joint['desirability_w0.5'],s=MARKER_SIZE,color=COLORS['B']);ax.set(ylim=(0,1),xlabel='ddAPL (Å²)',ylabel='Geometric desirability, weight B = 0.5 (0–1)')
 P.save(fig,'07_selectivity','51_desirability_vs_ddAPL','B/M late vs pristine desirability versus endpoint-vs-start ddAPL (dimensionless / Å²)','Does structural ranking duplicate ddAPL?','ddAPL','desirability_w0.5','none','Å² / dimensionless','data/candidate_scores.csv','Any disagreement reflects a chosen physical utility, not established antibacterial selectivity.','desirability_0_1')
 # H. CV plus rank/bootstraps, avoiding p-value-only presentation.
 cv=pd.read_csv(out/'tables/supervised_nested_CV.csv');fig,axs=plt.subplots(1,2,figsize=SIZE,layout='constrained',sharey=True)
 for ax,side in zip(axs,['B','M']):
  a=cv[cv.side==side];ax.errorbar(range(len(a)),a.AUROC,yerr=np.vstack([a.AUROC-a.auc_ci_low,a.auc_ci_high-a.AUROC]),fmt='o',capsize=3,color=COLORS[side]);ax.axhline(.5,color='gray',ls='--');ax.set(xticks=range(len(a)),xticklabels=[s.replace('_','\n') for s in a.features],ylim=(0,1),ylabel='OOF AUROC (0–1)',title=side)
 P.save(fig,'02_class_comparisons','52_nested_CV_physical_phenotype','B/M deformation-label discrimination: dAPL and late morphology (OOF AUROC)','Does morphology add phenotype information beyond APL?','feature set','AUROC','membrane','dimensionless','tables/supervised_nested_CV.csv','Nested grouped validation tests phenotype alignment, not prediction from peptoid chemistry.','shared_BM_y_0_1')
 stab=pd.read_csv(out/'tables/ranking_stability.csv');chosen=[f'{metric}__late_vs_ref__{scale}__{family}' for metric in ['contrast','desirability_w0.5'] for scale in ['iqr','null_mad','q95'] for family in ['euclidean_reduced','mahalanobis_reduced']];a=stab[stab.score.isin(chosen)].copy();fig,ax=plt.subplots(figsize=(10,6),layout='constrained');ax.barh(range(len(a)),a.top10_overlap_mean,color=COLORS['B']);ax.set(yticks=range(len(a)),yticklabels=[v.replace('__late_vs_ref__',' / ').replace('__',' / ') for v in a.score],xlim=(0,1),xlabel='Mean top-10 overlap under calibration bootstrap')
 P.save(fig,'09_bootstrap','53_top10_stability','B/M late vs pristine ranking stability (molecule bootstrap)','How stable are prioritised molecules to calibration data?','top10 overlap','candidate','scaler/family','dimensionless','tables/ranking_stability.csv','Bootstrap refits scalers, feature reduction, covariance and PCA; it does not estimate MD trajectory uncertainty.')
 cases=pd.read_csv(out/'tables/disagreement_cases.csv');selected=cases.drop_duplicates('pair_id').head(8);fig,ax=plt.subplots(figsize=(7,5),layout='constrained');ax.scatter(M,B,s=MARKER_SIZE,color='#AAAAAA',alpha=.5)
 for i,row in enumerate(selected.itertuples(),1):ax.scatter(row.Effect_M,row.Effect_B,s=30,color=COLORS['B'],label=f'{i}. {row.molecule}');ax.annotate(str(i),(row.Effect_M,row.Effect_B),xytext=(3,3),textcoords='offset points',fontsize=7)
 ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=7)
 ax.plot([0,1],[0,1],color='gray',ls='--');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Effect M (0–1)',ylabel='Effect B (0–1)');ax.set_aspect('equal')
 P.save(fig,'08_disagreement_cases','54_disagreement_examples','B/M disagreement cases: late vs pristine q95 reduced effect (0–1)','Which molecules change interpretation?','Effect_M','Effect_B','selected discordant cases','dimensionless','tables/disagreement_cases.csv','Examples use explicitly stated quantile screens; labels are not hard thresholds.','effect_plane_0_1_equal')
 fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
 for ax,(screen,z) in zip(axs.flat,cases.groupby('screen',sort=True)):
  ax.scatter(M,B,s=10,color='#bbbbbb',alpha=.4)
  ax.scatter(z.Effect_M,z.Effect_B,s=25,color=COLORS['B'])
  ax.plot([0,1],[0,1],ls='--',color='gray');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Effect M (0–1)',ylabel='Effect B (0–1)',title=screen.replace('_',' '));ax.set_aspect('equal')
 axs.flat[-1].axis('off')
 P.save(fig,'08_disagreement_cases','55_all_disagreement_screens','B/M late vs pristine q95 reduced effect: five disagreement screens (0–1)','Are all discordance modes represented?','Effect_M','Effect_B','screen','dimensionless','tables/disagreement_cases.csv','Quantile screens identify examples, not clinical thresholds.','effect_plane_0_1_equal')
 # Numeric candidate comparisons include class effect size, rank agreement and Pareto counts.
 decision=pd.read_csv(out/'tables/decision_table.csv')
 candidates=[m+'__'+fam for m in ['difference','contrast','angle','desirability_w0.5','linear_utility_w0.5','ideal_w0.5','magnitude']]
 a=decision.set_index('score').reindex(candidates)
 if 'B_class_epsilon_squared' in a:
  fig,axs=plt.subplots(1,2,figsize=SIZE,layout='constrained',sharey=True)
  for ax,side in zip(axs,['B','M']):
   ax.bar(range(len(a)),a[side+'_class_epsilon_squared'],color=COLORS[side]);ax.set(xticks=range(len(a)),xticklabels=[v.split('__')[0].replace('_','\n') for v in candidates],ylim=(0,1),ylabel='Kruskal–Wallis epsilon squared',title=side)
  P.save(fig,'02_class_comparisons','56_candidate_class_effect_sizes','B/M late vs pristine q95 reduced candidates: nominal-class effect sizes (0–1)','How strongly are candidates associated with nominal phenotype?','candidate','epsilon squared','membrane','dimensionless','tables/decision_table.csv','Effect sizes complement FDR tests; class ordering and biological efficacy are not assumed.','shared_BM_y_0_1')
 fig,ax=plt.subplots(figsize=(7,6),layout='constrained');mat=scores[candidates].corr(method='spearman');im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');names=[v.split('__')[0] for v in candidates];ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names);ax.tick_params(axis='x',rotation=60);fig.colorbar(im,ax=ax,label='Spearman rho')
 P.save(fig,'03_redundancy','57_candidate_rank_agreement','B/M late vs pristine q95 reduced candidate rank agreement (Spearman rho)','Which rankings are redundant?','candidate','candidate','rank correlation','dimensionless','data/candidate_scores.csv','Angle and contrast carry equivalent rank information.')
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.bar(range(len(a)),a.pareto_violations,color=COLORS['B']);ax.set(xticks=range(len(a)),xticklabels=[v.split('__')[0].replace('_','\n') for v in candidates],ylabel='Dominance violations (ordered pairs)')
 P.save(fig,'07_selectivity','58_candidate_Pareto_violations','B/M late vs pristine q95 reduced candidates: Pareto violations (counts)','Do larger-is-better bacterial utilities respect dominance?','candidate','violations','none','counts','tables/decision_table.csv','Magnitude measures response strength and is not a bacterial utility; its violations are expected.')
 P.finish();print('Figures:',len(P.rows),flush=True)
if __name__=='__main__':
 import sys
 plot_all(Path(sys.argv[1]))
