#!/usr/bin/env python3
"""Independently validate continuation calculations, sources, points and GUI state."""
import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from surface_papl_analysis import sha,js
from review_metric_figures import self_test,ReviewState,result_identity
from pristine_fluctuations import stats_series

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--checkpoint-manifest',type=Path,required=True);a=p.parse_args();project=a.project.resolve();r=a.result.resolve();o=r/'surface_papl_analysis';d=pd.read_csv(o/'tables/matched_late_dataset.csv');u=pd.read_csv(r/'data/unified_metrics.csv');u=u[u.selected_pair];checks=[]
 def check(name,ok,detail):
  checks.append(dict(check=name,pass_check=bool(ok),detail=detail))
  if not ok:raise AssertionError(name+': '+str(detail))
 check('cohort',len(d)==312 and len(u)==156 and d.groupby('side').size().eq(156).all() and not d.duplicated(['pair_id','side']).any(),'312 unique canonical systems / 156 pairs')
 check('time',d.n_late_frames.eq(21).all() and d.late_start_ns.eq(80).all() and d.late_end_ns.eq(100).all(),'21 inclusive 1 ns frames, 80–100 Relax-local ns')
 check('binned_coverage',d.binned_pAPL_available.sum()==48 and len(pd.read_csv(o/'tables/missing_inputs.csv'))==264,'48 present / 264 missing exact-run bins')
 pars=json.loads((r/'data/frozen_parameters.json').read_text());errors=[];frame_errors=[]
 for row in d.itertuples():
  task=project/row.surface_source;f=pd.read_csv(task);late=f[f.time_ns.between(80,100)];base=pd.read_csv(project/row.baseline_source).iloc[0]
  frame_errors.extend([abs(late.projected_APL_A2.mean()-row.pAPL_late_A2),abs(((late.upper_surface_area_nm2+late.lower_surface_area_nm2)/2).mean()-row.A_surface_mean_late_nm2)])
  match=u[u.pair_id==row.pair_id].iloc[0]
  check('identity_'+row.side+'_'+row.pair_id,match['physical_key_'+row.side]==row.physical_key and match.physical_run==row.physical_run and match['label_'+row.side]==row.phenotype,'run/membrane/label from original selected pair')
  for view in ['late_vs_ref','endpoint_vs_ref']:
   st=pars[view+'__q95'];x=np.array([match[f'{row.side}__{name}__{view}'] for name in st['names']]);z=(x-np.array(st['origin']))/st['scale'];keep=st['reduced_indices'];D=np.linalg.norm(z[keep])/np.sqrt(len(keep));E=D/(1+D);errors.append(abs(E-getattr(row,'physical_deviation_'+view+'_q95')))
 check('source_frame_means',max(frame_errors)<1e-10,f'max error={max(frame_errors):.3g}')
 check('independent_q95_formula',max(errors)<1e-12,f'max error={max(errors):.3g}; independent norm/sqrt(k) formula')
 check('area_decomposition',np.allclose(d.delta_sAPL_proxy_late_baseline_A2,d.delta_pAPL_late_start_A2+d.delta_geometry_area_contribution_A2,atol=1e-8,rtol=0),'reference-consistent additive identity')
 for side in ['B','M']:
  rr=pd.read_csv(o/f'tables/complete_rankings_{side}.csv');gg=d[d.side==side].set_index('pair_id').loc[rr.pair_id]
  for c in [c for c in rr if c.endswith('__rank_low_to_high')]:
   base=c.removesuffix('__rank_low_to_high');check('ranking_'+side+'_'+base,np.allclose(rr[c],rankdata(gg[base],method='average')) and np.allclose(rr[base+'__rank_high_to_low'],157-rr[c]),'scipy independent rankdata; ascending+descending=157')
 check('tie_fixture',np.array_equal(rankdata([2,2,5],method='average'),[1.5,1.5,3]),'exact-tie semantics')
 splits=pd.read_csv(o/'tables/CV_splits.csv');pred=pd.read_csv(o/'tables/type_discrimination_predictions.csv');folds=pd.read_csv(o/'tables/type_discrimination_folds.csv')
 for row in splits.itertuples():
  tr=set(row.train_ids.split(';'));te=set(row.test_ids.split(';'));check(f'CV_disjoint_{row.side}_{row.repeat}_{row.fold}',not tr&te and len(tr|te)==(147 if row.side=='B' else 146),'identity-disjoint three-class cohort')
 for (s,rep,f,ff),g in pred.groupby(['side','repeat','fold','feature_set']):
  saved=folds[(folds.side==s)&(folds.repeat==rep)&(folds.fold==f)&(folds.feature_set==ff)].iloc[0];ba=g.assign(correct=g.truth.eq(g.prediction)).groupby('truth').correct.mean().mean();check('prediction_score_'+str((s,rep,f,ff)),abs(ba-saved.balanced_accuracy)<1e-12,'balanced accuracy independently recomputed from held-out rows')
 pp=pd.read_csv(o/'tables/frozen_PCA_projection_validation.csv');check('frozen_projection',len(pp)==12 and np.isfinite(pd.read_csv(o/'tables/frozen_PCA_scores.csv')[['PC1','PC2','PC3']]).all().all(),'12 models; original saved projection validations retained')
 checkpoint=json.loads(a.checkpoint_manifest.read_text());unchanged=0;allowed={'surface_papl_figures.py','review_metric_figures.py'}
 for item in checkpoint:
  path=project/item['path']
  if '__pycache__' in path.parts:continue
  if item['path'] in {'analysis/image_classification/metric_dev/scripts/'+name for name in allowed}:continue
  check('original_hash_'+item['path'],path.is_file() and sha(path)==item['sha256'],'checkpoint byte hash');unchanged+=1
 guicheck=self_test(r);js(o/'gui_validation.json',guicheck);g=json.loads((r/'review_bundle.json').read_text());ids={e['figure_id'] for e in g['figures']};png=list((o/'figures/png').rglob('*.png'));svg=list((o/'figures/svg').rglob('*.svg'))
 check('figure_guide_bijection',len(ids)==len(png)==len(svg)==123 and {x.stem for x in png}=={x.stem for x in svg}==ids,'123 unique figure IDs / PNG / SVG')
 from review_saved_pca import resolve_palette
 palette,_=resolve_palette(['flat_no_interaction','flat_interaction','bulge_flat','curved','bulge_curved']);gen=json.loads((o/'figure_generation.json').read_text());check('canonical_palette',gen['canonical_colours']==palette,str(palette))
 pts=pd.read_csv(o/'tables/figure_point_counts.csv');expected=[]
 for row in pts.itertuples():
  if row.figure_id.endswith('_grid'):n=156
  elif row.figure_id.startswith('matched_counterexample_maps'):n=156
  elif row.figure_id in ['binned_frame_reconciliation','temporal_papl_deformation']:n=int(d[d.side==row.panel].binned_pAPL_available.sum())
  elif row.figure_id=='51_desirability_vs_ddAPL':n=156
  else:n=int(d[d.side==row.panel].shape[0])
  expected.append(n);check('points_'+row.figure_id+'_'+row.panel,int(row.n_plotted)==n,'source cohort independently resolved')
 pts['n_expected_from_source']=expected;pts['pass_point_count']=pts.n_plotted.eq(pts.n_expected_from_source);pts.to_csv(o/'tables/figure_point_counts_validated.csv',index=False)
 # Validate actual SVG point marks for every PCA panel, not just the metadata.
 import xml.etree.ElementTree as ET
 mark_validation=[]
 for entry in g['figures']:
  if not entry['figure_id'].endswith('_grid'):continue
  tree=ET.parse(r/entry['svg_path']);namespace={'s':'http://www.w3.org/2000/svg'};axes=[v for v in tree.findall('.//s:g',namespace) if v.get('id','').startswith('axes_')]
  for i,ax in enumerate(axes[:4]):
   marks=0
   for col in ax.findall('./s:g',namespace):
    if col.get('id','').startswith(('PathCollection_','Path3DCollection_')):marks+=len(col.findall('.//s:use',namespace))+len(col.findall('./s:path',namespace))
   check('actual_SVG_points_'+entry['figure_id']+'_'+str(i),marks==156,'rendered SVG sample marks');mark_validation.append(dict(figure_id=entry['figure_id'],panel=i+1,rendered_sample_marks=marks,expected=156))
 pd.DataFrame(mark_validation).to_csv(o/'tables/rendered_PCA_point_validation.csv',index=False)
 with tempfile.TemporaryDirectory() as td:
  stpath=Path(td)/'state.json';st=ReviewState(stpath,result_identity(r),list(ids));st.update(next(iter(ids)),note='retain me',status='important');v=json.loads(stpath.read_text());v['per_figure']['archived_id']={'status':'revisit','note':'old note'};stpath.write_text(json.dumps(v));rest=ReviewState(stpath,result_identity(r),list(ids));rest.update(next(iter(ids)),viewed=True);check('review_legacy_state_preserved',rest.data['per_figure']['archived_id']['note']=='old note','legacy IDs and notes preserved')
  before=stpath.read_bytes();stpath.write_bytes(before+b' ')
  try:rest.update(next(iter(ids)),note='must reject')
  except RuntimeError:pass
  else:raise AssertionError('External state change accepted')
 check('pristine_block_fixture',stats_series(np.arange(20.),np.arange(20.)*.1,1)['n_blocks']==2,'physical-time blocks')
 pd.DataFrame(checks).to_csv(o/'tables/final_validation_checks.csv',index=False)
 js(o/'final_validation.json',dict(status='PASS',checks=len(checks),preserved_checkpoint_files=unchanged,figures=123,PCA_grids=28,actual_PCA_point_panels=len(mark_validation),q95_max_error=max(errors),gui=guicheck,native_gui_interaction='not tested: no graphical display or Xvfb installed',raw_pristine_extraction='not tested: raw external trajectories absent',visual_inspection='representative rendered figures reviewed separately'))
 print('PASS:',len(checks),'checks;',unchanged,'checkpoint files unchanged; 123 figures; 112 actual PCA panels have 156 SVG points.')
if __name__=='__main__':main()
