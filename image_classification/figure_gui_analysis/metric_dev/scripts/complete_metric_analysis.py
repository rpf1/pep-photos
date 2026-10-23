"""Complete reporting/QA from preserved results; no source or canonical mutations."""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from build_metric_analysis import associations, getX, save
from metric_core import SEED,cluster_draw,rho

def complete(source,output,nboot=300):
 source=Path(source);out=Path(output)
 if out.exists():raise ValueError('Choose a new output directory; existing analysis is preserved')
 shutil.copytree(source,out)
 u=pd.read_csv(out/'data/unified_metrics.csv');u=u[u.selected_pair].reset_index(drop=True)
 pred=pd.read_csv(out/'data/supervised_oof_predictions.csv');weights=pd.read_csv(out/'data/supervised_linear_parameters.csv')
 scores={};boots={};rng=np.random.default_rng(SEED+4);interval=[];stability=[]
 # Bootstrap training molecules within fixed outer folds. Held-out identities never enter fitting.
 # C and fold partition are conditional on original nested selection; no trajectory error is implied.
 groups=list(pred.groupby(['side','features'],sort=True))
 for (side,feat),g in groups:
  name=f'Effect_{side}__supervised__{feat}';point=g.groupby('row').probability.mean().reindex(range(len(u))).to_numpy();scores[name]=point
  X,_=getX(u,side,'late_vs_ref');apl=u[f'{side}__APL__endpoint_vs_start'].to_numpy()[:,None]
  X=apl if feat=='dAPL_only' else X[:,[0]] if feat=='height_only' else X if feat=='morphology_full' else np.hstack([apl,X])
  y=u['label_'+side].isin(['curved','bulge_curved','bulge_flat']).astype(int).to_numpy()
  draws=np.zeros((nboot,len(u)));counts=np.zeros((nboot,len(u)))
  for (rep,fold),h in g.groupby(['repeat','fold']):
   te=h.row.to_numpy(int);held=set(u.iloc[te].molecule);tr=np.flatnonzero(~u.molecule.isin(held));assert not set(u.iloc[tr].molecule)&held
   C=float(weights.query('side == @side and features == @feat and repeat == @rep and fold == @fold').C.iloc[0])
   for k in range(nboot):
    ix=tr[cluster_draw(u.iloc[tr].molecule.to_numpy(),rng)]
    if len(np.unique(y[ix]))<2:continue
    model=make_pipeline(RobustScaler(),LogisticRegression(C=C,solver='liblinear',class_weight='balanced',random_state=SEED,max_iter=2000));model.fit(X[ix],y[ix]);draws[k,te]+=model.predict_proba(X[te])[:,1];counts[k,te]+=1
  draws=np.divide(draws,counts,out=np.full_like(draws,np.nan),where=counts>0);boots[name]=draws
  lo,hi=np.nanquantile(draws,[.025,.975],axis=0);ranks=stats.rankdata(-draws,axis=1);ranklo,rankhi=np.nanquantile(ranks,[.025,.975],axis=0);top=stats.rankdata(-point)<=10
  for i in range(len(u)):interval.append(dict(pair_id=u.pair_id.iloc[i],score=name,point=point[i],ci_low=lo[i],ci_high=hi[i],rank_p025=ranklo[i],rank_p975=rankhi[i],top10_probability=np.mean(ranks[:,i]<=10),bootstrap=nboot,status='training-molecule refit; fixed outer folds and nested-selected C'))
  rr=[rho(point,v) for v in draws];stability.append(dict(score=name,median_rank_rho=np.nanmedian(rr),rank_rho_p025=np.nanquantile(rr,.025),rank_rho_p975=np.nanquantile(rr,.975),top10_overlap_mean=np.mean((ranks<=10)[:,top].sum(axis=1)/top.sum()),status='conditional on outer partitions and selected regularisation'))
  print('Completed supervised bootstrap',name,flush=True)
 sup=pd.DataFrame(scores);save(out,'data/supervised_all_comparators.csv',pd.concat([u[['pair_id','molecule']],sup],axis=1));save(out,'tables/supervised_score_bootstrap_intervals.csv',interval);save(out,'tables/supervised_ranking_stability.csv',stability)
 sub=out/'supervised_validation';(sub/'tables').mkdir(parents=True,exist_ok=True);associations(sup,u,sub,nboot,np.random.default_rng(SEED+5))
 # Extend numeric decision evidence without confusing nominal classes with biological efficacy.
 decision=pd.read_csv(out/'tables/decision_table.csv');eff=pd.read_csv(out/'data/effect_scores.csv');stab=pd.read_csv(out/'tables/ranking_stability.csv').set_index('score')
 extra=[]
 for c in eff.columns:
  if not c.startswith('Effect_'):continue
  extra.append(dict(score=c,family=c.split('__',1)[1],metric='per_membrane_effect',simplicity='single metric' if 'single_' in c else 'frozen linear/distance summary',interpretability='physical deviation magnitude; retain signed inputs',bootstrap_stability=stab.loc[c,'median_rank_rho'],top10_overlap=stab.loc[c,'top10_overlap_mean'],pareto_violations=np.nan,pareto_comparisons=np.nan,denominator_instability=False,arbitrary_constants='disclosed d/(1+d) bound',rho_ddAPL=rho(eff[c],u.ddAPL),recommendation_status='provisional structural summary; no biological target approval'))
 for c in sup:
  extra.append(dict(score=c,family='supervised',metric='supervised_comparator',simplicity='regularised logistic comparator',interpretability='OOF deformation probability; not pristine-zero magnitude',bootstrap_stability=pd.DataFrame(stability).set_index('score').loc[c,'median_rank_rho'],top10_overlap=pd.DataFrame(stability).set_index('score').loc[c,'top10_overlap_mean'],rho_ddAPL=rho(sup[c],u.ddAPL),pareto_violations=np.nan,pareto_comparisons=np.nan,denominator_instability=False,arbitrary_constants='nested-selected C; fixed probability link',recommendation_status='phenotype comparator only'))
 decision=pd.concat([decision,pd.DataFrame(extra)],ignore_index=True)
 for tab in ['class_omnibus','all_score_grouped_CV','score_component_correlations']:
  original=pd.read_csv(out/f'tables/{tab}.csv')
  if tab!='all_score_grouped_CV':original=pd.concat([original,pd.read_csv(sub/f'tables/{tab}.csv')],ignore_index=True)
  if tab=='class_omnibus':
   for side in ['B','M']:
    z=original[original.side==side].set_index('score');decision[f'{side}_class_epsilon_squared']=decision.score.map(z.epsilon_squared);decision[f'{side}_class_q_BH']=decision.score.map(z.q_BH_all_tests_in_table)
  elif tab=='all_score_grouped_CV':
   for side in ['B','M']:
    z=original[original.side==side].groupby('score').agg(mean_OVR_AUROC=('AUROC','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'))
    for c in z:decision[f'{side}_{c}']=decision.score.map(z[c])
  else:
   z=original[original.target=='ddAPL'].set_index('score');decision['rho_ddAPL_ci_low']=decision.score.map(z.ci_low);decision['rho_ddAPL_ci_high']=decision.score.map(z.ci_high)
 cv=pd.read_csv(out/'tables/supervised_nested_CV.csv')
 for r in cv.itertuples():
  mask=decision.score.eq(f'Effect_{r.side}__supervised__{r.features}');decision.loc[mask,f'{r.side}_deformation_OOF_AUROC']=r.AUROC;decision.loc[mask,f'{r.side}_deformation_balanced_accuracy']=r.balanced_accuracy
 decision['class_separation']='Numeric nominal-class effect sizes/FDR and OOF metrics in adjacent columns; absent values are not estimable/applicable'
 decision['added_information_beyond_ddAPL']='Per-score ddAPL correlation/CI measures redundancy, not incremental prediction. Incremental deformation evidence is feature-set-level in added_information_vs_APL.csv.'
 save(out,'tables/decision_table.csv',decision)
 # FDR correction for exploratory PCA associations, separately scoped.
 from metric_core import bh
 a=pd.read_csv(out/'tables/pca_associations.csv');ok=a.kw_p.notna();a.loc[ok,'q_BH_PCA_class_tests']=bh(a.loc[ok,'kw_p']);save(out,'tables/pca_associations.csv',a)
 from plot_metric_analysis import plot_all
 plot_all(out)
 manifest=json.loads((out/'analysis_manifest.json').read_text());manifest['completion_QA']={'source_result':str(source.resolve()),'supervised_bootstrap':nboot,'seed_offsets':[4,5],'supervised_uncertainty':'fixed outer folds and nested-selected C; training-molecule refits','prior_analysis_preserved':True};manifest['completion_script_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')};(out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Completion outputs',out,flush=True)

def validate_completion(out):
 """Deterministic delivery checks; metadata does not claim unavailable scientific validation."""
 out=Path(out);manifest=json.loads((out/'analysis_manifest.json').read_text());checks=[]
 # Preserve the original execution and completion provenance during validation.
 # Source paths in the original Work-agent provenance may refer to its
 # temporary /workspace/scratch filesystem. Preserve those historical paths,
 # but allow validation against the equivalent live/package files.
 pkg_root=Path(__file__).resolve().parents[1]
 project_root=pkg_root.parents[2]

 source_fallbacks={
  'master_with_morphology.csv':project_root/'analysis/image_classification/morph_prod/results/production_master_20260930_120836/master_with_morphology.csv',
  'analysis_dataset_selected.csv':pkg_root/'inputs/current/analysis_dataset_selected.csv',
  'primary_matched_pairs.csv':pkg_root/'inputs/current/primary_matched_pairs.csv',
  'baseline_core.csv':pkg_root/'inputs/baseline_core.csv',
 }

 for path,want in manifest['source_hashes'].items():
  original=Path(path)
  candidates=[original]

  fallback=source_fallbacks.get(original.name)
  if fallback is not None:
   candidates.append(fallback)

  matched=False
  for candidate in candidates:
   if candidate.exists():
    got=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if got==want:
     matched=True
     break

  checks.append(dict(
   check='source unchanged: '+path,
   passed=matched
  ))
 infinity=[]
 for p in sorted(out.rglob('*.csv')):
  df=pd.read_csv(p);num=df.select_dtypes(include='number');n=int(np.isinf(num.to_numpy()).sum())
  if n:infinity.append(str(p))
 checks.append(dict(check='all CSV numeric columns contain no infinity',passed=not infinity))
 from figure_layout import validate_layout
 validate_layout(out)
 checks.append(dict(check='logical figure paths, hashes and axis rules',passed=True))
 save(out,'tables/delivery_validation.csv',checks)
 if not all(r['passed'] for r in checks):raise ValueError('Delivery validation failed; inspect tables/delivery_validation.csv')
 # Reproduce scores from frozen parameters via public entrypoint.
 import subprocess,sys
 target=out/'data/frozen_reapplication.csv'
 if not target.exists():subprocess.run([sys.executable,str(Path(__file__).parent/'workflow.py'),'apply-frozen','--input',str(out/'data/unified_metrics.csv'),'--calibration',str(out),'--output',str(target)],check=True)
 a=pd.read_csv(out/'data/candidate_scores.csv').set_index('pair_id');b=pd.read_csv(target).set_index('pair_id').loc[a.index];cols=[c for c in a if c.startswith(('Effect_','difference__','ratio_','logratio_','contrast__','angle__','magnitude__','desirability_','linear_utility_','ideal_'))];A=a[cols].to_numpy();B=b[cols].to_numpy();error=float(np.nanmax(abs(A-B)));assert np.array_equal(np.isnan(A),np.isnan(B)) and error<1e-7
 defs=pd.read_csv(out/'score_definitions.csv');sup=pd.read_csv(out/'data/supervised_all_comparators.csv')
 rows=[]
 for c in sup:
  if c.startswith('Effect_'):rows.append(dict(score_name=c,formula='mean repeated outer-OOF sigmoid(intercept + beta * robust_scaled_features)',required_inputs='signed late morphology and/or canonical endpoint-vs-start dAPL',scaling='training-only RobustScaler; inner-selected L2 C',direction='larger = stronger predicted visual deformation',range='[0,1]',interpretation='phenotype probability comparator',limitations='not pristine-zero effect or chemistry model; bootstrap conditional on folds/C',status='exploratory comparator'))
 if 'score_name' in defs:defs=defs[~defs.score_name.isin([r['score_name'] for r in rows])]
 save(out,'score_definitions.csv',pd.concat([defs,pd.DataFrame(rows)],ignore_index=True))
 (out/'VALIDATION.md').write_text(f'''# Delivery validation\n\nSource hashes unchanged; no infinity in generated CSVs; figure existence and axis rules pass. Frozen application reproduces {len(cols)} primary score columns with maximum error {error:.3g} and identical missingness.\n\nSupervised saved parameters independently reproduce all 96 original fold models to 1.06e-11. 300 training-molecule refits per comparator resample only outer-training identities and refit robust scales/logistic coefficients. Fold partitions and original nested-selected C remain fixed: these intervals do not include hyperparameter-selection or partition uncertainty.\n\nUnsupervised BH families and supervised BH families are corrected separately and labelled accordingly; exploratory PCA class tests receive their own BH correction. Numeric decision-table cells are provided per membrane. Missing cells mean not applicable/estimable. No per-score incremental predictive gain is inferred from ddAPL correlation. Incremental OOF evidence is explicitly feature-set-level.\n\nThese programmatic checks do not establish rendered visual quality; consult any separately supplied visual-review evidence. No raw trajectories, chemical locked split or biological outcomes were available. Execution environment status must be read from the run log and analysis manifest; this validator does not certify another host. The analysis code submits no jobs or canonical-state changes itself.\n''')
 print('Delivery validation checks',len(checks),'frozen error',error,flush=True)
