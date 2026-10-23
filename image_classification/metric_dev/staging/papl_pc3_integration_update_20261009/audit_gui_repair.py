#!/usr/bin/env python3
"""Read-only source audit; optional report destination must be in existing stage."""
import argparse,csv,hashlib,json,re,sys
from pathlib import Path
from collections import Counter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def audit(base):
 r=base/'results/archie_8193723_complete';a=base.parent.parent
 g=json.loads((r/'review_bundle.json').read_text());u=[x for x in rows(r/'data/unified_metrics.csv') if x['selected_pair'].lower()=='true' and x['qc_pass_pair'].lower()=='true']; t=[];axes=[]
 for x in u:
  for side,bilayer in [('B','POPE75POPG25_W_WF'),('M','POPC60CHOL40_W_WF')]:
   p=a/'output_2'/f"{x['molecule']}_{x['nmol']}"/bilayer/x['physical_run']/'binned_APL.csv';d=rows(p);rel=[v for v in d if v['phase_label']=='Relax']
   t.append(dict(pair_id=x['pair_id'],side=side,physical_run=x['physical_run'],source=str(p.relative_to(a)),sha256=sha(p),rows=len(d),pure_relax=len(rel),mixed=sum(v['phase_label']=='Mixed' for v in d),first_relax_start=min(float(v['bin_start_ns']) for v in rel),last_relax_end=max(float(v['bin_end_ns']) for v in rel),one_frame_relax=sum(int(v['n_frames'])==1 for v in rel),x_is_start=all(float(v['x_ns'])==float(v['bin_start_ns']) for v in d),count_consistent=all(int(v['n_frames'])==int(v['pull_frames'])+int(v['relax_frames']) for v in d),outside_display=sum(float(v['apl_mean'])<40 or float(v['apl_mean'])>90 for v in d),pull_offset='unresolved; combined clock retained'))
 for f in g['figures']:
  text=json.dumps(f['axes']);absolute=f['figure_id'] in {'pristine_papl_b','pristine_papl_m','pristine_papl_combined','endpoint_late','binned_frame_reconciliation','temporal_selected_trajectories','papl_all_available_24_pairs_B','papl_all_available_24_pairs_M'} or f['figure_id'].startswith('papl_selected_run_')
  axes.append(dict(figure_id=f['figure_id'],category='absolute_projected_APL' if absolute else 'different_metric',original_axis=text,target_axis='40–90 Å²' if absolute else 'not applicable',compatible=absolute,evidence='registry metadata; renderer range must be verified separately',source=f.get('generator',''),png=f['png_path'],svg=f['svg_path'],action_status='audit_pending' if absolute else 'excluded_from_rescaling'))
 return dict(status='OFFLINE_SOURCE_AUDIT',figures=len(g['figures']),selected_pairs=len(u),run_counts=dict(Counter(x['physical_run'] for x in u)),systems=len(t),terminal_singletons=sum(v['one_frame_relax'] for v in t),all_x_start=all(v['x_is_start'] for v in t),all_phase_counts_valid=all(v['count_consistent'] for v in t),protected_hashes={n:sha(r/n) for n in ['figure_review_state.json','data/pca_parameters.json','data/frozen_parameters.json']},gui_sha256=sha(base/'scripts/review_metric_figures.py')),t,axes

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--report-dir',type=Path);args=p.parse_args();b=args.base.resolve();summary,t,axes=audit(b);print(json.dumps(summary,indent=2))
 if args.report_dir:
  dest=args.report_dir.resolve();stage=b/'staging/papl_pc3_integration_update_20261009'
  if not dest.is_relative_to(stage.resolve()):p.error('Report destination must be inside existing stage')
  dest.mkdir(parents=True,exist_ok=True)
  for name,data in [('time_audit.csv',t),('axis_audit.csv',axes)]:
   with (dest/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
  (dest/'read_only_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
