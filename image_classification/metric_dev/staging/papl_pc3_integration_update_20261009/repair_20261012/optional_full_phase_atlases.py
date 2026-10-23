#!/usr/bin/env python3
"""Optional, guarded Pull/Mixed/Relax views from existing canonical binned_APL.csv files.

READ ONLY by default. --render writes only to the existing stage's named subfolder.
--apply installs exactly six additive images and one source-manifest CSV in the
existing results tree, guarded by a transaction. Does not run any MD/analysis.
No equilibration masking without a documented phase/equilibration source.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUNS=('run_0','run_ML_cycle_01','run_ML_cycle_02')
EXPECTED={'run_0':35,'run_ML_cycle_01':52,'run_ML_cycle_02':69}
BILAYERS={'B':'POPE75POPG25_W_WF','M':'POPC60CHOL40_W_WF'}
RESULT_REL='results/archie_8193723_complete'
SOURCE_REL='surface_papl_analysis/tables/selected_cohort_full_phase_5ns_apl.csv'
STAGE_REL='staging/papl_pc3_integration_update_20261009/repair_20261012'


def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def check(ok,msg):
 if not ok:raise RuntimeError('STOP: '+msg)


def read_csv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def atomic(path,data):
 import tempfile
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.full_phase_',delete=False) as f:
  tmp=Path(f.name)
  try:f.write(data);f.flush();os.fsync(f.fileno())
  except Exception:tmp.unlink(missing_ok=True);raise
 try:os.replace(tmp,path)
 finally:tmp.unlink(missing_ok=True)


def csv_dump(rows):
 buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return buf.getvalue().encode()


def load(base):
 """Source-verified row interpretation; no inferral of exact transition time."""
 results=base/RESULT_REL;analysis=base.parent.parent
 selected=[r for r in read_csv(results/'data/unified_metrics.csv') if
           r['selected_pair'].strip().lower()=='true' and r['qc_pass_pair'].strip().lower()=='true']
 check(len(selected)==156,'Expected 156 selected systems')
 check(dict(Counter(r['physical_run'] for r in selected))==EXPECTED,'Unexpected selected run distribution')
 refrows=read_csv(results/'surface_papl_analysis/tables/matched_late_pAPL.csv')
 refs={(r['pair_id'],r['side']):r for r in refrows}
 series={};snapshot=[];warnings=[]
 for r in sorted(selected,key=lambda r:(r['physical_run'],r['pair_id'])):
  run=r['physical_run'];molecule=r['molecule'];nmol=int(r['nmol']);pair=r['pair_id']
  check(pair==f'{molecule}|{nmol}|{run}',f'Invalid pair identity {pair}')
  for side,bilayer in BILAYERS.items():
   path=analysis/'output_2'/f'{molecule}_{nmol}'/bilayer/run/'binned_APL.csv'
   check(path.is_file(),f'Missing canonical source: {path}')
   rows=read_csv(path)
   required={'bin_index','bin_start_ns','bin_end_ns','x_ns','apl_mean','phase_label','n_frames','pull_frames','relax_frames'}
   check(rows and required<=rows[0].keys(),f'Missing phase columns: {path}')
   check(len(rows)==41,f'Unexpected number of 5 ns bins ({len(rows)}): {path}')
   counts=Counter(x['phase_label'].strip().lower() for x in rows)
   check(dict(counts)=={'pull':20,'mixed':1,'relax':20},f'Unexpected saved phase membership: {path} {dict(counts)}')
   parsed=[];s_hash=digest(path)
   for i,v in enumerate(rows):
    a=float(v['bin_start_ns']);b=float(v['bin_end_ns']);x=float(v['x_ns']);mean=float(v['apl_mean'])
    n=int(float(v['n_frames']));npull=int(float(v['pull_frames']));nrelax=int(float(v['relax_frames']))
    label=v['phase_label'].strip().lower()
    check(all(math.isfinite(z) for z in (a,b,x,mean)) and n>0,f'Nonfinite/empty bin {path} index {i}')
    check(abs(b-a-5.0)<1e-4 and abs(a-i*5)<1e-4,f'Unexpected nominal time-bin limits: {path} index {i}')
    check(abs(x-a)<1e-4,f'Saved x_ns is not bin start: {path} index {i}')
    check(n==npull+nrelax,f'Frame count discrepancy {path} index {i}')
    if label=='pull':check(npull==n and nrelax==0,f'Invalid pure Pull bin {path} {i}')
    if label=='relax':check(nrelax==n and npull==0,f'Invalid pure Relax bin {path} {i}')
    if label=='mixed':check(npull>0 and nrelax>0,f'Mixed bin not actually mixed {path}')
    parsed.append(dict(x=x,start=a,end=b,apl=mean,n=n,phase=label,pull=npull,relax=nrelax))
    snapshot.append(dict(pair_id=pair,side=side,physical_run=run,molecule=molecule,nmol=nmol,
      bin_index=i,combined_x_ns=x,bin_start_ns=a,bin_end_ns=b,apl_mean_A2=mean,
      n_frames=n,pull_frames=npull,relax_frames=nrelax,phase_label=label,
      source_apl_csv=str(path.relative_to(analysis)),source_sha256=s_hash))
   check(parsed[0]['phase']=='pull' and parsed[20]['phase']=='mixed' and parsed[21]['phase']=='relax',f'Unexpected phase order {path}')
   check(parsed[-1]['x']==200 and parsed[-1]['n']==1,f'Missing 200 ns singleton marker {path}')
   reference=refs.get((pair,side));check(reference is not None,f'Missing saved reference for {pair} {side}')
   ref=float(reference['pAPL_reference_A2']);check(math.isfinite(ref),f'Invalid reference {pair} {side}')
   series[(pair,side)]=dict(points=parsed,reference=ref,molecule=molecule,nmol=nmol,run=run,source=str(path),hash=s_hash)
   if any(not 40<=v['apl']<=90 for v in parsed):warnings.append(f'Plot-axis clipping flagged: {pair} {side}')
 check(len(series)==312 and len(snapshot)==312*41,'Full cohort row reconciliation failed')
 return selected,series,snapshot,warnings


def stem(run,side):return f'papl_full_pull_relax_{run}_{side}'


def render(selected,series,dest):
 import matplotlib
 matplotlib.use('Agg')
 matplotlib.rcParams['svg.fonttype']='none'
 import matplotlib.pyplot as plt
 dest.mkdir(parents=True,exist_ok=True)
 saved={}
 for run in RUNS:
  pairs=sorted((r for r in selected if r['physical_run']==run),key=lambda r:r['pair_id'])
  for side in ('B','M'):
   name=stem(run,side);png=dest/(name+'.png');svg=dest/(name+'.svg')
   if png.exists() and svg.exists():
    from PIL import Image
    import xml.etree.ElementTree as ET
    with Image.open(png) as im:im.verify()
    ET.parse(svg)
    saved[(run,side)]={'png':digest(png),'svg':digest(svg)}
    print('VERIFIED PREVIOUS STAGED RENDER:',name,flush=True)
    continue
   check(not png.exists() and not svg.exists(),f'Incomplete pair of staged images; remove/move manually after inspection: {name}')
   n=len(pairs);cols=6;rows=math.ceil(n/cols)
   fig,axes=plt.subplots(rows,cols,figsize=(18,rows*2.4+1.6),sharex=True,sharey=True,squeeze=False)
   for i,r in enumerate(pairs):
    ax=axes.flat[i];d=series[(r['pair_id'],side)];pts=d['points']
    # The 100–105 ns bin contains both stages. It is not a pure stage mean.
    for phase,color,style in [('pull','tab:gray','-'),('relax','tab:blue','-')]:
     q=[p for p in pts if p['phase']==phase]
     ax.plot([p['x'] for p in q],[p['apl'] for p in q],style,color=color,lw=1,ms=2,marker='o',zorder=3)
    mix=[p for p in pts if p['phase']=='mixed']
    ax.scatter([p['x'] for p in mix],[p['apl'] for p in mix],marker='D',color='tab:orange',s=14,zorder=4)
    singleton=[p for p in pts if p['n']==1]
    if singleton:ax.scatter([p['x'] for p in singleton],[p['apl'] for p in singleton],marker='x',color='red',s=17,zorder=5)
    ax.axvspan(100,105,color='tab:orange',alpha=.10,zorder=0)
    ax.axhline(d['reference'],color='0.5',ls='--',lw=.8,zorder=1)
    if any(not 40<=p['apl']<=90 for p in pts):ax.text(.02,.92,'OUTSIDE 40–90',transform=ax.transAxes,color='red',fontsize=7)
    ax.set_title(f"{r['molecule']} | n={int(r['nmol'])}",fontsize=8,pad=5)
    ax.set_xlim(0,205);ax.set_ylim(40,90);ax.set_xticks([0,50,100,150,200]);ax.set_yticks([40,50,60,70,80,90]);ax.grid(alpha=.18);ax.tick_params(labelsize=7)
    if i%cols==0:ax.set_ylabel('pAPL (Å²)',fontsize=9)
    if i//cols==rows-1:ax.set_xlabel('Combined time (ns)',fontsize=9)
   for i in range(n,cols*rows):axes.flat[i].set_visible(False)
   fig.suptitle(f'Optional full trajectory — {run}, {side} membrane | {n} selected pairs',fontsize=14,y=.999)
   fig.text(.5,.982,'Source-labelled 5 ns means: grey Pull • orange diamond Mixed (100–105 ns nominal) • blue Relax • red × one-frame endpoint',ha='center',fontsize=8)
   fig.text(.5,.972,'Mixed stage has no exact boundary inferred; no equilibration exclusion without documented interval; references dashed',ha='center',fontsize=8)
   fig.tight_layout(rect=(0,0,1,.956),h_pad=1.2,w_pad=.6)
   name=stem(run,side);png=dest/(name+'.png');svg=dest/(name+'.svg')
   fig.savefig(png,dpi=105);fig.savefig(svg);plt.close(fig)
   check(png.stat().st_size>4000 and svg.stat().st_size>4000,'Rendered figure suspiciously small: '+name)
   saved[(run,side)]={'png':digest(png),'svg':digest(svg)}
   print('RENDERED:',name,flush=True)
 return saved


def desired_records(registry,selected,series,render_dir):
 orig={r['figure_id']:r for r in registry['figures']};created=[]
 for run in RUNS:
  for side in ('B','M'):
   canonical=orig[f'papl_selected_{run}_{side}'];new=copy.deepcopy(canonical)
   name=stem(run,side);root=f'surface_papl_analysis/figures/{{ext}}/temporal/{name}'
   vals=[p['apl'] for (pair,s),d in series.items() if d['run']==run and s==side for p in d['points']]
   new.update(figure_id=name,title=f'OPTIONAL full Pull + Relax projected APL — {run}, {side} ({EXPECTED[run]} pairs)',
     png_path=root.format(ext='png')+'.png',svg_path=root.format(ext='svg')+'.svg',
     png_sha256=digest(render_dir/(name+'.png')),svg_sha256=digest(render_dir/(name+'.svg')),
     original_question='How does the source-labelled Pull/Mixed/Relax APL history differ from the primary Relax-only response?',
     scientific_purpose='Secondary full-history view; retain the existing primary Relax-only plot unchanged.',
     axes={'x':'Saved combined bin-start x_ns (ns), 0–205 display; phase labels from source, exact stage change unresolved',
           'y':'Absolute projected APL (Å²); fixed 40–90',
           'phase':'Pull and Relax means disconnected across explicit Mixed bin (nominal 100–105 ns)',
           'equilibration':'Not excluded: no authenticated exclusion interval supplied'},
     colour_grouping_semantics='Grey Pull, blue Relax, orange diamond mixed-stage mean, red × single-frame endpoint; dashed line original reference.',
     observed_result=f'{EXPECTED[run]} QC-selected systems; saved combined APL range {min(vals):.3f}–{max(vals):.3f} Å².',
     interpretation='Full phase-labelled projection of the already saved canonical 5 ns binned APL; not a new simulation or a new local-time alignment.',
     limitations='The mixed bin contains frames from both stages; exact transition within its nominal 100–105 ns interval is not established. Initial equilibration is not excluded because no authenticated interval was supplied. Terminal 200 ns value is one frame. No extrapolation beyond 200 ns or unwarranted Relax-local late window. 40–90 Å² may hide out-of-range points (labelled on panels).',
     source_tables=[SOURCE_REL,'data/unified_metrics.csv','surface_papl_analysis/tables/matched_late_pAPL.csv'],
     related_figures=[f'papl_selected_{run}_{side}'],
     generator='staging/papl_pc3_integration_update_20261009/repair_20261012/optional_full_phase_atlases.py',
     method='Read-only rendering of saved binned_APL.csv phase labels; no physical recalculation or masked equilibration.',
     status='optional_full_phase_display',
     figure_specific_context=f'{run} {side}: {EXPECTED[run]} selected pairs. Each series contains 20 Pull, one Mixed and 20 Relax bins. Mixed is an interval-aggregate, not an exact transition timestamp.',
     what_to_look_for='Compare one physical pair between B and M on the same absolute APL scale; explicitly inspect pull, mixed and relax discontinuity and the one-frame terminal marker.',
     common_misreadings='A mixed bin is not a pure Pull or Relax mean; the 200–205 ns nominal bin contains only the frame at 200 ns; unknown equilibration must not be arbitrarily excluded.',
     provenance=[{'source_document':SOURCE_REL,'section_or_heading':f'physical_run={run}; side={side}; source binned_APL.csv hashes preserved','provenance_type':'completed_result'}],
     result_provenance=[{'source_document':SOURCE_REL,'section_or_heading':f'Original saved phase labels, run {run}, membrane {side}','provenance_type':'completed_result'}],
     numeric_evidence=[{'source_table':SOURCE_REL,'row_filter':f'physical_run={run}; side={side}',
       'columns':['pair_id','phase_label','combined_x_ns','apl_mean_A2','n_frames','pull_frames','relax_frames','source_sha256'],
       'records':[{'selected_pairs':EXPECTED[run],'bins_per_system':41,
                   'pull_bins':20,'mixed_bins':1,'relax_bins':20,
                   'minimum_A2':round(min(vals),4),'maximum_A2':round(max(vals),4)}]}])
   # Do not carry a P1 original-display derivative from a different image.
   new.pop('original_display',None);new.pop('display_source_hashes',None)
   created.append(new)
 return created


def render_phase(base,out):
 results=base/RESULT_REL
 selected,series,snapshot,warnings=load(base)
 staged=out/'rendered';snap=csv_dump(snapshot)
 fingerprint=hashlib.sha256(snap).hexdigest()
 if staged.exists():
  guard=staged/'render_inputs.sha256'
  check(guard.exists() and guard.read_text().strip()==fingerprint,'Existing partial stage cannot be trusted for this source; inspect it before rerendering')
  check(not (staged/'render_report.json').exists(),'Rendered atlas set already complete; use preflight/apply instead of overwriting it')
 else:
  staged.mkdir(parents=True)
  atomic(staged/'render_inputs.sha256',(fingerprint+'\n').encode())
 try:
  images=render(selected,series,staged)
  atomic(staged/'selected_cohort_full_phase_5ns_apl.csv',snap)
  report=dict(schema_version=1,cohort_n=156,system_series_n=312,source_rows=len(snapshot),
       primary_figures_untouched=6,new_optional_figures=6,source_sha256=hashlib.sha256(snap).hexdigest(),
       cohort_source_sha256=digest(base/RESULT_REL/'data/unified_metrics.csv'),
       reference_source_sha256=digest(base/RESULT_REL/'surface_papl_analysis/tables/matched_late_pAPL.csv'),
       rendered_images={stem(k[0],k[1]):v for k,v in images.items()},
       source_phase_counts=dict(Counter(r['phase_label'] for r in snapshot)),
       equilibration_exclusion='none: no verified interval evidence',
       transition='Mixed bin nominally [100,105); no exact physical boundary inferred',
       out_of_range_warnings=warnings)
  atomic(staged/'render_report.json',(json.dumps(report,indent=2)+'\n').encode())
 except Exception:
  print('Staged partial files retained for guarded resumption:',staged,file=sys.stderr)
  raise
 print('STAGED RENDER VALIDATED:',staged)


def apply(base,out):
 import fcntl
 result=base/RESULT_REL;manifest=out/'rendered/render_report.json'
 check(manifest.exists(),'Run --render first on validated Archie inputs')
 report=json.loads(manifest.read_text());check(report['cohort_n']==156 and report['source_rows']==12792,'Incomplete render')
 check(digest(result/'data/unified_metrics.csv')==report['cohort_source_sha256'],'Selected cohort changed since render')
 check(digest(result/'surface_papl_analysis/tables/matched_late_pAPL.csv')==report['reference_source_sha256'],'Reference APL table differs from rendered source')
 lockfd=os.open(result,os.O_RDONLY)
 try:
  fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  registry_file=result/'review_bundle.json';reg=json.loads(registry_file.read_text())
  names={f['figure_id'] for f in reg['figures']};check(len(reg['figures'])==168,'Expected accepted 168 original figures before optional addition')
  check(all(stem(run,side) not in names for run in RUNS for side in BILAYERS),'Optional figures already registered')
  staged=out/'rendered';protection={x:digest(result/x) for x in ('figure_review_state.json','data/pca_parameters.json','data/frozen_parameters.json')}
  profile=result/'visual_comparison_profiles.json'
  if profile.exists():protection['visual_comparison_profiles.json']=digest(profile)
  originals={fid:f for fid,f in ((f['figure_id'],f) for f in reg['figures']) if fid.startswith('papl_selected_')}
  selected,series,snapshot,warnings=load(base)
  check(hashlib.sha256(csv_dump(snapshot)).hexdigest()==report['source_sha256'],'Canonical binned data changed since render')
  targets={}
  for run in RUNS:
   for side in BILAYERS:
    name=stem(run,side)
    for ext in ('png','svg'):
     src=staged/(name+'.'+ext)
     check(src.exists() and digest(src)==report['rendered_images'][name][ext],'Staged plot modified')
     dest=result/f'surface_papl_analysis/figures/{ext}/temporal/{name}.{ext}'
     check(not dest.exists(),'Refusing existing destination: '+str(dest))
     targets[dest]=src.read_bytes()
  src=staged/'selected_cohort_full_phase_5ns_apl.csv';check(digest(src)==report['source_sha256'],'Snapshot changed')
  target=result/SOURCE_REL;check(not target.exists(),'Snapshot destination already exists')
  targets[target]=src.read_bytes()
  rec=desired_records(reg,selected,series,staged)
  check(len(rec)==6,'Unexpected figure count')
  reg['figures'].extend(rec)
  targets[registry_file]=(json.dumps(reg,indent=2,ensure_ascii=False)+'\n').encode()
  backups=out/'backups';backups.mkdir(exist_ok=True)
  txn=backups/('full_phase_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
  txn.mkdir();before=txn/'review_bundle.original.json';before.write_bytes(registry_file.read_bytes())
  tx=dict(base=str(base),protected=protection,files={}, created_utc=datetime.now(timezone.utc).isoformat())
  for target,payload in targets.items():
   relative=str(target.relative_to(base));tx['files'][relative]={'after_sha256':hashlib.sha256(payload).hexdigest(),
     'original_backup':'review_bundle.original.json' if target==registry_file else None,
     'before_sha256':digest(target) if target.exists() else None}
  (txn/'transaction.json').write_text(json.dumps(tx,indent=2)+'\n')
  written=[]
  try:
   for path,data in targets.items():atomic(path,data);written.append(path)
   for rel,h in protection.items():check(digest(result/rel)==h,'Protected state changed: '+rel)
   check(len(json.loads(registry_file.read_text())['figures'])==174,'Figure registry count incorrect')
   # Headless validator verifies new registry figure IDs, image hashes, source tables and provenance.
   import subprocess
   viewer=base/'scripts/review_metric_figures.py'
   check(viewer.is_file(),'Existing GUI entrypoint missing')
   verified=subprocess.run([sys.executable,str(viewer),'--results',str(result),'--check'],cwd=result,text=True,capture_output=True)
   check(verified.returncode==0,'GUI registry validation after optional atlas installation failed: '+verified.stdout+'\n'+verified.stderr)
   (txn/'gui_validation.json').write_text(verified.stdout)
   print('OPTIONAL FULL-PHASE APPLY PASS: 174 figures; six primary Relax atlases untouched')
   print('ROLLBACK TRANSACTION:',txn/'transaction.json')
  except Exception:
   for target in reversed(written):
    if digest(target)!=tx['files'][str(target.relative_to(base))]['after_sha256']:
     raise RuntimeError('Rollback stopped: target changed externally '+str(target))
    if target==registry_file:atomic(target,before.read_bytes())
    else:target.unlink()
   raise
 finally:os.close(lockfd)


def rollback(base,manifest):
 import fcntl
 m=json.loads(manifest.read_text());check(m['base']==str(base),'Rollback belongs to different root')
 result=base/RESULT_REL;fd=os.open(result,os.O_RDONLY)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for rel,info in m['files'].items():check(digest(base/rel)==info['after_sha256'],'Target has changed: '+rel)
  for rel,info in m['files'].items():
   target=base/rel
   if info['original_backup']:
    saved=manifest.parent/info['original_backup'];check(digest(saved)==info['before_sha256'],'Backup changed');atomic(target,saved.read_bytes())
   else:target.unlink()
  print('OPTIONAL FULL-PHASE ROLLBACK PASS')
 finally:os.close(fd)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
 group=parser.add_mutually_exclusive_group();group.add_argument('--render',action='store_true');group.add_argument('--rerender',action='store_true',help='Discard only verified installer-owned staged plot files, then rebuild from local sources');group.add_argument('--apply',action='store_true');group.add_argument('--rollback',type=Path)
 args=parser.parse_args();base=args.base.resolve();out=base/STAGE_REL
 check(out.is_dir(),'Expected existing dedicated stage package: '+str(out))
 check((base/RESULT_REL).is_dir(),'Existing GUI results not found')
 if args.rollback:rollback(base,args.rollback.resolve());return
 if args.apply:apply(base,out);return
 if args.rerender:
  staged=out/'rendered';check(staged.is_dir(),'No prepared rendered files to replace; use --render')
  manifest=staged/'render_report.json';check(manifest.exists(),'No completed owned-render manifest; inspect partial files manually')
  report=json.loads(manifest.read_text())
  expected={stem(run,side)+'.'+ext for run in RUNS for side in BILAYERS for ext in ('png','svg')}
  expected|={'render_report.json','render_inputs.sha256','selected_cohort_full_phase_5ns_apl.csv'}
  check({p.name for p in staged.iterdir()}==expected,'Unexpected file in staged rendering directory; refusing cleanup')
  for run in RUNS:
   for side in BILAYERS:
    for ext in ('png','svg'):
     name=stem(run,side)
     check(digest(staged/(name+'.'+ext))==report['rendered_images'][name][ext],'Staged render modified; refusing cleanup')
  check(digest(staged/'selected_cohort_full_phase_5ns_apl.csv')==report['source_sha256'],'Staged source table modified; refusing cleanup')
  shutil.rmtree(staged);render_phase(base,out);return
 if args.render:render_phase(base,out);return
 selected,series,snapshot,warnings=load(base)
 print('READ-ONLY SOURCE GATE PASS:',len(selected),'pairs;',len(series),'B/M series;',len(snapshot),'source rows; Pull/Mixed/Relax labels verified')
 staged=out/'rendered/render_report.json'
 if staged.exists():
  info=json.loads(staged.read_text());match=(info['source_sha256']==hashlib.sha256(csv_dump(snapshot)).hexdigest() and
     info['cohort_source_sha256']==digest(base/RESULT_REL/'data/unified_metrics.csv') and
     info['reference_source_sha256']==digest(base/RESULT_REL/'surface_papl_analysis/tables/matched_late_pAPL.csv'))
  print('Prepared six-atlas source fidelity:', 'MATCH — --apply eligible' if match else 'MISMATCH — inspect, then use --rerender')
 else:print('Six-atlas render not staged; use --render')
 print('No authenticated equilibration exclusion interval; none masked.')
 if warnings:print('Axis warnings:',len(warnings))

if __name__=='__main__':
 try:main()
 except Exception as exc:print(exc,file=sys.stderr);sys.exit(1)
