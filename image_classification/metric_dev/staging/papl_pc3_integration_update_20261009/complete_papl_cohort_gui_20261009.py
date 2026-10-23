#!/usr/bin/env python3
"""Integrate selected-cohort pAPL time-series and pending GUI navigation fix.

Default is STRICTLY READ-ONLY. --apply backs up old 24-pair assets and
registry/GUI, replaces the two legacy entries with six full-cohort B/M cycle
atlases, stores one reproducibility snapshot CSV, applies navigation fix and
runs the original GUI's headless validation. Does not run aplanalysis.py or
modify frozen PCA, primary APL inputs or saved review state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUNS = ('run_0', 'run_ML_cycle_01', 'run_ML_cycle_02')
EXPECTED = {'run_0':35, 'run_ML_cycle_01':52, 'run_ML_cycle_02':69}
BILAYERS = {'B':'POPE75POPG25_W_WF','M':'POPC60CHOL40_W_WF'}
OLD_FIDS = ('papl_all_available_24_pairs_B','papl_all_available_24_pairs_M')
EXPECTED_OLD_FIGURES=164
EXPECTED_NEW_FIGURES=168
YMIN,YMAX = 40.0,90.0
SOURCE_MANIFEST='surface_papl_analysis/tables/selected_cohort_combined_5ns_apl.csv'
LATE_SOURCE='surface_papl_analysis/tables/matched_late_pAPL.csv'
COHORT_SOURCE='data/unified_metrics.csv'
OUT_BASE='surface_papl_analysis/figures'
REPORT_NAME='full_cohort_papl_gui_validation_20261009.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def expect(cond,message):
    if not cond:raise RuntimeError('STOP: '+message)


def is_true(x):return str(x).strip().lower() in ('true','1')


def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def source_paths(base):
    # metric_dev is analysis/image_classification/metric_dev
    analysis=base.parent.parent
    results=base/'results/archie_8193723_complete'
    stage=base/'staging/papl_pc3_integration_update_20261009'
    return analysis,results,stage


def load_selected(base,results):
    analysis=base.parent.parent
    u=read_csv(results/COHORT_SOURCE)
    must={'pair_id','molecule','nmol','physical_run','selected_pair','qc_pass_pair'}
    expect(bool(u) and must.issubset(u[0]), 'Canonical cohort table or required fields missing')
    selected=[r for r in u if is_true(r['selected_pair']) and is_true(r['qc_pass_pair'])]
    expect(len(selected)==156, f'Expected 156 selected/QC pairs; found {len(selected)}')
    ids=[r['pair_id'] for r in selected]
    expect(len(set(ids))==156,'Duplicate cohort pair_id')
    runs=Counter(r['physical_run'] for r in selected)
    expect(dict(runs)==EXPECTED,f'Unexpected cohort membership: {dict(runs)}')
    refs_rows=read_csv(results/LATE_SOURCE)
    refs={(r['pair_id'],r['side']):r for r in refs_rows}
    expect(len(refs)==len(refs_rows), 'Duplicate entries in original matched-late table')
    source_bins=[];series={};range_values=[];frame_counts={}
    for r in sorted(selected,key=lambda v:(v['physical_run'],v['pair_id'])):
        pair=r['pair_id'];run=r['physical_run'];side_keys=[]
        molecule=r['molecule'];nmol=int(r['nmol'])
        expect(nmol>0 and pair==f'{molecule}|{nmol}|{run}',f'Unexpected pair_id semantics: {pair}')
        for side,bilayer in BILAYERS.items():
            src=analysis/'output_2'/f'{molecule}_{nmol}'/bilayer/run/'binned_APL.csv'
            expect(src.is_file(),f'Missing {side} APL input: {src}')
            rows=read_csv(src)
            expect(bool(rows),f'Empty APL input: {src}')
            cols={'x_ns','apl_mean','phase_label','bin_start_ns','bin_end_ns','n_frames'}
            expect(cols.issubset(rows[0]),f'Missing expected APL columns in {src}')
            pure_relax=[v for v in rows if v['phase_label'].strip().lower()=='relax']
            expect(bool(pure_relax),f'No pure-Relax bins: {src}')
            values=[]
            for v in pure_relax:
                start=float(v['bin_start_ns']);end=float(v['bin_end_ns']);x=float(v['x_ns']);apl=float(v['apl_mean'])
                expect(all(map(math.isfinite,(start,end,x,apl))),f'Nonfinite APL/time in {src}')
                expect(abs((end-start)-5.0)<1e-5,f'Unexpected bin width in {src}')
                expect(abs(x-start)<1e-4 or abs(x-(start+end)/2)<1e-4,f'Unsupported x_ns meaning in {src}')
                expect(0 <= start <= 200+1e-5 and end <= 205+1e-5, f'Outside pipeline configured combined window: {src}')
                expect(int(v['pull_frames']) == 0 and int(v['relax_frames']) == int(v['n_frames']), f'Inconsistent pure Relax counts: {src}')
                # Out-of-range means remain in the snapshot and are flagged in the panel.
                n=int(float(v['n_frames']))
                expect(n>0,f'Empty nominal bin in {src}')
                local=x  # Saved combined coordinate: never infer exact Pull end from bins.
                values.append((start,end,local,apl,n))
            values.sort()
            starts=[v[0] for v in values]
            expect(len(starts)==len(set(starts)),f'Duplicate Relax bins in {src}')
            expect(starts[0] <= starts[-1], f'Invalid coverage: {src}')
            expect(all(abs((b-a)-5)<1e-5 for a,b in zip(starts,starts[1:])),f'Gapped Relax bins: {src}')
            ref_row=refs.get((pair,side))
            expect(ref_row is not None,f'Missing matched-late row: {pair} {side}')
            ref=float(ref_row['pAPL_reference_A2'])
            expect(math.isfinite(ref),f'Nonfinite pristine/initial reference: {pair} {side}')
            expect(YMIN<=ref<=YMAX,f'Reference outside 40–90 Å²: {pair} {side}')
            sha=digest(src)
            series[(pair,side)]=dict(run=run,side=side,molecule=molecule,nmol=nmol,
                                     reference=ref,source=src,hash=sha,points=values)
            frame_counts[(run,side)]=frame_counts.get((run,side),0)+len(values)
            range_values.extend(p[3] for p in values)
            for start,end,local,apl,n in values:
                source_bins.append(dict(pair_id=pair,molecule=molecule,nmol=nmol,physical_run=run,
                    side=side,reference_APL_A2=repr(ref),combined_x_ns=f'{local:.8f}',
                    combined_bin_start_ns=f'{start:.8f}',combined_bin_end_ns=f'{end:.8f}',
                    apl_mean_A2=repr(apl),n_frames=n,
                    source_apl_csv=str(src.relative_to(analysis)),source_sha256=sha))
            side_keys.append(side)
        expect(side_keys==['B','M'],f'Unpaired data: {pair}')
    expect(len(series)==312, 'Expected exactly 312 side-wise time series')
    return selected,series,source_bins,frame_counts,(min(range_values),max(range_values))


def csv_bytes(rows):
    assert rows
    out=io.StringIO(newline='')
    w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return out.getvalue().encode('utf-8')


def make_figures(selected,series,where):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    from PIL import Image
    results={}
    for run in RUNS:
        pairs=sorted((r for r in selected if r['physical_run']==run),key=lambda r:r['pair_id'])
        for side in ('B','M'):
            n=len(pairs);ncols=6;nrows=math.ceil(n/ncols)
            fig,axes=plt.subplots(nrows,ncols,figsize=(18,nrows*2.40+1.35),
                                   sharex=True,sharey=True,squeeze=False)
            for idx,r in enumerate(pairs):
                ax=axes.flat[idx];entry=series[(r['pair_id'],side)]
                ts=[a[2] for a in entry['points']];ys=[a[3] for a in entry['points']]
                # No late Relax shading without independently verified offset.
                ax.plot(ts,ys,'o-',lw=1.0,ms=2.0,zorder=2)
                singleton=[p for p in entry['points'] if p[4] == 1]
                if singleton: ax.scatter([p[2] for p in singleton],[p[3] for p in singleton], marker='x', color='red', s=15, zorder=3)
                if any(y < YMIN or y > YMAX for y in ys): ax.text(.02,.92,'OUTSIDE 40–90',transform=ax.transAxes,color='red',fontsize=7)
                ax.axhline(entry['reference'],ls='--',lw=.85,color='0.45',zorder=1)
                ax.set_title(f"{r['molecule']} | n={int(r['nmol'])}",fontsize=8,pad=5)
                ax.set_xlim(0,205);ax.set_ylim(YMIN,YMAX)
                ax.set_xticks([0,50,100,150,200]);ax.set_yticks([40,50,60,70,80,90])
                ax.grid(alpha=.18);ax.tick_params(labelsize=7)
                if idx%ncols==0:ax.set_ylabel('pAPL (Å²)',fontsize=9)
                if idx//ncols==nrows-1:ax.set_xlabel('Combined time (ns)',fontsize=9)
            for idx in range(n,nrows*ncols):axes.flat[idx].set_visible(False)
            fig.suptitle(f'Projected APL — {side} membrane | {run} | {n} selected pairs',y=.999,fontsize=14)
            fig.text(.5,.983,'aplanalysis.py saved pure-Relax means; combined x_ns; fixed 40–90 Å²; red × = one frame; Mixed omitted',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,0,1,.968),h_pad=1.2,w_pad=.6)
            stem=f'papl_selected_{run}_{side}'
            png=where/(stem+'.png');svg=where/(stem+'.svg')
            fig.savefig(png,dpi=105)
            fig.savefig(svg)
            plt.close(fig)
            with Image.open(png) as im:im.verify()
            expect(png.stat().st_size>4000 and svg.stat().st_size>4000,'Unexpectedly small atlas '+stem)
            results[(run,side)]=(png,svg)
            print(f'RENDER PASS: {stem} ({n} panels)',flush=True)
    return results


def figure_records(selected,series,frames,images,source_digest):
    recs=[]
    for run in RUNS:
        n=EXPECTED[run]
        for side in ('B','M'):
            fid=f'papl_selected_{run}_{side}'
            png,svg=images[(run,side)]
            relroot=f'{OUT_BASE}/{{ext}}/temporal/{fid}'
            values=[p[3] for (pair,s),data in series.items() if data['run']==run and s==side for p in data['points']]
            rec={
              'figure_id':fid,'title':f'Projected APL — {run}, {side} membrane ({n} selected pairs)',
              'family':'temporal','side':side,'physical_run':run,'png_path':relroot.format(ext='png')+'.png',
              'svg_path':relroot.format(ext='svg')+'.svg',
              'png_sha256':digest(png),'svg_sha256':digest(svg),
              'scientific_purpose':'Compare canonical box-area APL time evolution across all QC-selected peptoids in this physical run and membrane.',
              'original_question':'Are selected peptoids temporally stable or do their projected membrane footprints change during Relax?',
              'original_hypothesis':'Descriptive follow-up; no prospective directional hypothesis for these cycle-specific figures.',
              'hypothesis_status':'exploratory_no_directional_hypothesis',
              'hypothesis_provenance':[{'source_document':'surface_papl_analysis/SCIENTIFIC_DECISION_REPORT.md','section_or_heading':'Temporal response','provenance_type':'method_definition'}],
              'expected_pattern':{'support':'Differences and temporal trajectories visible on a consistent scale.',
                                  'against':'Weak or ambiguous response and time-dependent reversals.'},
              'axes':{'x':'Saved combined x_ns (ns), bin-start convention; 0–205 ns display; pure Relax only',
                      'y':'Canonical BOX projected APL (Å²); fixed 40–90 on every subplot',
                      'shaded':'No late-window shading: exact per-system offset unresolved'},
              'colour_grouping_semantics':'One panel per QC-selected physical B/M pair, same panel order across B/M for this cycle; blue line = binned projected APL; grey dashed = saved original projected-APL reference.',
              'figure_specific_context':f'{run}: {n} selected pair identities. No independent replicate claim; APLAnalysis.py binned_APL.csv is canonical source.',
              'what_to_look_for':'Compare a named peptoid across B and M; distinguish baseline offset from time-evolving APL. A flatter trace under 40–90 common scale can still have nonzero slopes.',
              'observed_result':f'All {n} selected B/M pairs in {run} have saved 5 ns pure-Relax bins. This side has {frames[(run,side)]} observations; its binned APL range is {min(values):.3f}–{max(values):.3f} Å².',
              'interpretation':'Descriptive projected area response of selected physical trajectories; does not quantify intrinsic surface-following molecular area, reproducibility across replicates, or antimicrobial efficacy.',
              'limitations':'One physical run per selected system. 5 ns bins are temporally correlated; no replicate uncertainty. Original reference is an initial membrane state, not a validated stationary pristine null. Saved combined x_ns retained without shifting. Mixed bins are omitted; one-frame terminal bins marked red ×. A nominal 200–205 ns bin need not contain frames beyond 200 ns. Exact Pull offset is unresolved. This is a secondary visualization, not an independent APL calculation.',
              'common_misreadings':'Projected APL is not curvature-corrected surface area; consistency of 40–90 Å² axes is not evidence of identical sensitivity or equilibrium.',
              'source_tables':[SOURCE_MANIFEST,COHORT_SOURCE,LATE_SOURCE,'surface_papl_analysis/VARIABLES_AND_METHODS.md'],
              'provenance':[{'source_document':SOURCE_MANIFEST,'section_or_heading':f'selected_pair=true; physical_run={run}; side={side}','provenance_type':'completed_result'}],
              'result_provenance':[{'source_document':SOURCE_MANIFEST,'section_or_heading':f'APLAnalysis.py binned_APL.csv SHA256 snapshot for {run}/{side}','provenance_type':'completed_result'}],
              'numeric_evidence':[{'source_table':SOURCE_MANIFEST,'row_filter':f'physical_run={run}; side={side}',
                                   'columns':['pair_id','side','combined_x_ns','apl_mean_A2','source_sha256'],
                                   'records':[{'selected_pairs':n,'binned_points':frames[(run,side)],
                                               'minimum_A2':round(min(values),5),'maximum_A2':round(max(values),5),
                                               'snapshot_csv_sha256':source_digest}]}],
              'metric_ids':['continuation'],'analysis_technique_ids':['continuation'],
              'related_figures':[f'papl_selected_{run}_{"M" if side=="B" else "B"}'],
              'status':'new_complete_cohort_plot','n':n,
              'generator':'scripts/complete_papl_cohort_gui_20261009.py',
              'method':'Read-only visualization of canonical APLAnalysis.py 5 ns BOX APL means and historical reference; no rerun/rebinning.'}
            recs.append(rec)
    return recs


def old_manifest(results):
    registry_file=results/'review_bundle.json'
    guide=json.loads(registry_file.read_text())
    figures=guide['figures'];ids=[e['figure_id'] for e in figures]
    expect(guide.get('schema_version')==2 and len(figures)==EXPECTED_OLD_FIGURES and len(set(ids))==len(ids),
           'Expected original 164-entry validated review registry')
    old={e['figure_id']:e for e in figures if e['figure_id'] in OLD_FIDS}
    expect(set(old)==set(OLD_FIDS),'Cannot find both 24-pair legacy atlas entries')
    expect(all(not fid.startswith('papl_selected_run_') for fid in ids),'Full-cohort figures already in registry')
    original_paths=[]
    for entry in old.values():
        for ext in ('png','svg'):
            rel=Path(entry[ext+'_path']);path=results/rel
            expect(path.is_file() and digest(path)==entry[ext+'_sha256'],f'Original registered atlas invalid: {path}')
            original_paths.append((str(rel),path))
    for e in figures:
        if e['figure_id'] in OLD_FIDS:continue
        expect(not set(e.get('related_figures',[]))&set(OLD_FIDS),
               'An unrelated GUI figure links to the old atlas; manual review required')
    return guide,original_paths


def nav_transform(stage,gui):
    module_path=stage/'fix_review_gui_navigation.py'
    expect(module_path.is_file(),'Upload the previously supplied fix_review_gui_navigation.py to existing staging folder')
    spec=importlib.util.spec_from_file_location('pfp_nav_patch',module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original=gui.read_bytes()
    expect(hashlib.sha256(original).hexdigest()==module.SOURCE_SHA256,
           'Existing GUI code differs from the expected pre-navigation-fix source. Do not double-patch.')
    patched=module.transform(original.decode('utf-8')).encode('utf-8')
    return original,patched


def atomic_write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.pfp_fullcohort_',delete=False) as f:
            tmp=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
        os.chmod(tmp,(p.stat().st_mode&0o777) if p.exists() else 0o644)
        os.replace(tmp,p)
    finally:
        if tmp and tmp.exists():tmp.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    base=args.base.expanduser().resolve()
    analysis,results,stage=source_paths(base)
    gui=base/'scripts/review_metric_figures.py'
    canonical=base/'scripts/complete_papl_cohort_gui_20261009.py'
    expect(base.is_dir() and stage.is_dir() and gui.is_file(),'Canonical directories/GUI missing')
    expect(Path(__file__).resolve()==(stage/Path(__file__).name).resolve(),
           'Run this script directly from the existing canonical staging directory')
    expect(not canonical.exists(),'Canonical continuation script already exists; refuse to overwrite')
    guide,legacy=old_manifest(results)
    oldsha=digest(results/'review_bundle.json')
    original_gui,patched_gui=nav_transform(stage,gui)
    state=results/'figure_review_state.json'
    before_state=digest(state) if state.is_file() else None
    for name in ('data/pca_parameters.json','data/frozen_parameters.json'):
        expect((results/name).is_file(),'Frozen scientific data missing: '+name)
    frozen={s:digest(results/s) for s in ('data/pca_parameters.json','data/frozen_parameters.json')}
    selected,series,bins,frames,limits=load_selected(base,results)
    table_data=csv_bytes(bins)
    expect(not (results/SOURCE_MANIFEST).exists(),'Selected cohort source snapshot already exists')
    for run in RUNS:
        print(f'{run}: selected pairs={EXPECTED[run]}, B bins={frames[(run,"B")]}, M bins={frames[(run,"M")]}')
    print(f'PREFLIGHT PASS: {len(selected)} canonical pairs; {len(series)} B/M series; {len(bins)} original 5ns means')
    print(f'Observed pAPL range: {limits[0]:.3f} to {limits[1]:.3f} Å²; out-of-range means flagged')
    print('Planned registry: 164 - 2 superseded atlases + 6 cycle-specific atlases = 168 figures')
    print('Original review notes, original APLAnalysis.py and frozen PCA: no edits')
    if not args.apply:
        print('READ-ONLY PASS. Close GUI, then rerun with --apply.')
        return
    lockfd=None
    if os.name=='posix':
        import fcntl
        lockfd=os.open(str(results),os.O_RDONLY)
        try:fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as ex:
            os.close(lockfd)
            raise RuntimeError('STOP: Close the GUI before --apply; result directory locked') from ex
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    registry_backup=stage/f'review_bundle_before_complete_papl_{timestamp}.json'
    gui_backup=stage/f'review_metric_figures_before_complete_papl_{timestamp}.py'
    archive=stage/f'superseded_24_pair_atlases_{timestamp}.tar.gz'
    changed=[]
    installed_registry_hash=None
    try:
        expect(digest(results/'review_bundle.json')==oldsha,'Registry changed since preflight')
        expect(gui.read_bytes()==original_gui,'GUI changed since preflight')
        expect((digest(state) if state.is_file() else None)==before_state,'Review state changed since preflight')
        with tempfile.TemporaryDirectory(dir=stage,prefix='build_complete_papl_') as tmpdir:
            tmp=Path(tmpdir)
            pics=make_figures(selected,series,tmp)
            print('All atlases rendered; preparing canonical installation.',flush=True)
            additions=figure_records(selected,series,frames,pics,hashlib.sha256(table_data).hexdigest())
            new_guide=json.loads(json.dumps(guide))
            new_guide['figures']=[f for f in new_guide['figures'] if f['figure_id'] not in OLD_FIDS]+additions
            new_guide['figures'].sort(key=lambda f:(f['family'],f['figure_id']))
            expect(len(new_guide['figures'])==EXPECTED_NEW_FIGURES,'Registry extension length mismatch')
            targets=[]
            for record in additions:
                for ext in ('png','svg'):
                    path=results/record[ext+'_path']
                    expect(not path.exists(),f'Refusing overwrite of existing generated path: {path}')
                    targets.append((pics[(record['figure_id'].rsplit('_',1)[0].replace('papl_selected_',''),record['figure_id'].rsplit('_',1)[1])][0 if ext=='png' else 1],path))
            # Recheck all 312 external canonical source files under the install lock.
            for entry in series.values():
                expect(digest(entry['source'])==entry['hash'],
                       'Canonical APL CSV modified since data load: '+str(entry['source']))
            expect(digest(results/'review_bundle.json')==oldsha,'Registry changed during rendering; no apply')
            expect(gui.read_bytes()==original_gui,'GUI changed during rendering; no apply')
            expect((digest(state) if state.is_file() else None)==before_state,'Notes changed during rendering; no apply')
            expect(all(digest(results/name)==value for name,value in frozen.items()),'Frozen data changed during rendering; no apply')
            # Backups are created with exclusive modes; user files never overwritten.
            with registry_backup.open('xb') as out:out.write((results/'review_bundle.json').read_bytes())
            with gui_backup.open('xb') as out:out.write(original_gui)
            with tarfile.open(archive,'x:gz') as tar:
                for name,path in legacy:tar.add(path,arcname=name)
            expect(digest(registry_backup)==oldsha,'Registry backup hash mismatch')
            expect(gui_backup.read_bytes()==original_gui,'GUI backup mismatch')
            # No canonical changes until all 6 figures and all 3 backups are ready.
            atomic_write(results/SOURCE_MANIFEST,table_data);changed.append(results/SOURCE_MANIFEST)
            for source,target in targets:
                atomic_write(target,source.read_bytes());changed.append(target)
                expect(digest(source)==digest(target),'Generated figure altered during copy')
            reg_bytes=(json.dumps(new_guide,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
            installed_registry_hash=hashlib.sha256(reg_bytes).hexdigest()
            atomic_write(results/'review_bundle.json',reg_bytes)
            atomic_write(gui,patched_gui)
            check=subprocess.run([sys.executable,str(gui),'--results',str(results),'--check'],
                                 capture_output=True,text=True)
            try:checked=json.loads(check.stdout)
            except json.JSONDecodeError:checked={}
            expect(check.returncode==0 and checked.get('logical_figures')==EXPECTED_NEW_FIGURES
                   and checked.get('paths_hashes_axes')=='passed' and checked.get('state_roundtrip')=='passed',
                   'GUI validation failed after temporary install:\n'+check.stdout[-1500:]+'\n'+check.stderr[-1500:])
            expect((digest(state) if state.is_file() else None)==before_state,'Saved notes were altered')
            expect(all(digest(results/name)==val for name,val in frozen.items()),'Frozen scientific source changed')
            # Remove only original 4 atlas image files from main results: safe in archive.
            for name,path in legacy:
                expect(path.is_file() and digest(path)==next(e for e in guide['figures'] if str(name) in (e['png_path'],e['svg_path']))[Path(name).suffix[1:]+'_sha256'],
                       'Old file modified before safe archival: '+str(path))
            # Originals are retained in place as well as backed up; no cleanup in this scope.
            atomic_write(canonical,Path(__file__).read_bytes())
            expect(digest(canonical)==digest(__file__),'Canonical generator did not copy correctly')
            report=dict(status='PASS',time_utc=datetime.now(timezone.utc).isoformat(),
              selected_pairs=len(selected),matched_systems=len(series),apl_bin_rows=len(bins),
              by_run=EXPECTED,figure_count_before=EXPECTED_OLD_FIGURES,
              figure_count_after=checked['logical_figures'],figure_added=6,figure_removed=2,
              y_limits_A2=[YMIN,YMAX],x_axis='Saved combined x_ns; pure Relax only; no offset inference',
              canonical_apl_source='analysis/aplanalysis.py; analysis/output_2/**/binned_APL.csv',
              results= str(results),stage=str(stage),registry_before_sha256=oldsha,
              registry_after_sha256=digest(results/'review_bundle.json'),
              gui_before_sha256=hashlib.sha256(original_gui).hexdigest(),gui_after_sha256=digest(gui),
              source_snapshot_sha256=digest(results/SOURCE_MANIFEST),frozen_sha256=frozen,
              state_sha256=before_state,gui_check=checked,
              backups=[str(registry_backup),str(gui_backup),str(archive)],
              new_ids=[r['figure_id'] for r in additions])
            atomic_write(stage/REPORT_NAME,(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode())
            print('APPLY PASS: 6 full-cohort B/M atlases, 168 GUI figures; review state and frozen PCA unchanged')
            print('OLD 24-PAIR IMAGES SAFELY ARCHIVED:',archive)
            print('VALIDATION REPORT:',stage/REPORT_NAME)
    except Exception:
        # No automated overwrite of unexpectedly altered registry or GUI.
        if registry_backup.exists() and digest(registry_backup)==oldsha and installed_registry_hash and digest(results/'review_bundle.json')==installed_registry_hash:
            atomic_write(results/'review_bundle.json',registry_backup.read_bytes())
        if gui_backup.exists() and gui.read_bytes() in (original_gui,patched_gui):
            atomic_write(gui,original_gui)
        for path in changed:
            if path.exists():path.unlink()
        if canonical.exists() and digest(canonical)==digest(__file__):canonical.unlink()
        if archive.exists():
            with tarfile.open(archive,'r:gz') as tar:
                for member in tar.getmembers():
                    dest=results/member.name
                    if not dest.exists():
                        with tar.extractfile(member) as f:atomic_write(dest,f.read())
        raise
    finally:
        if lockfd is not None:os.close(lockfd)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
