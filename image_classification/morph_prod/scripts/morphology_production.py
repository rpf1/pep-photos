#!/usr/bin/env python3
"""Standalone morphology lifecycle. Every generated path must be new; no implicit submission."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import morphology_geometry as G

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids')
CORE = ['height_rms_nm', 'midplane_excess_area_fraction', 'H_rms_nm_inv', 'thickness_median_nm']
LEAF = ['surface_area_nm2', 'excess_area_fraction', 'height_rms_nm', 'H_rms_nm_inv']
ASYM = ['surface_area_upper_minus_lower_nm2', 'surface_area_abs_upper_minus_lower_nm2',
        'excess_area_upper_minus_lower', 'excess_area_abs_upper_minus_lower',
        'height_rms_upper_minus_lower_nm', 'height_rms_abs_upper_minus_lower_nm',
        'H_rms_upper_minus_lower_nm_inv', 'H_rms_abs_upper_minus_lower_nm_inv']
METRICS = CORE + [f'{l}_{m}' for l in ['upper', 'lower'] for m in LEAF] + ASYM
INPUTS = {'gro_path': 'Relax.gro', 'trajectory_path': 'Relax.xtc',
          'baseline_gro_path': 'Pull.gro', 'baseline_trajectory_path': 'Pull.xtc'}
KEYS = ['molecule', 'nmol', 'bilayer', 'physical_run']
PAIR = ['molecule', 'nmol', 'physical_run']
LABELS = {'curved', 'flat_interaction', 'flat_no_interaction', 'bulge_flat', 'bulge_curved', 'pore_formed'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))}


def clean_json(x):
    if isinstance(x, dict): return {str(k): clean_json(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(clean_json(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def write_csv(path, rows, columns=None):
    d = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    with Path(path).open('x', newline='') as f: d.to_csv(f, index=False)


def new_dir(path):
    path = Path(path).resolve()
    # Enforce the authorised new package as the only output owner.
    if ROOT != path and ROOT not in path.parents:
        raise ValueError(f'Output must be inside package {ROOT}: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def config(path=None):
    c = json.loads(Path(path or ROOT/'config/default.json').read_text())
    if c['grid'] != 32 or c['min_wavelength_nm'] != 4:
        raise ValueError('Scientific freeze is 4 nm / grid 32; changing it requires revalidation')
    numeric=[c[k] for k in ['fit_rmse_max_nm','thickness_valid_fraction_min','circular_resultant_min','branch_margin_min_nm','endpoint_tolerance_ns']]
    if not np.isfinite(numeric).all() or min(numeric)<=0 or c['thickness_valid_fraction_min']>1:raise ValueError('Invalid QC thresholds')
    if c['stride'] < 1 or c['endpoint_ns'] != 100 or c['start_ns'] != 0:
        raise ValueError('Invalid production sampling/window')
    return c


def truth(x): return str(x).strip().lower() in {'true', '1', '1.0'}


def integer(x):
    n = float(x)
    if not math.isfinite(n) or n <= 0 or not n.is_integer():
        raise ValueError(f'Expected positive integer, got {x!r}')
    return int(n)


def parse_system(path):
    p = Path(path)
    if not p.is_absolute() or len(p.parts) < 4: raise ValueError('Missing absolute system path')
    molecule, n = p.name.rsplit('_', 1)
    run, bilayer = p.parent.name, p.parent.parent.name
    if not re.fullmatch(r'run_[A-Za-z0-9_-]+', run): raise ValueError('Unsupported physical run path')
    if not molecule or '|' in molecule: raise ValueError('Invalid molecule identity')
    return molecule, integer(n), bilayer, run


def physical_key(molecule, nmol, bilayer, run):
    return f'{molecule}|{integer(nmol)}|{bilayer}|{run}'


def unique(d, col='physical_system_key'):
    if col not in d or d[col].isna().any() or d[col].astype(str).eq('').any() or d[col].duplicated().any():
        raise ValueError(f'Missing or duplicate {col}')


def paired(d):
    x = d[d.membrane_role.isin(['bacterial', 'mammalian'])]
    good = x.groupby(PAIR, dropna=False).membrane_role.agg(lambda v: len(v)==2 and set(v)=={'bacterial','mammalian'})
    ids = set(good[good].index)
    return x.loc[np.array([tuple(r) in ids for r in x[PAIR].itertuples(index=False, name=None)],dtype=bool)]


def filter_rows(d, a):
    if getattr(a, 'tripeptoids_only', False): d = d[(d.molecule_type=='peptoid') & (d.length.astype(float)==3)]
    for arg, col in [('bilayer','bilayer'), ('run','physical_run'), ('molecule','molecule')]:
        val = getattr(a, arg, None)
        if val: d = d[d[col].isin(val)]
    if getattr(a, 'paired_only', False): d = paired(d)
    return d.sort_values('physical_system_key').reset_index(drop=True)


def population(d):
    return {'candidate_systems':len(d), 'tripeptoid_systems':int(((d.molecule_type=='peptoid') & (d.length.astype(float)==3)).sum()),
            'paired_systems':len(paired(d)), 'paired_tripeptoid_systems':len(paired(d[(d.molecule_type=='peptoid') & (d.length.astype(float)==3)])),
            'per_bilayer':d.bilayer.value_counts().to_dict(), 'per_run':d.physical_run.value_counts().to_dict(),
            'per_type_length':{f'{t}/{l}':int(n) for (t,l),n in d.groupby(['molecule_type','length']).size().items()}}


def build_manifest(annotations, master, c):
    ann = pd.read_csv(annotations, keep_default_na=False)
    sim = pd.read_csv(master, keep_default_na=False)
    unique(ann, 'system_key')
    required = {'molecule','bilayer_type','label','annotation_status','evidence_kind','image_paths','authoritative_run','canonical_include'}
    if required-set(ann): raise ValueError(f'Annotation schema missing {sorted(required-set(ann))}')
    bykey = {}
    for idx, s in sim.iterrows():
        try:
            ident = parse_system(s['system_dir']); key = physical_key(*ident)
        except (ValueError, KeyError): continue
        bykey.setdefault(key, []).append((idx,s))
    rows, excluded = [], []
    for _, a in ann.iterrows():
        reasons = []
        if not truth(a.canonical_include): reasons.append('CANONICAL_EXCLUDED')
        if a.annotation_status != 'labelled' or a.label not in LABELS: reasons.append('NOT_SCIENTIFIC_LABEL')
        if a.evidence_kind != 'STACKED': reasons.append('NOT_STACKED')
        try:
            images = json.loads(a.image_paths)
            vals = set()
            for image in images:
                matches = re.findall(r'__nmol_(\d+)__', image)
                if len(matches)!=1: raise ValueError('image does not resolve one concentration')
                vals.add(integer(matches[0]))
            if len(vals)!=1: raise ValueError('mixed or absent concentrations')
            nmol = vals.pop()
        except (ValueError, TypeError):
            nmol = None; reasons.append('CONCENTRATION_AMBIGUOUS_OR_UNPROVEN')
        run = a.authoritative_run
        if not re.fullmatch(r'run_[A-Za-z0-9_-]+', run): reasons.append('NO_AUTHORITATIVE_PHYSICAL_RUN')
        key = physical_key(a.molecule,nmol,a.bilayer_type,run) if nmol else ''
        matches = bykey.get(key, [])
        if len(matches)!=1: reasons.append('NO_EXACT_SIMULATION' if not matches else 'DUPLICATE_PHYSICAL_SIMULATION')
        if a.bilayer_type not in c['bilayers']: reasons.append('UNSUPPORTED_BILAYER')
        if reasons:
            excluded.append({'annotation_system_key':a.system_key,'molecule':a.molecule,'bilayer':a.bilayer_type,'physical_run':run,'physical_system_key':key,'reason':';'.join(reasons)})
            continue
        idx, s = matches[0]
        try:
            length=integer(s['length']); typ=s['molecule_type']
            if typ not in {'peptoid','peptide'}: raise ValueError('unproven molecule type')
            if s.get('contains_mixture','') and truth(s['contains_mixture']): raise ValueError('mixture unsupported')
            if not truth(s.get('analysis_complete','')) or not truth(s.get('timepoint_100ns_found','')):
                raise ValueError('canonical 100 ns incomplete')
            if integer(s['nmol']) != nmol: raise ValueError('concentration mismatch')
            npo4=integer(s['n_phospholipids']); total=integer(s['n_membrane_molecules'])
            b=c['bilayers'][a.bilayer_type]
            if npo4 % 2 or total % 2: raise ValueError('odd membrane marker/total count')
            if b['po4_total'] and (npo4 != b['po4_total'] or total != b['membrane_total']): raise ValueError('unexpected membrane counts')
        except (ValueError, KeyError) as e:
            excluded.append({'annotation_system_key':a.system_key,'molecule':a.molecule,'bilayer':a.bilayer_type,'physical_run':run,'physical_system_key':key,'reason':'SIMULATION_METADATA:'+str(e)});continue
        root=Path(s['system_dir'])
        row={'manifest_row_id':hashlib.sha256(key.encode()).hexdigest()[:24], 'physical_system_key':key,
             'molecule':a.molecule,'molecule_type':typ,'length':length,'nmol':nmol,'bilayer':a.bilayer_type,
             'membrane_role':b['role'],'physical_run':run,'system_dir':str(root),'expected_endpoint_ns':100.,
             'expected_start_ns':0.,'expected_po4_total':npo4,'expected_membrane_total':total,
             'annotation_status':a.annotation_status,'manual_label':a.label,'evidence_type':a.evidence_kind,
             'evidence_sha256':a.get('evidence_sha256',''),'annotation_system_key':a.system_key,
             'annotation_event_id':a.get('event_id',''),'canonical_reason':a.get('canonical_reason',''),
             'simulation_source_row':idx+2,'simulation_source_key':s.get('system_key',''),
             'inclusion_status':'CANDIDATE_REQUIRES_LIVE_VALIDATION','exclusion_reason':''}
        row.update({k:str(root/v) for k,v in INPUTS.items()});rows.append(row)
    d=pd.DataFrame(rows)
    if d.empty: raise ValueError('No eligible candidate identities')
    unique(d);unique(d,'manifest_row_id')
    return d, pd.DataFrame(excluded)


def manifest_cmd(a):
    c=config(a.config)
    if a.manifest:
        d=pd.read_csv(a.manifest);unique(d);excluded=pd.DataFrame(columns=['reason'])
    else:
        if not a.annotations or not a.master: raise ValueError('Use --annotations and --master, or --manifest')
        d,excluded=build_manifest(a.annotations,a.master,c)
    d=filter_rows(d,a)
    if d.empty: raise ValueError('Empty selection')
    out=new_dir(a.out)
    write_csv(out/'candidates.csv',d);write_csv(out/'excluded.csv',excluded)
    inputs={str(Path(p).resolve()):sha(p) for p in [a.annotations,a.master,a.manifest] if p}
    write_json(out/'sources.json',inputs)
    write_json(out/'population.json',{**population(d),'excluded_annotations':len(excluded),
        'exclusion_reason_combinations':excluded.reason.value_counts().to_dict() if len(excluded) else {},
        'live_validated_systems':None,'status':'CANDIDATES_ONLY'})
    print(json.dumps(population(d),indent=2))


def open_universe(gro, xtc):
    import MDAnalysis as mda
    from MDAnalysis.coordinates.XTC import XTCReader
    class ReadOnlyXTCReader(XTCReader):
        def _load_offsets(self):
            self._read_offsets(store=False)
        def _read_offsets(self, store=False):
            return super()._read_offsets(store=False)
    return mda.Universe(str(gro),str(xtc),format=ReadOnlyXTCReader,
                        in_memory=False,to_guess=())


def fingerprint(row):
    return {k:{'path':str(Path(row[k]).resolve()),'size':Path(row[k]).stat().st_size,
               'mtime_ns':Path(row[k]).stat().st_mtime_ns,'sha256':sha(row[k])} for k in INPUTS}


def marker_group(u,row,c):
    b=c['bilayers'][row['bilayer']]
    p=u.select_atoms('name PO4 and resname '+' '.join(b['resnames']))
    if len(p)!=integer(row['expected_po4_total']): raise ValueError(f'PO4 count {len(p)} != expected {row["expected_po4_total"]}')
    if len(np.unique(p.resindices))!=len(p): raise ValueError('Duplicate PO4 markers per residue')
    species=b['resnames']+(['CHOL'] if row['bilayer']=='POPC60CHOL40_W_WF' else [])
    if len(u.select_atoms('resname '+' '.join(species)).residues)!=integer(row['expected_membrane_total']):
        raise ValueError('Membrane residue count does not match source metadata')
    return p


def coverage(row,c):
    info={}
    for stage,g,x in [('Pull','baseline_gro_path','baseline_trajectory_path'),('Relax','gro_path','trajectory_path')]:
        u=open_universe(row[g],row[x])
        try:
            marker_group(u,row,c)
            times=np.array([ts.time/1000 for ts in u.trajectory],float)
            if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0): raise ValueError(f'{stage}: invalid trajectory times')
            info[stage]={'frames':len(times),'first_ns':float(times[0]),'last_ns':float(times[-1])}
            if stage=='Relax':
                if abs(times[0]-c['start_ns'])>c['endpoint_tolerance_ns']: raise ValueError('Relax start missing or time origin changed')
                end=int(np.argmin(abs(times-c['endpoint_ns'])))
                if abs(times[end]-c['endpoint_ns'])>c['endpoint_tolerance_ns']: raise ValueError('Relax 100 ns endpoint absent')
                inds=sorted(set(range(0,end+1,c['stride']))|{end})
                selected=times[inds]
                info['selected_indices']=inds;info['selected_times_ns']=selected.tolist()
                for start in c['windows_ns']:
                    if ((selected>=start)&(selected<=100+c['endpoint_tolerance_ns'])).sum()<c['min_window_frames']:
                        raise ValueError(f'Insufficient sampled frames in {start}–100 ns window')
        finally: u.trajectory.close()
    return info


def validate_rows(d,c,read_trajectories=False):
    unique(d);unique(d,'manifest_row_id')
    resolved=[str(Path(x).resolve()) for x in d.system_dir]
    if len(set(resolved))!=len(resolved):raise ValueError('Duplicate resolved physical system directories')
    records=[];fps={};cov={} 
    for row in d.to_dict('records'):
        errors=[];key=row['physical_system_key']
        try:
            if physical_key(*parse_system(row['system_dir'])) != key: raise ValueError('physical path/identity mismatch')
            if Path(row['system_dir']).exists() and physical_key(*parse_system(Path(row['system_dir']).resolve()))!=key:raise ValueError('Resolved path alias does not match physical identity')
            if physical_key(*[row[k] for k in KEYS])!=key: raise ValueError('manifest identity fields disagree')
            b=c['bilayers'][row['bilayer']]
            if row['membrane_role']!=b['role']: raise ValueError('bilayer role mismatch')
            if row['annotation_status']!='labelled' or row['manual_label'] not in LABELS or row['evidence_type']!='STACKED': raise ValueError('unsupported annotation')
            for col,name in INPUTS.items():
                if Path(row[col])!=Path(row['system_dir'])/name: raise ValueError('input path outside exact system')
                if not Path(row[col]).is_file(): errors.append('MISSING:'+name)
            if not errors and read_trajectories:
                before=fingerprint(row);cov[key]=coverage(row,c);after=fingerprint(row)
                if before!=after: raise ValueError('input changed during validation')
                fps[key]=after
            elif not errors: errors.append('TRAJECTORY_NOT_CHECKED')
        except Exception as e: errors.append(type(e).__name__+':'+str(e))
        records.append({'physical_system_key':key,'validation_status':'PASS' if not errors else 'BLOCKED','errors':';'.join(errors)})
    return pd.DataFrame(records),fps,cov


def validate_cmd(a):
    d=pd.read_csv(a.manifest);c=config(a.config)
    if a.read_trajectories and not os.environ.get('SLURM_JOB_ID'):
        fixture_root=(ROOT/'tests/fixtures').resolve()
        if not all(fixture_root in Path(row[k]).resolve().parents for row in d.to_dict('records') for k in INPUTS):
            raise ValueError('Full trajectory validation runs on a compute node; omit --read-trajectories for lightweight dry-run')
    sources_file=Path(a.manifest).parent/'sources.json'
    sources=json.loads(sources_file.read_text()) if sources_file.exists() else {}
    for source,h in sources.items():
        if not Path(source).is_file() or sha(source)!=h:raise ValueError('Authoritative export changed: rebuild manifest')
    out=new_dir(a.out)
    checks,fps,cov=validate_rows(d,c,a.read_trajectories)
    write_csv(out/'validation.csv',checks)
    report={'manifest_path':str(Path(a.manifest).resolve()),'manifest_sha256':sha(a.manifest),'config':c,
            'code_sha256':source_hashes(),'input_fingerprints':fps,'coverage':cov,
            'status':'PASS' if checks.validation_status.eq('PASS').all() else 'BLOCKED',
            'systems':len(d),'passed':int(checks.validation_status.eq('PASS').sum()),'timestamp':now(),'source_sha256':sources}
    write_json(out/'validation.json',report)
    print(json.dumps({k:report[k] for k in ['status','systems','passed']},indent=2))
    return 0 if report['status']=='PASS' else 2


def frame_metrics(u,p,row,c,previous=None):
    g=G.build_surface_geometry(p.positions.copy(),u.dimensions.copy(),c['min_wavelength_nm'],c['grid'])
    m=G.geometry_metrics(g,integer(row['expected_membrane_total']))
    labels=np.zeros(len(p),dtype=bool);labels[g['qc']['upper_indices']]=True
    xyz=np.asarray(p.positions)/10;lx,ly,lz=G.validate_box(u.dimensions)
    z=G.periodic_relative(xyz[:,2]%lz,g['qc']['z_center_nm'],lz)
    mid=.5*(G.evaluate_surface(g['upper'],xyz[:,0]%lx,xyz[:,1]%ly)[0]+G.evaluate_surface(g['lower'],xyz[:,0]%lx,xyz[:,1]%ly)[0])
    disagreement=int(np.sum(labels!=(z>=mid)))
    changes=None if previous is None else int(np.sum(labels!=previous))
    flags=[]
    tests={'surface_crossing':m['surface_crossing_fraction']>0,
           'low_thickness_valid_fraction':m['thickness_valid_fraction']<c['thickness_valid_fraction_min'],
           'large_fit_residual':max(m['fit_rmse_upper_nm'],m['fit_rmse_lower_nm'])>c['fit_rmse_max_nm'],
           'ambiguous_periodic_center':m['z_center_resultant']<c['circular_resultant_min'],
           'periodic_branch_proximity':min(m['po4_z_branch_margin_nm'],m['surface_z_branch_margin_nm'])<c['branch_margin_min_nm'],
           'PO4_midplane_disagreement':disagreement>0,'PO4_assignment_change':changes is not None and changes>0,
           'missing_core_metric':not all(np.isfinite(m[k]) for k in CORE)}
    flags=[k for k,v in tests.items() if v]
    m.update(PO4_assignment_changes=changes,PO4_local_midplane_disagreement_count=disagreement,
             qc_flags=';'.join(flags),missing_core_fraction=float(np.mean([not np.isfinite(m[k]) for k in CORE])))
    return m,labels


def summarise(baseline,frames,c):
    end=frames[-1];out={};df=pd.DataFrame(frames)
    for k in METRICS:
        out['baseline__'+k]=baseline.get(k)
        out['endpoint__'+k]=end.get(k)
        a,b=baseline.get(k),end.get(k)
        out['delta__'+k]=b-a if a is not None and b is not None else None
    for start in c['windows_ns']:
        w=df[(df.time_ns>=start)&(df.time_ns<=100+c['endpoint_tolerance_ns'])]
        out[f'relax_{start}_100__frame_count']=len(w)
        for k in METRICS:
            valid=np.isfinite(pd.to_numeric(w[k],errors='coerce'))
            vals=w.loc[valid,k];ts=w.loc[valid,'time_ns'];n=len(vals)
            for tag,q in [('median',.5),('p05',.05),('p95',.95)]: out[f'relax_{start}_100__{k}__{tag}']=float(vals.quantile(q)) if n else None
            out[f'relax_{start}_100__{k}__slope_per_ns']=float(np.polyfit(ts,vals,1)[0]) if n>=c['min_window_frames'] and ts.nunique()>1 else None
            out[f'relax_{start}_100__{k}__valid_fraction']=n/len(w) if len(w) else 0.
            base=baseline.get(k)
            out[f'relax_{start}_100__{k}__fraction_above_baseline']=float((vals>base).mean()) if n and base is not None and np.isfinite(base) else None
    allrows=[baseline]+frames
    flags=sorted({s for r in allrows for s in r['qc_flags'].split(';') if s})
    qc={'qc_status':'REVIEW_REQUIRED' if flags else 'PASS','flags':flags,'analysed_frames':len(frames),
        'baseline_time_ns':baseline['time_ns'],'endpoint_time_ns':end['time_ns'],
        'review_frames':sum(bool(r['qc_flags']) for r in allrows),
        'missing_data_fraction':float(np.mean([r['missing_core_fraction'] for r in allrows])),
        'minimum_thickness_valid_fraction':min(r['thickness_valid_fraction'] for r in allrows),
        'maximum_fit_rmse_nm':max(max(r['fit_rmse_upper_nm'],r['fit_rmse_lower_nm']) for r in allrows),
        'PO4_assignment_changes_total':sum(r['PO4_assignment_changes'] or 0 for r in frames),
        'PO4_local_midplane_disagreement_max':max(r['PO4_local_midplane_disagreement_count'] for r in allrows),
        'late_window_frame_count':out['relax_90_100__frame_count'],
        'slope_window_sufficient':out['relax_30_100__frame_count']>=c['min_window_frames'],
        'height_field_limit':'equal z split cannot certify arbitrary folds/overhangs; flags require review'}
    return clean_json(out),qc


def load_plan(path):
    path=Path(path).resolve();p=json.loads((path/'plan.json').read_text())
    if sha(path/'manifest.csv')!=p['manifest_sha256']: raise ValueError('Immutable manifest changed')
    if source_hashes()!=p['code_sha256']: raise ValueError('Code changed since preparation')
    if digest(json.loads((path/'config.json').read_text()))!=p['config_digest']: raise ValueError('Configuration changed')
    d=pd.read_csv(path/'manifest.csv');unique(d);unique(d,'manifest_row_id')
    return path,p,d


def task_path(plan,row): return Path(plan)/'tasks'/str(row['manifest_row_id'])


def check_result(task,row,p,check_inputs=True):
    result=json.loads((task/'result.json').read_text());prov=json.loads((task/'provenance.json').read_text())
    completion=json.loads((task/'RUN_COMPLETE').read_text())
    for name in ['result.json','provenance.json','frames.csv','baseline.csv']:
        if sha(task/name)!=completion['hashes'].get(name): raise ValueError('Incomplete/corrupt result payload')
    if result['physical_system_key']!=row['physical_system_key'] or prov['physical_system_key']!=row['physical_system_key']: raise ValueError('Result identity mismatch')
    if prov['manifest_sha256']!=p['manifest_sha256'] or prov['code_sha256']!=p['code_sha256'] or prov['config_digest']!=p['config_digest']: raise ValueError('Result provenance mismatch')
    if prov['input_fingerprints']!=p['input_fingerprints'].get(row['physical_system_key']): raise ValueError('Result input provenance mismatch')
    if check_inputs and fingerprint(row)!=prov['input_fingerprints']: raise ValueError('INPUT_CHANGED')
    required_metrics={phase+'__'+k for phase in ['baseline','endpoint','delta'] for k in METRICS}
    if required_metrics-set(result['metrics']):raise ValueError('Required metric missing')
    required_qc={'qc_status','flags','analysed_frames','baseline_time_ns','endpoint_time_ns','minimum_thickness_valid_fraction','maximum_fit_rmse_nm','missing_data_fraction','late_window_frame_count','slope_window_sufficient','PO4_assignment_changes_total'}
    if required_qc-set(result['qc']): raise ValueError('Required QC missing')
    if result['qc']['qc_status'] not in {'PASS','REVIEW_REQUIRED'}: raise ValueError('Invalid QC status')
    for k in CORE:
        val=result['metrics']['endpoint__'+k]
        if val is None or not np.isfinite(val):
            if result['qc']['qc_status']=='PASS': raise ValueError('Invalid metric marked PASS')
    for phase in ['baseline','endpoint']:
        for m,stem,unit in [('surface_area_nm2','surface_area','_nm2'),('excess_area_fraction','excess_area',''),('height_rms_nm','height_rms','_nm'),('H_rms_nm_inv','H_rms','_nm_inv')]:
            upper=result['metrics'].get(phase+'__upper_'+m);lower=result['metrics'].get(phase+'__lower_'+m)
            if upper is None or lower is None:continue
            signed=result['metrics'].get(phase+'__'+stem+'_upper_minus_lower'+unit)
            absolute=result['metrics'].get(phase+'__'+stem+'_abs_upper_minus_lower'+unit)
            if signed is None or absolute is None or not np.isclose(signed,upper-lower,rtol=1e-10,atol=1e-12) or not np.isclose(absolute,abs(signed),rtol=1e-10,atol=1e-12):raise ValueError('Asymmetry identity failed')
    return result


def state(plan,row,p,check_inputs=True):
    out=task_path(plan,row)
    if not out.exists(): return 'NOT_STARTED'
    try:
        if check_inputs and fingerprint(row)!=p['input_fingerprints'].get(row['physical_system_key']): return 'INPUT_CHANGED'
        if (out/'RUN_COMPLETE').is_file(): check_result(out,row,p,check_inputs=False);return 'COMPLETE'
        if (out/'FAILED.json').exists(): return 'FAILED'
        if (out/'RUNNING.json').exists(): return 'RUNNING/UNKNOWN'
        return 'INCOMPLETE'
    except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError): return 'STALE_OUTPUT'


def run_system(a):
    plan,p,d=load_plan(a.plan)
    if not 0<=a.index<len(d): raise ValueError('Task index out of range')
    row=d.iloc[a.index].to_dict();c=json.loads((plan/'config.json').read_text())
    st=state(plan,row,p)
    if st=='COMPLETE': print('COMPLETE: skipped');return 0
    if st!='NOT_STARTED': raise ValueError(f'{st}: existing output preserved; generate recovery plan')
    if p['input_fingerprints'].get(row['physical_system_key'])!=fingerprint(row): raise ValueError('INPUT_CHANGED')
    print(row['physical_system_key'])
    if not a.execute: print('DRY RUN: add --execute on a compute node');return 0
    if not os.environ.get('SLURM_JOB_ID') and not a.fixture: raise ValueError('Compute node required (SLURM_JOB_ID); --fixture is synthetic-only')
    if a.fixture and not all((ROOT/'tests/fixtures').resolve() in Path(row[k]).resolve().parents for k in INPUTS): raise ValueError('--fixture accepts package synthetic fixtures only')
    out=task_path(plan,row);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();running={'start':now(),'hostname':socket.gethostname(),'pid':os.getpid(),'index':a.index,'physical_system_key':row['physical_system_key'],'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    write_json(out/'RUNNING.json',running)
    try:
        cov=coverage(row,c);frames=[]
        u=open_universe(row['baseline_gro_path'],row['baseline_trajectory_path'])
        try:
            po4=marker_group(u,row,c);u.trajectory[0];baseline,_=frame_metrics(u,po4,row,c)
            baseline.update(time_ns=float(u.trajectory.time/1000),frame=0,stage='Pull')
        finally: u.trajectory.close()
        u=open_universe(row['gro_path'],row['trajectory_path'])
        try:
            po4=marker_group(u,row,c);previous=None
            for i in cov['selected_indices']:
                ts=u.trajectory[i];m,previous=frame_metrics(u,po4,row,c,previous)
                m.update(time_ns=float(ts.time/1000),frame=int(i),stage='Relax');frames.append(m)
        finally: u.trajectory.close()
        if fingerprint(row)!=p['input_fingerprints'][row['physical_system_key']]: raise ValueError('INPUT_CHANGED during calculation')
        metrics,qc=summarise(baseline,frames,c)
        write_csv(out/'baseline.csv',[baseline]);write_csv(out/'frames.csv',frames)
        write_json(out/'result.json',{'physical_system_key':row['physical_system_key'],'metrics':metrics,'qc':qc})
        prov={**running,'end':now(),'elapsed_seconds':time.monotonic()-start,'python':sys.version,
              'dependencies':{x:importlib.metadata.version(x) for x in ['numpy','pandas','MDAnalysis']},
              'manifest_sha256':p['manifest_sha256'],'config_digest':p['config_digest'],'code_sha256':p['code_sha256'],
              'input_fingerprints':p['input_fingerprints'][row['physical_system_key']],'status':'SUCCESS',
              'fixture':a.fixture,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
        write_json(out/'provenance.json',prov)
        write_json(out/'RUN_COMPLETE',{'hashes':{name:sha(out/name) for name in ['result.json','provenance.json','frames.csv','baseline.csv']}})
    except Exception as e:
        write_json(out/'FAILED.json',{'end':now(),'error':str(e),'traceback':traceback.format_exc()});raise
    print(qc['qc_status'])


def prepare(a):
    d=pd.read_csv(a.manifest);unique(d);c=config(a.config)
    v=json.loads(Path(a.validation).read_text()) if a.validation else {}
    verified=(v.get('status')=='PASS' and v.get('manifest_sha256')==sha(a.manifest) and v.get('code_sha256')==source_hashes() and v.get('config')==c)
    if not verified and not a.draft: raise ValueError('Current full trajectory validation required; --draft creates blocked review plan')
    if verified:
        for row in d.to_dict('records'):
            if fingerprint(row)!=v['input_fingerprints'].get(row['physical_system_key']): raise ValueError('Input changed after validation')
    out=new_dir(a.out)
    manifest=Path(a.manifest).read_bytes();(out/'manifest.csv').write_bytes(manifest)
    write_json(out/'config.json',c)
    p={'manifest_sha256':sha(out/'manifest.csv'),'config_digest':digest(c),'code_sha256':source_hashes(),
       'input_fingerprints':v.get('input_fingerprints',{}),'live_validation_pass':verified,
       'production_ready':False,'created':now(),'row_count':len(d),'benchmark_only':a.benchmark,
       'blockers':['live_validation'] if not verified else [],'submission_executed':False,'source_sha256':v.get('source_sha256',{})}
    recovery_file=Path(a.manifest).parent/'recovery.json'
    if recovery_file.exists():
        recovery_meta=json.loads(recovery_file.read_text())
        p['recovery_of_manifest_sha256']=recovery_meta.get('source_manifest_sha256')
        p['confirmed_stopped_keys']=recovery_meta.get('user_confirmed_stopped_keys',[])
    write_json(out/'plan.json',p)
    (out/'tasks').mkdir();(out/'logs').mkdir()
    write_csv(out/'expected_outputs.csv',[{'index':i,'physical_system_key':r['physical_system_key'],'task_directory':f'tasks/{r["manifest_row_id"]}'} for i,r in enumerate(d.to_dict('records'))])
    s=c['slurm'];n=len(d)
    if not n: raise ValueError('Cannot prepare empty array')
    mem_directive = f"#SBATCH --mem={s['memory']}\n" if s.get('memory') else ''
    script=f'''#!/bin/bash
#SBATCH --job-name=pfp-morphology
#SBATCH --account={s['account']}
#SBATCH --partition={s['partition']}
#SBATCH --cpus-per-task={s['cpus']}
{mem_directive}#SBATCH --time={s['time']}
#SBATCH --array=0-{n-1}%{s['concurrency']}
#SBATCH --output=logs/%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
: "${{MORPH_PYTHON:?Export the absolute Python executable with validated dependencies}}"
: "${{MORPH_WORKFLOW:?Set by guarded submit command}}"
: "${{MORPH_PLAN:?Set by guarded submit command}}"
exec "$MORPH_PYTHON" "$MORPH_WORKFLOW" run-system --plan "$MORPH_PLAN" --index "$SLURM_ARRAY_TASK_ID" --execute
'''
    (out/'production_array.sbatch').write_text(script)
    submit='''#!/bin/bash
set -euo pipefail
: "${MORPH_PYTHON:?Export the absolute Python executable}"
plan="$(cd -- "$(dirname -- "$0")" && pwd)"
package="$(cd -- "$plan/../.." && pwd)"
exec "$MORPH_PYTHON" "$package/scripts/morphology_production.py" submit --plan "$plan" "$@"
'''
    (out/'submit.sh').write_text(submit)
    for name in ['manifest.csv','config.json','plan.json','production_array.sbatch']: (out/name).chmod(0o444)
    print(f'Prepared {n} rows; NOT SUBMITTED; live validation={verified}; scientific release gate required')


def submit(a):
    path,p,d=load_plan(a.plan)
    cmd=['sbatch','--chdir',str(path),str(path/'production_array.sbatch')]
    print(shlex.join(cmd))
    if not a.submit: print('DRY RUN ONLY');return 0
    if not p['live_validation_pass']: raise ValueError('Submission blocked: live validation has not passed')
    for source,h in p.get('source_sha256',{}).items():
        if not Path(source).is_file() or sha(source)!=h:raise ValueError('Authoritative export changed since preparation')
    if p['benchmark_only']:
        if not getattr(a,'benchmark_only',False) or len(d)>30:raise ValueError('Benchmark submission requires --benchmark-only and at most 30 rows')
    else:
        if getattr(a,'benchmark_only',False):raise ValueError('Not a benchmark plan')
        if not a.gate: raise ValueError('Submission blocked: supply --gate after live scientific reproduction')
        gate=json.loads(Path(a.gate).read_text())
        if gate.get('status')!='PASS' or gate.get('code_sha256')!=source_hashes() or gate.get('config_digest')!=p['config_digest']:
            raise ValueError('Submission blocked: scientific release gate absent/stale')
        for source,h in gate.get('evidence_hashes',{}).items():
            if not Path(source).is_file() or sha(source)!=h:raise ValueError('Release evidence changed')
    for row in d.to_dict('records'):
        if state(path,row,p)!='NOT_STARTED': raise ValueError('Use recovery: task already started')
        if fingerprint(row)!=p['input_fingerprints'].get(row['physical_system_key']): raise ValueError('INPUT_CHANGED')
    env=os.environ.copy();env.update(MORPH_PYTHON=sys.executable,MORPH_WORKFLOW=str(Path(__file__).resolve()),MORPH_PLAN=str(path))
    subprocess.run(cmd,check=True,env=env)


def status(a):
    path,p,d=load_plan(a.plan)
    rows=[{'physical_system_key':r['physical_system_key'],'manifest_row_id':r['manifest_row_id'],'status':state(path,r,p)} for r in d.to_dict('records')]
    if a.out:
        out=new_dir(a.out);write_csv(out/'status.csv',rows)
    print(pd.DataFrame(rows).status.value_counts().to_json())
    return rows


def recover(a):
    path,p,d=load_plan(a.plan)
    states={r['physical_system_key']:state(path,r,p) for r in d.to_dict('records')}
    stopped=set(getattr(a,'confirmed_stopped_task',None) or [])
    if stopped-set(d.manifest_row_id.astype(str)):raise ValueError('Unknown stopped task ID')
    eligible=d.physical_system_key.map(states).isin(['NOT_STARTED','FAILED','INCOMPLETE','STALE_OUTPUT','INPUT_CHANGED'])
    eligible |= d.manifest_row_id.isin(stopped)&d.physical_system_key.map(states).eq('RUNNING/UNKNOWN')
    selected=d[eligible]
    out=new_dir(a.out);write_csv(out/'recovery_candidates.csv',selected)
    write_csv(out/'deferred_running.csv',d[d.physical_system_key.map(states).eq('RUNNING/UNKNOWN')&~d.manifest_row_id.isin(stopped)])
    write_json(out/'recovery.json',{'source_plan':str(path),'rerun_candidates':len(selected),'completed_skipped':sum(x=='COMPLETE' for x in states.values()),
                                   'source_manifest_sha256':p['manifest_sha256'],'user_confirmed_stopped_tasks':sorted(stopped),'user_confirmed_stopped_keys':d[d.manifest_row_id.isin(stopped)].physical_system_key.tolist(),'rule':'new validation and new plan required; old outputs never overwritten; RUNNING/UNKNOWN requires scheduler review'})


def collect(a):
    path,p,d=load_plan(a.plan);out=new_dir(a.out);systems=[];leaves=[];qcs=[];excluded=[]
    recovery={}
    for recovery_path in getattr(a,'recovery_plan',None) or []:
        rp,meta,rd=load_plan(recovery_path)
        if set(rd.physical_system_key)-set(d.physical_system_key):raise ValueError('Recovery contains unexpected physical identity')
        if meta['config_digest']!=p['config_digest']:raise ValueError('Recovery configuration mismatch')
        for rr in rd.to_dict('records'):
            if state(rp,rr,meta)=='COMPLETE':
                key=rr['physical_system_key']
                if key in recovery:raise ValueError('Duplicate complete recovery result')
                recovery[key]=(rp,meta,rr)
    expected=set(d.manifest_row_id.astype(str))
    if {x.name for x in (path/'tasks').iterdir()}-expected: raise ValueError('Unexpected/duplicate task directories')
    for row in d.to_dict('records'):
        st=state(path,row,p);result_path,result_plan,result_row=path,p,row
        if row['physical_system_key'] in recovery:
            if st=='COMPLETE':raise ValueError('Duplicate complete primary/recovery result')
            result_path,result_plan,result_row=recovery[row['physical_system_key']]
            if st=='RUNNING/UNKNOWN' and not (result_plan.get('recovery_of_manifest_sha256')==p['manifest_sha256'] and row['physical_system_key'] in result_plan.get('confirmed_stopped_keys',[])):
                raise ValueError('Original task still unresolved; verify stopped before recovery collection')
            st='COMPLETE'
        if st!='COMPLETE': excluded.append({'physical_system_key':row['physical_system_key'],'reason':st});continue
        r=check_result(task_path(result_path,result_row),result_row,result_plan)
        systems.append({**row,**{'morph__'+k:v for k,v in r['metrics'].items()},'morph__qc_status':r['qc']['qc_status'],'morph__manifest_sha256':result_plan['manifest_sha256'],'morph__config_digest':result_plan['config_digest']})
        qcs.append({'physical_system_key':row['physical_system_key'],**r['qc']})
        for leaf in ['upper','lower']:
            leaves.append({'physical_system_key':row['physical_system_key'],'leaflet':leaf,**{f'{phase}__{m}':r['metrics'][f'{phase}__{leaf}_{m}'] for phase in ['baseline','endpoint','delta'] for m in LEAF}})
    outliers=[]
    if systems:
        sd=pd.DataFrame(systems);unique(sd)
        for role,group in sd.groupby('membrane_role'):
            for metric in CORE:
                col='morph__endpoint__'+metric;values=pd.to_numeric(group[col],errors='coerce');median=values.median();mad=(values-median).abs().median()
                for ix,value in values.items():
                    z=.67448975*(value-median)/mad if mad>0 else None
                    outliers.append({'physical_system_key':group.loc[ix,'physical_system_key'],'membrane_role':role,'metric':metric,'value':value,'median':median,'MAD':mad,'robust_z':z,'screen_flag':bool(z is not None and abs(z)>3.5),'note':'descriptive review only; no automatic deletion; undefined z when MAD=0'})
    write_csv(out/'outlier_diagnostics.csv',outliers,columns=None if outliers else ['physical_system_key','metric','screen_flag'])
    write_csv(out/'system_morphology.csv',systems,columns=None if systems else ['physical_system_key'])
    write_csv(out/'leaflet_morphology.csv',leaves,columns=None if leaves else ['physical_system_key','leaflet'])
    write_csv(out/'qc.csv',qcs,columns=None if qcs else ['physical_system_key','qc_status'])
    write_csv(out/'excluded_failed.csv',excluded,columns=['physical_system_key','reason'])
    write_json(out/'collection.json',{'expected':len(d),'complete':len(systems),'excluded':len(excluded),'manifest_sha256':p['manifest_sha256'],
                                     'status':'COMPLETE' if len(systems)==len(d) else 'PARTIAL','source_plan':str(path),'timestamp':now()})
    print(f'{len(systems)}/{len(d)} complete')


def merge(a):
    base=pd.read_csv(a.master,dtype=str,keep_default_na=False)
    morph=pd.read_csv(a.morphology,dtype=str,keep_default_na=False)
    key='physical_key' if 'physical_key' in base else 'physical_system_key'
    unique(base,key);unique(morph)
    if set(morph.physical_system_key)-set(base[key]): raise ValueError('Morphology keys absent from scientific master')
    # Keep existing pilot fields and all canonical values verbatim; production has its own namespace.
    take=['physical_system_key']+[x for x in morph if x.startswith('morph__')]
    morph=morph[take].rename(columns={x:'production_'+x for x in take if x!='physical_system_key'})
    merged=base.merge(morph,left_on=key,right_on='physical_system_key',how='left',validate='one_to_one',sort=False,suffixes=('','_join'))
    for col in base:
        if merged[col].fillna('').tolist()!=base[col].tolist(): raise ValueError(f'Canonical column altered: {col}')
    out=new_dir(a.out);write_csv(out/'master_with_morphology.csv',merged)
    write_json(out/'merge.json',{'source_master_sha256':sha(a.master),'morphology_sha256':sha(a.morphology),'rows':len(merged),'canonical_columns_preserved_exactly':True})


def audit(a):
    root=Path(a.root).resolve()
    if not root.is_dir(): raise ValueError(f'Audit root unavailable: {root}')
    files=[];dirs=[]
    for top,ds,fs in os.walk(root,followlinks=False):
        ds[:]=sorted(x for x in ds if (Path(top)/x).resolve()!=ROOT)
        dirs.extend(Path(top)/x for x in ds)
        files.extend(Path(top)/x for x in sorted(fs))
    # Freeze inventory before creating output, preventing recursive handoff copying.
    inventory=[];texts={}
    for p in files:
        if not p.is_symlink() and p.suffix in {'.py','.sh','.sbatch','.json','.md','.tcl','.toml','.yaml','.yml'} and p.stat().st_size<2_000_000:
            texts[str(p)]=p.read_text(errors='replace')
    for p in dirs+files:
        typ='SYMLINK' if p.is_symlink() else ('DIRECTORY' if p.is_dir() else 'FILE')
        size=p.lstat().st_size;h=sha(p) if typ=='FILE' else ''
        refs=[q for q,t in texts.items() if q!=str(p) and (str(p.relative_to(root)) in t or p.name in t)]
        classification='UNKNOWN';reason='live role/consumer verification required'
        if p.name in {'annotations.csv','state.sqlite3','state.db'}: classification='AUTHORITATIVE_STATE';reason='annotation/state filename; preserve'
        elif p.name in {'annotate.py','workflow.py','morphology_production.py'}: classification='ACTIVE_ENTRYPOINT';reason='entrypoint filename; inspect before relocation'
        elif p.name.startswith(('check_legacy_','audit_annotation_')): classification='ONE_OFF_AUDIT';reason='historical audit name; full external caller check required'
        elif p.suffix in {'.json','.toml','.yaml','.yml'}: classification='CONFIG';reason='structured configuration or state; content review required'
        action='KEEP_ACTIVE' if classification in {'AUTHORITATIVE_STATE','ACTIVE_ENTRYPOINT','CONFIG'} else 'UNKNOWN_REVIEW'
        inventory.append({'path':str(p),'relative_path':str(p.relative_to(root)),'kind':typ,'bytes':size,'sha256':h,'classification':classification,
                          'action':action,'target_path':'','evidence':reason,'references_in_scanned_tree':json.dumps(refs),'external_callers_verified':False,'regenerable':'UNPROVEN'})
    df=pd.DataFrame(inventory)
    df['duplicate_content_count']=df.sha256.map(df[df.sha256!=''].sha256.value_counts()).fillna(0).astype(int)
    out=new_dir(a.out);write_csv(out/'filesystem_inventory.csv',df);write_csv(out/'cleanup_plan.csv',df[['path','classification','action','target_path','evidence','references_in_scanned_tree','external_callers_verified','regenerable']])
    git={}
    probes={root,Path(a.project_root).resolve()}
    probes.update(p.parent for p in files+dirs if p.name=='.git')
    for probe in sorted(probes):
        record={}
        for name,args in [('root',['rev-parse','--show-toplevel']),('branch',['branch','--show-current']),('HEAD',['rev-parse','HEAD']),('status',['status','--porcelain=v1']),('submodules',['submodule','status'])]:
            r=subprocess.run(['git','-C',str(probe),*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
            record[name]=r.stdout.strip() if r.returncode==0 else 'UNAVAILABLE: '+r.stderr.strip()
        git[str(probe)]=record
    write_json(out/'orientation.json',{'scope':str(root),'inventory_entries':len(df),'git':git,'live_project':Path(a.project_root).exists(),
        'cleanup_applied':False,'limitations':'filename classifications provisional; only scanned-tree text references, no external caller certification'})
    (out/'cleanup_summary.md').write_text('No move/delete is approved or applied. UNKNOWN_REVIEW rows remain in place.\n'
        'Hash-identical files are duplication evidence, not deletion permission. Full external caller verification is required before assigning move targets.\n'
        'Any later removal uses 17082026_cleanup/_cleanup_review/ and deletion_audit.csv after exact-manifest approval.\n')


def verify_statistics(a):
    """Compare independently regenerated numerical values against archived tables."""
    from morphology_statistics import bh
    reference=Path(a.reference);candidate=Path(a.candidate);checks=[]
    def add(table,key,col,actual,expected):
        ok=(pd.isna(actual) and pd.isna(expected)) or (pd.notna(actual) and pd.notna(expected) and np.isclose(float(actual),float(expected),rtol=1e-9,atol=1e-12))
        checks.append({'table':table,'identity':str(key),'field':col,'actual':actual,'expected':expected,'pass':bool(ok)})
    old=pd.read_csv(reference/'04_label_concordance_statistics.csv');new=pd.read_csv(candidate/'concordance.csv')
    for col in ['n_pairs','exact_label_agreement_fraction','chi2','chi2_dof','chi2_p','cramers_v']:
        add('concordance','all',col,new.iloc[0][col],old.iloc[0][col])
    old=pd.read_csv(reference/'06_class_metric_omnibus_tests.csv');new=pd.read_csv(candidate/'class_omnibus.csv')
    reconstructed=[]
    for _,r in old.iterrows():
        hit=new[(new.membrane==r.membrane)&(new.metric==r.metric)]
        if len(hit)!=1:raise ValueError('Missing omnibus comparator')
        for col in ['n_total','H','p_raw','epsilon_squared']:add('class_omnibus',(r.membrane,r.metric),col,hit.iloc[0][col],r[col])
        reconstructed.append(hit.iloc[0].p_raw)
    for i,q in enumerate(bh(reconstructed)):add('class_omnibus',i,'legacy_BH_FDR',q,old.iloc[i]['p_BH_FDR_across_omnibus_tests'])
    for filename,newname in [('10_target_correlations.csv','pair_correlations.csv'),('14_morphology_core_correlations.csv','system_correlations.csv'),('15_morphology_vs_APL_thickness.csv','system_correlations.csv')]:
        old=pd.read_csv(reference/filename);new=pd.read_csv(candidate/newname);ps=[]
        for _,r in old.iterrows():
            subset=new if filename.startswith('10') else new[new.dataset==r.dataset.split('_')[0]]
            hit=subset[((subset.metric_x==r.metric_x)&(subset.metric_y==r.metric_y))|((subset.metric_y==r.metric_x)&(subset.metric_x==r.metric_y))]
            if len(hit)!=1:raise ValueError(f'Missing/ambiguous correlation {r.metric_x} / {r.metric_y}')
            for col in ['n','spearman_rho','p_raw']:add(filename,(r.metric_x,r.metric_y),col,hit.iloc[0][col],r[col])
            ps.append(hit.iloc[0].p_raw)
        # Legacy tables correct within dataset, not across differently named families.
        qvals=pd.Series(index=old.index,dtype=float)
        for dataset,idx in old.groupby('dataset').groups.items():qvals.loc[idx]=bh(np.asarray(ps)[list(idx)])
        for i in old.index:add(filename,i,'legacy_BH_FDR',qvals[i],old.loc[i,'p_BH_FDR'])
    for filename in ['07_class_metric_pairwise_tests.csv','16_morphology_class_comparisons.csv']:
        old=pd.read_csv(reference/filename);new=pd.read_csv(candidate/'class_pairwise.csv')
        for _,r in old.iterrows():
            hit=new[(new.membrane==r.membrane)&(new.metric==r.metric)&(new.class_a==r.class_a)&(new.class_b==r.class_b)]
            reversed_=False
            if hit.empty:
                hit=new[(new.membrane==r.membrane)&(new.metric==r.metric)&(new.class_b==r.class_a)&(new.class_a==r.class_b)];reversed_=True
            if len(hit)!=1:raise ValueError('Missing pairwise comparator')
            v=hit.iloc[0]
            add(filename,(r.membrane,r.metric,r.class_a,r.class_b),'p_raw',v.p_raw,r.p_raw)
            add(filename,(r.membrane,r.metric,r.class_a,r.class_b),'U',v.n_a*v.n_b-v.U if reversed_ else v.U,r.U)
    old=pd.read_csv(reference/'09_pair_label_ddapl_omnibus.csv');new=pd.read_csv(candidate/'label_pair_ddapl_tests.csv')
    add('label_pair_ddapl','all','H',new.iloc[0].H,old.iloc[0].H)
    add('label_pair_ddapl','all','p',new.iloc[0].p_raw,old.iloc[0]['p'])
    out=new_dir(a.out);write_csv(out/'comparisons.csv',checks)
    result={'status':'PASS' if all(r['pass'] for r in checks) else 'FAIL','checks':len(checks),
            'failed':sum(not r['pass'] for r in checks),'code_sha256':source_hashes(),
            'reference_hashes':{f.name:sha(f) for f in reference.glob('*.csv')},
            'candidate_hashes':{f.name:sha(f) for f in candidate.glob('*.csv')},
            'scope':'concordance, canonical class tests, pair-label ddAPL, target correlations, core morphology and APL/thickness correlations; historical FDR families reconstructed',
            'limits':'legacy plotting/report/local-lipid analyses are outside production scope'}
    write_json(out/'statistics_verification.json',result)
    print(json.dumps({k:result[k] for k in ['status','checks','failed']}))
    return 0 if result['status']=='PASS' else 2


def verify_pilot(a):
    """Compare real benchmark results to pilot baseline/endpoint/window summaries."""
    pilot=pd.read_csv(a.reference);candidate=pd.read_csv(a.candidate);unique(candidate)
    reference_manifest=pd.read_csv(a.reference_manifest)
    if reference_manifest.duplicated(['id','membrane']).any():raise ValueError('Ambiguous pilot physical manifest')
    checks=[]
    for _,r in pilot.iterrows():
        role=r['membrane'];molecule=r['id']
        hit=candidate[(candidate.molecule==molecule)&(candidate.membrane_role==role)]
        if len(hit)!=1:raise ValueError(f'Pilot identity absent/ambiguous: {molecule}/{role}')
        row=hit.iloc[0]
        identity=reference_manifest[(reference_manifest.id==molecule)&(reference_manifest.membrane==role)]
        if len(identity)!=1 or physical_key(*parse_system(identity.iloc[0].system_dir))!=row.physical_system_key:
            raise ValueError('Pilot physical run/concentration mismatch')
        for k in r.index:
            prefixes=('baseline__','endpoint__','delta__') if a.leaflets else ('baseline__','endpoint__','delta__','relax_')
            scientific=(k.startswith(prefixes) and k in {c.removeprefix('morph__') for c in candidate})
            if not scientific or not isinstance(r[k],(float,int,np.number)):continue
            value=row['morph__'+k];ref=r[k]
            ok=(pd.isna(value) and pd.isna(ref)) or np.isclose(value,ref,rtol=1e-8,atol=1e-6)
            checks.append({'molecule':molecule,'membrane':role,'metric':k,'actual':value,'reference':ref,'pass':bool(ok)})
        required=[phase+'__'+k for phase in ['baseline','endpoint','delta'] for k in CORE]
        if a.leaflets:required += [phase+'__'+leaf+'_'+k for phase in ['baseline','endpoint','delta'] for leaf in ['upper','lower'] for k in LEAF]+['endpoint__'+k for k in ASYM]
        checked={x['metric'] for x in checks if x['molecule']==molecule and x['membrane']==role}
        if set(required)-checked:raise ValueError('Pilot comparison lacks required metric columns')
    out=new_dir(a.out);write_csv(out/'comparisons.csv',checks)
    r={'status':'PASS' if len(checks) and all(x['pass'] for x in checks) else 'FAIL',
       'systems':len(pilot),'checks':len(checks),'failed':sum(not x['pass'] for x in checks),
       'leaflets':a.leaflets,'reference_sha256':sha(a.reference),'reference_manifest_sha256':sha(a.reference_manifest),'candidate_sha256':sha(a.candidate),
       'code_sha256':source_hashes(),'tolerance':{'rtol':1e-8,'atol':1e-6},
       'rationale':('leaflet smoke comparison is restricted to baseline/endpoint/delta because the historical smoke run used non-equivalent sparse temporal sampling' if a.leaflets else 'same double-precision kernel; absolute allowance covers coordinate rounding/serialisation, not method changes')}
    write_json(out/'pilot_verification.json',r);print(json.dumps(r))
    return 0 if r['status']=='PASS' else 2


def release_check(a):
    import ast
    receipts=[json.loads(Path(p).read_text()) for p in [a.pilot_check,a.leaflet_check,a.statistics_check]]
    if any(x.get('status')!='PASS' or x.get('code_sha256')!=source_hashes() for x in receipts):raise ValueError('Scientific verification missing/failed/stale')
    if receipts[0].get('systems',0)<30 or receipts[1].get('systems',0)<4 or not receipts[1].get('leaflets'):raise ValueError('Full existing 30-system core / 4-system leaflet comparisons required')
    audit_report=json.loads(Path(a.audit).read_text())
    if not audit_report.get('live_project') or audit_report.get('scope')!=str(PROJECT/'analysis/image_classification'):raise ValueError('Live project inventory required')
    if not (PROJECT/'SCRIPT_INDEX.md').is_file():raise ValueError('Live SCRIPT_INDEX missing')
    # The production geometry kernel was extracted from the reviewed leaflet-v2
    # snapshot recorded in qc/geometry_source.json.  The current live
    # 06_morphology_v1.py is the historical core-pilot implementation and is
    # not expected to be text-identical to that leaflet-v2 extraction.
    #
    # Scientific equivalence is therefore BLOCKED on the strict real-data
    # 30-system core and 4-system leaflet numerical reproductions above.
    # We still compare the live source here and record the result as provenance.
    live_source=Path(a.live_pilot_source)
    if not live_source.is_file():
        raise ValueError('Live pilot source missing')

    geometry_record_path=ROOT/'qc/geometry_source.json'
    if not geometry_record_path.is_file():
        raise ValueError('Frozen geometry source record missing')

    geometry_record=json.loads(geometry_record_path.read_text())
    if geometry_record.get('extraction')!='functions weighted_percentile through geometry_metrics, verbatim':
        raise ValueError('Unexpected frozen geometry extraction record')

    live_ast=ast.parse(live_source.read_text())
    frozen_ast=ast.parse((ROOT/'scripts/morphology_geometry.py').read_text())

    live={x.name:ast.dump(x,include_attributes=False)
          for x in live_ast.body if isinstance(x,ast.FunctionDef)}
    frozen={x.name:ast.dump(x,include_attributes=False)
            for x in frozen_ast.body if isinstance(x,ast.FunctionDef)}

    live_geometry_match=not any(live.get(k)!=v for k,v in frozen.items())
    plan,p,d=load_plan(a.benchmark_plan)
    if not p['live_validation_pass']:raise ValueError('Benchmark lacks live validation')
    provenance=[]
    for row in d.to_dict('records'):
        check_result(task_path(plan,row),row,p)
        r=json.loads((task_path(plan,row)/'provenance.json').read_text())
        if r.get('fixture') or not r.get('slurm_job_id'):raise ValueError('Real compute-node benchmark required')
        provenance.append(r)
    if len(provenance)<4:raise ValueError('Benchmark at least four systems including both primary membranes')
    if not {'bacterial','mammalian'}.issubset(set(d.membrane_role)):raise ValueError('Benchmark must cover both membranes')
    out=new_dir(a.out)
    write_json(out/'production_gate.json',{'status':'PASS','code_sha256':source_hashes(),'config_digest':p['config_digest'],
        'live_pilot_sha256':sha(live_source),
        'geometry_extraction_source_sha256':geometry_record.get('source_sha256'),
        'live_geometry_functions_match_frozen':live_geometry_match,
        'geometry_equivalence_basis':'strict real-data 30-system core and 4-system leaflet numerical reproduction; live core source comparison retained as provenance, not textual-identity gate','evidence_hashes':{str(Path(x).resolve()):sha(x) for x in [a.pilot_check,a.leaflet_check,a.statistics_check,a.audit]},
        'benchmark_max_seconds':max(x['elapsed_seconds'] for x in provenance),'timestamp':now(),
        'note':'Scientific gate passes; user approval still required for submission; resource peak memory must be reviewed from scheduler accounting'})


def package_cmd(a):
    import zipfile
    dest=Path(a.output).resolve()
    if ROOT in dest.parents: raise ValueError('Archive must be outside package to prevent recursion')
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for p in paths: z.write(p,Path(ROOT.name)/p.relative_to(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('manifest');q.set_defaults(func=manifest_cmd)
    for name in ['annotations','master','manifest','config']: q.add_argument('--'+name)
    q.add_argument('--out',required=True)
    q.add_argument('--tripeptoids-only',action='store_true');q.add_argument('--paired-only',action='store_true')
    for k in ['bilayer','run','molecule']: q.add_argument('--'+k,action='append')
    q=sub.add_parser('validate');q.set_defaults(func=validate_cmd)
    q.add_argument('--manifest',required=True);q.add_argument('--config');q.add_argument('--out',required=True);q.add_argument('--read-trajectories',action='store_true')
    q=sub.add_parser('prepare');q.set_defaults(func=prepare)
    for k in ['manifest','out']:q.add_argument('--'+k,required=True)
    for k in ['validation','config']:q.add_argument('--'+k)
    q.add_argument('--draft',action='store_true');q.add_argument('--benchmark',action='store_true')
    q=sub.add_parser('run-system');q.set_defaults(func=run_system);q.add_argument('--plan',required=True);q.add_argument('--index',type=int,required=True)
    q.add_argument('--execute',action='store_true');q.add_argument('--fixture',action='store_true')
    q=sub.add_parser('submit');q.set_defaults(func=submit);q.add_argument('--plan',required=True);q.add_argument('--submit',action='store_true');q.add_argument('--gate');q.add_argument('--benchmark-only',action='store_true')
    for cmd,fn in [('status',status),('recover',recover),('collect',collect)]:
        q=sub.add_parser(cmd);q.set_defaults(func=fn);q.add_argument('--plan',required=True);q.add_argument('--out',required=cmd!='status')
        if cmd=='collect':q.add_argument('--recovery-plan',action='append')
        if cmd=='recover':q.add_argument('--confirmed-stopped-task',action='append',help='Explicit task ID verified stopped using scheduler accounting')
    q=sub.add_parser('merge');q.set_defaults(func=merge)
    for k in ['master','morphology','out']:q.add_argument('--'+k,required=True)
    q=sub.add_parser('stats');q.set_defaults(func=stats_cmd);q.add_argument('--master',required=True);q.add_argument('--out',required=True);q.add_argument('--pilot',action='store_true')
    q=sub.add_parser('audit');q.set_defaults(func=audit);q.add_argument('--root',required=True);q.add_argument('--project-root',default=str(PROJECT));q.add_argument('--out',required=True)
    q=sub.add_parser('verify-statistics');q.set_defaults(func=verify_statistics)
    for k in ['reference','candidate','out']:q.add_argument('--'+k,required=True)
    q=sub.add_parser('verify-pilot');q.set_defaults(func=verify_pilot)
    for k in ['reference','candidate','out']:q.add_argument('--'+k,required=True)
    q.add_argument('--leaflets',action='store_true');q.add_argument('--reference-manifest',required=True)
    q=sub.add_parser('release-check');q.set_defaults(func=release_check)
    for k in ['pilot-check','leaflet-check','statistics-check','audit','live-pilot-source','benchmark-plan','out']:q.add_argument('--'+k,required=True)
    q=sub.add_parser('package');q.set_defaults(func=package_cmd);q.add_argument('--output',required=True)
    return p


def stats_cmd(a):
    from morphology_statistics import run
    run(a, new_dir, write_csv, write_json, sha)


if __name__=='__main__':
    try:
        args=parser().parse_args();ret=args.func(args)
        sys.exit(ret if isinstance(ret,int) else 0)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);sys.exit(2)
