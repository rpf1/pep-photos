#!/usr/bin/env python3
"""Create a verified annotation continuation; never replace the original session."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids')
DEFAULT_STAGE = '20260928_145342_047966'
EXPORTS = ('annotations.csv', 'annotation_order.csv', 'system_images.csv')
EVIDENCE = ('source_directory', 'relative_path', 'image_count', 'image_paths', 'evidence_sha256', 'members')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_tool(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contained(path, root):
    path = Path(path)
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise RuntimeError(f'Path escapes expected tree or is a symlink: {path}')
    return path


def audit_unresolved(root, records, old_metadata):
    index = []
    boxes = root / 'Boxes/2.1'
    # Fixed depth only; no recursive traversal of historical bundle trees.
    for b in sorted(boxes.iterdir()):
        if not b.is_dir() or b.is_symlink(): continue
        for run in sorted(b.iterdir()):
            if not run.is_dir() or run.is_symlink(): continue
            for s in sorted(run.iterdir()):
                if not s.is_dir() or s.is_symlink(): continue
                mol, sep, n = s.name.rpartition('_')
                if sep and n.isdigit():
                    index.append((mol, b.name, run.name, int(n), s))
    output = []
    for r in records:
        if r['status'] == 'rendered': continue
        old = old_metadata.get(r['system_key'], {})
        candidates = []
        evidence_names = [m['relative_path'] for m in old.get('members', [])]
        topology = []
        for mol, b, run, n, s in index:
            if mol != r['molecule'] or b != r['bilayer']: continue
            candidates.append(dict(path=str(s), run=run, nmol=n,
                exact_run=run == r['run'], relax_gro=(s/'Relax.gro').is_file(),
                relax_xtc=(s/'Relax.xtc').is_file(),
                count_seen_in_old_filename=any(f'__nmol_{n}__' in x for x in evidence_names)))
            if run == r['run'] and 'unsupported GRO residues' in r.get('reason', ''):
                for f in sorted(s.iterdir()):
                    if f.is_file() and not f.is_symlink() and f.suffix in ('.itp', '.top') and f.stat().st_size < 2_000_000:
                        lines = f.read_text(errors='replace').splitlines()
                        hits = [(i+1, line) for i,line in enumerate(lines) if {'HIS','GLY'} & set(line.split())]
                        if hits: topology.append(dict(path=str(f), matches=hits[:50]))
        output.append(dict(**r, old_images=evidence_names, candidate_directories=candidates,
                           topology_evidence=topology,
                           action='Review only: no substitutions or rendering performed'))
    return output


def prepare(root, stage):
    from PIL import Image
    app = root/'analysis/image_classification/pfp34_v2'
    report = json.loads((stage/'report.json').read_text())
    records = report['records']
    by_key = {r['system_key']: r for r in records}
    if len(by_key) != len(records): raise RuntimeError('Duplicate report system keys')
    tool = load_tool(app/'scripts/annotate.py', 'original_annotation_tool')
    config = tool.load_config(app/'annotation_config.json')
    session = Path(config['output_dir'])
    photos = Path(config['image_root'])
    lock = tool.SessionLock(session/'.annotation.lock')
    lock.acquire()  # Refuse to fork a session while its GUI is writing.
    continuation = root/'analysis/image_classification/pfp34_new_photos'
    store = None
    try:
        if continuation.exists():
            raise RuntimeError(f'{continuation} already exists. Use its START_ANNOTATING.sh if READY.json exists; otherwise preserve it for diagnosis.')
        source_db = session/'state.sqlite3'
        source_hashes = {str(p): sha(p) for p in [source_db, *(session/n for n in EXPORTS)]}
        connection = sqlite3.connect(source_db.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Original database integrity check failed')
            systems = list(connection.execute('SELECT system_key,position,relative_path,metadata FROM systems ORDER BY position'))
            before_records = list(connection.execute('SELECT system_key,target,record FROM annotations ORDER BY system_key,target'))
            before_events = list(connection.execute('SELECT * FROM events ORDER BY event_id'))
            boundary = connection.execute('SELECT COALESCE(MAX(event_id),0) FROM events').fetchone()[0]
            if not set(by_key).issubset({s[0] for s in systems}):
                raise RuntimeError('Regeneration report contains systems absent from current annotation session')
            text = (app/'scripts/annotate.py').read_text()
            marker = "WHERE e.kind='decision'"
            if text.count(marker) != 1:
                raise RuntimeError('Unexpected undo implementation; refusing unverified patch')
            # Preserve historical events; prevent undo from crossing evidence transition.
            text = text.replace(marker, marker + f' AND e.event_id > {boundary}', 1)
            compile(text, 'annotate.py', 'exec')
            continuation.mkdir()
            (continuation/'scripts').mkdir()
            (continuation/'session').mkdir()
            (continuation/'images').mkdir()
            (continuation/'original_session_snapshot').mkdir()
            destination = sqlite3.connect(continuation/'original_session_snapshot/state.sqlite3')
            try: connection.backup(destination)
            finally: destination.close()
        finally: connection.close()
        snapshot = continuation/'original_session_snapshot'
        for name in EXPORTS: shutil.copy2(session/name, snapshot/name)
        shutil.copy2(app/'annotation_config.json', snapshot/'annotation_config.json')
        shutil.copy2(app/'scripts/annotate.py', snapshot/'annotate.py')
        shutil.copy2(snapshot/'state.sqlite3', continuation/'session/state.sqlite3')
        for name in EXPORTS: shutil.copy2(snapshot/name, continuation/'session'/name)
        (continuation/'scripts/annotate.py').write_text(text)
        new_config = dict(config, image_root=str(continuation/'images'), output_dir=str(continuation/'session'))
        write_json(continuation/'annotation_config.json', new_config)
        copied_tool = load_tool(continuation/'scripts/annotate.py', 'new_annotation_tool')
        store = copied_tool.Store(new_config)  # Verifies original CSV exports against cloned DB.
        updates, checklist, transitions = {}, [], []
        quarantined = set()
        old_metadata = {key: json.loads(meta) for key,pos,rel,meta in systems}
        for index, (key, position, relative, meta_text) in enumerate(systems, 1):
            old = json.loads(meta_text)
            source_dir = contained(photos/relative, photos)
            try:
                current = tool.inspect_system(source_dir, photos, config)
                evidence_problem = current.get('problem', '') or (
                    'Original image hashes changed'
                    if current['evidence_sha256'] != old['evidence_sha256']
                    else ''
                )
            except OSError as exc:
                evidence_problem = str(exc)

            r = by_key.get(key)
            if evidence_problem and not (r and r['status'] == 'rendered'):
                blocked_dir = contained(
                    continuation/'images'/relative, continuation/'images'
                )
                blocked_dir.mkdir(parents=True, exist_ok=False)
                updates[key] = old
                quarantined.add(key)
                checklist.append(dict(
                    system_key=key,
                    molecule=old['molecule'],
                    bilayer=old['bilayer_type'],
                    run=old['run'],
                    image_version='BLOCKED',
                    reason=evidence_problem,
                    review_existing_label=False,
                ))
                transitions.append(dict(
                    system_key=key, old=old, new=old,
                    image_version='BLOCKED', reason=evidence_problem,
                ))
                print(
                    f'BLOCKED evidence only; labels retained: {relative}',
                    flush=True,
                )
                continue
            dest = contained(continuation/'images'/relative, continuation/'images')
            dest.mkdir(parents=True, exist_ok=False)
            r = by_key.get(key)
            kind = 'LEGACY'
            if r and r['status'] == 'rendered':
                out = contained(Path(r['output']), stage)
                validation = json.loads((out/'validation.json').read_text())
                if len(validation) != 1 or validation[0].get('ten_panels_verified') is not True:
                    raise RuntimeError(f'Missing ten-panel validation: {out}')
                pattern = f"{r['molecule']}__nmol_*__{r['bilayer']}__{r['run']}*.png"
                matches = [p for p in out.rglob('*.png') if p.match(pattern) and p.parent.name == r['run'] and p.parent.parent.name == r['bilayer'] and p.parent.parent.parent.name == r['molecule']]
                if len(matches) != 1: raise RuntimeError(f'Expected one system composite: {out}; found {len(matches)}')
                src = contained(matches[0], stage)
                with Image.open(src) as image:
                    image.load()
                    if image.width != 3300: raise RuntimeError(f'Unexpected ten-panel composite width: {src}')
                target = dest/('NEW10__'+src.name)
                shutil.copy2(src, target)
                if sha(src) != sha(target): raise RuntimeError('Image copy hash mismatch')
                kind = 'NEW10'
            else:
                for member in old['members']:
                    src = contained(photos/member['relative_path'], photos)
                    target = dest/('LEGACY__'+src.name)
                    shutil.copy2(src, target)
                    if sha(target) != member['sha256']: raise RuntimeError('Legacy image hash mismatch')
            fresh = copied_tool.inspect_system(dest, continuation/'images', new_config)
            if fresh['discovery_status'] != 'ready' or fresh['system_key'] != key:
                raise RuntimeError(f'New evidence inspection failed: {relative}')
            fresh['molecule_type'] = old.get('molecule_type','')
            updates[key] = fresh
            checklist.append(dict(system_key=key, molecule=old['molecule'], bilayer=old['bilayer_type'],
                run=old['run'], image_version=kind, reason=(r or {}).get('reason','Not in regeneration report' if not r else ''),
                review_existing_label=kind == 'NEW10' and any(k == key and json.loads(raw).get('annotation_status') == 'labelled' for k,t,raw in before_records)))
            transitions.append(dict(system_key=key, old=old, new=fresh, image_version=kind))
            if index % 25 == 0: print(f'Prepared {index}/{len(systems)} systems', flush=True)
        fields = set(EVIDENCE)
        with store.conn:
            for key,fresh in updates.items():
                store.conn.execute('UPDATE systems SET metadata=? WHERE system_key=?', (json.dumps(fresh),key))
            for key,target,raw in before_records:
                row = json.loads(raw)
                after = dict(row)
                after.update({f: updates[key][f] for f in EVIDENCE if f in updates[key]})
                if {k:v for k,v in row.items() if k not in fields} != {k:v for k,v in after.items() if k not in fields}:
                    raise RuntimeError('Annotation preservation check failed')
                store.conn.execute('UPDATE annotations SET record=? WHERE system_key=? AND target=?', (json.dumps(after),key,target))
            store.set_meta('evidence_transition', dict(timestamp=datetime.now(timezone.utc).isoformat(),
                original_session=str(session), undo_boundary=boundary, audit_file=str(continuation/'evidence_transition.json')))
        store.export()  # Prior export hashes still verify the untouched pre-migration CSVs.
        if [tuple(x) for x in store.conn.execute('SELECT * FROM events ORDER BY event_id')] != before_events:
            raise RuntimeError('Historical events changed')
        for unit in store.units():
            problem = store.source_problem(unit)
            if bool(problem) != (unit[0] in quarantined): raise RuntimeError(problem or 'Expected blocked evidence')
        write_json(continuation/'evidence_transition.json', transitions)
        write_json(continuation/'original_source_hashes.json', source_hashes)
        with (continuation/'image_checklist.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(checklist[0]));w.writeheader();w.writerows(checklist)
        summary = dict(systems=len(systems), versions=dict(Counter(x['image_version'] for x in checklist)),
                       annotation_status_counts=dict(store.counts()), old_session=str(session),
                       new_session=str(continuation/'session'), historical_events_preserved=len(before_events))
        store.backup('new_images_verified')
        store.close();store=None
        # Prove the continuation reopens with its normal export/evidence guards.
        store=copied_tool.Store(new_config)
        for unit in store.units():
            if bool(store.source_problem(unit)) != (unit[0] in quarantined): raise RuntimeError('Reopen verification failed')
        store.close();store=None
        for path,digest in source_hashes.items():
            if sha(path) != digest: raise RuntimeError(f'Original source changed: {path}')
        launcher = continuation/'START_ANNOTATING.sh'
        launcher.write_text('#!/usr/bin/env bash\nset -e\nexec '+shlex.quote(sys_executable())+' '+
            shlex.quote(str(continuation/'scripts/annotate.py'))+' --config '+shlex.quote(str(continuation/'annotation_config.json'))+'\n')
        write_json(continuation/'READY.json',summary)
        print('\nREADY — use this launcher from now on:\nbash '+shlex.quote(str(launcher)),flush=True)
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        if store: store.close()
        lock.release()
    # Read-only investigations occur after a ready annotation session exists.
    try:
        audit=audit_unresolved(root, records, old_metadata)
        write_json(continuation/'unresolved_audit.json',audit)
        print('Unresolved audit:',continuation/'unresolved_audit.json',flush=True)
    except Exception as exc:
        write_json(continuation/'audit_error.json',dict(error=str(exc)))
        print('Annotation is ready; optional audit failed:',exc,flush=True)
    plan=[]
    archive_zip=root/'analysis/PFPeptoids_photograph_patch_v1.zip'
    if archive_zip.is_file() and not archive_zip.is_symlink():
        plan.append(dict(source=str(archive_zip), kind='file', sha256=sha(archive_zip), reason='Downloaded installer'))
    pilots=root/'analysis/photograph_validation'
    if pilots.is_dir():
        for folder in sorted(pilots.glob('pilot_*')):
            if not folder.is_dir() or folder.is_symlink() or not (folder/'validation.json').is_file(): continue
            validation=json.loads((folder/'validation.json').read_text())
            if not isinstance(validation,list) or not validation or not all(v.get('ten_panels_verified') or v.get('nine_panels_verified') for v in validation): continue
            files={}
            safe=True
            for f in folder.rglob('*'):
                if f.is_symlink(): safe=False;break
                if not f.is_file(): continue
                known=(f.name in {'validation.json','new_systems_created.txt','Photograph.tcl','Photograph_generated.tcl','photograph_residues.tcl'} or
                       (f.suffix=='.png' and (f.name.startswith('snapshot_') or '__nmol_' in f.name or '__ALL_BILAYERS__' in f.name)))
                if not known: safe=False;break
                files[f.relative_to(folder).as_posix()]=sha(f)
            if safe:
                plan.append(dict(source=str(folder),kind='directory',files=files,reason='Completed pilot evidence; retain intact in cleanup archive'))
    write_json(continuation/'cleanup_plan.json',plan)
    return continuation


def sys_executable():
    import sys
    return sys.executable


def cleanup(root):
    app=root/'analysis/image_classification/pfp34_new_photos'
    if not (app/'READY.json').is_file():raise RuntimeError('No verified continuation')
    plan=json.loads((app/'cleanup_plan.json').read_text())
    archive=root/'17082026_cleanup/_cleanup_review'/('photograph_installer_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for item in plan:
        p=Path(item['source'])
        if item['kind']=='file':
            valid=(p == root/'analysis/PFPeptoids_photograph_patch_v1.zip' and p.is_file() and not p.is_symlink() and sha(p)==item['sha256'])
        else:
            valid=(p.parent == root/'analysis/photograph_validation' and p.name.startswith('pilot_') and p.is_dir() and not p.is_symlink())
            if valid:
                entries=list(p.rglob('*'))
                valid=not any(f.is_symlink() for f in entries) and {f.relative_to(p).as_posix():sha(f) for f in entries if f.is_file()}==item['files']
        if not valid: raise RuntimeError('Cleanup inventory changed; nothing moved')
    archive.mkdir(parents=True,exist_ok=False)
    audit=archive/'deletion_audit.csv'
    with audit.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['action','source','destination','sha256','reason'])
        for item in plan:
            src=Path(item['source']);dst=archive/src.name
            shutil.move(str(src),str(dst))
            w.writerow(['archive_move',str(src),str(dst),item.get('sha256',hashlib.sha256(json.dumps(item.get('files',{}),sort_keys=True).encode()).hexdigest()),item['reason']]);f.flush();os.fsync(f.fileno())
    print('Archived verified installer/pilot candidates:',archive)
    print('Retained old Photos, original session, new session and staged renders. Pilot evidence remains in the cleanup archive. Nothing deleted.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--stage',type=Path)
    p.add_argument('--cleanup-confirmed',action='store_true',help='Use only after confirming the new GUI works; archives only explicit cleanup candidates')
    a=p.parse_args()
    if a.cleanup_confirmed:cleanup(a.root.resolve())
    else:prepare(a.root.resolve(),a.stage or a.root/'analysis/photograph_regeneration'/DEFAULT_STAGE)

if __name__=='__main__':main()
