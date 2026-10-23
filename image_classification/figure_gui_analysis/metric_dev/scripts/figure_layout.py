"""Durable, explicit logical-figure contract and lossless legacy migration."""
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

SCHEMA = '2'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_write(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.write_', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name == 'posix':
            # Network filesystems may reject directory fsync after a successful replace.
            try:
                fd = os.open(str(path.parent), os.O_RDONLY)
                try: os.fsync(fd)
                finally: os.close(fd)
            except OSError: pass
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def read_manifest(out):
    with (Path(out)/'plot_manifest.csv').open(newline='') as f:
        return list(csv.DictReader(f))

def axis_check(row):
    import numpy as np
    a = json.loads(row['axis_limits']); rule = row['axis_rule']
    def eq(x,y):
        if not np.allclose(x,y): raise ValueError('Axis rule failed: '+row['figure_id'])
    if rule.startswith('shared_BM_y'): eq(a[0]['ylim'],a[1]['ylim'])
    if rule == 'shared_BM_y_0_1': eq(a[0]['ylim'],[0,1])
    if rule == 'shared_BM_xy_equal':
        eq(a[0]['xlim'],a[0]['ylim']);eq(a[0]['xlim'],a[1]['xlim']);eq(a[0]['ylim'],a[1]['ylim'])
    if rule == 'PCA_BM_common_limits':
        eq(a[0]['xlim'],a[1]['xlim']);eq(a[0]['ylim'],a[1]['ylim'])
    if rule in ('effect_plane_0_1_equal','synthetic_0_1_equal'):
        for panel in a:
            if panel['xlabel'].startswith('Effect M'):
                eq(panel['xlim'],[0,1]);eq(panel['ylim'],[0,1])
    if rule == 'selectivity_minus1_plus1': eq(a[0]['xlim'],[-1,1])
    if rule == 'desirability_0_1': eq(a[0]['ylim'],[0,1])

def validate_layout(out, rows=None):
    from PIL import Image
    out = Path(out); rows = read_manifest(out) if rows is None else rows
    ids = [r['figure_id'] for r in rows]
    if not ids or len(ids)!=len(set(ids)): raise ValueError('Empty or duplicate logical figure IDs')
    expected = set()
    for r in rows:
        if r.get('layout_schema_version') != SCHEMA: raise ValueError('Unsupported layout schema; use migrate-figures')
        for ext in ('png','svg'):
            name = r[ext+'_filename']; p = Path(name)
            if p.parts != ('figures',ext,r['family'],r['figure_id']+'.'+ext): raise ValueError('Noncanonical figure path: '+name)
            target = out/p
            if target.is_symlink() or not target.is_file() or not target.stat().st_size: raise ValueError('Missing/empty figure: '+str(target))
            if sha(target)!=r[ext+'_sha256']: raise ValueError('Figure hash mismatch: '+name)
            if ext=='png':
                with Image.open(target) as im: im.verify()
            expected.add(name)
        axis_check(r)
    actual={p.relative_to(out).as_posix() for p in (out/'figures').rglob('*') if p.suffix.lower() in ('.png','.svg')}
    if expected!=actual: raise ValueError('Unreconciled figure files: '+str(sorted(expected^actual)))
    return {'logical_figures':len(rows),'png':len(rows),'svg':len(rows),'families':len({r['family'] for r in rows}),'paths_hashes_axes':'passed'}

def migrate_layout(out):
    """Preflight every file, move exact pairs, validate, rollback on any failure."""
    out=Path(out); rows=read_manifest(out)
    if rows and 'png_filename' in rows[0]: return validate_layout(out)
    groups={}
    for r in rows: groups.setdefault(r['figure_id'],[]).append(r)
    logical=[]; moves=[]; olddirs=set()
    for fid, pair in groups.items():
        if len(pair)!=2 or {Path(r['filename']).suffix for r in pair}!={'.png','.svg'}: raise ValueError('Expected one PNG/SVG pair: '+fid)
        meta={k:v for k,v in pair[0].items() if k!='filename'}
        if any({k:v for k,v in r.items() if k!='filename'}!=meta for r in pair): raise ValueError('Conflicting format metadata: '+fid)
        family=Path(pair[0]['filename']).parent.name
        meta.update(family=family,layout_schema_version=SCHEMA)
        for r in pair:
            old=Path(r['filename']);ext=old.suffix[1:]
            if old.parts!=('figures',family,fid+'.'+ext): raise ValueError('Unexpected legacy path: '+str(old))
            source=out/old;dest=out/'figures'/ext/family/old.name
            if not source.is_file() or source.is_symlink() or dest.exists(): raise ValueError('Missing source or existing destination: '+str(source))
            meta[ext+'_filename']=dest.relative_to(out).as_posix();meta[ext+'_sha256']=sha(source)
            moves.append((source,dest));olddirs.add(source.parent)
        logical.append(meta)
    actual={p for p in (out/'figures').rglob('*') if p.suffix.lower() in ('.png','.svg')}
    if actual!={x for x,y in moves}: raise ValueError('Unmanifested image files; migration refused')
    original=(out/'plot_manifest.csv').read_bytes();done=[]
    try:
        for source,dest in moves:
            dest.parent.mkdir(parents=True,exist_ok=True);source.rename(dest);done.append((source,dest))
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(logical[0]));w.writeheader();w.writerows(logical)
        atomic_write(out/'plot_manifest.csv',s.getvalue().encode())
        result=validate_layout(out)
    except BaseException:
        for source,dest in reversed(done): dest.rename(source)
        atomic_write(out/'plot_manifest.csv',original)
        raise
    for d in sorted(olddirs):
        if not any(d.iterdir()): d.rmdir()
    evidence={'schema_version':1,'validation':result,'moves':[{'old':str(a.relative_to(out)),'new':str(b.relative_to(out)),'before_sha256':next(r[b.suffix[1:]+'_sha256'] for r in logical if r['figure_id']==b.stem),'after_sha256':sha(b)} for a,b in moves]}
    atomic_write(out/'figure_layout_validation.json',(json.dumps(evidence,indent=2)+'\n').encode())
    return result
