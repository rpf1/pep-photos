#!/usr/bin/env python3
"""Separate guarded P1/P2 transactions. Default is read-only. Use --apply only after approval."""
import argparse,hashlib,json,os,sys,subprocess,copy
from datetime import datetime,timezone
from pathlib import Path
from complete_papl_cohort_gui_20261009 import atomic_write,digest,expect

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--gate',choices=['P1','P2'],required=True);p.add_argument('--apply',action='store_true');p.add_argument('--rollback',type=Path);a=p.parse_args();b=a.base.resolve();stage=b/'staging/papl_pc3_integration_update_20261009';r=b/'results/archie_8193723_complete';bundle=Path(__file__).resolve().parent;gui=b/'scripts/review_metric_figures.py'
 expect(bundle==stage.resolve(),'Run from existing canonical stage')
 fd=os.open(r,os.O_RDONLY)
 import fcntl
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.rollback:
   expect(a.apply,'Rollback requires --apply');m=json.loads(a.rollback.read_text());expect(m['base']==str(b),'Rollback belongs to another base')
   for rel,v in m['files'].items():expect(digest(b/rel)==v['after_sha256'],'Changed since install: '+rel)
   for rel,v in m['files'].items():
    if v['before_backup']:atomic_write(b/rel,(a.rollback.parent/v['before_backup']).read_bytes())
    else:(b/rel).unlink() # Only exact installer-owned files, guarded above.
   print('ROLLBACK COMPLETE');return
  g=json.loads((r/'review_bundle.json').read_text());expect(len(g['figures'])==168,'P0=168 required before P1/P2')
  protected={n:digest(r/n) for n in ['figure_review_state.json','data/pca_parameters.json','data/frozen_parameters.json']};targets={}
  if a.gate=='P1':
   d=stage/'validation/axis_derivatives';records=json.loads((d/'axis_derivatives.json').read_text())
   for i,f in enumerate(g['figures']):
    if f['figure_id'] in records:
     rec=records[f['figure_id']];expect('original_display' not in f,'P1 already installed')
     for ext in ['png','svg']:
      expect(f[ext+'_sha256']==rec['original_display'][ext+'_sha256'],'Original changed')
      src=d/Path(rec[ext+'_path']).name;expect(digest(src)==rec[ext+'_sha256'],'Derivative changed');dest=r/rec[ext+'_path'];expect(not dest.exists(),'Derivative destination exists');targets[dest]=src.read_bytes()
     for source,expected in rec['display_source_hashes'].items():expect(digest(r/source)==expected,'Source arrays changed')
     g['figures'][i]=rec
   targets[r/'review_bundle.json']=(json.dumps(g,indent=2,ensure_ascii=False)+'\n').encode()
  else:
   payload=bundle/'payload';source=gui.read_bytes();expect(hashlib.sha256(source).hexdigest()==json.loads((payload/'payload_hashes.json').read_text())['expected_navigation_gui_sha256'],'GUI differs from P0 navigation result')
   targets[gui]=(payload/'review_metric_figures.py').read_bytes();module=b/'scripts/visual_comparison_profiles.py';expect(not module.exists(),'P2 module already exists');targets[module]=(payload/'visual_comparison_profiles.py').read_bytes()
   for name in ['visual_comparison_photo_sources.csv','visual_comparison_candidates.csv']:
    dest=r/name;expect(not dest.exists(),'Profile catalog already exists');targets[dest]=(payload/name).read_bytes()
  print('READ-ONLY PREFLIGHT PASS:',a.gate,'targets:',len(targets),'notes/PCA protected')
  if not a.apply:return
  name='backup_'+a.gate+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');backup=stage/name;backup.mkdir();m={'gate':a.gate,'base':str(b),'protected':protected,'files':{}};changed=[]
  for i,(path,raw) in enumerate(targets.items()):
   rel=str(path.relative_to(b));old=path.read_bytes() if path.exists() else None;before=f'{i:03d}.original' if old is not None else None
   if old is not None:(backup/before).write_bytes(old)
   m['files'][rel]={'before_backup':before,'before_sha256':hashlib.sha256(old).hexdigest() if old is not None else None,'after_sha256':hashlib.sha256(raw).hexdigest()}
  (backup/'transaction.json').write_text(json.dumps(m,indent=2)+'\n')
  try:
   for path,raw in targets.items():atomic_write(path,raw);changed.append(path)
   check=subprocess.run([sys.executable,str(gui),'--results',str(r),'--check'],text=True,capture_output=True);expect(check.returncode==0,check.stderr)
   for n,h in protected.items():expect(digest(r/n)==h,'Protected source changed')
   (backup/'validation.json').write_text(check.stdout);print('APPLY PASS:',a.gate,'GUI --check passed; manual GUI acceptance still required');print('ROLLBACK MANIFEST:',backup/'transaction.json')
  except Exception:
   for path in reversed(changed):
    v=m['files'][str(path.relative_to(b))]
    expect(digest(path)==v['after_sha256'],'Cannot overwrite external change during rollback')
    if v['before_backup']:atomic_write(path,(backup/v['before_backup']).read_bytes())
    else:path.unlink()
   raise
 finally:os.close(fd)
if __name__=='__main__':
 try:main()
 except Exception as e:print('STOP:',e,file=sys.stderr);sys.exit(1)
