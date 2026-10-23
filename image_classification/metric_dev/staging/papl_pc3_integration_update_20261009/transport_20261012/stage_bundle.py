#!/usr/bin/env python3
"""Guarded transfer into the existing stage only; never installs GUI/results."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--write-stage',action='store_true');a=p.parse_args();b=a.base.resolve();src=Path(__file__).resolve().parent/'metric_dev/staging/papl_pc3_integration_update_20261009';dest=b/'staging/papl_pc3_integration_update_20261009';r=b/'results/archie_8193723_complete'
if not dest.is_dir():p.error('Existing staging directory required')
if sha(b/'scripts/review_metric_figures.py')!='1a01e3d1a464fade165779791b693ce7a42952fb0035cd01babc94d4b51718d7':p.error('Expected unpatched original GUI; stop and reconcile current state')
if len(json.loads((r/'review_bundle.json').read_text())['figures'])!=164:p.error('Expected 164 figures before staging')
files=[x for x in src.rglob('*') if x.is_file() and '__pycache__' not in x.parts];changed=[x for x in files if not (dest/x.relative_to(src)).exists() or sha(x)!=sha(dest/x.relative_to(src))]
print('STAGE PREFLIGHT: files',len(files),'changed/new',len(changed),'canonical GUI/results will not be written')
if a.write_stage:
 fd=os.open(r,os.O_RDONLY)
 import fcntl
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  backup=dest/('backup_stage_sources_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'));backup.mkdir();records={}
  for x in changed:
   rel=x.relative_to(src);target=dest/rel
   if target.exists():old=backup/rel;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,old)
   records[str(rel)]={'previous_sha256':sha(target) if target.exists() else None,'staged_sha256':sha(x)};target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(x,target)
  (backup/'manifest.json').write_text(json.dumps(records,indent=2)+'\n');testdest=dest/'tests';testdest.mkdir(exist_ok=True)
  for t in (Path(__file__).resolve().parent/'metric_dev/tests').glob('*.py'):shutil.copy2(t,testdest/t.name)
  print('STAGED ONLY:',dest,'previous stage sources preserved:',backup)
 finally:os.close(fd)
