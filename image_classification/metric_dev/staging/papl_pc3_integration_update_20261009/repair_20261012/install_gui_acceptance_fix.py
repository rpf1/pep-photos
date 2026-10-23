#!/usr/bin/env python3
"""Read-only by default, guarded GUI module patch on P0/P1/P2 Archie baseline.

Never alters review state, candidate CSV, scientific data or saved profile JSON.
--apply replaces only scripts/visual_comparison_profiles.py after hash guard,
with a reversible transaction in the *existing* metric_dev stage.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime,timezone
from pathlib import Path

ORIGINAL_SHA='235d994b8fe6682daf9b26097740ef02dbb4a08efe5d87f3a79785b73636b213'
RESULTS='results/archie_8193723_complete'
STAGE='staging/papl_pc3_integration_update_20261009/repair_20261012'
TARGET='scripts/visual_comparison_profiles.py'
PROTECTED=['figure_review_state.json','data/pca_parameters.json','data/frozen_parameters.json',
           'visual_comparison_candidates.csv','visual_comparison_photo_sources.csv','review_bundle.json']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def check(flag,msg):
 if not flag:raise RuntimeError('STOP: '+msg)

def atomic(p,data):
 p=Path(p)
 with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.gui_acceptance_',delete=False) as f:
  tmp=Path(f.name)
  try:f.write(data);f.flush();os.fsync(f.fileno())
  except Exception:tmp.unlink(missing_ok=True);raise
 try:os.replace(tmp,p)
 finally:tmp.unlink(missing_ok=True)

def snapshot(base):
 r=base/RESULTS
 d={p:sha(r/p) for p in PROTECTED}
 p=r/'visual_comparison_profiles.json'
 if p.exists():d['visual_comparison_profiles.json']=sha(p)
 return d

def guard(base,package):
 target=base/TARGET;planned=package/'visual_comparison_profiles.py'
 check(target.is_file() and planned.is_file(),'Existing module or packaged replacement absent')
 digest=sha(target);check(digest==ORIGINAL_SHA,f'Installed source differs from certified P2 baseline ({digest}); inspect diff before applying')
 check(len(json.loads((base/RESULTS/'review_bundle.json').read_text())['figures']) in (168,174),'Expected 168 accepted figures (or 174 after optional atlas)')
 check(len(json.loads((base/RESULTS/'visual_comparison_profiles.json').read_text()).get('profiles',[]))>=0 if (base/RESULTS/'visual_comparison_profiles.json').exists() else True,'Saved profile store invalid')
 check(planned!=target and sha(planned)!=digest,'No valid new patch / already installed')
 print('BASELINE AND PROFILE READ-ONLY GATE PASS')
 return target,planned

def verify(base):
 gui=base/'scripts/review_metric_figures.py';r=base/RESULTS
 check(gui.is_file(),'GUI entrypoint missing')
 run=subprocess.run([sys.executable,str(gui),'--results',str(r),'--check'],cwd=r,text=True,capture_output=True)
 check(run.returncode==0,'GUI regression failed: '+run.stdout+'\n'+run.stderr)
 info=json.loads(run.stdout[run.stdout.find('{'):run.stdout.rfind('}')+1])
 check(info['logical_figures'] in (168,174),'Wrong figure count after patch')
 print('GUI HEADLESS VALIDATION:',info['logical_figures'],'figures')

def rollback(base,manifest):
 import fcntl
 details=json.loads(manifest.read_text());check(details['base']==str(base),'Transaction belongs to a different project')
 target=base/TARGET;before=manifest.parent/'original_visual_comparison_profiles.py'
 check(sha(target)==details['installed_sha256'],'Post-install file differs; do not overwrite user edits')
 check(sha(before)==details['original_sha256'],'Backup hash invalid')
 fd=os.open(base/RESULTS,os.O_RDONLY)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  check(sha(target)==details['installed_sha256'],'Concurrent change')
  atomic(target,before.read_bytes());verify(base)
  print('ROLLBACK PASS; saved profiles and scientific data unchanged')
 finally:os.close(fd)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--base',type=Path,required=True)
 parser.add_argument('--apply',action='store_true')
 parser.add_argument('--rollback',type=Path)
 args=parser.parse_args();base=args.base.resolve();pkg=base/STAGE
 check(pkg.is_dir() and (base/RESULTS).is_dir(),'Canonical stage/results directory not found')
 if args.rollback:
  check(args.apply,'Rollback requires explicit --apply');rollback(base,args.rollback.resolve());return
 target,planned=guard(base,pkg)
 before=snapshot(base)
 print('PATCHED MODULE SHA256:',sha(planned))
 print('PROTECTED STATE:',json.dumps(before,indent=2))
 if not args.apply:
  print('READ-ONLY PREFLIGHT PASS. Close the GUI before running --apply.');return
 import fcntl
 fd=os.open(base/RESULTS,os.O_RDONLY)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  check(sha(target)==ORIGINAL_SHA,'Installed module changed after preflight')
  txroot=pkg/'backups';txroot.mkdir(exist_ok=True)
  txn=txroot/('gui_acceptance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'));txn.mkdir()
  original=target.read_bytes();(txn/'original_visual_comparison_profiles.py').write_bytes(original)
  data=dict(base=str(base),original_sha256=sha(target),installed_sha256=sha(planned),protected=before,created_utc=datetime.now(timezone.utc).isoformat())
  (txn/'transaction.json').write_text(json.dumps(data,indent=2)+'\n')
  try:
   atomic(target,planned.read_bytes())
   verify(base)
   check(snapshot(base)==before,'Protected state changed; rolling module back')
  except Exception:
   check(sha(target)==data['installed_sha256'],'External modification; refusing automatic rollback')
   atomic(target,original)
   raise
  print('GUI ACCEPTANCE PATCH APPLY PASS. ROLLBACK MANIFEST:',txn/'transaction.json')
 finally:os.close(fd)

if __name__=='__main__':
 try:main()
 except Exception as exc:print(exc,file=sys.stderr);sys.exit(1)
