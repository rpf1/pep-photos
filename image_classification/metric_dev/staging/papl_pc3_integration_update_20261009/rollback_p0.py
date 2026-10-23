#!/usr/bin/env python3
"""Guarded P0 rollback. Roll back P2 and P1 first; retain all user notes."""
import argparse,json,os,sys
from pathlib import Path
from complete_papl_cohort_gui_20261009 import digest,expect,atomic_write,SOURCE_MANIFEST

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args();b=a.base.resolve();s=b/'staging/papl_pc3_integration_update_20261009';r=b/'results/archie_8193723_complete';report=json.loads((s/'full_cohort_papl_gui_validation_20261009.json').read_text());gui=b/'scripts/review_metric_figures.py';fd=os.open(r,os.O_RDONLY)
 import fcntl
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);expect(digest(r/'review_bundle.json')==report['registry_after_sha256'],'Registry changed; roll back P1 first');expect(digest(gui)==report['gui_after_sha256'],'GUI changed; roll back P2 first')
  backup=Path(report['backups'][0]);oldgui=Path(report['backups'][1]);expect(digest(backup)==report['registry_before_sha256'],'Registry backup mismatch');expect(digest(oldgui)==report['gui_before_sha256'],'GUI backup mismatch')
  for n,h in report['frozen_sha256'].items():expect(digest(r/n)==h,'Frozen source changed')
  g=json.loads((r/'review_bundle.json').read_text());targets={}
  for f in g['figures']:
   if f['figure_id'] in report['new_ids']:
    for ext in ['png','svg']:targets[r/f[ext+'_path']]=f[ext+'_sha256']
  targets[r/SOURCE_MANIFEST]=report['source_snapshot_sha256'];generator=b/'scripts/complete_papl_cohort_gui_20261009.py';targets[generator]=digest(s/'complete_papl_cohort_gui_20261009.py')
  for target,h in targets.items():expect(digest(target)==h,'Installed file changed: '+str(target))
  dest=s/'rollback_retired_P0';expect(not dest.exists(),'Rollback archive already exists');print('ROLLBACK PREFLIGHT PASS; user review/profile notes retained')
  if not a.apply:return
  dest.mkdir();atomic_write(r/'review_bundle.json',backup.read_bytes());atomic_write(gui,oldgui.read_bytes())
  for target in targets:
   saved=dest/target.relative_to(b);saved.parent.mkdir(parents=True,exist_ok=True);os.replace(target,saved)
  print('P0 ROLLED BACK: registry=164; installer-owned outputs retained in existing stage/rollback_retired_P0')
 finally:os.close(fd)
if __name__=='__main__':
 try:main()
 except Exception as e:print('STOP:',e,file=sys.stderr);sys.exit(1)
