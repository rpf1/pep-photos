#!/usr/bin/env python3
"""Safely fix the existing PFPeptoids figure-review sidebar navigation.

Read-only by default; --apply backs up then patches only scripts/review_metric_figures.py.
Does not edit scientific results, figure registry, review state or PCA parameters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE_SHA256 = '1a01e3d1a464fade165779791b693ce7a42952fb0035cd01babc94d4b51718d7'
BACKUP_NAME = 'review_metric_figures_before_navigation_fix_20261009.py'

OLD_REFRESH = '''    def refresh_tree(self):
        if not hasattr(self,'tree'):return
        self.tree.delete(*self.tree.get_children());self.filtered=[f['figure_id'] for f in self.figures if self.matches(f)]
        for fid in self.filtered:
            f=self.byid[fid];parent='family:'+f['family']
            if not self.tree.exists(parent):self.tree.insert('', 'end',iid=parent,text=self.guide['figure_families'][f['family']]['name'],open=True)
            status=self.state.data['per_figure'].get(fid,{}).get('status','not_reviewed');self.tree.insert(parent,'end',iid=fid,text=fid+'  ['+status.replace('_',' ')+']')
        if self.current in self.filtered:self.tree.selection_set(self.current);self.tree.see(self.current)
        self.update_position()
'''
NEW_REFRESH = '''    def refresh_tree(self):
        if not hasattr(self,'tree'):return
        # Do not reset tree expansion/scroll whenever a filter changes.
        old_expanded=getattr(self,'_tree_open',{}).copy()
        old_expanded.update({iid:bool(self.tree.item(iid,'open')) for iid in self.tree.get_children()})
        self._tree_open=old_expanded
        old_scroll=self.tree.yview()
        self.tree.delete(*self.tree.get_children());self.filtered=[f['figure_id'] for f in self.figures if self.matches(f)]
        for fid in self.filtered:
            f=self.byid[fid];parent='family:'+f['family']
            if not self.tree.exists(parent):
                self.tree.insert('', 'end',iid=parent,text=self.guide['figure_families'][f['family']]['name'],open=old_expanded.get(parent,True))
            status=self.state.data['per_figure'].get(fid,{}).get('status','not_reviewed')
            self.tree.insert(parent,'end',iid=fid,text=fid+'  ['+status.replace('_',' ')+']')
        if self.current in self.filtered:self.tree.selection_set(self.current)
        if old_scroll:self.tree.yview_moveto(old_scroll[0])
        self.update_position()
'''
OLD_SELECT = '''    def tree_select(self,event=None):
        selection=self.tree.selection()
        if selection:
            fid=selection[0]
            if fid.startswith('family:'):
                kids=self.tree.get_children(fid)
                if kids:self.show(kids[0])
            elif fid!=self.current:self.show(fid)
'''
NEW_SELECT = '''    def tree_select(self,event=None):
        selection=self.tree.selection()
        if not selection:return
        fid=selection[0]
        # A family row is a folder to expand/collapse, not a figure to open.
        if fid in self.byid and fid!=self.current:self.show(fid)
'''
OLD_SHOW_END = '''        self.fit();self.refresh_tree();self.feedback.set('Saved view: '+fid)
'''
NEW_SHOW_END = '''        self.fit()
        # Update just the selected item; rebuilding the whole tree here
        # expands closed folders and displaces the user's next click.
        if self.tree.exists(fid):
            status=self.state.data['per_figure'].get(fid,{}).get('status','not_reviewed')
            self.tree.item(fid,text=fid+'  ['+status.replace('_',' ')+']')
            if self.tree.selection()!=(fid,):self.tree.selection_set(fid)
            self.tree.see(fid)
        self.update_position();self.feedback.set('Saved view: '+fid)
'''
OLD_STATUS = '''        try:self.state.update(self.current,status=self.status.get());self.refresh_tree();self.feedback.set('Review status saved.')
'''
NEW_STATUS = '''        try:
            self.state.update(self.current,status=self.status.get())
            if self.tree.exists(self.current):
                status=self.state.data['per_figure'][self.current]['status']
                self.tree.item(self.current,text=self.current+'  ['+status.replace('_',' ')+']')
            self.update_position();self.feedback.set('Review status saved.')
'''
OLD_ARGS = '''    args=parser.parse_args()
    try:
'''
NEW_ARGS = '''    args=parser.parse_args()
    # Relative --results paths must work even when result_identity() uses parents.
    args.results=args.results.expanduser().resolve()
    try:
'''
REPLACEMENTS = ((OLD_REFRESH,NEW_REFRESH),(OLD_SELECT,NEW_SELECT),(OLD_SHOW_END,NEW_SHOW_END),
                (OLD_STATUS,NEW_STATUS),(OLD_ARGS,NEW_ARGS))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(src):
    for old,new in REPLACEMENTS:
        found=src.count(old)
        if found!=1:raise RuntimeError(f'STOP: expected one exact GUI source block, found {found}; do not patch manually')
        src=src.replace(old,new,1)
    compile(src,'review_metric_figures.py','exec')
    return src


def atomic_write(target,content,mode):
    tmp=None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=target.parent,prefix='.gui_patch_',delete=False) as f:
            tmp=Path(f.name)
            f.write(content);f.flush();os.fsync(f.fileno())
        os.chmod(tmp,mode)
        os.replace(tmp,target)
    finally:
        if tmp is not None and tmp.exists():tmp.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',required=True,type=Path,help='Canonical metric_dev directory')
    ap.add_argument('--apply',action='store_true',help='Back up and patch the existing GUI')
    args=ap.parse_args()
    base=args.base.expanduser().resolve()
    target=base/'scripts/review_metric_figures.py'
    stage=base/'staging/papl_pc3_integration_update_20261009'
    results=base/'results/archie_8193723_complete'
    if not stage.is_dir():ap.error('Existing canonical stage directory is missing: '+str(stage))
    if not target.is_file():ap.error('Cannot locate existing GUI: '+str(target))
    if not results.is_dir():ap.error('Cannot locate canonical results: '+str(results))
    original=target.read_bytes()
    actual=sha(original)
    if actual!=SOURCE_SHA256:
        raise RuntimeError(f'STOP: unexpected GUI source SHA256 {actual}; expected {SOURCE_SHA256}. No files changed.')
    patched=transform(original.decode('utf-8')).encode('utf-8')
    print('GUI:',target)
    print('SOURCE SHA256:',actual)
    print('PATCHED SHA256:',sha(patched))
    print('CHANGES: keep collapsed folders closed; avoid tree rebuilds; folder selection no longer opens a figure; resolve relative results paths')
    if not args.apply:
        print('DRY RUN PASS: zero files changed. Rerun with --apply to install the GUI-only correction.')
        return
    lockfd=None
    if os.name=='posix':
        import fcntl
        lockfd=os.open(str(results),os.O_RDONLY)
        try:
            fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            os.close(lockfd)
            raise RuntimeError('STOP: close the existing GUI and any registry installer before patching.') from exc
    try:
        if sha(target.read_bytes())!=actual:
            raise RuntimeError('STOP: GUI changed during preflight')
        backup=stage/BACKUP_NAME
        if backup.exists():
            if sha(backup.read_bytes())!=actual:
                raise RuntimeError('STOP: pre-existing backup differs from source; will not overwrite')
        else:
            shutil.copy2(target,backup)
            if sha(backup.read_bytes())!=actual:raise RuntimeError('Backup checksum mismatch')
        mode=target.stat().st_mode&0o777
        atomic_write(target,patched,mode)
        proc=subprocess.run([sys.executable,str(target),'--results',str(results),'--check'],
                            text=True,capture_output=True)
        try: validation=json.loads(proc.stdout)
        except json.JSONDecodeError:validation={}
        if proc.returncode or validation.get('logical_figures')!=164 or validation.get('state_roundtrip')!='passed':
            atomic_write(target,original,mode)
            print(proc.stdout,proc.stderr,sep='\n',file=sys.stderr)
            raise RuntimeError('STOP: GUI --check failed. Original GUI automatically restored; backup retained.')
        print('INSTALL PASS: GUI navigation patched, backup preserved.')
        print('GUI HEADLESS CHECK PASS:',json.dumps({k:validation.get(k) for k in ('logical_figures','png','svg','paths_hashes_axes','state_roundtrip')}))
        print('BACKUP:',backup)
        print('NOTE: mouse navigation still needs a visual check on Archie.')
    finally:
        if lockfd is not None:os.close(lockfd)

if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError,UnicodeError) as e:
        print(str(e),file=sys.stderr)
        sys.exit(1)
