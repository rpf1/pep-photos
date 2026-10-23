"""Contract, failure-path and persistence checks; no production review writes."""
import copy
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from PIL import Image
from figure_layout import migrate_layout,validate_layout,read_manifest,sha,atomic_write
from review_metric_figures import load_review,ReviewState,self_test,FigureReviewGUI

RESULT=Path(__file__).resolve().parents[1]/'results/archie_8193723_complete'

def legacy(root):
    folder=root/'figures/01_family';folder.mkdir(parents=True)
    Image.new('RGB',(24,16),'blue').save(folder/'01_example.png');(folder/'01_example.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    rows=[dict(figure_id='01_example',filename=f'figures/01_family/01_example.{ext}',title='Example',question='Question',x_variable='x',y_variable='y',grouping_color_variable='none',units='nm',axis_limits='[{"xlim":[0,1],"ylim":[0,1]}]',axis_rule='metric-specific',source_table='data/example.csv',interpretation='Example metadata retained') for ext in ['png','svg']]
    write_manifest(root,rows)
    return rows

def write_manifest(root,rows):
    with (root/'plot_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class LayoutTests(unittest.TestCase):
    def test_migration_byte_preserving_and_idempotent(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);old=legacy(r);hashes={Path(x['filename']).suffix:sha(r/x['filename']) for x in old}
            self.assertEqual(migrate_layout(r)['logical_figures'],1);new=read_manifest(r)[0]
            for ext in ['png','svg']:self.assertEqual(hashes['.'+ext],sha(r/new[ext+'_filename']))
            self.assertEqual(new['interpretation'],old[0]['interpretation']);self.assertFalse((r/'figures/01_family').exists());self.assertEqual(migrate_layout(r)['logical_figures'],1)
    def test_preflight_rejects_untracked_and_conflicting_pairs(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);rows=legacy(r);original=(r/'plot_manifest.csv').read_bytes();extra=r/'figures/01_family/extra.svg';extra.write_text('extra')
            with self.assertRaisesRegex(ValueError,'Unmanifested'):migrate_layout(r)
            self.assertEqual(original,(r/'plot_manifest.csv').read_bytes());extra.unlink();rows[1]['title']='Conflict';write_manifest(r,rows)
            with self.assertRaisesRegex(ValueError,'Conflicting'):migrate_layout(r)
            self.assertTrue((r/rows[0]['filename']).exists())
    def test_failed_png_validation_rolls_back(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);rows=legacy(r);(r/rows[0]['filename']).write_bytes(b'broken');before=(r/'plot_manifest.csv').read_bytes()
            with self.assertRaises(Exception):migrate_layout(r)
            self.assertEqual(before,(r/'plot_manifest.csv').read_bytes());self.assertTrue(all((r/x['filename']).exists() for x in rows))
    def test_new_plotter_emits_one_row_and_format_paths(self):
        import matplotlib.pyplot as plt
        from plot_metric_analysis import Plotter
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);plotter=Plotter(r);fig,ax=plt.subplots();ax.plot([0,1],[0,1]);plotter.save(fig,'01_family','01_example','Title','Question','x','y','none','nm','data/x.csv','Interpretation');plotter.finish()
            self.assertEqual(validate_layout(r)['logical_figures'],1);self.assertEqual(len(read_manifest(r)),1)
    def test_invalid_paths_duplicate_id_and_hash(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);legacy(r);migrate_layout(r);rows=read_manifest(r)
            with self.assertRaisesRegex(ValueError,'duplicate'):validate_layout(r,rows+rows)
            bad=copy.deepcopy(rows);bad[0]['png_filename']='../missing.png'
            with self.assertRaisesRegex(ValueError,'Noncanonical'):validate_layout(r,bad)
            (r/rows[0]['svg_filename']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash'):validate_layout(r)

class GuideTests(unittest.TestCase):
    def test_current_guide_and_science_invariance(self):
        guide,figures,validation=load_review(RESULT);self.assertEqual(len(figures),validation['logical_figures'])
        recorded=json.loads((RESULT/'figure_layout_validation.json').read_text())
        for path,digest in recorded['scientific_files_unchanged'].items():self.assertEqual(sha(RESULT/path),digest,path)
        for move in recorded['moves']:self.assertEqual(move['before_sha256'],sha(RESULT/move['new']))
    def test_guide_failures_clear(self):
        original=json.loads((RESULT/'figure_guide.json').read_text());read_text=Path.read_text
        def check_bad(modification,pattern):
            bad=copy.deepcopy(original);modification(bad)
            def mocked(path,*a,**kw):return json.dumps(bad) if path.name=='figure_guide.json' else read_text(path,*a,**kw)
            with patch.object(Path,'read_text',mocked),self.assertRaisesRegex(ValueError,pattern):load_review(RESULT)
        check_bad(lambda g:g['figures'].append(g['figures'][0]),'IDs')
        check_bad(lambda g:g['figures'][0].update(png_path='missing.png'),'path mismatch')
        check_bad(lambda g:g['figures'][0].update(metric_ids=['missing']),'unresolved')
        check_bad(lambda g:g['figures'][0].update(observed_result=''),'missing explanation')
        check_bad(lambda g:g['figures'][0].update(source_tables=['missing.csv']),'Missing')
        check_bad(lambda g:g['figures'][0].update(hypothesis_status='documented_expectation',hypothesis_provenance=[dict(provenance_type='completed_result')]),'design provenance')
    def test_self_test_is_read_only(self):
        before={p:sha(p) for p in RESULT.rglob('*') if p.is_file()};self_test(RESULT)
        self.assertEqual(before,{p:sha(p) for p in RESULT.rglob('*') if p.is_file()})

class StateTests(unittest.TestCase):
    def test_roundtrip_all_statuses_and_identity(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'state.json';state=ReviewState(path,'identity',['one','two'])
            for status in ['not_reviewed','understood','revisit','important']:state.update('one',status=status,note='Unicode ΔAPL\nSecond line',viewed=True)
            state.update('two',viewed=True);again=ReviewState(path,'identity',['one','two']);self.assertEqual(again.data['last_figure_id'],'two');self.assertIn('ΔAPL',again.data['per_figure']['one']['note'])
            with self.assertRaises(ValueError):ReviewState(path,'other',['one'])
            with self.assertRaises(ValueError):state.update('one',status='wrong')
            with self.assertRaises(ValueError):state.update('missing')
    def test_failed_atomic_write_preserves_state_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'state.json';state=ReviewState(path,'id',['one']);state.update('one',note='old');before=path.read_bytes()
            with patch('figure_layout.os.replace',side_effect=OSError('injected failure')),self.assertRaises(OSError):state.update('one',note='new')
            self.assertEqual(path.read_bytes(),before);self.assertEqual(state.data['per_figure']['one']['note'],'old');self.assertEqual(len(list(Path(t).iterdir())),1)
    def test_external_edit_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'state.json';a=ReviewState(path,'id',['one']);a.update('one');b=ReviewState(path,'id',['one']);b.update('one',note='external')
            with self.assertRaisesRegex(RuntimeError,'externally'):a.update('one',note='overwrite')
            self.assertEqual(json.loads(path.read_text())['per_figure']['one']['note'],'external')
    def test_typing_does_not_navigate(self):
        from types import SimpleNamespace
        gui=object.__new__(FigureReviewGUI);gui.navigate=lambda n:self.fail('Typing navigated')
        for cls in ['Text','Entry','TEntry','TCombobox']:
            gui.key(SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda:cls),keysym='Right'))

if __name__=='__main__':unittest.main()
