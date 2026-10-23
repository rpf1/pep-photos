import copy,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
BASE=Path(os.environ['PFPML_TEST_BASE']).resolve();STAGE=BASE/'staging/papl_pc3_integration_update_20261009';sys.path.insert(0,str(STAGE/'payload'));sys.path.insert(0,str(STAGE));sys.path.insert(0,str(BASE/'scripts'))
import visual_comparison_profiles as v
import complete_papl_cohort_gui_20261009 as p0
class Profiles(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'profiles.json';self.s=v.ProfileStore(self.path)
 def tearDown(self):self.temp.cleanup()
 def test_restart_order_notes_independent_profiles(self):
  a=self.s.create('A');b=self.s.create('B');e={'kind':'figure','figure_id':'existing'};self.s.add(a,e);self.s.add(a,{'kind':'figure','figure_id':'second'});self.s.add(b,e);self.s.edit(a,notes='hypothesis and caveat');self.s.move(a,1,-1);r=v.ProfileStore(self.path);self.assertEqual([e['figure_id'] for e in r.get(a)['entries']],['second','existing']);self.assertEqual(r.get(a)['notes'],'hypothesis and caveat');self.assertEqual(r.get(b)['notes'],'');self.assertEqual(len(list(self.path.parent.iterdir())),1)
 def test_crud_duplicate_references(self):
  a=self.s.create('A');self.s.add(a,{'kind':'photo','path':'existing.png'});b=self.s.duplicate(a,'Copy');self.s.edit(b,name='Renamed',notes='independent');self.assertEqual(self.s.get(a)['entries'],self.s.get(b)['entries']);self.assertEqual(self.s.get(a)['notes'],'');self.s.edit(b,entries=[]);self.s.delete(a);self.assertEqual(len(self.s.data['profiles']),1)
 def test_conflict_preserves_disk_and_memory(self):
  a=self.s.create('A');other=v.ProfileStore(self.path);other.edit(a,notes='external');raw=self.path.read_bytes()
  with self.assertRaises(RuntimeError):self.s.edit(a,notes='overwrite')
  self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.s.get(a)['notes'],'')
 def test_write_failure_rolls_back_memory(self):
  a=self.s.create('A');self.s.save=lambda:(_ for _ in ()).throw(OSError('read only'))
  with self.assertRaises(OSError):self.s.edit(a,notes='lost')
  self.assertEqual(self.s.get(a)['notes'],'')
 def test_path_traversal_absolute_symlink(self):
  with self.assertRaises(ValueError):v.contained(self.path.parent,'../outside.png')
  with self.assertRaises(ValueError):v.contained(self.path.parent,'/absolute.png')
  (self.path.parent/'link').symlink_to('/tmp')
  with self.assertRaises(ValueError):v.contained(self.path.parent,'link/outside.png')
 def test_missing_registry_reference_retains_note(self):
  class GUI:byid={};results=BASE/'results/archie_8193723_complete'
  e={'kind':'figure','figure_id':'retired','notes':'keep me'};p,w=v.resolve_entry(GUI(),e);self.assertIsNone(p);self.assertIn('RETIRED',w);self.assertEqual(e['notes'],'keep me')
 def test_exact_halfopen_boundaries(self):
  self.assertEqual([v.bin_number(x,.5) for x in ['-.5','-.0001','0','.499999','.5','1']],[-1,-1,0,0,1,2]);self.assertEqual(v.bin_number('.5',.5,'.25'),0)
 def test_query_count_and_hash_guard(self):
  catalog=v.scalar_catalog(BASE/'results/archie_8193723_complete');r=catalog[0];q=dict(schema_version=1,metric='signed',width_A2='.5',origin_A2='0',side='B',run='All',time_definition=r['time_definition'],reference_definition=r['reference_definition'],source_sha256=r['source_sha256'],bin_index=0);match,counts=v.evaluate_query(catalog,q);self.assertEqual(sum(counts.values()),156);self.assertEqual(len(match),counts.get(0,0));q['source_sha256']='changed'
  with self.assertRaises(ValueError):v.evaluate_query(catalog,q)
 def test_raw_requires_membrane(self):
  catalog=v.scalar_catalog(BASE/'results/archie_8193723_complete');r=catalog[0];q=dict(schema_version=1,metric='raw',width_A2=1,origin_A2=0,side='All',run='All',time_definition=r['time_definition'],reference_definition=r['reference_definition'],source_sha256=r['source_sha256'],bin_index=0)
  with self.assertRaises(ValueError):v.evaluate_query(catalog,q)
 def test_query_restart(self):
  a=self.s.create('A');q={'metric':'signed','source_sha256':'preserved'};self.s.edit(a,query=q);self.assertEqual(v.ProfileStore(self.path).get(a)['query'],q)
 def test_catalog_joins_selected_cohort(self):
  entries=v.photo_catalog(BASE/'results/archie_8193723_complete');self.assertGreaterEqual(len(entries),312);self.assertEqual(len({(e['pair_id'],e['side']) for e in entries}),312)
  for e in entries:self.assertEqual(e['path_status'],'LIVE_INVENTORY_PRESENT');self.assertNotIn('NEW10__',e['path']);self.assertIn('science_annotation',e)
 def test_invalid_profile_schema_preserves_bytes(self):
  raw=b'{"schema_version": 99, "profiles": []}';self.path.write_bytes(raw)
  with self.assertRaises(ValueError):v.ProfileStore(self.path)
  self.assertEqual(self.path.read_bytes(),raw)
class SourceAndNavigation(unittest.TestCase):
 def test_real_endpoint_bins_included_without_shift(self):
  selected,series,bins,frames,limits=p0.load_selected(BASE,BASE/'results/archie_8193723_complete');self.assertEqual(len(selected),156);self.assertEqual(len(series),312);self.assertEqual(len(bins),6240)
  for e in series.values():self.assertEqual(e['points'][-1][2],200);self.assertEqual(e['points'][-1][4],1);self.assertEqual(e['points'][0][2],105)
 def test_original_gui_compiles_and_profile_module_compiles(self):
  for p in [STAGE/'payload/review_metric_figures.py',STAGE/'payload/visual_comparison_profiles.py']:compile(p.read_text(),str(p),'exec')
 def test_navigation_folder_no_figure_selection(self):
  import review_metric_figures as gui
  class Tree:
   def selection(self):return ('family:temporal',)
  class Fake:tree=Tree();byid={'leaf':{}};current='leaf';show=lambda self,fid:(_ for _ in ()).throw(AssertionError('folder opened figure'))
  gui.FigureReviewGUI.tree_select(Fake())
 def test_axes_for_all_projected_derivatives(self):
  g=json.loads((BASE/'results/archie_8193723_complete/review_bundle.json').read_text())
  for f in g['figures']:
   if f['figure_id'].startswith('papl_selected_run_'):self.assertIn('40–90',f['axes']['y']);self.assertIn('combined',f['axes']['x'])
   if f.get('original_display'):
    for a in f['axes']['saved_axis_limits']:
     if 'APL' in a['ylabel']:self.assertEqual(a['ylim'],[40.0,90.0])
 def test_review_and_all_frozen_parameters_preserved(self):
  r=BASE/'results/archie_8193723_complete';original=os.environ.get('PFPML_ORIGINAL_BASE')
  if original:
   old=Path(original)/'results/archie_8193723_complete'
   for name in ['figure_review_state.json','data/pca_parameters.json','data/frozen_parameters.json']:self.assertEqual((r/name).read_bytes(),(old/name).read_bytes())
  else:
   report=json.loads((STAGE/p0.REPORT_NAME).read_text());self.assertEqual(v.sha(r/'figure_review_state.json'),report['state_sha256'])
   for name,h in report['frozen_sha256'].items():self.assertEqual(v.sha(r/name),h)
  self.assertEqual(len(json.loads((r/'data/pca_parameters.json').read_text())),12)
 def test_unaffected_registry_entries_exact(self):
  original=os.environ.get('PFPML_ORIGINAL_BASE')
  beforepath=Path(original)/'results/archie_8193723_complete/review_bundle.json' if original else Path(json.loads((STAGE/p0.REPORT_NAME).read_text())['backups'][0])
  before=json.loads(beforepath.read_text());after=json.loads((BASE/'results/archie_8193723_complete/review_bundle.json').read_text());by={f['figure_id']:f for f in after['figures']}
  p1={'pristine_papl_b','pristine_papl_m','pristine_papl_combined','endpoint_late','binned_frame_reconciliation','temporal_selected_trajectories'}
  for f in before['figures']:
   if f['figure_id'] not in p0.OLD_FIDS and f['figure_id'] not in p1:self.assertEqual(f,by[f['figure_id']])
if __name__=='__main__':unittest.main(verbosity=2)
