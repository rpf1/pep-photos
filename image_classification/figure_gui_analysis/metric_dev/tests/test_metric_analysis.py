import sys,json,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metric_core import *

class MetricTests(unittest.TestCase):
 def test_origin_singularities(self):
  s=combine(np.array([0.,1.,0.]),np.array([0.,0.,1.]))
  self.assertEqual(s['contrast'][0],0);self.assertEqual(s['angle'][0],0);self.assertTrue(np.isnan(s['ratio_BM'][:2]).all());self.assertTrue(np.isnan(s['ratio_MB'][[0,2]]).all())
 def test_angle_contrast_equivalence(self):
  rng=np.random.default_rng(SEED);b,m=rng.random((2,1000));s=combine(b,m);np.testing.assert_allclose(s['angle'],4/np.pi*np.arctan(s['contrast']),atol=1e-12)
 def test_equal_effect_information_loss(self):
  s=combine(np.array([0.,1.]),np.array([0.,1.]));np.testing.assert_array_equal(s['difference'],[0,0]);self.assertEqual(s['magnitude'][0],0);self.assertAlmostEqual(s['magnitude'][1],np.sqrt(2))
 def test_pareto_utilities(self):
  rng=np.random.default_rng(SEED);b,m=rng.random((2,100));dom=dominance(b,m)
  for name,v in combine(b,m).items():
   if name in ['magnitude','ratio_MB']:continue
   self.assertFalse(np.any(dom&(v[:,None]+1e-12<v[None,:])),name)
 def test_weak_b_ratio_pathology(self):
  s=combine(np.array([1e-6,.8]),np.array([1e-12,.1]));self.assertGreater(s['ratio_BM'][0],s['ratio_BM'][1]);self.assertLess(s['desirability_w0.5'][0],s['desirability_w0.5'][1])
 def test_all_effects_pristine_zero(self):
  rng=np.random.default_rng(SEED);x=rng.normal(size=(100,4));names=['height','thickness','excess','curvature']
  for method in SCALERS:
   st=fit_state(x,names,method,np.arange(100)<10)
   for v in transform(np.zeros((1,4)),st).values():np.testing.assert_allclose(v,0,atol=1e-12)
 def test_parameter_roundtrip(self):
  rng=np.random.default_rng(SEED);x=rng.normal(size=(100,4));st=fit_state(x,['height','thickness','excess','curvature']);new=json.loads(json.dumps(st))
  for k,v in transform(x,st).items():np.testing.assert_allclose(v,transform(x,new)[k])
 def test_degenerate_scale_fails(self):
  with self.assertRaises(ValueError):fit_state(np.zeros((20,4)),['height','thickness','excess','curvature'])
 def test_bounded_effect_monotonic(self):
  d=np.array([0,1,10,1000]);x=bounded(d);self.assertTrue(((x>=0)&(x<1)).all());self.assertTrue((np.diff(x)>0).all())
 def test_cluster_bootstrap_keeps_rows_together(self):
  g=np.array(['a','a','b','b','c','c']);ix=cluster_draw(g,np.random.default_rng(SEED));self.assertTrue(all((ix==i).sum()==(ix==i+1).sum() for i in [0,2,4]))
 def test_generated_axis_rules(self):
  import pandas as pd
  p=Path(__file__).resolve().parents[1]/'results/archie_8193723_complete/plot_manifest.csv'
  self.assertTrue(p.exists(), 'Current completed manifest required')
  from figure_layout import validate_layout
  validate_layout(p.parent)
  for r in pd.read_csv(p).itertuples():
   a=json.loads(r.axis_limits)
   if r.axis_rule.startswith('shared_BM_y'):np.testing.assert_allclose(a[0]['ylim'],a[1]['ylim'])
   if r.axis_rule=='shared_BM_xy_equal':
    np.testing.assert_allclose(a[0]['xlim'],a[0]['ylim']);np.testing.assert_allclose(a[0]['xlim'],a[1]['xlim']);np.testing.assert_allclose(a[0]['ylim'],a[1]['ylim'])
   if r.axis_rule in ['effect_plane_0_1_equal','synthetic_0_1_equal']:
    np.testing.assert_allclose(a[0]['xlim'],[0,1]);np.testing.assert_allclose(a[0]['ylim'],[0,1])
   if r.axis_rule=='selectivity_minus1_plus1':np.testing.assert_allclose(a[0]['xlim'],[-1,1])
 def test_supervised_completion_intervals(self):
  import pandas as pd
  root=Path(__file__).resolve().parents[1]/'results/archie_8193723_complete'
  p=root/'tables/supervised_score_bootstrap_intervals.csv'
  self.assertTrue(p.exists(), 'Current completed result required')
  df=pd.read_csv(p);self.assertEqual(df.score.nunique(),8);self.assertEqual(len(df),156*8)
  self.assertTrue((df.ci_low<=df.ci_high).all());self.assertTrue(((df.ci_low>=0)&(df.ci_high<=1)).all());self.assertTrue((df.bootstrap==300).all())
 def test_decision_numeric_coverage(self):
  import pandas as pd
  p=Path(__file__).resolve().parents[1]/'results/archie_8193723_complete/tables/decision_table.csv'
  self.assertTrue(p.exists(), 'Current completed result required')
  df=pd.read_csv(p);self.assertFalse(df.score.duplicated().any());self.assertEqual(len(df),1768)
  for side in ['B','M']:
   self.assertTrue(df.loc[df.score.str.startswith('Effect_'+side+'__'),side+'_class_epsilon_squared'].notna().all())
if __name__=='__main__':unittest.main()
