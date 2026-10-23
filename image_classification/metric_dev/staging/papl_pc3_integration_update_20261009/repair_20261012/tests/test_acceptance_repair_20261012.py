"""Non-destructive regression for PFPML GUI acceptance repair, no Archie GUI needed."""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
sys.path.insert(0,str(STAGE))
import visual_comparison_profiles as v

class CandidateProfiles(unittest.TestCase):
 def setUp(self):
  self.dir=tempfile.TemporaryDirectory()
  self.path=Path(self.dir.name)/'visual_comparison_profiles.json'
  self.store=v.ProfileStore(self.path)
  self.rows=v.read_csv(STAGE/'visual_comparison_candidates.csv')
  self.photos=v.photo_catalog.__name__
  self.row=self.rows[0]
  self.entries=[{'kind':'photo','path':'mock.png','side':self.row['side'], 'pair_id':self.row['pair_id_1'],'notes':'original note'}]
 def tearDown(self):self.dir.cleanup()
 def test_unique_stable_ids(self):
  ids=[v.suggestion_id(row) for row in self.rows]
  self.assertEqual(len(ids),18)
  self.assertEqual(len(set(ids)),18)
  modified=dict(self.row,profile_name='renamed',hypothesis='edited wording')
  self.assertEqual(v.suggestion_id(self.row),v.suggestion_id(modified))
 def test_atomic_single_creation_and_reopening_identity(self):
  pid=self.store.create_from_suggestion(self.row,self.entries,'sourcehash')
  before=self.path.read_bytes()
  for i in range(3):
   matches=self.store.suggestion_matches(self.row)
   self.assertEqual([p['id'] for p in matches],[pid])
   self.assertEqual(self.path.read_bytes(),before)
  self.store.edit(pid,notes='User interpretation',entries=[{'kind':'figure','figure_id':'pristine_papl_b','notes':'custom'}])
  restarted=v.ProfileStore(self.path)
  self.assertEqual(restarted.suggestion_matches(self.row)[0]['id'],pid)
  self.assertEqual(restarted.get(pid)['notes'],'User interpretation')
  self.assertEqual(restarted.get(pid)['entries'][0]['notes'],'custom')
  with self.assertRaises(ValueError):restarted.create_from_suggestion(self.row,self.entries,'sourcehash')
  self.assertEqual(len(restarted.data['profiles']),1)
 def test_duplicate_is_independent(self):
  pid=self.store.create_from_suggestion(self.row,self.entries,'sourcehash')
  copy_id=self.store.duplicate(pid,'Independent copy')
  self.assertEqual(self.store.suggestion_matches(self.row)[0]['id'],pid)
  self.assertNotIn('suggestion',self.store.get(copy_id))
 def test_pre_patch_legacy_matches_only_strong_evidence(self):
  pid=self.store.create(self.row['profile_name'])
  self.store.edit(pid,purpose=self.row['hypothesis'],notes=v.candidate_notes(self.row)+'\nMy findings',entries=[dict(kind='photo',path='a.png',pair_id=k,side=self.row['side']) for k in (self.row['pair_id_1'],self.row['pair_id_2'])])
  self.assertEqual([p['id'] for p in self.store.suggestion_matches(self.row)],[pid])
  self.store.link_legacy_suggestion(pid,self.row,'sha')
  self.assertEqual(self.store.get(pid)['notes'],v.candidate_notes(self.row)+'\nMy findings')
  self.assertEqual(len(self.store.suggestion_matches(self.row)),1)
 def test_same_name_does_not_link_unrelated_profile(self):
  pid=self.store.create(self.row['profile_name'])
  self.store.edit(pid,notes='Completely unrelated data')
  self.assertEqual(self.store.suggestion_matches(self.row),[])
 def test_no_empty_or_partial_profiles(self):
  with self.assertRaises(ValueError):self.store.create_from_suggestion(self.row,[],'hash')
  self.assertEqual(len(self.store.data['profiles']),0)
 def test_atomic_failed_write(self):
  self.store.save=lambda:(_ for _ in ()).throw(OSError('I/O error'))
  with self.assertRaises(OSError):self.store.create_from_suggestion(self.row,self.entries,'sha')
  self.assertEqual(self.store.data['profiles'],[])
 def test_multiple_legacy_matches_visible_not_merged(self):
  a=self.store.create(self.row['profile_name']);b=self.store.create(self.row['profile_name'])
  for p in (a,b):self.store.edit(p,purpose=self.row['hypothesis'],notes=v.candidate_notes(self.row),entries=[dict(kind='photo',path='a.png',pair_id=k,side=self.row['side']) for k in (self.row['pair_id_1'],self.row['pair_id_2'])])
  self.assertEqual({p['id'] for p in self.store.suggestion_matches(self.row)},{a,b})
 def test_source_rows_preserved(self):
  self.assertEqual(self.row['time_window'].split(';')[0],'Relax-local 80–100 ns')

class FigureSelection(unittest.TestCase):
 def test_pristine_search_scientific_not_photo(self):
  class GUI:
   byid={'pristine_papl_b':{'figure_id':'pristine_papl_b', 'title':'Pristine bacterial pAPL', 'family':'pristine_time_series', 'png_path':'figures/pristine_papl_b.png','scientific_purpose':'Pristine projected APL','png_sha256':'abc','axes':{'y':'40–90 Å²'}}}
  entries=v.registered_figures(GUI())
  match=[e for e,t in entries if 'pristine' in t.casefold()]
  self.assertEqual(len(match),1)
  self.assertEqual(match[0]['kind'],'figure')
  self.assertEqual(match[0]['figure_id'],'pristine_papl_b')

if __name__=='__main__':unittest.main(verbosity=2)
