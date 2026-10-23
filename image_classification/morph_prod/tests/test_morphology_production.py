"""Focused numerical, identity, safety, recovery and end-to-end fixture tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import morphology_production as W
import morphology_geometry as G
from morphology_statistics import bh,paired_table


def fixture_dir(root):
    import MDAnalysis as mda
    d=root/'POPE75POPG25_W_WF'/'run_0'/'Na-Nb-Nc_80';d.mkdir(parents=True)
    nx=16;x,y=np.meshgrid((np.arange(nx)+.25)*16/nx,(np.arange(nx)+.35)*16/nx)
    xy=np.column_stack([x.ravel(),y.ravel()]);n=len(xy)
    u=mda.Universe.empty(2*n,n_residues=2*n,atom_resindex=np.arange(2*n),trajectory=True)
    u.add_TopologyAttr('names',['PO4']*(2*n));u.add_TopologyAttr('resnames',(['POPE']*192+['POPG']*64)*2)
    u.add_TopologyAttr('resids',np.arange(1,2*n+1));u.add_TopologyAttr('ids',np.arange(1,2*n+1))
    dims=np.array([160,160,200,90,90,90],float)
    for stage in ['Pull','Relax']:
        with mda.Writer(str(d/(stage+'.xtc')),n_atoms=2*n) as wr:
            for t in (range(1001) if stage=='Relax' else [0,1]):
                amp=.4*t/1000 if stage=='Relax' else 0
                u.atoms.positions=np.vstack([np.column_stack([xy,8+amp*np.cos(2*np.pi*xy[:,0]/8)]),np.column_stack([xy,12+amp*np.cos(2*np.pi*xy[:,0]/8)])])*10
                u.dimensions=dims;u.trajectory.ts.time=t*100
                if t==0:u.atoms.write(str(d/(stage+'.gro')))
                wr.write(u)
    return d


def row_for(d):
    ident=W.parse_system(d);key=W.physical_key(*ident)
    return {'physical_system_key':key,'manifest_row_id':W.digest(key)[:24],'molecule':ident[0],'nmol':ident[1],
            'bilayer':ident[2],'physical_run':ident[3],'molecule_type':'peptoid','length':3,
            'membrane_role':'bacterial','annotation_status':'labelled','manual_label':'curved','evidence_type':'STACKED',
            'system_dir':str(d),'expected_po4_total':512,'expected_membrane_total':512,
            **{k:str(d/v) for k,v in W.INPUTS.items()}}


class GeometryTests(unittest.TestCase):
    def coords(self,au=.9,al=.15):
        x,y=np.meshgrid((np.arange(16)+.25),(np.arange(16)+.35));xy=np.column_stack([x.ravel(),y.ravel()])
        return 10*np.vstack([np.column_stack([xy,8+al*np.cos(2*np.pi*xy[:,0]/8)]),np.column_stack([xy,12+au*np.cos(2*np.pi*xy[:,0]/8)])])
    def test_analytic_sinusoid_and_asymmetry(self):
        g=G.build_surface_geometry(self.coords(),[160,160,200,90,90,90]);m=G.geometry_metrics(g,512)
        for leaf,amp in [('upper',.9),('lower',.15)]:
            self.assertAlmostEqual(m[leaf+'_height_rms_nm'],amp/np.sqrt(2),places=11)
            x=g['xg'];k=2*np.pi/8
            expected=-amp*k*k*np.cos(k*x)/(2*(1+(amp*k*np.sin(k*x))**2)**1.5)
            np.testing.assert_allclose(g[leaf+'_mean_curvature_nm_inv'],expected,atol=1e-12)
        self.assertAlmostEqual(m['height_rms_upper_minus_lower_nm'],(.9-.15)/np.sqrt(2),places=11)
        self.assertEqual(m['height_rms_abs_upper_minus_lower_nm'],abs(m['height_rms_upper_minus_lower_nm']))
    def test_flat_and_periodic_translation(self):
        pos=self.coords(0,0);g=G.build_surface_geometry(pos,[160,160,200,90,90,90]);m=G.geometry_metrics(g,512)
        self.assertAlmostEqual(m['thickness_median_nm'],4,places=10)
        self.assertAlmostEqual(m['midplane_excess_area_fraction'],0,places=10)
        pos[:,2]=(pos[:,2]+95)%200
        n=G.geometry_metrics(G.build_surface_geometry(pos,[160,160,200,90,90,90]),512)
        for k in W.METRICS:np.testing.assert_allclose(m[k],n[k],atol=1e-10)
    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):G.build_surface_geometry(self.coords(),[160,160,200,80,90,90])
        x=self.coords();x[0,0]=np.nan
        with self.assertRaises(ValueError):G.build_surface_geometry(x,[160,160,200,90,90,90])
    def test_missing_not_zero(self):
        self.assertTrue(np.isnan(G.weighted_percentile([np.nan],[0],50)))


class IdentityTests(unittest.TestCase):
    def test_exact_run_parser(self):
        self.assertEqual(W.parse_system('/p/Boxes/2.1/POPC60CHOL40_W_WF/run_ML_cycle_02/Na-Nb-Nc_80'),('Na-Nb-Nc',80,'POPC60CHOL40_W_WF','run_ML_cycle_02'))
        with self.assertRaises(ValueError):W.parse_system('relative/run_0/Na_80')
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):W.unique(pd.DataFrame({'physical_system_key':['a','a']}))
    def test_pairs_exclude_other_runs_concentrations_and_additional(self):
        rows=[]
        for bilayer,role,run,n in [('B','bacterial','run_0',80),('M','mammalian','run_0',80),('A','additional','run_0',80),('M','mammalian','run_1',80),('B','bacterial','run_1',40)]:
            rows.append(dict(molecule='x',nmol=n,physical_run=run,membrane_role=role,bilayer=bilayer,length=3,molecule_type='peptoid',physical_system_key=f'{role}-{run}-{n}'))
        d=pd.DataFrame(rows);self.assertEqual(len(W.paired(d)),2)
        a=argparse.Namespace(tripeptoids_only=True,paired_only=True,bilayer=None,run=None,molecule=None)
        self.assertEqual(len(W.filter_rows(d,a)),2)
        a.bilayer=['A'];self.assertEqual(len(W.filter_rows(d,a)),0)
    def test_bh_missing(self):
        np.testing.assert_allclose(bh([.01,.04,np.nan,.03]),[.03,.04,np.nan,.04],equal_nan=True)
    def test_output_escape(self):
        with self.assertRaises(ValueError):W.new_dir('/tmp/unauthorised-morphology-output')


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=Path(tempfile.mkdtemp(prefix='synthetic_',dir=ROOT/'tests/fixtures'))
        cls.system=fixture_dir(cls.temp);cls.row=row_for(cls.system);cls.c=W.config()
        cls.manifest=cls.temp/'manifest.csv';W.write_csv(cls.manifest,[cls.row])
        cls.val=cls.temp/'validation'
        W.validate_cmd(argparse.Namespace(manifest=cls.manifest,config=None,out=cls.val,read_trajectories=True))
        cls.plan=cls.temp/'plan'
        W.prepare(argparse.Namespace(manifest=cls.manifest,config=None,validation=cls.val/'validation.json',out=cls.plan,draft=False,benchmark=False))
        W.run_system(argparse.Namespace(plan=cls.plan,index=0,execute=True,fixture=True))
        cls.p=json.loads((cls.plan/'plan.json').read_text());cls.task=W.task_path(cls.plan,cls.row)
    @classmethod
    def tearDownClass(cls):
        # Only disposable test-owned files created in setUpClass.
        shutil.rmtree(cls.temp)
    def test_read_only_inputs(self):
        self.assertEqual({p.name for p in self.system.iterdir()},{'Pull.gro','Pull.xtc','Relax.gro','Relax.xtc'})
    def test_resume_complete(self):
        before={p.name:W.sha(p) for p in self.task.iterdir()}
        W.run_system(argparse.Namespace(plan=self.plan,index=0,execute=True,fixture=True))
        self.assertEqual(before,{p.name:W.sha(p) for p in self.task.iterdir()})
        self.assertEqual(W.state(self.plan,self.row,self.p),'COMPLETE')
    def test_missing_and_invalid_result(self):
        r=json.loads((self.task/'result.json').read_text());original=(self.task/'result.json').read_bytes()
        try:
            del r['metrics']['endpoint__height_rms_nm'];(self.task/'result.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):W.check_result(self.task,self.row,self.p)
        finally:(self.task/'result.json').write_bytes(original)
    def test_input_changed(self):
        path=self.system/'Relax.gro';original=path.read_bytes();st=path.stat()
        try:
            path.write_bytes(original+b'\n');self.assertEqual(W.state(self.plan,self.row,self.p),'INPUT_CHANGED')
        finally:path.write_bytes(original);os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
    def test_failed_and_unknown_status(self):
        for filename,expected in [('FAILED.json','FAILED'),('RUNNING.json','RUNNING/UNKNOWN')]:
            newrow={**self.row,'manifest_row_id':'test_'+expected.split('/')[0]};d=W.task_path(self.plan,newrow);d.mkdir();(d/filename).write_text('{}')
            try:self.assertEqual(W.state(self.plan,newrow,self.p),expected)
            finally:shutil.rmtree(d)
    def test_collector_duplicate_directory(self):
        d=self.plan/'tasks'/'extra';d.mkdir()
        try:
            with self.assertRaises(ValueError):W.collect(argparse.Namespace(plan=self.plan,out=self.temp/'bad_collect'))
        finally:d.rmdir()
    def test_collect_merge_recovery(self):
        out=self.temp/'collected';W.collect(argparse.Namespace(plan=self.plan,out=out))
        m=pd.read_csv(out/'system_morphology.csv');self.assertEqual(len(m),1)
        self.assertAlmostEqual(m['morph__endpoint__height_rms_nm'].iloc[0],.4/np.sqrt(2),places=4)
        base=self.temp/'master.csv';W.write_csv(base,[{'physical_key':self.row['physical_system_key'],'APL_change_BOX':'1.23456789012345678','manual_label':'curved'}])
        W.merge(argparse.Namespace(master=base,morphology=out/'system_morphology.csv',out=self.temp/'merged'))
        merged=pd.read_csv(self.temp/'merged/master_with_morphology.csv',dtype=str)
        self.assertEqual(merged.APL_change_BOX.iloc[0],'1.23456789012345678')
        W.recover(argparse.Namespace(plan=self.plan,out=self.temp/'recovery'))
        self.assertEqual(len(pd.read_csv(self.temp/'recovery/recovery_candidates.csv')),0)
    def test_manifest_hash_guard(self):
        path=self.plan/'manifest.csv';original=path.read_bytes()
        try:
            path.chmod(0o644);path.write_bytes(original+b'\n')
            with self.assertRaises(ValueError):W.load_plan(self.plan)
        finally:path.write_bytes(original);path.chmod(0o444)
    def test_submission_default_is_dry(self):
        from unittest.mock import patch
        with patch('subprocess.run',side_effect=AssertionError('sbatch invoked')):
            W.submit(argparse.Namespace(plan=self.plan,submit=False,gate=None))
            with self.assertRaises(ValueError):W.submit(argparse.Namespace(plan=self.plan,submit=True,gate=None))



class ArchivedPilotTests(unittest.TestCase):
    def test_archived_synthetic_pilot_values(self):
        index=json.loads((ROOT/'tests/fixtures/archived_pilot/index.json').read_text())
        checked=0
        for entry in index:
            d=ROOT/entry['path'];total=512 if entry['role']=='bacterial' else 720
            for stage,table in [('Pull','reference_baseline.csv'),('Relax','reference_frames.csv')]:
                expected=pd.read_csv(d/table);u=W.open_universe(d/(stage+'.gro'),d/(stage+'.xtc'))
                try:
                    p=u.select_atoms('name PO4 and resname POPE POPG POPC')
                    for _,r in expected.iterrows():
                        u.trajectory[int(r['frame'])]
                        m=G.geometry_metrics(G.build_surface_geometry(p.positions,u.dimensions,4,32),total)
                        for key,val in m.items():
                            if key in expected:
                                np.testing.assert_allclose(val,r[key],rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=f'{entry}/{stage}/{key}')
                                checked+=1
                finally:u.trajectory.close()
        self.assertGreater(checked,1000)

class ExportRegressionTests(unittest.TestCase):
    def test_current_export_population_and_exclusions(self):
        h=ROOT/'tests/fixtures/historical'
        d,e=W.build_manifest(h/'canonical_annotations.csv',h/'canonical_apl.csv',W.config())
        self.assertEqual(len(d),395);self.assertEqual(len(e),76)
        self.assertEqual(len(W.paired(d)),362)
        self.assertEqual(len(W.paired(d[(d.molecule_type=='peptoid')&(d.length==3)])),334)
        self.assertEqual(d.bilayer.value_counts().to_dict(),{'POPE75POPG25_W_WF':188,'POPC60CHOL40_W_WF':183,'POPC80POPS20_W_WF':24})
        self.assertEqual(set(d.nmol),{80})
        only=e[e.reason=='NO_EXACT_SIMULATION'];self.assertEqual(len(only),3)
        self.assertEqual(set(only.molecule),{'Nk-Nfex-Nfex'})
    def test_primary_statistics_excludes_additional_bilayer(self):
        from morphology_statistics import run
        temp=Path(tempfile.mkdtemp(prefix='stats_',dir=ROOT/'tests/fixtures'))
        try:
            out=temp/'output'
            a=argparse.Namespace(master=ROOT/'tests/fixtures/historical/system_master.csv',out=out,pilot=True)
            run(a,W.new_dir,W.write_csv,W.write_json,W.sha)
            r=json.loads((out/'statistics.json').read_text());self.assertEqual(r['primary_systems'],334)
            self.assertEqual(r['primary_pairs'],167)
            self.assertNotIn('additional',set(pd.read_csv(out/'class_omnibus.csv').membrane))
        finally:shutil.rmtree(temp)

class CollectorRecoveryTests(unittest.TestCase):
    def test_duplicate_completed_recovery_is_rejected(self):
        # Reuse an independent fixture lifecycle; an identical plan offered twice is not an extra observation.
        temp=Path(tempfile.mkdtemp(prefix='recover_',dir=ROOT/'tests/fixtures'))
        try:
            system=fixture_dir(temp);row=row_for(system);manifest=temp/'manifest.csv';W.write_csv(manifest,[row])
            val=temp/'validation';W.validate_cmd(argparse.Namespace(manifest=manifest,config=None,out=val,read_trajectories=True))
            plan=temp/'plan';W.prepare(argparse.Namespace(manifest=manifest,config=None,validation=val/'validation.json',out=plan,draft=False,benchmark=False))
            W.run_system(argparse.Namespace(plan=plan,index=0,execute=True,fixture=True))
            with self.assertRaises(ValueError):W.collect(argparse.Namespace(plan=plan,recovery_plan=[plan],out=temp/'duplicate'))
        finally:shutil.rmtree(temp)

if __name__=="__main__":unittest.main(verbosity=2)
