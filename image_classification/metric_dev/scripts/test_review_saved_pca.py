"""Regression checks for the existing PCA review. No PCA fit is used."""
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

SCRIPT=Path(__file__).with_name('review_saved_pca.py')
ROOT=SCRIPT.resolve().parents[1]
SOURCE=ROOT/'results/archie_8193723_complete'
OUTPUT=SOURCE/'pca_review'
spec=importlib.util.spec_from_file_location('review_saved_pca',SCRIPT)
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def source_hashes():
    files = [
        SOURCE/'data/pca_parameters.json',
        SOURCE/'data/unified_metrics.csv',
        SOURCE/'data/pca_scores.csv',
        SOURCE/'data/pca_loadings.csv',
        SOURCE/'tables/pca_summary.csv',
        SOURCE/'tables/pca_associations.csv',
    ]
    return {
        str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


class FrozenProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before=source_hashes()
        cls.scores=pd.read_csv(SOURCE/'data/pca_scores.csv')
        cls.unified=pd.read_csv(SOURCE/'data/unified_metrics.csv')
        cls.parameters=json.loads((SOURCE/'data/pca_parameters.json').read_text())
        cls.loadings=pd.read_csv(SOURCE/'data/pca_loadings.csv')
        cls.projected,cls.validation=review.frozen_projection(cls.scores,cls.unified,cls.parameters,cls.loadings)

    def test_01_pc1_reconstructs(self):
        self.assertLess(self.validation.max_absolute_PC1_error.max(),1e-8)

    def test_02_pc2_reconstructs(self):
        self.assertLess(self.validation.max_absolute_PC2_error.max(),1e-8)

    def test_03_pc3_finite_and_population_preserved(self):
        self.assertTrue(np.isfinite(self.projected.PC3).all())
        self.assertEqual(self.projected.groupby('model').size().to_dict(),self.scores.groupby('model').size().to_dict())
        self.assertTrue(self.projected.groupby('model').size().eq(156).all())
        pd.testing.assert_frame_equal(self.projected[self.scores.columns],self.scores)

    def test_04_projection_failure_aborts(self):
        corrupt=self.scores.copy();corrupt.loc[0,'PC1']+=1e-5
        with self.assertRaisesRegex(ValueError,'reconstruction failed'):
            review.frozen_projection(corrupt,self.unified,self.parameters,self.loadings)

    def test_05_missing_features_aborts(self):
        with self.assertRaisesRegex(ValueError,'missing feature'):
            review.frozen_projection(self.scores,self.unified.drop(columns=['B__height__endpoint_vs_ref']),self.parameters,self.loadings)

    def test_06_duplicate_identity_aborts(self):
        with self.assertRaisesRegex(ValueError,'Duplicate pair_id'):
            review.frozen_projection(self.scores,pd.concat([self.unified,self.unified.iloc[[0]]]),self.parameters,self.loadings)

    def test_07_no_refit(self):
        from sklearn.decomposition import PCA
        with patch.object(PCA,'fit',side_effect=AssertionError('PCA fit forbidden')),patch.object(PCA,'fit_transform',side_effect=AssertionError('PCA refit forbidden')):
            review.frozen_projection(self.scores,self.unified,self.parameters,self.loadings)
        tree=ast.parse(SCRIPT.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['fit','fit_transform']]
        self.assertEqual(calls,[])

    def test_08_source_bytes_unchanged(self):
        self.assertEqual(self.before,source_hashes())

    def test_09_canonical_mapping_fixture_consistency(self):
        # Explicit synthetic fixture; not the absent canonical project mapping.
        observed=sorted(set(self.scores.label_B)|set(self.scores.label_M))
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT) as temp:
            file=Path(temp)/'synthetic_colour_fixture.json';fixture={c:'#123456' for c in observed};file.write_text(json.dumps({'colours':fixture}))
            colours,status=review.resolve_palette(observed,file)
            self.assertEqual(colours,fixture);self.assertTrue(status.startswith('supplied canonical'))
            self.assertEqual(review.resolve_palette(list(reversed(observed)),file)[0],fixture)

    def test_10_actual_canonical_project_colours(self):
        expected = {
            'flat_no_interaction': '#657685',
            'flat_interaction': '#4796BA',
            'bulge_flat': '#DD9A35',
            'curved': '#427943',
            'bulge_curved': '#963966',
            'pore_formed': '#111111',
        }
        colours, status = review.resolve_palette(list(expected))
        self.assertEqual(colours, expected)
        self.assertIn('project canonical phenotype colours', status)

    def test_11_unmapped_classes_abort(self):
        with self.assertRaisesRegex(ValueError,'No phenotype colour'):
            review.resolve_palette(['unknown_phenotype'])

    def test_12_paired_figures_actual_points_both_panels(self):
        counts=pd.read_csv(OUTPUT/'pca_plot_point_validation.csv')
        paired=counts[counts.model.str.contains('__paired__')]
        self.assertEqual(len(paired),4*4*2)
        self.assertTrue(paired.n_points.eq(156).all())
        for _,group in paired.groupby(['model','view']):self.assertEqual(set(group.label_membrane),{'B','M'})
        meta=json.loads((OUTPUT/'phenotype_colour_mapping.json').read_text())
        model='late_vs_ref__paired__full';block=self.projected[self.projected.model.eq(model)]
        audit=pd.read_csv(OUTPUT/'pca_model_summary.csv');row=audit[audit.model.eq(model)].iloc[0]
        fig,actual=review.plot_samples(block,row,['PC1','PC3'],meta['colours'],meta['markers'],meta['status'])
        self.assertEqual(len(fig.axes),2)
        for ax in fig.axes:self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections),156)
        review.plt.close(fig)

    def test_13_all_pc3_figures_nonempty(self):
        from PIL import Image
        for model in self.scores.model.unique():
            for view in ['PC1_PC2','PC1_PC3','PC2_PC3','PC1_PC2_PC3']:
                png=OUTPUT/'figures'/f'{model}__{view}.png';pdf=png.with_suffix('.pdf')
                self.assertGreater(png.stat().st_size,10000);self.assertGreater(pdf.stat().st_size,1000)
                with Image.open(png) as image:
                    pixels=np.asarray(image.convert('RGB'));self.assertGreater(pixels.std(),5)

    def test_14_three_component_cli_validates(self):
        run=subprocess.run([sys.executable,str(SCRIPT),'--source',str(SOURCE),'--components','3','--validate-only'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        evidence=json.loads(run.stdout);self.assertEqual(len(evidence['projection']),12)
        self.assertTrue(evidence['source_files_unchanged'])

    def test_15_two_component_cli_still_works(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as temporary:
            run=subprocess.run([sys.executable,str(SCRIPT),'--source',str(SOURCE),'--components','2','--permutations','19','--output',temporary],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            table=pd.read_csv(Path(temporary)/'pca_sample_scores_PC1_PC2_PC3.csv')
            self.assertEqual(len(table),1872);self.assertTrue(table.PC3.isna().all())
            self.assertEqual(len(list((Path(temporary)/'figures').glob('*__PC1_PC2.png'))),12)
            self.assertFalse(list((Path(temporary)/'figures').glob('*__PC1_PC3.png')))

    def test_16_three_component_outputs_consistent(self):
        table=pd.read_csv(OUTPUT/'pca_scores_PC1_PC2_PC3.csv')
        self.assertEqual(list(table.columns),['model','pair_id','molecule','label_B','label_M','PC1','PC2','PC3'])
        self.assertTrue(np.isfinite(table[['PC1','PC2','PC3']]).all().all())
        np.testing.assert_allclose(table.PC3,self.projected.PC3,rtol=1e-12,atol=1e-12)
        stats=pd.read_csv(OUTPUT/'pca_phenotype_separation.csv')
        self.assertEqual(len(stats),48);self.assertTrue(stats.BH_test_family_size.eq(48).all())
        self.assertEqual(set(stats.component),{'PC1','PC2','PC3'})
        points=pd.read_csv(OUTPUT/'pca_sample_scores_PC1_PC2_PC3.csv')
        self.assertTrue(points[points.model.str.contains('__paired__')].manual_phenotype_label.isna().all())


if __name__=='__main__':unittest.main(verbosity=2)
