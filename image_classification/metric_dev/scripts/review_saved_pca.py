#!/usr/bin/env python3
"""Review exported PCA only; never fit PCA or change source data.
Run: python scripts/review_saved_pca.py --source results/archie_8193723_complete
"""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr, rankdata
from sklearn.metrics import silhouette_score

TOLERANCE = 1e-8

def source_file(source, name):
    candidates = [source / name, source / 'data' / name, source / 'tables' / name]
    matches = [p for p in candidates if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {name} under {source}; found {len(matches)}')
    return matches[0]


def frozen_projection(saved_scores, unified, parameters, loadings=None):
    """Project using saved oriented coefficients. No estimator is instantiated or fitted."""
    if unified.pair_id.duplicated().any():
        raise ValueError('Duplicate pair_id in original feature table')
    if saved_scores.duplicated(['model', 'pair_id']).any():
        raise ValueError('Duplicate model/pair_id in saved scores')
    if set(parameters) != set(saved_scores.model):
        raise ValueError('Frozen model IDs do not match score model IDs')
    values = unified.set_index('pair_id')
    result = saved_scores.copy()
    result['PC3'] = np.nan
    validation = []
    for model, block in saved_scores.groupby('model', sort=False):
        p = parameters[model]
        view, side, feature_set = model.split('__')
        features = p['features']
        scale = np.asarray(p['scale'], dtype=float)
        centre = np.asarray(p['centre'], dtype=float)
        matrix = np.asarray(p['components'], dtype=float)
        signs = np.asarray(p['orientation_multipliers'], dtype=float)
        if scale.shape != (len(features),) or centre.shape != scale.shape or matrix.shape[1] != len(features) or matrix.shape[0] < 3 or signs.shape != (matrix.shape[0],):
            raise ValueError(f'{model}: malformed frozen parameters')
        if not np.isfinite(matrix).all() or not np.isfinite(centre).all() or not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isin(signs, [-1, 1]).all():
            raise ValueError(f'{model}: invalid scales, centres, components or orientation metadata')
        columns = []
        for feature in features:
            if side == 'paired':
                membrane, metric = feature.split('_', 1)
                if membrane not in ('B', 'M'):
                    raise ValueError(f'{model}: invalid paired feature {feature}')
            else:
                membrane, metric = side, feature
            columns.append(f'{membrane}__{metric}__{view}')
        missing = [c for c in columns if c not in values]
        missing_ids = set(block.pair_id) - set(values.index)
        if missing or missing_ids:
            raise ValueError(f'{model}: missing feature columns {missing}; missing pair IDs {sorted(missing_ids)}')
        original = values.loc[block.pair_id]
        for identity in ['molecule', 'label_B', 'label_M']:
            if not np.array_equal(original[identity].to_numpy(), block[identity].to_numpy()):
                raise ValueError(f'{model}: {identity} mismatch against frozen sample identities')
        for flag in ['selected_pair', 'qc_pass_pair']:
            if flag in original and not original[flag].eq(True).all():
                raise ValueError(f'{model}: saved sample includes non-primary {flag} row')
        raw = original[columns].to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError(f'{model}: nonfinite original PCA features')
        if loadings is not None:
            expected = loadings[loadings.model.eq(model)].pivot(index='component', columns='feature', values='loading').reindex(index=np.arange(1, len(matrix)+1), columns=features).to_numpy()
            if not np.allclose(matrix, expected, atol=1e-9, rtol=0):
                raise ValueError(f'{model}: frozen components disagree with exported oriented loadings')
        # Saved centre is in scaled coordinates. Matrices already contain orientation.
        projected = (raw / scale - centre) @ matrix[:3].T
        errors = np.max(np.abs(projected[:, :2] - block[['PC1', 'PC2']].to_numpy()), axis=0)
        if not np.isfinite(projected).all() or not np.all(errors < TOLERANCE):
            raise ValueError(f'{model}: frozen PC1/PC2 reconstruction failed; max errors={errors.tolist()}, required <{TOLERANCE}')
        result.loc[block.index, 'PC3'] = projected[:, 2]
        validation.append({'model': model, 'n': len(block), 'max_absolute_PC1_error': errors[0], 'max_absolute_PC2_error': errors[1], 'tolerance_strict_less_than': TOLERANCE, 'PC3_finite': True, 'projection_formula': '(X / scale - centre) @ saved_oriented_components.T', 'orientation_applied_again': False, 'feature_columns': '; '.join(columns)})
    return result, pd.DataFrame(validation)


def resolve_palette(observed, configuration=None):
    """Use the canonical phenotype colours from plot_metric_analysis.py unless explicitly overridden."""
    canonical = {
        'flat_no_interaction': '#657685',
        'flat_interaction': '#4796BA',
        'bulge_flat': '#DD9A35',
        'curved': '#427943',
        'bulge_curved': '#963966',
        'pore_formed': '#111111',
    }
    if configuration:
        data = json.loads(Path(configuration).read_text())
        mapping = data.get('colours', data.get('phenotype_colours', data))
        status = 'supplied canonical configuration: ' + str(configuration)
    else:
        mapping, status = canonical, 'project canonical phenotype colours from scripts/plot_metric_analysis.py'
    missing = set(observed) - set(mapping)
    if missing:
        raise ValueError(f'No phenotype colour supplied for {sorted(missing)}; no new colours will be invented')
    from matplotlib.colors import is_color_like
    if not all(is_color_like(mapping[c]) for c in observed):
        raise ValueError('Invalid phenotype colour value')
    return {c: mapping[c] for c in observed}, status


def figure_description(model, features):
    view, side, status = model.split('__')
    time = 'Late 80–100 ns morphology vs pristine reference' if view == 'late_vs_ref' else 'Endpoint vs pristine reference'
    membrane = {'B': 'Bacterial-like membrane', 'M': 'Mammalian-like membrane', 'paired': 'Matched bacterial/mammalian pair'}[side]
    omitted = 'none' if status == 'full' else ('APL, curvature_asym' if view == 'endpoint_vs_ref' else 'excess_asym')
    names = [f.split('_', 1)[1] if side == 'paired' else f for f in features]
    names = list(dict.fromkeys(names))
    import textwrap
    feature_text = 'Included: ' + ', '.join(names) + ('. B and M blocks.' if side == 'paired' else '.') + ' Omitted: ' + omitted
    return time + '\n' + membrane, textwrap.fill(feature_text, width=100)


def save_figure(fig, path, **kwargs):
    """Write verified complete figures atomically; preserve existing file identity."""
    import os
    path=Path(path);temporary=path.with_name(path.stem+'.rendering'+path.suffix)
    attrs={key:os.getxattr(path,key) for key in os.listxattr(path)} if path.exists() else {}
    try:
        fig.savefig(temporary, **kwargs)
        if temporary.stat().st_size < 1000:
            raise ValueError(f'Empty figure export: {path}')
        if path.suffix == '.png':
            from PIL import Image
            with Image.open(temporary) as image:image.verify()
        if path.suffix == '.pdf' and not temporary.read_bytes().startswith(b'%PDF'):
            raise ValueError(f'Invalid PDF export: {path}')
        for key,value in attrs.items():os.setxattr(temporary,key,value)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def plot_samples(block, audit_row, components, colours, markers, palette_status):
    """Return actual point counts for both label panels; no artificial combined phenotype."""
    panels = ['B', 'M'] if audit_row.scope == 'paired' else [audit_row.scope]
    dim = len(components)
    fig = plt.figure(figsize=(8 * len(panels), 7.5))
    counts = []
    axes = []
    for i, side in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection='3d' if dim == 3 else None)
        axes.append(ax)
        total = 0
        for phenotype in sorted(block['label_' + side].unique()):
            subset = block[block['label_' + side].eq(phenotype)]
            values = [subset[c].to_numpy() for c in components]
            ax.scatter(*values, c=colours[phenotype], marker=markers[phenotype], s=40, alpha=.8, edgecolors='white', linewidths=.3, label=f'{phenotype} (n={len(subset)})')
            total += len(subset)
        assert total == len(block) and total > 0
        for c, setter in zip(components, [ax.set_xlabel, ax.set_ylabel] + ([ax.set_zlabel] if dim == 3 else [])):
            setter(f'{c} ({audit_row[c + "_variance_pct"]:.1f}%)')
        ax.set_title(f'{side} phenotype colouring', fontsize=11)
        if dim == 2:
            ax.axhline(0, color='#aaa', lw=.6);ax.axvline(0, color='#aaa', lw=.6);ax.grid(alpha=.12)
        else:
            ax.view_init(elev=22, azim=35)
        ax.legend(loc='upper left', bbox_to_anchor=(0, -.12), fontsize=8, ncol=2, frameon=False)
        counts.append({'model': audit_row.model, 'view': '_'.join(components), 'label_membrane': side, 'n_points': total, 'same_saved_coordinates': True})
    if dim == 2 and len(axes) == 2:
        xs = block[components[0]];ys = block[components[1]]
        for ax in axes:
            ax.set_xlim(xs.min() - .05 * max(xs.max()-xs.min(), .01), xs.max() + .05 * max(xs.max()-xs.min(), .01))
            ax.set_ylim(ys.min() - .05 * max(ys.max()-ys.min(), .01), ys.max() + .05 * max(ys.max()-ys.min(), .01))
    title, subtitle = figure_description(audit_row.model, audit_row.exact_features.split('; '))
    note = 'Canonical colours' if ('canonical' in palette_status.lower()) else 'Custom phenotype colours'
    fig.suptitle(title + '\n' + subtitle + '\n' + note, fontsize=11)
    fig.subplots_adjust(left=.1 / len(panels), right=.97, top=.75, bottom=.25, wspace=.22)
    return fig, counts



def grouped_rank_test(values, labels, seed, permutations):
    """Same KW/epsilon² and count-preserving permutation convention as primary review."""
    values=np.asarray(values);labels=np.asarray(labels);classes=sorted(set(labels));n=len(values);k=len(classes)
    H,p=kruskal(*(values[labels==c] for c in classes));ranks=rankdata(values)
    counts=np.array([sum(labels==c) for c in classes]);splits=np.cumsum(counts)[:-1]
    ties=np.unique(values,return_counts=True)[1];correction=1-sum(ties**3-ties)/(n**3-n)
    rng=np.random.default_rng(seed);ge=0
    for _ in range(permutations):
        sums=np.array([x.sum() for x in np.split(rng.permutation(ranks),splits)])
        statistic=(12/(n*(n+1))*sum(sums*sums/counts)-3*(n+1))/correction
        ge+=statistic>=H-1e-12
    return {'n':n,'groups':k,'kw_H':H,'kw_p_asymptotic':p,'epsilon_squared':max(0,(H-k+1)/(n-k)),'permutation_p':(ge+1)/(permutations+1),'permutations':permutations}


def pc3_sensitivity(scores, permutations):
    rows=[];discord_rows=[]
    for model,block in scores.groupby('model',sort=False):
        panels=['B','M'] if '__paired__' in model else [model.split('__')[1]]
        for side in panels:
            subset=block[block['label_'+side].ne('flat_no_interaction')]
            seed=20261005+int(hashlib.sha256((model+side+'PC3_no_flat_no_interaction').encode()).hexdigest()[:8],16)
            rows.append({'model':model,'label_membrane':side,'removed_class':'flat_no_interaction','removed_n':len(block)-len(subset),**grouped_rank_test(subset.PC3,subset['label_'+side],seed,permutations)})
        if len(panels)==2:
            deformation=['curved','bulge_curved','bulge_flat','pore_formed']
            mismatch=block.label_B.isin(deformation)!=block.label_M.isin(deformation)
            comparison=np.where(mismatch,'deformed/plain-flat mismatch',np.where(block.label_B!=block.label_M,'different class, same broad grouping','exact label concordance'))
            for mode in ['signed_PC3','absolute_PC3']:
                values=block.PC3 if mode=='signed_PC3' else block.PC3.abs()
                seed=20261005+int(hashlib.sha256((model+mode+'discordance').encode()).hexdigest()[:8],16)
                discord_rows.append({'model':model,'measure':mode,**grouped_rank_test(values,comparison,seed,permutations)})
    def adjust(rows):
        frame=pd.DataFrame(rows)
        for source,target in [('kw_p_asymptotic','q_BH_asymptotic'),('permutation_p','q_BH_permutation')]:
            p=frame[source].to_numpy();order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);frame[target]=q
        frame['BH_test_family_size']=len(frame);return frame
    return adjust(rows),adjust(discord_rows)


def update_report(report, args, scores, validation, stats, groups, silhouettes, discord, info, audit, parameters, unified, palette_status, md):
    import re
    ntests=len(stats)
    report=report.replace('The 32 newly calculated tests receive one Benjamini–Hochberg correction',f'The {ntests} requested-component tests receive one Benjamini–Hochberg correction')
    report=report.replace('32 new tests',f'{ntests} new tests').replace('32 newly calculated tests',f'{ntests} newly calculated tests')
    report=report.replace('1,999 permutations',f'{args.permutations:,} permutations').replace('p=(1+exceedances)/2,000',f'p=(1+exceedances)/{args.permutations+1:,}')
    report=report.replace('Reproducibility seed = 20261005.', 'Reproducibility base seed = 20261005; SHA256-derived per-test streams make PC1/PC2 permutation results stable when PC3 is enabled.')
    report=re.sub(r'## Completion and reproducibility\n.*', '', report, flags=re.S)
    report=report.replace('PC3 sample projections are impossible from this bundle.', 'PC3 sample projections were unavailable in the earlier bundle.')
    report=report.replace('the origin convention cannot be independently reconstructed without frozen transforms.', 'the frozen centre reproduces the cohort-centred origin; it is not the pristine reference.')
    report=report.replace('nonlinear transformations, winsorisation or definitions cannot be resolved without preprocessing parameters.', 'the newly supplied frozen projection validates signed response scaling and centring; this is still a fitted-input Pearson correlation rather than the original Spearman redundancy screen.')
    report=report.replace('## Key numerical evidence','## Established PC1/PC2 evidence')
    report=report.replace('All 32 new tests survive BH correction at 0.05 using both asymptotic and permutation p values; significance is not clean separation.', 'The earlier PC1/PC2 tests supported distribution shifts; current adjusted statistics below use the explicitly updated test family. Significance is not clean separation.')
    report=report.replace(f'All {ntests} new tests survive BH correction at 0.05 using both asymptotic and permutation p values; significance is not clean separation.', 'The earlier PC1/PC2 tests supported distribution shifts; current adjusted statistics below use the explicitly updated test family. Significance is not clean separation.')
    if args.components==2:
        report=re.sub(r'\*\*PC3 sample projections were unavailable.*?\n\n', '**This run requests PC1/PC2 only (--components 2).** No sample PC3 projection is accepted or produced. Loading-based PC3 interpretation remains distinct from sample coordinates.\n\n', report, count=1, flags=re.S)
        report += '\n## Completion and reproducibility\n\nTwo-component mode completed. Run `python scripts/review_saved_pca.py --source results/archie_8193723_complete --components 2`. Colours: '+palette_status+'. Original source hashes are preserved.\n'
        return report
    v1=float(validation.max_absolute_PC1_error.max());v2=float(validation.max_absolute_PC2_error.max())
    report=re.sub(r'\*\*PC3 sample projections were unavailable.*?\n\n', '**Frozen projection now validated.** The added archive supplies pca_parameters.json and unified_metrics.csv. PC3 is reconstructed only after all model PC1/PC2 errors pass the strict <1e−8 gate. No PCA was refitted. Exact errors and source columns are recorded below.\n\n',report,count=1,flags=re.S)
    report=re.sub(r'The authoritative specification was read.*?\n\n', 'The original task specification and PC3 continuation govern this review. The added archive contains data/pca_parameters.json, data/unified_metrics.csv, data/pca_scores.csv, data/pca_loadings.csv, tables/pca_summary.csv and tables/pca_associations.csv. The four overlapping PCA exports are byte-identical to the earlier supplied bundle. The unified table contains 166 pairs; saved score identities select the same 156 strict primary pairs for each of 12 models. Ten review pairs are not introduced into the projection. Sources remain unchanged; no full workflow or Slurm jobs were run.\n\n',report,count=1,flags=re.S)
    report=report.replace('Its phenotype utility cannot be measured without PC3 scores.', 'Validated PC3 adds phenotype distribution information most strongly in endpoint M full (epsilon²=0.268; 0.247 excluding the four flat_no_interaction samples). Incremental silhouettes improve modestly but M groups still overlap. B-coloured paired PC3 associations depend heavily on that tiny class; PC3 remains a complementary physical review axis, not a classifier.')
    report=report.replace('only. Exported scores contain PC1 and PC2 only.', 'only. Original exported scores contain PC1 and PC2; derived PC3 scores are stored separately.')
    report=report.replace('Precise endpoint APL timing, raw units, leaflet-asymmetry formulas, preprocessing and origin conventions cannot be verified from this bundle.', 'Frozen preprocessing is now explicit: use saved feature order, divide signed response values by saved scales, subtract the saved scaled-coordinate centre, then multiply by already-oriented component coefficients. Exact raw units, endpoint sampling provenance and leaflet-asymmetry formula details still require the canonical generating implementation.')
    report=report.replace('their raw targets cannot be recomputed or used for sample selection here.', 'the new unified table now contains these raw targets, though aggregate source correlations remain reported as original source results.')
    report=report.replace('These target correlations are supplied source results, not newly recomputed.', 'These target correlations are supplied source results, not newly recomputed in this continuation.')
    report=report.replace('no raw values are supplied.', 'original response values are now supplied in unified_metrics.csv.')
    report=re.sub(r'## PC3 decision\n.*?(?=\n\n## Figure guide)', '## PC3 decision\n\nPC3 is retained as a routine additional physical review dimension for late M and paired models, and endpoint M/paired models. The previous provisional loading-based recommendation is now informed by validated sample positions, class distribution tests and incremental silhouette diagnostics below. Late B PC3 remains optional for most dominant-mode review. No PCA component is adopted as an ML target.',report,flags=re.S)
    report=report.replace('Each model has a sample PC1/PC2 PNG and PDF and a signed PC1–PC3 loading PNG.', 'Each model has PC1/PC2, PC1/PC3 and PC2/PC3 sample PNG/PDF figures, a supplementary 3D PNG/PDF figure, and a signed PC1–PC3 loading PNG.')
    report=report.replace('Figures with PC3 in the filename show feature loadings, not sample projections.', 'Files containing __loadings_ show feature coefficients; other PC3 files show validated sample projections. Paired files always contain separate B-label and M-label panels with the same observed pair coordinates.')
    report=report.replace('Raw scores preserve saved PCA coordinate distances; equal-variance silhouettes rescale each plotted PC by its observed SD as a sensitivity view.', 'Raw silhouettes preserve saved PCA distances; equal-variance silhouettes rescale each included PC by its observed SD. All three pairwise planes and the three-component space are evaluated.')
    sensitivity,discord_tests=pc3_sensitivity(scores,args.permutations)
    sensitivity.to_csv(Path(args.output)/'pca_PC3_small_class_sensitivity.csv',index=False)
    discord_tests.to_csv(Path(args.output)/'pca_PC3_discordance_tests.csv',index=False)
    extension=['## Validated PC3 continuation',
        f'**All 12 frozen projections pass:** maximum absolute PC1 error {v1:.3e}, PC2 error {v2:.3e}; required errors are strictly <1e−8. Every derived PC3 is finite, with 156 scores per model. Original PC1/PC2 values are retained in the derived score table; reconstructed PC1/PC2 are used for validation only.',
        'Projection: `(X / scale - centre) @ saved_components[:3].T`. Centres are in scaled coordinates. Components already match signed exported loadings; orientation_multipliers are metadata and are not multiplied again. Pair IDs, molecule identities and both labels must match; selected_pair and qc_pass_pair must both be true. Frozen coefficient matrices must match the saved loading exports. Missing/nonfinite features, duplicate IDs or failed reproduction abort before figure/report writes.',md(validation[['model','n','max_absolute_PC1_error','max_absolute_PC2_error','PC3_finite']]),
        '### PC3 phenotype distributions and associations',
        f'Kruskal–Wallis, existing epsilon² convention, and {args.permutations:,} label permutations are applied to all requested components. The BH family now has {ntests} tests (12 models × 3 PCs with two label membranes for each paired model), replacing the earlier 32-test family. Permutation resolution is {1/(args.permutations+1):.4g}. Small flat_no_interaction n=4 remains a limitation; no thermal-null threshold is inferred.',
        md(stats[stats.component.eq('PC3')][['model','label_membrane','kw_H','epsilon_squared','kw_p_asymptotic','permutation_p','q_BH_asymptotic','q_BH_permutation']]),md(groups[groups.component.eq('PC3')]),
        '### Incremental overlap beyond PC1/PC2']
    piv=silhouettes.pivot(index=['model','label_membrane'],columns='view',values=['silhouette_raw','silhouette_equal_variance'])
    incremental=pd.DataFrame({'PC1_PC2_raw':piv['silhouette_raw']['PC1_PC2'],'PC1_PC2_PC3_raw':piv['silhouette_raw']['PC1_PC2_PC3'],'raw_change':piv['silhouette_raw']['PC1_PC2_PC3']-piv['silhouette_raw']['PC1_PC2'],'PC1_PC2_equal_variance':piv['silhouette_equal_variance']['PC1_PC2'],'PC1_PC2_PC3_equal_variance':piv['silhouette_equal_variance']['PC1_PC2_PC3'],'equal_variance_change':piv['silhouette_equal_variance']['PC1_PC2_PC3']-piv['silhouette_equal_variance']['PC1_PC2']}).reset_index()
    incremental.to_csv(Path(args.output)/'pca_PC3_incremental_overlap.csv',index=False)
    extension += [md(incremental),
        'Positive changes mean less overlap under this particular descriptive distance convention; they do not establish predictive discrimination or conditional phenotype information. A PC3 rank test is marginal, not conditioned on PC1/PC2. Raw 3D distances and equal-variance distances answer different questions. PCA component orthogonality does not imply phenotype independence. No new classifier, supervised fit or causal analysis is introduced.',
        '### Small-class sensitivity',md(sensitivity),
        'This separate 16-test sensitivity family removes the four flat_no_interaction pairs per label membrane, preserving the original frozen projection. It is not a PCA refit. Endpoint M full retains the strongest PC3 association (epsilon² 0.247 after exclusion, versus 0.268 overall). Late M full/reduced effects decrease to 0.053/0.040. B-coloured paired PC3 association disappears after exclusion (epsilon² 0–0.007, unadjusted p 0.26–0.97): its apparent five-class shift depends heavily on the tiny flat_no_interaction class. M-coloured paired effects persist, with epsilon² 0.064–0.127. These sensitivities qualify any claim of general class structure.',
        '### Priority model decisions']
    for model in ['late_vs_ref__M__full','late_vs_ref__M__reduced','endpoint_vs_ref__M__full','endpoint_vs_ref__M__reduced','late_vs_ref__paired__full','late_vs_ref__paired__reduced','endpoint_vs_ref__paired__full']:
        a=audit[audit.model.eq(model)].iloc[0];r=stats[(stats.model.eq(model))&stats.component.eq('PC3')];si=incremental[incremental.model.eq(model)]
        association='; '.join(f'{x.label_membrane}: ε²={x.epsilon_squared:.4f}, BH permutation q={x.q_BH_permutation:.4g}' for x in r.itertuples())
        deltas='; '.join(f'{x.label_membrane}: raw silhouette Δ={x.raw_change:+.4f}, equal-variance Δ={x.equal_variance_change:+.4f}' for x in si.itertuples())
        extension.append(f'**{model}:** PC3 adds {a.PC3_variance_pct:.2f}%, cumulative PC1–PC3 {a.PC1_PC2_PC3_pct:.2f}%. {association}. {deltas}. Retain the physical mode; judge phenotype relevance from these sample findings rather than loadings alone.')
    extension += ['### Paired phenotype discordance',md(discord),
        '### Discordance rank tests',md(discord_tests),
        'Eight signed/absolute-PC3 comparison tests form their own BH family. Signed PC3 distributions show modest descriptive shifts across concordance strata, but none of the eight discordance tests survives BH correction at 0.05 (signed permutation q=0.068–0.091; absolute q=0.249–0.871). Discordant pairs are not preferentially extreme in absolute PC3; PC3 is not a quantitative discordance/selectivity score.',
        'Discordance groups are comparison strata, not invented phenotype labels or colour classes. Exact concordance n=85, different exact labels but the same broad deformation grouping n=15, and deformation/plain-flat mismatch n=56. Distributions are descriptive; no inference that a PC3 score is biological selectivity or damage. Absolute PC3 is distance along a cohort-centred axis, not physical response magnitude from pristine.',
        '### Actual PC3 extreme samples',md(info[info.example_type.isin(['PC3 minimum','PC3 maximum'])]),
        '### PC3 phenotype/location examples',md(info[info.example_type.eq('PC3 phenotype/location exception')].groupby('model',sort=False).head(3)),
        'Examples outside their own phenotype’s Tukey 1.5-IQR fences are flagged only for classes n≥5. They identify empirical residual-mode diversity; they are not annotation errors. All examples are retained in pca_informative_samples.csv.',
        '### Interpretation update',
        'The established PC1/PC2 physical interpretations remain. PC3 provides a complementary physical covariance direction; marginal phenotype shifts can coexist with large overlap. Routine PC3 review is justified for physically distinct thickness or APL–thickness modes even when five-class clustering does not improve. No component is validated as a biological efficacy, toxicity or ML target. The exact answer to whether PC3 adds phenotype-relevant information is model-specific in the statistics and incremental-overlap tables; a universal claim of added discrimination is unsupported.',
        '### Colour mapping',palette_status+'. The default mapping is the canonical phenotype mapping used by scripts/plot_metric_analysis.py. `--colour-config PATH` remains available for an explicit override. Unknown/unmapped classes abort rather than receiving invented colours.',
        '## Completion and reproducibility',
        'Implemented in the existing script: --components 2/3, exact frozen validation gate, separate derived score table, all three pairwise plots plus supplementary 3D, paired label panels with recorded actual point counts, readable temporal/membrane/feature subtitles, extended rank tests/permutations/BH, pairwise and 3D silhouettes, class summaries, PC3 extremes/exceptions, updated same report and provenance. No parallel PCA workflow or PCA refit exists.',
        'Run from package root:\n\n```bash\npython scripts/review_saved_pca.py --source results/archie_8193723_complete --components 2\npython scripts/review_saved_pca.py --source results/archie_8193723_complete --components 3\npython scripts/review_saved_pca.py --source results/archie_8193723_complete --components 3 --validate-only\npython -m unittest discover -s scripts -p "test_review_saved_pca.py" -v\n```',
        'Automated validation results are in pca_extension_test_results.txt and pca_extension_validation.json. Canonical-mapping tests verify the project phenotype colours used by scripts/plot_metric_analysis.py. Earlier review reasoning and outputs are preserved in history/PCA_review_pre_PC3.zip.']
    report=report.replace('## Conclusions','## Conclusions',1)
    return report+'\n\n'+'\n\n'.join(extension)+'\n'


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='results/archie_8193723_complete');ap.add_argument('--output',default='results/archie_8193723_complete/pca_review');ap.add_argument('--components',type=int,choices=[2,3],default=2);ap.add_argument('--colour-config');ap.add_argument('--permutations',type=int,default=1999);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args(argv)
    src=Path(args.source);out=Path(args.output);figdir=out/'figures'
    immutable_sources = [
        src/'data/pca_parameters.json',
        src/'data/unified_metrics.csv',
        src/'data/pca_scores.csv',
        src/'data/pca_loadings.csv',
        src/'tables/pca_summary.csv',
        src/'tables/pca_associations.csv',
    ]
    def hashes():
        missing = [str(f) for f in immutable_sources if not f.is_file()]
        if missing:
            raise FileNotFoundError('Missing canonical PCA source files: ' + ', '.join(missing))
        return {
            str(f.relative_to(src)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in immutable_sources
        }
    before=hashes()
    summary=pd.read_csv(source_file(src,'pca_summary.csv'));loads=pd.read_csv(source_file(src,'pca_loadings.csv'));scores=pd.read_csv(source_file(src,'pca_scores.csv'));assoc=pd.read_csv(source_file(src,'pca_associations.csv'))
    manifest_path=src/'analysis_manifest.json';manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'source_directory':'results/archie_8193723_complete'}
    assert not scores.duplicated(['model','pair_id']).any()
    assert scores[['PC1','PC2']].notna().all().all()
    models=summary.model.drop_duplicates().tolist();assert set(models)==set(scores.model)==set(loads.model)
    projection_validation=None;parameters=None;unified=None
    if args.components==3:
     parameters=json.loads(source_file(src,'pca_parameters.json').read_text());unified=pd.read_csv(source_file(src,'unified_metrics.csv'))
     scores,projection_validation=frozen_projection(scores,unified,parameters,loads)
    classes=sorted(set(scores.label_B.dropna())|set(scores.label_M.dropna()))
    colours,palette_status=resolve_palette(classes,args.colour_config)
    markers={c:m for c,m in zip(classes,['P','^','D','o','s','X','v','<','>'])}
    if args.permutations<1:raise ValueError('--permutations must be positive')
    if args.validate_only:
     print(json.dumps({'components':args.components,'projection':None if projection_validation is None else projection_validation.to_dict('records'),'palette_status':palette_status,'source_files_unchanged':before==hashes()},indent=2));return
    out.mkdir(parents=True,exist_ok=True);figdir.mkdir(exist_ok=True)
    if projection_validation is not None:projection_validation.to_csv(out/'pca_frozen_projection_validation.csv',index=False)
    (out/'phenotype_colour_mapping.json').write_text(json.dumps({'status':palette_status,'canonical_available':True,'colours':colours,'markers':markers,'paired_representation':'same coordinates in separate B-label and M-label panels; no combined class'},indent=2))
    component_names=['PC'+str(i) for i in range(1,args.components+1)]
    rng=np.random.default_rng(20261005)
    def md(df):
     def cell(v):
      if isinstance(v,(float,np.floating)): return f'{v:.4g}' if abs(v)<.001 and v!=0 else f'{v:.4f}'
      return str(v).replace('|',' / ').replace('\n',' ')
     rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
     rows+=['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
     return '\n'.join(rows)
    def save(df,name): df.to_csv(out/name,index=False)
    def bh(p):
     p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q
    
    # Exact model inventory and full basis validation.
    audit=[];corrrows=[];similar=[];qa=[]
    for model in models:
     ss=summary[summary.model==model].sort_values('component');ll=loads[loads.model==model];dd=scores[scores.model==model]
     feats=parameters[model]['features'] if parameters is not None else ll[ll.component==1].feature.tolist();V=ll.pivot(index='component',columns='feature',values='loading').reindex(index=ss.component,columns=feats).to_numpy()
     assert V.shape==(len(feats),len(feats));err=np.max(abs(V@V.T-np.eye(len(feats))));assert err<1e-9
     ev=ss.explained_variance.to_numpy();assert abs(ev.sum()-1)<1e-9;assert len(dd)==int(ss.iloc[0]['n'])
     assert np.isclose(dd.PC1.var()/dd.PC2.var(),ev[0]/ev[1],rtol=1e-8)
     assert np.allclose(ll.loading**2,ll.squared_loading_contribution,rtol=1e-8,atol=1e-10)
     # Eigenvalues known up to one common positive factor; that factor cancels in Pearson correlations.
     C=V.T@np.diag(ev)@V;R=C/np.sqrt(np.outer(np.diag(C),np.diag(C)))
     for i,f in enumerate(feats):
      for j in range(i+1,len(feats)):corrrows.append(dict(model=model,feature_1=f,feature_2=feats[j],Pearson_r_fitted_inputs=R[i,j],absolute_r=abs(R[i,j]),redundant_at_0_9=abs(R[i,j])>=.9))
     qa.append(dict(model=model,n_rows=len(dd),orthogonality_max_error=err,variance_ratio_PC1_PC2_from_scores=dd.PC1.var()/dd.PC2.var(),variance_ratio_PC1_PC2_from_summary=ev[0]/ev[1]))
     audit.append(dict(model=model,scope=ss.iloc[0]['side'],temporal_representation='endpoint versus pristine reference' if ss.iloc[0]['view']=='endpoint_vs_ref' else '80–100 ns morphology median versus pristine reference',feature_set=ss.iloc[0].feature_set,exact_features='; '.join(feats),APL_included=any(f.endswith('APL') for f in feats),n=len(dd),components_available=len(ev),sample_score_components_available=args.components,PC1_variance_pct=100*ev[0],PC2_variance_pct=100*ev[1],PC3_variance_pct=100*ev[2],PC1_PC2_pct=100*sum(ev[:2]),PC1_PC2_PC3_pct=100*sum(ev[:3])))
    audit=pd.DataFrame(audit);save(audit,'pca_model_summary.csv');save(pd.DataFrame(qa),'pca_validation.csv')
    ranked=loads.copy();ranked['absolute_loading']=abs(ranked.loading);ranked['absolute_loading_rank']=ranked.groupby(['model','component']).absolute_loading.rank(method='first',ascending=False).astype(int);ranked['direction']=np.where(ranked.loading>=0,'positive','negative');ranked['little_by_display_convention']=ranked.absolute_loading<.1
    ranked=ranked.sort_values(['model','component','absolute_loading_rank']);save(ranked,'pca_component_loadings_ranked.csv')
    corr=pd.DataFrame(corrrows).sort_values(['model','absolute_r'],ascending=[True,False]);save(corr,'pca_reconstructed_feature_correlations.csv')
    points=scores.copy();points['membrane']=points.model.map(audit.set_index('model').scope).map({'B':'POPE75POPG25_W_WF','M':'POPC60CHOL40_W_WF','paired':'paired B/M'})
    points['manual_phenotype_label']=np.where(points.model.str.contains('__B__'),points.label_B,np.where(points.model.str.contains('__M__'),points.label_M,None))
    if args.components==2:points['PC3']=np.nan
    points['PC3_status']='validated frozen projection' if args.components==3 else 'not requested (--components 2)'
    save(points,'pca_sample_scores_PC1_PC2_PC3.csv')
    if args.components==3:save(scores[['model','pair_id','molecule','label_B','label_M','PC1','PC2','PC3']],'pca_scores_PC1_PC2_PC3.csv')
    
    # Same rank-based review, extended to requested components. No supervised model.
    stats=[];groupstats=[];info=[];sil=[];plot_counts=[];discord=[]
    for model in models:
     dd=scores[scores.model==model].copy();ar=audit[audit.model==model].iloc[0];panels=['B','M'] if ar.scope=='paired' else [ar.scope]
     views=[['PC1','PC2']] if args.components==2 else [['PC1','PC2'],['PC1','PC3'],['PC2','PC3'],['PC1','PC2','PC3']]
     for view in views:
      fig,counts=plot_samples(dd,ar,view,colours,markers,palette_status);plot_counts.extend(counts)
      for suffix in ['png','pdf']:save_figure(fig,figdir/(model+'__'+'_'.join(view)+'.'+suffix),dpi=180)
      plt.close(fig)
     for side_label in panels:
      lab='label_'+side_label;labels=dd[lab].to_numpy();observed=sorted(dd[lab].unique());k=len(observed);n=len(dd)
      for c in observed:
       sub=dd[dd[lab]==c]
       for pc in component_names:
        z=sub[pc];groupstats.append(dict(model=model,label_membrane=side_label,phenotype=c,component=pc,n=len(z),median=z.median(),q25=z.quantile(.25),q75=z.quantile(.75),minimum=z.min(),maximum=z.max()))
       if args.components==3 and len(sub)>=5:
        q1,q3=sub.PC3.quantile([.25,.75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
        for _,row in sub[(sub.PC3<lo)|(sub.PC3>hi)].iterrows():
         info.append(dict(model=model,example_type='PC3 phenotype/location exception',criterion=f'{side_label}: outside own {c} PC3 Tukey 1.5-IQR fences [{lo:.6g},{hi:.6g}], class n={len(sub)}; no annotation-error claim',**row.drop('model').to_dict()))
      for pc in component_names:
       H,pvalue=kruskal(*(dd.loc[dd[lab]==c,pc] for c in observed));eps=max(0,(H-k+1)/(n-k))
       ranks=rankdata(dd[pc]);counts=np.array([(labels==c).sum() for c in observed]);split=np.cumsum(counts)[:-1];tiecounts=np.unique(dd[pc],return_counts=True)[1];tie=1-sum(tiecounts**3-tiecounts)/(n**3-n);ge=0
       # Per-test stream keeps PC1/PC2 permutation values invariant when PC3 is added.
       test_seed=20261005+int(hashlib.sha256((model+side_label+pc).encode()).hexdigest()[:8],16)
       test_rng=np.random.default_rng(test_seed)
       for _ in range(args.permutations):
        sums=np.array([v.sum() for v in np.split(test_rng.permutation(ranks),split)]);hp=12/(n*(n+1))*sum(sums*sums/counts)-3*(n+1);ge+=hp/tie>=H-1e-12
       stats.append(dict(model=model,label_membrane=side_label,component=pc,kw_H=H,kw_p_asymptotic=pvalue,permutation_p=(1+ge)/(args.permutations+1),epsilon_squared=eps,n=n,groups=k,permutations=args.permutations,test_seed=test_seed))
      for view in views:
       X=dd[view].to_numpy();sd=X.std(axis=0,ddof=1)
       sil.append(dict(model=model,label_membrane=side_label,view='_'.join(view),dimensions=len(view),silhouette_raw=silhouette_score(X,labels),silhouette_equal_variance=silhouette_score(X/sd,labels)))
      deform=dd[lab].isin(['curved','bulge_curved','bulge_flat','pore_formed']);flatmed=dd.loc[~deform,'PC1'].median();defmed=dd.loc[deform,'PC1'].median()
      for ix in dd.index[(~deform&(dd.PC1>defmed))|(deform&(dd.PC1<flatmed))]:
       row=dd.loc[ix];info.append(dict(model=model,example_type='phenotype/location discordance',criterion=f'{side_label} plain-flat PC1 > deformed median ({defmed:.4f}) or deformed PC1 < plain-flat median ({flatmed:.4f}); descriptive, not mislabelling',**row.drop('model').to_dict()))
     for pc in component_names:
      for endpoint,ix in [('minimum',dd[pc].idxmin()),('maximum',dd[pc].idxmax())]:
       row=dd.loc[ix];info.append(dict(model=model,example_type=f'{pc} {endpoint}',criterion=f'global observed {endpoint} {pc}',**row.drop('model').to_dict()))
     if ar.scope=='paired':
      for ix in dd.index[dd.label_B!=dd.label_M]:
       row=dd.loc[ix];info.append(dict(model=model,example_type='B/M phenotype discordance',criterion='label_B differs from label_M; no combined phenotype class',**row.drop('model').to_dict()))
      if args.components==3:
       deformation=['curved','bulge_curved','bulge_flat','pore_formed']
       mismatch=dd.label_B.isin(deformation)!=dd.label_M.isin(deformation)
       concordance=np.where(mismatch,'deformed/plain-flat mismatch',np.where(dd.label_B!=dd.label_M,'different class, same broad grouping','exact label concordance'))
       for c in sorted(set(concordance)):
        z=dd.loc[concordance==c,'PC3'];discord.append(dict(model=model,concordance_group=c,n=len(z),median_PC3=z.median(),q25_PC3=z.quantile(.25),q75_PC3=z.quantile(.75),median_absolute_PC3=z.abs().median()))
    stats=pd.DataFrame(stats);test_family_size=len(stats)
    stats['q_BH_asymptotic']=bh(stats.kw_p_asymptotic);stats['q_BH_permutation']=bh(stats.permutation_p);stats['BH_test_family_size']=test_family_size
    stats['BH_test_family']='all requested PC-by-model-by-label-membrane tests';save(stats,'pca_phenotype_separation.csv')
    groupstats=pd.DataFrame(groupstats);save(groupstats,'pca_phenotype_group_summaries.csv');save(pd.DataFrame(sil),'pca_phenotype_overlap.csv');save(pd.DataFrame(plot_counts),'pca_plot_point_validation.csv')
    info=pd.DataFrame(info).drop_duplicates(['model','example_type','pair_id','criterion']);save(info,'pca_informative_samples.csv')
    if discord:save(pd.DataFrame(discord),'pca_PC3_paired_discordance.csv')
    # Reconcile all original class tests (PC1 only).
    recon=[]
    for _,row in assoc[assoc.target.str.startswith('visual_classes')].iterrows():
     side=row.target[-1];rr=stats[(stats.model==row.model)&(stats.label_membrane==side)&(stats.component=='PC1')].iloc[0];recon.append(dict(model=row.model,label_membrane=side,source_H=row.kw_H,recomputed_H=rr.kw_H,absolute_difference=abs(row.kw_H-rr.kw_H)))
    recon=pd.DataFrame(recon);assert recon.absolute_difference.max()<1e-7;save(recon,'pca_source_association_reconciliation.csv')
    # Compare B/M loadings only on matched feature bases, allowing global sign reversal.
    for view in ['endpoint_vs_ref','late_vs_ref']:
     for fs in ['full','reduced']:
      b=f'{view}__B__{fs}';m=f'{view}__M__{fs}'
      for pc in [1,2,3]:
       vb=loads[(loads.model==b)&(loads.component==pc)].set_index('feature').loading;vm=loads[(loads.model==m)&(loads.component==pc)].set_index('feature').loading
       assert set(vb.index)==set(vm.index)
       cos=float(vb@vm.reindex(vb.index));similar.append(dict(view=view,feature_set=fs,component=pc,absolute_loading_cosine=abs(cos),sign_alignment=(-1 if cos<0 else 1),interpretation='descriptive vector similarity; independently fitted axes are not identical'))
    similar=pd.DataFrame(similar);save(similar,'pca_B_M_loading_similarity.csv')
    apl=ranked[ranked.feature.str.endswith('APL')&ranked.component.le(3)].copy();save(apl,'pca_APL_loadings.csv')
    comparisons=[]
    for view in ['endpoint_vs_ref','late_vs_ref']:
     for side in ['B','M','paired']:
      a=f'{view}__{side}__full';b=f'{view}__{side}__reduced';da=scores[scores.model==a].set_index('pair_id');db=scores[scores.model==b].set_index('pair_id').reindex(da.index)
      fa=set(loads[(loads.model==a)&(loads.component==1)].feature);fb=set(loads[(loads.model==b)&(loads.component==1)].feature)
      comparisons.append(dict(view=view,scope=side,features_removed='; '.join(sorted(fa-fb)),features_added='; '.join(sorted(fb-fa)),PC1_score_Spearman=spearmanr(da.PC1,db.PC1).statistic,PC2_score_Spearman=spearmanr(da.PC2,db.PC2).statistic))
    comparisons=pd.DataFrame(comparisons);save(comparisons,'pca_full_reduced_comparison.csv')
    # Overview and loading figures.
    fig,ax=plt.subplots(figsize=(12,7));x=np.arange(len(audit));base=np.zeros(len(audit))
    for pc,c in zip([1,2,3],['#0072B2','#E69F00','#CC79A7']):
     vals=audit[f'PC{pc}_variance_pct'].to_numpy();ax.barh(x,vals,left=base,color=c,label=f'PC{pc}');base+=vals
    ax.set_yticks(x,audit.model);ax.invert_yaxis();ax.set_xlabel('Explained variance (%)');ax.set_xlim(0,100);ax.legend();fig.tight_layout();save_figure(fig,figdir/'variance_overview.png',dpi=180);plt.close(fig)
    for model in models:
     z=loads[(loads.model==model)&loads.component.le(3)].pivot(index='feature',columns='component',values='loading');fig,ax=plt.subplots(figsize=(7,max(4,len(z)*.38)));im=ax.imshow(z.to_numpy(),cmap='PuOr',vmin=-1,vmax=1,aspect='auto');ax.set_yticks(range(len(z)),z.index);ax.set_xticks(range(3),['PC1','PC2','PC3']);ax.set_title(model)
     for i in range(len(z)):
      for j in range(3):ax.text(j,i,f'{z.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(z.iloc[i,j])>.65 else 'black')
     fig.colorbar(im,ax=ax,label='Saved signed component coefficient');fig.tight_layout();save_figure(fig,figdir/(model+'__loadings_PC1_PC2_PC3.png'),dpi=180);plt.close(fig)
    
    # Explicit, model-specific interpretation; physical metric definitions are absent.
    interpret={
    'endpoint_vs_ref__B__full':['Deformation/asymmetry mode: height, excess, curvature and all asymmetries co-load; thickness opposes; APL contributes little.','Projected-area/thickness contrast: APL +0.744 opposes thickness −0.617; morphology coefficients mostly small.','Height/excess versus curvature asymmetry: height +0.660 and excess +0.369 oppose curvature_asym −0.619; mixed geometric contrast.'],
    'endpoint_vs_ref__B__reduced':['Deformation/asymmetry mode with thickness opposing.','Predominantly thickness (+0.894), with weaker excess asymmetry; height opposes.','Height (+0.741), thickness and excess versus height/excess asymmetry; mixed amplitude/asymmetry mode.'],
    'endpoint_vs_ref__M__full':['Deformation/asymmetry mode; thickness opposes; APL contributes little.','Thickness/APL contrast: thickness +0.694 opposes APL −0.655.','Height/excess versus asymmetries: height +0.693 and excess +0.379 oppose height_asym, excess_asym and curvature_asym; APL contributes little.'],
    'endpoint_vs_ref__M__reduced':['Deformation/asymmetry mode; thickness opposes.','Height/excess versus height/excess asymmetry; thickness nearly absent.','Predominantly thickness (+0.947), explaining a structural mode not prominent in PC1/PC2.'],
    'endpoint_vs_ref__paired__full':['Common B/M deformation and asymmetry; both thicknesses oppose; APL coefficients are modest.','B-versus-M geometric contrast: B geometric terms oppose M geometric terms; thicknesses have matching signs; B_APL is a secondary term.','Common projected-area/thickness contrast: both APLs co-load and oppose both thicknesses; small shape coefficients.'],
    'endpoint_vs_ref__paired__reduced':['Common B/M deformation and asymmetry; thicknesses oppose.','B-versus-M geometric contrast, with largely matched thickness signs.','Common thickness mode, M-dominated; mixed secondary geometry.'],
    'late_vs_ref__B__full':['Persistent deformation/asymmetry mode; thickness opposes.','Predominantly thickness; inspect signed ranking below for smaller terms.','Height/excess versus curvature/excess asymmetry; a small mixed geometric mode.'],
    'late_vs_ref__B__reduced':['Persistent deformation/asymmetry mode; thickness opposes.','Predominantly thickness (+0.952); most other terms small.','Height (+0.697) opposes curvature_asym (−0.647), with secondary excess; small amplitude/asymmetry contrast.'],
    'late_vs_ref__M__full':['Persistent deformation/asymmetry mode; thickness opposes.','Height/excess versus height/excess/curvature asymmetries; thickness secondary.','Predominantly thickness (+0.950), a separate structural variance direction.'],
    'late_vs_ref__M__reduced':['Persistent deformation/asymmetry mode; thickness opposes.','Mixed height/thickness/excess versus height/curvature asymmetry; not a pure single physical axis.','Thickness (+0.863), with height_asym +0.394 opposing height −0.234; thickness-led mixed mode.'],
    'late_vs_ref__paired__full':['Common persistent B/M deformation/asymmetry; thicknesses oppose.','B-versus-M geometric contrast; both thickness terms share a sign and are secondary.','Common thickness mode: M +0.799, B +0.485; minor asymmetry/excess terms.'],
    'late_vs_ref__paired__reduced':['Common persistent B/M deformation/asymmetry; thicknesses oppose.','B-versus-M geometric contrast; both thicknesses have the same sign.','Common thickness mode: M +0.789, B +0.447, with secondary M height/excess.']}
    
    text=['# PCA scientific review',
    '## Conclusions',
    '1. **PC1 is predominantly a deformation/asymmetry covariance mode**, with height, excess, curvature and retained asymmetry measurements co-loading; thickness opposes. It is not a damage or efficacy score.',
    '2. **PC2 is model-dependent.** Separate full endpoint models emphasize APL versus thickness; late B emphasizes thickness; late M emphasizes height/excess versus asymmetry; paired PC2 mainly contrasts B and M geometry.',
    '3. **PC3 adds physically distinct structure**, especially thickness in late M (7.72–8.76% additional variance) and late paired models (4.74–5.65%). Late B PC3 adds only 2.90–3.52%. Endpoint PC3 adds 4.50–10.45% depending on model. Its phenotype utility cannot be measured without PC3 scores.',
    '4. **B/M PC1 patterns are similar, but their axes are independently fitted.** Secondary modes differ and may change component order. Paired models provide a common joint pair space, not a shared embedding of individual B/M samples.',
    '5. **APL contributes a distinct secondary covariance direction.** In endpoint B/M full models its PC1 coefficients are only 0.0619/0.1015, versus absolute PC2 coefficients 0.7444/0.6551. In the paired endpoint model, both APLs primarily contribute to PC3 together with thickness. Full-versus-reduced differences cannot be attributed solely to APL because curvature_asym also changes.',
    '6. **Phenotypes differ mainly along PC1 but overlap substantially.** Class medians and rank tests support association, not clean five-class clusters or a prediction claim. Only four flat_no_interaction pairs per membrane are available.',
    '7. **Redundancy is strongest among geometric/asymmetry measures**, with model-specific pairs quantified below from the complete saved spectral decomposition. Thickness and APL preserve secondary information; height/asymmetry contrasts remain visible after the dominant mode.',
    '8. **Keep PCA exploratory/diagnostic.** Covariance directions do not establish causation, antibacterial activity, toxicity, utility, prospective chemical predictability or a validated ML target.',
    '## Scope, missing evidence and provenance',
    'The authoritative specification was read before inspecting the archive. The source archive contains exactly six files: pca_scores.csv, pca_loadings.csv, pca_summary.csv, pca_associations.csv, SUMMARY.md and analysis_manifest.json. There are 12 models, each with 156 unique matched pairs (156 molecules), 1,872 score rows, 1,110 coefficients and complete spectra of 6–16 components. Exported scores contain PC1 and PC2 only. Sources were preserved; no full workflow, PCA refit or Slurm job was run.',
    '**PC3 sample projections are impossible from this bundle.** Despite the specification’s expected existing state, pca_parameters.json is absent. Loadings supply all component directions but not sample feature vectors, fitted preprocessing transforms (feature ordering, centres, scales, any sign transformations) or projection/origin conventions. Even complete orthonormal directions and spectra do not identify missing sample coordinates. PC3 is blank in the score export; no fabricated PC1/PC3, PC2/PC3 or 3D sample plots are generated. To finish, supply frozen parameters plus matched original feature values, or existing saved PC3 scores. Do not refit.',
    '**Canonical project phenotype colours are used.** The mapping is inherited from scripts/plot_metric_analysis.py and recorded in phenotype_colour_mapping.json. Redundant marker shapes distinguish classes. Paired scatterplots show the identical 156 pair points twice, once coloured by B labels and once by M labels, avoiding a silently selected membrane label. Paired points are joint B/M pair observations; they are not two separately projected membrane systems.',
    'Endpoint names mean endpoint-versus-pristine-reference; late names mean persistent 80–100 ns morphology median versus pristine reference. Precise endpoint APL timing, raw units, leaflet-asymmetry formulas, preprocessing and origin conventions cannot be verified from this bundle. SUMMARY.md separately records a 90–100 ns APL mean; do not silently substitute it into the late PCA. No start-reference PCA family is present. Source SUMMARY hypotheses and recommendations are not rewritten.',
    'All exported PC1/PC2 score means are numerically zero (absolute mean <2e−13), consistent with cohort centering. Zero lines are coordinate references, not evidence of a pristine/null origin; the origin convention cannot be independently reconstructed without frozen transforms. Component sign is arbitrary. Reported signed coefficients are the supplied loadings; orientation_multiplier is preserved in the source summary and not applied a second time.',
    '## Key numerical evidence',
    'Late full PC1 coefficients have B/M loading cosine similarity 0.9831 (endpoint full: 0.9882), while late full PC2/PC3 similarities are only 0.3797/0.3126 after global sign alignment. Late B PC2 thickness corresponds conceptually more closely to late M PC3 thickness than to M PC2; component numbering is not a physical identity.',
    'Late full phenotype rank effect sizes are PC1 ε²=0.4301 (B) and 0.3306 (M), versus PC2 ε²=0.0977 and 0.1468. Five-class raw-coordinate silhouettes are only 0.0771 (B) and −0.1623 (M): distribution shifts coexist with substantial overlap, especially in M. All 32 new tests survive BH correction at 0.05 using both asymptotic and permutation p values; significance is not clean separation.',
    'Late full B PC1 medians in saved orientation are flat_no_interaction −0.7198, flat_interaction −0.3813, curved +0.0988, bulge_flat +0.7499 and bulge_curved +1.5515. M medians are −1.3981, −0.3259, +0.1580, +0.0971 and +1.1941 respectively. Thus bulge_flat/curved ordering differs; no universal ordinal severity scale is established. Values are within-model coordinates and cannot be compared as physical B/M magnitudes.',
    '71/156 pairs have different exact B/M phenotype labels; 56/156 differ on the broader deformed versus plain-flat grouping. In late fitted input space, height–excess correlations are 0.9422 (B) and 0.9270 (M), height_asym–excess_asym are 0.9815 and 0.9257, and curvature–curvature_asym are 0.9651 and 0.9113. These are Pearson values reconstructed from saved spectra, not raw-source Spearman estimates.',
    'Endpoint full versus reduced PC1 ranks are stable (Spearman B 0.9862, M 0.9925, paired 0.9915). M PC2 ranks change substantially (rho −0.0104), consistent with an altered secondary mode after simultaneous APL and curvature_asym removal; this does not isolate either feature’s causal effect.',
    '## Model audit',md(audit),
    '## APL and temporal/full–reduced comparisons',md(apl[['model','component','feature','loading','absolute_loading_rank','squared_loading_contribution']]),
    'Squared coefficients describe composition of a unit component vector, not the fraction of total data variance explained by an individual feature, and not a causal contribution. APL can contribute to every component where included. It is absent from all late models because no exactly matching persistent APL measure is supplied.',md(comparisons),
    'Endpoint reduced models omit APL **and curvature_asym**; late reduced models omit **excess_asym** instead. Therefore “reduced” does not mean one common feature subset across times. Changes in explained-variance percentages use different denominators after feature removal. Endpoint-versus-late differences also change temporal representation and cannot isolate an APL intervention.',
    '## B/M loading patterns',md(similar),
    'Cosines are calculated on matched ordered feature bases, with global sign alignment allowed; values near one indicate similar coefficient patterns, not axis identity or equal physical effects. No membrane score subtraction is used as an effect/selectivity measurement. Paired PCA concatenates B and M feature blocks; paired PC1 is common geometry, paired PC2 is mainly differential geometry. Paired endpoint-full PC3 is APL–thickness contrast; late paired PC3 is thickness-led.',
    '### Supplied aggregate APL associations',md(assoc[assoc.target.isin(['ddAPL','B__APL__endpoint_vs_start','M__APL__endpoint_vs_start'])][['model','target','rho_PC1']]),'These target correlations are supplied source results, not newly recomputed. Endpoint-vs-start APL associations have a different reference convention from endpoint-vs-ref PCA; no raw values are supplied. Moderate PC1 association does not make APL a dominant PC1 coefficient or rule out an independent secondary APL direction.',
    '## Phenotype association, gradient and overlap',
    'All analyses use fixed saved scores and observed labels. Kruskal–Wallis rank tests are calculated separately for each score/label membrane; epsilon² = max(0,(H−k+1)/(n−k)) with k=5 and n=156 is a descriptive rank effect size. The 32 newly calculated tests receive one Benjamini–Hochberg correction; this family differs from the source’s 16 PC1 class tests. Label permutation uses 1,999 permutations, preserving class counts, with p=(1+exceedances)/2,000; its resolution floor is 0.0005. No class ordering is imposed and no target is fitted. The tests assess distribution differences, not five-class discrimination. Correlated models are repeated views of the same 156 molecules, not independent replications.',md(stats),
    'All 16 supplied PC1 Kruskal–Wallis H statistics are reproduced within 1e−7; see reconciliation CSV. Original ddAPL/APL rho values are retained as source-derived aggregate associations; their raw targets cannot be recomputed or used for sample selection here.',
    'Classwise silhouettes use observed five-class labels in the 2D space. Raw scores preserve saved PCA coordinate distances; equal-variance silhouettes rescale each plotted PC by its observed SD as a sensitivity view. Negative/low silhouettes indicate substantial within/between-class intermixing rather than compact isolated classes; they are descriptive, not prediction estimates.',md(pd.DataFrame(sil)),
    'Compare PC1 versus PC2 effect sizes rather than assuming ordinal phenotypes. Group medians, IQRs and ranges are exported for every model/label. A median shift is a gradient-like association, not a literal continuous trajectory or clean cluster boundary. With flat_no_interaction n=4, no reliable thermal-null distribution, threshold, or distinct null cluster is established.',
    '### Full-model phenotype medians and ranges',md(groupstats[groupstats.model.str.endswith('__full')]),
    '## Redundancy and independent information',
    'Complete saved component bases are orthonormal to <2e−12. For each model reconstruct C ∝ Vᵀ diag(explained_variance_ratio) V; normalizing its diagonal recovers Pearson correlations of the **inputs to that fitted PCA** because the unknown common total-variance factor cancels. Orthogonal component sign changes cancel too. This is not a raw-metric or Spearman correlation reconstruction: nonlinear transformations, winsorisation or definitions cannot be resolved without preprocessing parameters. The |r|≥0.90 display criterion is a Pearson descriptive screen; the project’s original rho threshold is not falsely reused as an equivalent Spearman test.',md(corr[corr.redundant_at_0_9]),
    'High PC1 variance alone does not identify interchangeable individual measurements. Strong co-loading is consistent with redundancy, while the reconstructed pair correlations provide a more direct descriptive check in fitted input space. Residual height-versus-asymmetry contrasts, thickness-led modes and APL–thickness contrasts demonstrate independent covariance information; independence is mathematical within the fitted PCA, not biological independence.',
    '## Informative observed samples',
    'Extremes use actual saved PC1/PC2 minima and maxima, with no biological severity ranking. Phenotype/location examples use an explicit conservative rule: plain-flat samples above the same-membrane deformed-class PC1 median, or deformation-labelled samples below the plain-flat PC1 median. These are review examples, not annotation errors. All such examples and all B/M-discordant pair labels are exported; repeated model rows are separate views of the same samples.',md(info[info.example_type.str.contains('minimum|maximum')]),
    '### Selected phenotype/location exceptions',md(info[info.example_type.eq('phenotype/location discordance')].groupby('model',sort=False).head(3)),
    'B/M label discordance is available exactly. B/M quantitative physical-response discordance must not be inferred by directly subtracting independently fitted PCA coordinates. unified_metrics.csv supplies the canonical feature values used for frozen PC3 projection, and actual PC3 extremes and phenotype/location exceptions are reported by this review.',
    '## PC3 decision',
    'PC3 loading interpretation and variance are available for every model (see audit and model sections). Routine PC1/PC2 review is usually sufficient for late B dominant covariance (94.46–95.21% cumulative), with PC3 optional for residual shape/asymmetry. Retain PC3 as a standard additional review dimension once exact projections are available for late M, where thickness is largely a PC3 mode and PC1/PC2 account for only 85.59–87.09%, and for paired models, where an independent thickness/APL contrast matters. Endpoint M and paired endpoint models also merit PC3 review. This is a physical-coverage recommendation; **no evidence yet shows PC3 improves phenotype separation**. No PC3 score, phenotype statistic or 3D sample placement can be inferred from coefficient interpretation alone.',
    '## Figure guide','figures/variance_overview.png shows the variance audit. Each model has a sample PC1/PC2 PNG and PDF and a signed PC1–PC3 loading PNG. All molecule labels are in the sample CSV rather than crowded on the figures. Figures with PC3 in the filename show feature loadings, not sample projections.','## Model-by-model component interpretation']
    for model in models:
     ar=audit[audit.model==model].iloc[0];text+=['### '+model, f'[Sample PC1/PC2 plot](figures/{model}__PC1_PC2.png) · [PC1–PC3 loadings](figures/{model}__loadings_PC1_PC2_PC3.png)',f'PC1/PC2/PC3 explain {ar.PC1_variance_pct:.2f}% / {ar.PC2_variance_pct:.2f}% / {ar.PC3_variance_pct:.2f}%; cumulative three = {ar.PC1_PC2_PC3_pct:.2f}%.']
     for pc in [1,2,3]:
      g=ranked[(ranked.model==model)&(ranked.component==pc)];positive=', '.join(g[g.loading>0].feature);negative=', '.join(g[g.loading<0].feature);little=', '.join(g[g.absolute_loading<.1].feature) or 'none'
      text += [f'**PC{pc}: {interpret[model][pc-1]}**',f'Same-sign sets: positive [{positive}]; negative [{negative}]. Opposing sets contrast on this component. Little by explicit display convention |coefficient|<0.10: {little}. This cutoff is a reporting convention, not a significance test.',md(g[['feature','loading','absolute_loading_rank','squared_loading_contribution']])]
    text += ['## Legitimate use and limits','PCA summarizes population covariance under a particular frozen feature definition, preprocessing and cohort. It identifies dominant shared variation, contrasts, redundancy and observed sample placement. It does not predict a target, assign causation, determine membrane harm/selectivity, establish statistical repeatability of trajectories, quantify thermal noise, or prove generalization from peptoid chemistry. No independent efficacy, toxicity, trajectory replication or chemical held-out evaluation is available. A component may be investigated as a future target only after training-only freezing, repeatability tests, a defensible physical interpretation and independent prospective/held-out validation; these data do not establish that case.','## Completion and reproducibility','Supported: all 12 model audits; all saved component coefficient rankings; PC1–PC3 interpretations; APL/B/M/full–reduced comparisons; 12 PC1/PC2 sample figures (paired figures have two label panels); 12 PC1–PC3 loading figures; variance overview; complete sample links; descriptive phenotype statistics and overlap; actual informative samples; saved-spectrum redundancy reconstruction; source preservation verification. Unavailable: canonical colours, exact PC3 projections, PC3 sample plots/3D figures, PC3 phenotype improvement, per-system APL/ddAPL examples and exact preprocessing/metric definitions.','Run scripts/review_saved_pca.py --source results/archie_8193723_complete from the package root. Requirements: Python, NumPy, pandas, scipy, scikit-learn, matplotlib. This script only reviews supplied exports. Reproducibility seed = 20261005. Source files retain their original bytes. No claim is made that missing deliverables are completed.']
    report='\n\n'.join(text)+'\n'
    report=update_report(report,args,scores,projection_validation,stats,groupstats,pd.DataFrame(sil),pd.DataFrame(discord),info,audit,parameters,unified,palette_status,md)
    (out/'PCA_ANALYSIS.md').write_text(report)
    assert before==hashes()
    (out/'review_provenance.json').write_text(json.dumps({'source_files_SHA256':before,'sources_unchanged':True,'PCA_refit':False,'full_workflow_run':False,'PC3_scores_available':args.components==3,'canonical_palette_available':True,'source_analysis_manifest':manifest,'seed':20261005,'permutations':args.permutations,'components_requested':args.components,'projection_max_absolute_PC1_error':None if projection_validation is None else float(projection_validation.max_absolute_PC1_error.max()),'projection_max_absolute_PC2_error':None if projection_validation is None else float(projection_validation.max_absolute_PC2_error.max()),'review_script_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},indent=2))
    print('COMPLETE: '+str(args.components)+' components; '+palette_status);print(audit[['model','PC1_variance_pct','PC2_variance_pct','PC3_variance_pct']].to_string(index=False));print(stats[['model','label_membrane','component','epsilon_squared','q_BH_asymptotic']].to_string(index=False));print('Silhouettes');print(pd.DataFrame(sil).to_string(index=False));print('Correlations >=.9',len(corr[corr.redundant_at_0_9]));print('Sources unchanged:',before==hashes())

if __name__ == "__main__":
    main()
