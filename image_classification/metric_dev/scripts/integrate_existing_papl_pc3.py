#!/usr/bin/env python3
"""Non-destructive extension for existing PFPeptoids metric_dev figure review.

Default: read-only preflight. --apply: install 2 time-series atlases, add 3
existing pristine and 36 existing frozen PC3 figures to review_bundle.json.
No PCA fitting, MD analysis, input rewrites, or review-state mutations.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASELINE_REGISTRY_SHA256 = '03d483d70200fe912134e8b9085a82820044bfb955d048dfe9555f37760d4b11'
REGISTRY_NAME = 'review_bundle.json'
# Exact archived snapshots: refuse integration if analysis inputs changed on Archie.
SOURCE_SHA256 = {
    'data/pca_parameters.json': 'bf7623fb3a4a01fa0ec0637eebf67dde8f007782d289188e5020c680452b559d',
    'data/frozen_parameters.json': '0aaf543516ba9d1a69f2af75a3f109ae2be379a4e8218144979b33886c38a047',
    'surface_papl_analysis/tables/temporal_bin_coupling_source.csv': '6680261c295c12ba95a2caeba207409b83e6b62565af7a606c1cce828692aeab',
    'surface_papl_analysis/tables/matched_late_pAPL.csv': '6c6b6a43fd5b1fad9ed27efd8551cdcfefbf3d5ff469f4ea77c6ec274c039734',
    'pca_review/PCA_ANALYSIS.md': '4a2fad6b5d7df2a302c76a99d9a605c195ab318a2ad4ab54b2bbd06a721b56ce',
}
CSV_PATH = 'surface_papl_analysis/tables/temporal_bin_coupling_source.csv'
MATCHED_PATH = 'surface_papl_analysis/tables/matched_late_pAPL.csv'
PCA_MD = 'pca_review/PCA_ANALYSIS.md'
PC3_TYPES = ('PC1_vs_PC3', 'PC2_vs_PC3', 'PC1_PC2_PC3_3D')
PRISTINE_DIR = 'surface_papl_analysis/pristine_extraction_8202573'
ATLAS_STEM = 'papl_all_available_24_pairs'
NEW_FAMILY = 'pc3_frozen_review'
PRISTINE_FAMILY = 'pristine_time_series'


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as src:
        for data in iter(lambda: src.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def read_csv(results, rel):
    with (results / rel).open(newline='') as f:
        return list(csv.DictReader(f))


def model_records(results):
    text = (results / PCA_MD).read_text()
    start = text.index('## Model audit\n')
    end = text.index('\n## ', start + 3)
    lines = text[start:end].splitlines()
    header = None
    out = {}
    for line in lines:
        if not line.startswith('| '):
            continue
        vals = [v.strip() for v in line.strip().strip('|').split('|')]
        if vals[0] == 'model':
            header = vals
        elif header and len(vals) == len(header) and vals[0].startswith(('late_vs_ref__', 'endpoint_vs_ref__')):
            out[vals[0]] = dict(zip(header, vals))
    if len(out) != 12:
        raise ValueError('PCA scientific review does not contain all 12 audited model records')
    return out


def markdown_table_after(text, marker):
    section = text.split(marker, 1)
    if len(section) != 2:
        raise ValueError('Missing scientific review table section '+marker)
    rows=[]
    header=None
    started=False
    for line in section[1].splitlines():
        if line.startswith('| '):
            fields=[field.strip() for field in line.strip().strip('|').split('|')]
            if fields[0]=='model':
                header=fields
                started=True
            elif started and not fields[0].startswith('---'):
                if len(fields)!=len(header):
                    raise ValueError('Malformed markdown table in '+marker)
                rows.append(dict(zip(header,fields)))
        elif started and rows:
            break
    if not rows:
        raise ValueError('No scientific results within '+marker)
    return rows


def pc3_model_interpretations(results):
    import re
    text=(results/PCA_MD).read_text()
    interpretations={}
    for match in re.finditer(r'^### ((?:endpoint|late)_vs_ref__(?:B|M|paired)__(?:full|reduced))\s*$',text,re.M):
        model=match.group(1)
        end=text.find('\n### ',match.end())
        section=text[match.end():end if end!=-1 else len(text)]
        pc3=re.search(r'^\*\*PC3: (.*?)\*\*\s*$',section,re.M)
        if pc3:
            interpretations[model]=pc3.group(1)
    if len(interpretations)!=12:
        raise ValueError('Missing model-specific PC3 physical interpretations')
    assoc=markdown_table_after(text,'### PC3 phenotype distributions and associations')
    sensitivity=markdown_table_after(text,'### Small-class sensitivity')
    by_model=defaultdict(list)
    by_sensitive={(r['model'],r['label_membrane']):r for r in sensitivity}
    for r in assoc:
        model=r['model'];membrane=r['label_membrane']
        small=by_sensitive[(model,membrane)]
        by_model[model].append(dict(label_membrane=membrane,
             full_cohort_n=156,
             full_epsilon_squared=float(r['epsilon_squared']),
             full_q_BH_permutation=float(r['q_BH_permutation']),
             excluding_four_null_n=int(small['n']),
             excluding_four_null_epsilon_squared=float(small['epsilon_squared']),
             excluding_four_null_q_BH_permutation=float(small['q_BH_permutation'])))
    if len(by_model)!=12 or len(assoc)!=16 or len(sensitivity)!=16:
        raise ValueError('Expected 16 full and 16 small-class sensitivity associations')
    return interpretations,by_model


def read_series(results):
    rows = read_csv(results, CSV_PATH)
    if len(rows) != 912:
        raise ValueError(f'Expected 912 source records (24 paired systems x 2 x 19), found {len(rows)}')
    grouped = defaultdict(list)
    for row in rows:
        key = (row['pair_id'], row['side'])
        if key[1] not in ('B', 'M'):
            raise ValueError('Unexpected membrane label: ' + key[1])
        if not row['pair_id'].strip():
            raise ValueError('Empty physical trajectory ID')
        d = {'time': float(row['local_start_ns']) + 2.5,
             'start': float(row['local_start_ns']),
             'apl': float(row['pAPL_A2'])}
        import math
        if not all(math.isfinite(x) for x in d.values()):
            raise ValueError('Nonfinite time/APL value in ' + str(key))
        grouped[key].append(d)
    pairs = sorted({p for p, _ in grouped})
    if len(pairs) != 24 or len(grouped) != 48:
        raise ValueError(f'Expected 24 physical pairs with B and M; found {len(pairs)} / {len(grouped)}')
    expected = [float(x) for x in range(5, 100, 5)]
    for key, vals in grouped.items():
        vals.sort(key=lambda x: x['start'])
        if [v['start'] for v in vals] != expected:
            raise ValueError('Nonmatching five-nanosecond grid for ' + str(key))
    matched = read_csv(results, MATCHED_PATH)
    if len(matched) != 312:
        raise ValueError('Unexpected size of matched-late table')
    matched_by_key = {(r['pair_id'], r['side']): r for r in matched}
    if len(matched_by_key) != 312 or set(grouped) - set(matched_by_key):
        raise ValueError('Mismatch or duplicates in saved matched B/M table')
    available = {k for k, v in matched_by_key.items() if v['binned_pAPL_available'].lower() == 'true'}
    if available != set(grouped):
        raise ValueError('Mismatch between 5ns bins and availability flags')
    refs = {}
    for key in grouped:
        refs[key] = float(matched_by_key[key]['pAPL_reference_A2'])
        import math
        if not math.isfinite(refs[key]):
            raise ValueError('Nonfinite reference for ' + str(key))
    return grouped, pairs, refs


def provenance(source, heading, kind):
    return {'source_document': source, 'section_or_heading': heading, 'provenance_type': kind}


def figure_record(*, fid, title, family, png, svg, scientific_purpose, method, n,
                  observed_result, interpretation, limitations, source_tables,
                  evidence, evidence_table, evidence_filter, evidence_columns,
                  result_prov, hypotheses, axes, what_to_look_for,
                  plot_encoding, generator, related_figures=None, status='supplementary'):
    return dict(figure_id=fid, title=title, family=family,
        png_path=png, svg_path=svg,
        scientific_purpose=scientific_purpose,
        original_question=scientific_purpose,
        original_hypothesis='Exploratory diagnostic; no prospective or directional hypothesis established.',
        hypothesis_status='exploratory_no_directional_hypothesis',
        hypothesis_provenance=[hypotheses],
        expected_pattern={'support':'Reproducible descriptive structure and interpretable physical variation.',
                          'against':'Overlapping phenotype distributions, unstable associations, and mismatched windows.'},
        axes=axes, colour_grouping_semantics=plot_encoding,
        figure_specific_context=f'Diagnostic display of saved data. Cohort/coverage: {n}.',
        what_to_look_for=what_to_look_for,
        observed_result=observed_result,
        interpretation=interpretation,
        limitations=limitations,
        common_misreadings='Covariance or trajectory response is not evidence of membrane contact, antimicrobial efficacy, or incremental predictive improvement.',
        source_tables=source_tables,
        provenance=[hypotheses],
        result_provenance=[result_prov],
        numeric_evidence=[{'source_table':evidence_table, 'row_filter':evidence_filter,
                           'columns':evidence_columns, 'records':[evidence]}],
        metric_ids=['continuation'], analysis_technique_ids=['continuation'],
        related_figures=related_figures or [], status=status, n=n,
        generator=generator, method=method)


def extension_records(results, atlas_paths, grouped, pairs, refs):
    reg = json.loads((results / REGISTRY_NAME).read_text())
    existing_ids = {f['figure_id'] for f in reg['figures']}
    model_info = model_records(results)
    interpretation_by_model, assoc_by_model = pc3_model_interpretations(results)
    additions = []
    for membrane in ('B','M'):
        fid = f'papl_all_available_24_pairs_{membrane}'
        png, svg = atlas_paths[membrane]
        source=CSV_PATH
        vals=[v['apl'] for key, arr in grouped.items() if key[1]==membrane for v in arr]
        rec=figure_record(fid=fid, title=f'Projected APL versus time — 24 peptoid systems ({membrane})',
          family='temporal',png=png,svg=svg,
          scientific_purpose='What is the saved 5 ns projected-APL time evolution for every available matched peptoid-containing trajectory?',
          method='Saved pure-Relax 5 ns mean pAPL bins, 19 consecutive points per run, plotted at bin centre; source geometry is not recalculated.',
          n=24, observed_result=f'24 of 156 QC-passing physical pairs ({membrane}) have available saved bins; 19 bins per pair. Dashed line is the existing per-pair pristine reference. The shaded 80–100 ns interval is the late-time morphology analysis window.',
          interpretation='Shows descriptive within-run projected-APL evolution and between-run variability; suitable for examining temporal shape, not for extrapolating to the 132 pairs lacking bins.',
          limitations='Coverage is 24/156 physical pairs, not a representative full cohort by assumption. Five-nanosecond bins are correlated; no independent replicate uncertainty. pAPL is projected XY area per nominal leaflet lipid, not curved leaflet surface area. The pristine reference is an initial structure, not a validated thermal noise distribution.',
          source_tables=[CSV_PATH,MATCHED_PATH,'surface_papl_analysis/VARIABLES_AND_METHODS.md'],
          evidence={'n_pairs':24, 'n_bins_per_trajectory':19, 'n_binned_points':456,
                    'pAPL_min_A2':round(min(vals),5),'pAPL_max_A2':round(max(vals),5)},
          evidence_table=CSV_PATH,evidence_filter=f'side={membrane}; all available pairs, local_start_ns 5..95 ns',
          evidence_columns=['pair_id','side','local_start_ns','pAPL_A2'],
          result_prov=provenance(source,'all 24 available B/M physical pairs','completed_result'),
          hypotheses=provenance('surface_papl_analysis/SCIENTIFIC_DECISION_REPORT.md','Temporal analysis and limitations','method_definition'),
          axes={'x':'Relax-local bin centre (ns), 7.5..97.5; 5 ns saved means', 'y':'pAPL (Å²), independently scaled in each small panel','shade':'80–100 ns morphology window'},
          what_to_look_for='Which runs show early-to-late drift, extrema or reversal? Compare a given pair across B/M carefully: absolute B/M reference APL values are different.',
          plot_encoding='24 labelled panels, one physical peptoid/run per panel; solid line is saved projected APL and dashed line is that pair/side pristine reference.',
          generator='scripts/integrate_existing_papl_pc3.py', related_figures=['temporal_selected_trajectories','binned_frame_reconciliation'])
        additions.append(rec)
    for membrane in ('B','M','combined'):
        fid=f'pristine_papl_{membrane.lower()}'
        pristine = f'{PRISTINE_DIR}/figures/pristine_pAPL_{membrane}'
        source_files = [f'{PRISTINE_DIR}/pristine_frames_{m}.csv' for m in ('B','M') if membrane=='combined' or m==membrane]
        stats={}
        for side in ('B','M'):
            if membrane!='combined' and side!=membrane:
                continue
            rows=read_csv(results,f'{PRISTINE_DIR}/pristine_frames_{side}.csv')
            times=[float(r['time_ns']) for r in rows]
            values=[float(r['projected_APL_A2']) for r in rows]
            if len(rows)!=81 or times!=[i/8 for i in range(81)]:
                raise ValueError('Pristine sample-time validation failed for '+side)
            stats[side]={'n_frames':81,'first_A2':round(values[0],5),'last_A2':round(values[-1],5),'mean_A2':round(sum(values)/len(values),5)}
        rec=figure_record(fid=fid,
          title=f'Pristine projected APL versus time — {membrane.upper() if membrane!="combined" else "B and M"}',
          family=PRISTINE_FAMILY,png=pristine+'.png',svg=pristine+'.svg',
          scientific_purpose='How does the projected area per lipid evolve in the short peptoid-free B/M control trajectories?',
          method='Pre-existing 81 saved frames from 0–10 ns at observed 0.125 ns increments. Provenance stride_ps=100 is a requested stride parameter, not the observed saved-frame spacing. No trajectory recomputation.',
          n=81 if membrane!='combined' else 162,
          observed_result='Two separate 0–10 ns pristine reference trajectories have appreciable projected-APL drift. This is not a stable equilibrium null distribution.',
          interpretation='The short peptoid-free trajectories show transient area relaxation; their temporal fluctuations cannot calibrate uncertainty for 100 ns peptoid trajectories.',
          limitations='Only one short control per membrane and no independent replicate ensemble. Temporal standard deviation is not replicate confidence. Pristine 0–10 ns and peptoid late 80–100 ns windows are not directly equivalent.',
          source_tables=source_files+[f'{PRISTINE_DIR}/provenance.json'],
          evidence=stats,evidence_table=source_files[0],
          evidence_filter=f'side={membrane}; time_ns 0..10 ns',evidence_columns=['time_ns','projected_APL_A2'],
          result_prov=provenance(f'{PRISTINE_DIR}/provenance.json','pristine extraction job 8202573','completed_result'),
          hypotheses=provenance('surface_papl_analysis/SCIENTIFIC_DECISION_REPORT.md','Pristine and reference limitations','method_definition'),
          axes={'x':'Simulation time (ns); 0–10','y':'Projected APL (Å²)'},
          what_to_look_for='Check directional relaxation and the difference between temporal variation and equilibrium uncertainty.',
          plot_encoding='Saved B and/or M pristine projected-APL time series; no peptoid present.',
          generator='scripts/plot_pristine_papl.py (existing saved figures; generator not independently audited)',
          status='recorded_existing_plot')
        additions.append(rec)
    for view in ('endpoint_vs_ref','late_vs_ref'):
        for side in ('B','M','paired'):
            for features in ('full','reduced'):
                model=f'{view}__{side}__{features}'
                audit=model_info[model]
                for projection in PC3_TYPES:
                    fid=f'pc3_review__{model}__{projection}'
                    stem=f'pca_review/figures/{{ext}}/{model}__{projection}'
                    png=stem.format(ext='png')+'.png'
                    svg=stem.format(ext='svg')+'.svg'
                    label={'PC1_vs_PC3':'PC1 versus PC3','PC2_vs_PC3':'PC2 versus PC3','PC1_PC2_PC3_3D':'3D PC1/PC2/PC3'}[projection]
                    window='80–100 ns morphology versus pristine reference' if view=='late_vs_ref' else 'endpoint versus pristine reference'
                    interp='Paired model is a joint B/M feature embedding of physical pairs; its axes must not be interpreted as a subtraction of B and M scores.' if side=='paired' else 'B and M models are independently fitted; score coordinates from separate models cannot be treated as directly aligned axes.'
                    rec=figure_record(fid=fid,
                      title=f'Frozen PCA: {label} | {window} | {side} | {features}',family=NEW_FAMILY,
                      png=png,svg=svg,
                      scientific_purpose=f'What additional physical structure appears with frozen PC3 for {model}, without claiming new predictive discrimination?',
                      method='Existing saved frozen PCA projections; PC1/PC2 reproducibility and PC3 construction are documented in PCA_ANALYSIS.md. Plot files are reused unchanged. No fitting or coordinate recalculation.',
                      n=int(audit['n']),
                      observed_result=f"In recorded PCA model audit, PC3 explains {audit['PC3_variance_pct']}% variance (PC1 {audit['PC1_variance_pct']}%, PC2 {audit['PC2_variance_pct']}%; first 3 cumulative {audit['PC1_PC2_PC3_pct']}%). Reported frozen projection validation is not independently rerun by this installer.",
                      interpretation=f'{interp} Recorded physical PC3 mode: {interpretation_by_model[model]} PC3 phenotype associations are not proof of incremental predictive value beyond PC1/PC2.',
                      limitations='Strong class overlap; flat_no_interaction n=4 per membrane. PC3 does not establish five-way phenotype separation, B–M selectivity, additional out-of-sample predictive improvement, contact, or potency. Late models contain no late-window APL input. Reduced feature definitions differ between endpoint and late models.',
                      source_tables=[PCA_MD, 'data/pca_parameters.json'],
                      evidence={'model':model, 'n':int(audit['n']),
                                'pc1_variance_pct':float(audit['PC1_variance_pct']),
                                'pc2_variance_pct':float(audit['PC2_variance_pct']),
                                'pc3_variance_pct':float(audit['PC3_variance_pct']),
                                'pc1_pc2_pc3_pct':float(audit['PC1_PC2_PC3_pct']),
                                'phenotype_association_and_null_exclusion':assoc_by_model[model]},
                      evidence_table=PCA_MD,evidence_filter=f'## Model audit; model={model}',
                      evidence_columns=['model','n','PC1_variance_pct','PC2_variance_pct','PC3_variance_pct','PC1_PC2_PC3_pct'],
                      result_prov=provenance(PCA_MD,'Validated PC3 continuation / Model audit','completed_result'),
                      hypotheses=provenance(PCA_MD,'PC3 decision','method_definition'),
                      axes={'view':label, 'axes':'Saved PCA score coordinates, unitless, centered/scaled using original frozen parameters',
                            'time_window':window,'feature_set':features,'population':'156 QC-passing matched physical pairs'},
                      what_to_look_for='Inspect plot phenotype overlap, extreme points, loadings and support for a distinct physical covariance axis without projecting an unvalidated classification boundary.',
                      plot_encoding='Saved phenotype colours/markers; points are QC-passing samples in frozen PCA coordinate space.',
                      generator='scripts/review_saved_pca.py (existing frozen projections; no rerun)',
                      related_figures=[f'pc3_review__{model}__{p}' for p in PC3_TYPES if p!=projection],
                      status='recorded_saved_projection')
                    additions.append(rec)
    ids=[x['figure_id'] for x in additions]
    if len(additions)!=41 or len(ids)!=len(set(ids)):
        raise AssertionError('Expected 41 unique entries, got '+str(len(additions)))
    if set(ids)&existing_ids:
        raise ValueError('Figure ID collision; do not overwrite existing registry entries: '+str(sorted(set(ids)&existing_ids)))
    return reg, additions


def validate_existing_assets(results, additions, pending_paths):
    from PIL import Image
    missing=[]
    for f in additions:
        if f['figure_id'].startswith('papl_all_available_24_pairs_'):
            continue
        for ext in ('png','svg'):
            p=results / f[f'{ext}_path']
            if not p.is_file():
                missing.append(str(p))
            elif not p.stat().st_size:
                raise ValueError('Zero-byte image: '+str(p))
            elif ext=='png':
                with Image.open(p) as im:
                    im.verify()
    if missing:
        raise ValueError('Missing existing source figure files, installation cancelled:\n'+'\n'.join(missing))
    for f in additions:
        for ext in ('png','svg'):
            p=results / f[f'{ext}_path']
            if f['figure_id'].startswith('papl_all_available_24_pairs_'):
                if p.exists():
                    raise FileExistsError('Would overwrite existing atlas: '+str(p))
            else:
                f[f'{ext}_sha256']=digest(p)
        for src in f['source_tables']:
            if not (results / src).is_file():
                raise ValueError('Missing scientific source referenced by new entry: '+src)
        for cat in ('provenance','hypothesis_provenance','result_provenance'):
            for rec in f[cat]:
                path=results / rec['source_document']
                if not path.is_file():
                    raise ValueError('Missing provenance source: '+str(path))
    return len(additions)


def plot_atlases(grouped, pairs, refs, where):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs={}
    for membrane in ('B','M'):
        fig, axes=plt.subplots(6,4,figsize=(18,24), sharex=True)
        for idx,pair in enumerate(pairs):
            ax=axes.flat[idx]
            values=grouped[(pair,membrane)]
            x=[v['time'] for v in values]
            y=[v['apl'] for v in values]
            ax.axvspan(80,100,color='grey',alpha=.10,zorder=0)
            ax.plot(x,y,'o-',linewidth=1.3,markersize=2.4)
            ax.axhline(refs[(pair,membrane)],ls='--',linewidth=.9,color='0.45')
            ax.set_title(pair.replace('|','  |  '),fontsize=9,pad=7)
            ax.set_xlim(0,100)
            ax.set_xticks([0,20,40,60,80,100])
            ax.grid(alpha=.18)
            if idx//4==5:ax.set_xlabel('Relax-local time (ns)',fontsize=9)
            if idx%4==0:ax.set_ylabel('Projected APL (Å²)',fontsize=9)
            ax.tick_params(labelsize=8)
        fig.suptitle(f'Peptoid projected APL — {membrane} membrane, 24 available physical systems',fontsize=16,y=.998)
        fig.text(.5,.984,'Saved pure-Relax 5 ns means at 7.5–97.5 ns; dashed = pristine reference; shaded = late 80–100 ns',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.005,1,.975),h_pad=1.35,w_pad=.75)
        stem=f'{ATLAS_STEM}_{membrane}'
        png=where/(stem+'.png');svg=where/(stem+'.svg')
        fig.savefig(png,dpi=150)
        fig.savefig(svg)
        plt.close(fig)
        from PIL import Image
        with Image.open(png) as im:im.verify()
        if png.stat().st_size < 4000 or svg.stat().st_size<4000:
            raise ValueError('Unexpectedly small generated atlas')
        outputs[membrane]=(png,svg)
    return outputs


def atomic_registry(path, bytes_data):
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.registry_prepared_',delete=False) as h:
        h.write(bytes_data)
        h.flush()
        os.fsync(h.fileno())
        temp=Path(h.name)
    try:
        os.chmod(temp,path.stat().st_mode & 0o777)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    script_location=Path(__file__).resolve()
    default_base=(script_location.parent.parent if script_location.parent.name == 'scripts'
                  else script_location.parents[2])
    parser.add_argument('--results',type=Path,default=default_base/'results/archie_8193723_complete')
    parser.add_argument('--stage',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--apply',action='store_true',help='Install after successful read-only preflight')
    parser.add_argument('--preview',type=Path,help='Render two atlases to an external preview directory, without changing the project')
    args=parser.parse_args()
    results=args.results.resolve();stage=args.stage.resolve()
    raw=results / REGISTRY_NAME
    if not raw.is_file():raise SystemExit('Missing review_bundle.json: '+str(raw))
    before=digest(raw)
    if before!=BASELINE_REGISTRY_SHA256:
        raise SystemExit('Registry differs from supplied audit snapshot; no changes allowed. Original='+BASELINE_REGISTRY_SHA256+' current='+before)
    for rel, sha_expected in SOURCE_SHA256.items():
        path=results/rel
        if not path.is_file() or digest(path)!=sha_expected:
            raise SystemExit('Archived scientific source has changed; no modifications: '+str(path))
    grouped,pairs,refs=read_series(results)
    atlas_paths={membrane:(f'surface_papl_analysis/figures/png/temporal/{ATLAS_STEM}_{membrane}.png',
                          f'surface_papl_analysis/figures/svg/temporal/{ATLAS_STEM}_{membrane}.svg') for membrane in ('B','M')}
    registry,additional=extension_records(results,atlas_paths,grouped,pairs,refs)
    count=validate_existing_assets(results,additional,atlas_paths)
    print(f'PREFLIGHT PASS | existing registry={len(registry["figures"])} | additional={count} (PC3=36, pristine=3, peptoid atlas=2)')
    print(f'DATA PASS | 24 paired peptoid systems, B and M x19 5ns means; 132 paired systems lack matching 5ns series')
    print('SOURCE | original review_bundle.json SHA-256 = '+before)
    if args.preview:
        directory=args.preview.resolve()
        if directory.exists() and any(directory.iterdir()):
            raise SystemExit('Preview directory must not contain files: '+str(directory))
        directory.mkdir(parents=True,exist_ok=True)
        out=plot_atlases(grouped,pairs,refs,directory)
        print('PREVIEW PASS | '+', '.join(str(p) for both in out.values() for p in both))
    if not args.apply:
        print('READ-ONLY CHECK COMPLETE. To install, rerun with --apply after reviewing package and figures.')
        return
    if not stage.is_dir():raise ValueError('Stage directory missing: '+str(stage))
    # The figure-review GUI holds an advisory lock on the result directory.
    # Refuse to update the registry while its state might be in use.
    if os.name=='posix':
        import fcntl
        lockfd=os.open(str(results),os.O_RDONLY)
        try:
            fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            os.close(lockfd)
            raise RuntimeError('Close the figure-review GUI before applying this patch') from exc
        # Lock remains held until this single-shot Python process exits.
    dest_script=results.parent.parent/'scripts'/Path(__file__).name
    if not dest_script.parent.is_dir():
        raise ValueError('Missing existing canonical scripts directory: '+str(dest_script.parent))
    if dest_script.exists():
        raise FileExistsError('Would overwrite canonical script: '+str(dest_script))
    if digest(raw)!=before:raise RuntimeError('Registry changed during preflight')
    state=results/'figure_review_state.json'
    state_digest=digest(state) if state.exists() else None
    generated=[]
    installed=[]
    registry_backup=None
    installed_registry_sha=None
    try:
        with tempfile.TemporaryDirectory(dir=stage,prefix='pc3_papl_build_') as scratch:
            out=plot_atlases(grouped,pairs,refs,Path(scratch))
            for f in additional:
                if not f['figure_id'].startswith('papl_all_available_24_pairs_'):
                    continue
                m=f['figure_id'][-1]
                for ext,path in zip(('png','svg'),out[m]):
                    f[f'{ext}_sha256']=digest(path)
            registry['figure_families'].setdefault(NEW_FAMILY,{'name':'Frozen PC3 review (existing validated projections)'})
            registry['figure_families'].setdefault(PRISTINE_FAMILY,{'name':'Pristine controls — projected APL versus time'})
            registry['figures'].extend(additional)
            registry['figures'].sort(key=lambda x:(x['family'],x['figure_id']))
            assert len(registry['figures'])==164
            raw_bytes=(json.dumps(registry,ensure_ascii=False,indent=2)+'\n').encode()
            installed_registry_sha=hashlib.sha256(raw_bytes).hexdigest()
            # Preserve previous registry file and review status before changing anything.
            tag=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            registry_backup=stage/f'review_bundle_before_pc3_papl_{tag}.json'
            with registry_backup.open('xb') as h:
                h.write(raw.read_bytes())
            assert digest(registry_backup)==before
            for membrane,(png,svg) in out.items():
                for ext,src in (('png',png),('svg',svg)):
                    dest=results/atlas_paths[membrane][0 if ext=='png' else 1]
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with src.open('rb') as infile, dest.open('xb') as outfile:
                        shutil.copyfileobj(infile,outfile)
                    installed.append(dest)
                    if digest(dest)!=digest(src):raise ValueError('Output content changed while copying')
            with Path(__file__).open('rb') as source, dest_script.open('xb') as target:
                shutil.copyfileobj(source,target)
            installed.append(dest_script)
            if digest(dest_script)!=digest(__file__):raise ValueError('Generator script copy differs')
            if digest(raw)!=before:raise RuntimeError('Registry modified since audit; refusing overwrite')
            atomic_registry(raw,raw_bytes)
            if state_digest!=(digest(state) if state.exists() else None):
                raise RuntimeError('Review state changed during installation; audit before continuing')
            print('APPLY PASS | registry now contains',len(registry['figures']),'entries')
            print('APPLY PASS | preserved original registry at',registry_backup)
            print('APPLY PASS | existing review state unchanged; frozen PCA untouched')
            print('APPLY PASS | new atlas paths:',*[str(p) for p in installed if p.suffix in ('.png','.svg')],sep='\n  ')
    except Exception:
        if registry_backup is not None and registry_backup.is_file() and raw.exists():
            # Roll back only if we know exactly what was installed.
            actual_registry_sha=digest(raw)
            if installed_registry_sha and actual_registry_sha==installed_registry_sha:
                atomic_registry(raw,registry_backup.read_bytes())
            elif actual_registry_sha!=before:
                print('WARNING: registry changed externally; manual audit needed, refusing to overwrite',flush=True)
        for p in reversed(installed):
            if p.exists():p.unlink()
        raise

if __name__=='__main__':
    main()
