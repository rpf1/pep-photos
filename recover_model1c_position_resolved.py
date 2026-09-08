#!/usr/bin/env python3
"""Guarded Model1C retry. Default: read-only preflight; --apply: archive and submit.

Run using my_env_2025 Python on the login node, NOT with sbatch.
Training itself runs as a one-core Slurm job. Shared training code is unchanged.
The logical training run ID is retained to pair with the completed global run;
the old attempt and new job mapping are recorded in a separate recovery folder.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys

ROOT = Path('/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids')
ML = ROOT / 'analysis/machinelearning'
ACTIVE = ML / 'shared_data/active_learning'
DB = ACTIVE / 'active_learning.db'
PYTHON = Path('/users/pfb19164/.conda/envs/my_env_2025/bin/python')
SOURCE = ML / 'position_resolved/scripts/peptoidmachinelearning.py'
RUN_ID = '20260906_144733_646541'
MODEL = 'Model1C'
OLD_JOB = '8041982'
GLOBAL_JOB = '8041981'
POSITION = ML / 'position_resolved/training_runs' / RUN_ID
GLOBAL = ML / 'global_molecular/training_runs' / RUN_ID
MODELS = ['rf', 'lgbm', 'xgb', 'gbrt', 'mlp', 'ada', 'svr_gaussian', 'knn']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(args):
    result = subprocess.run([str(x) for x in args], text=True,
                            capture_output=True, timeout=60)
    require(result.returncode == 0,
            f'Command failed: {shlex.join([str(x) for x in args])}\n'
            f'{result.stdout}\n{result.stderr}')
    return result.stdout.strip()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2)


def connect(readonly=True):
    con = sqlite3.connect(DB.as_uri() + ('?mode=ro' if readonly else '?mode=rw'),
                          uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    if readonly:
        con.execute('PRAGMA query_only=ON')
    return con


def model_row(con):
    row = con.execute('SELECT * FROM models WHERE model_id=?', (MODEL,)).fetchone()
    require(row is not None, 'Model1C is missing.')
    return dict(row)


def assert_no_ml_jobs():
    output = command(['squeue', '-u', str(os.getuid()), '-h', '-o', '%A|%j'])
    blocked = [line for line in output.splitlines()
               if re.search(r'ML|Model1|predict', line, re.I)]
    require(not blocked, 'Active/pending ML jobs exist; do not retry concurrently:\n'
            + '\n'.join(blocked))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    for path in [DB, PYTHON, SOURCE, POSITION / 'run_manifest.json',
                 GLOBAL / 'run_manifest.json']:
        require(path.is_file(), f'Missing required file: {path}')
    assert_no_ml_jobs()
    accounting = command(['sacct', '-j', OLD_JOB, '-X', '-n', '-P',
                          '-o', 'JobIDRaw,State'])
    states = [line.split('|')[1].strip() for line in accounting.splitlines()
              if line.split('|')[0].strip() == OLD_JOB]
    require(states == ['TIMEOUT'], f'Expected timed-out original job: {accounting}')
    with connect() as con:
        row = model_row(con)
    jobs = json.loads(row['training_job_ids_json'])
    require(row['training_run_id'] == RUN_ID, 'Training run mapping changed.')
    require(str(jobs.get('position_resolved')) == OLD_JOB and
            str(jobs.get('global_molecular')) == GLOBAL_JOB, 'Job mapping changed.')
    require(row['status'] in ('training_submitted', 'training_failed'),
            f'Unexpected model status: {row["status"]}')
    require(not row['prediction_run_id'] and not row['chosen_algorithm'],
            'Model1C already has a prediction or winner; stop for review.')
    require(row['n_train'] == 47 and row['n_benchmark'] == 8, 'Counts changed.')
    require(json.loads(row['candidate_models_json']) == MODELS, 'Algorithms changed.')
    require(set(json.loads(row['candidate_representations_json'])) ==
            {'global_molecular', 'position_resolved'}, 'Representations changed.')

    pm = read_json(POSITION / 'run_manifest.json')
    gm = read_json(GLOBAL / 'run_manifest.json')
    for manifest, representation in [(pm, 'position_resolved'), (gm, 'global_molecular')]:
        require(manifest['run_id'] == RUN_ID and manifest['workflow'] == representation,
                'Manifest identity mismatch.')
        require(manifest['experiment']['logical_model_id'] == MODEL,
                'Manifest logical model mismatch.')
    require(pm['status'] in ('running', 'failed'), 'Position run is not incomplete.')
    require(gm['status'] == 'complete', 'Global run is not complete.')
    with (GLOBAL / 'results.csv').open(newline='') as handle:
        results = list(csv.DictReader(handle))
    require(len(results) == 8 and {x['model'] for x in results} == set(MODELS),
            'Global results must contain exactly all eight algorithms.')
    for name in ['dataset_ids_used.csv', 'train_ids.csv', 'test_ids.csv',
                 'split_assignments_used.csv']:
        require(sha(POSITION / name) == sha(GLOBAL / name),
                f'Representation membership mismatch: {name}')

    hashes = {}
    inputs = {}
    for key in ['dataset_membership', 'simulated_raw_descriptors',
                'reference_cleaned_descriptors', 'split_assignments']:
        record = pm['inputs'][key]
        path = Path(record['resolved_path'])
        require(path.is_file() and sha(path) == record['sha256'],
                f'Input missing or changed since original training: {path}')
        inputs[key] = path
        hashes[str(path)] = record['sha256']
    require(sha(SOURCE) == pm['script']['sha256'],
            'Live training script differs from the timed-out version; review first.')
    hashes[str(SOURCE)] = sha(SOURCE)
    experiment = pm['experiment']
    require(experiment['models'] == MODELS and experiment['n_trials'] == 50 and
            experiment['n_cv_folds'] == 5 and experiment['cv_random_seed'] == 42 and
            experiment['optuna_random_seed'] == 42, 'Original search settings differ.')

    # Import only central path configuration, not workflow or training code.
    registry = Path(command([PYTHON, '-B', '-c',
        'import sys; sys.path.insert(0, sys.argv[1]); '
        'from config.paths import RUN_REGISTRY_CSV; print(RUN_REGISTRY_CSV)', ML]))
    require(registry.is_file() and ML.resolve() in registry.resolve().parents,
            f'Cannot locate the existing run registry safely: {registry}')
    global_hashes = {str(p): sha(p) for p in GLOBAL.rglob('*') if p.is_file()}
    print('PREFLIGHT PASS: frozen inputs unchanged; global 8/8 complete; '
          'position incomplete; no active ML jobs.', flush=True)
    print('Retry: one node, one CPU, --n-jobs 1; all eight position algorithms; '
          '50 trials, five folds, original seeds and membership.', flush=True)
    if not args.apply:
        print('Read-only check finished. Run with --apply to archive and submit once.')
        return

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    recovery = ACTIVE / 'training_recoveries' / f'Model1C_position_{stamp}'
    recovery.mkdir(parents=True, exist_ok=False)
    print(f'Recovery records and Slurm logs: {recovery}', flush=True)
    with connect() as source, sqlite3.connect(recovery / 'pre_retry.db') as backup:
        source.backup(backup)
        require(backup.execute('PRAGMA integrity_check').fetchone()[0] == 'ok',
                'Database backup failed integrity check.')
    shutil.copy2(registry, recovery / 'run_registry_before.csv')
    shutil.copy2(SOURCE, recovery / 'training_script_before.py')
    write_json(recovery / 'model_before.json', row)
    write_json(recovery / 'global_files_before.json', global_hashes)
    write_json(recovery / 'input_hashes.json', hashes)
    for ext in ('out', 'err'):
        path = ML / 'slurm_out' / f'ML-{OLD_JOB}.{ext}'
        if path.is_file():
            shutil.copy2(path, recovery / path.name)

    archived = recovery / 'original_position_attempt'
    training_args = [
        '--run-id', RUN_ID, '--logical-model-id', MODEL,
        '--workflow-name', 'position_resolved',
        '--descriptor-version', experiment['descriptor_version'],
        '--dataset-name', experiment['dataset_name'],
        '--split-version', experiment['split_version'], '--target', 'DDAPL',
        '--models', ','.join(MODELS), '--n-trials', '50', '--n-jobs', '1',
        '--n-cv-folds', '5', '--cv-seed', '42', '--optuna-seed', '42',
        '--purpose', 'Model1C position retry; single-thread execution; original attempt archived',
        '--dataset-membership-path', str(inputs['dataset_membership']),
        '--descriptor-path', str(inputs['simulated_raw_descriptors']),
        '--reference-cleaned-descriptor-path', str(inputs['reference_cleaned_descriptors']),
        '--split-assignment-path', str(inputs['split_assignments'])]
    check_code = (
        'import hashlib,json,sys\n'
        'from pathlib import Path\n'
        'for filename, expected in json.loads(Path(sys.argv[1]).read_text()).items():\n'
        '    h=hashlib.sha256()\n'
        '    with open(filename,"rb") as f:\n'
        '        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)\n'
        '    if h.hexdigest()!=expected: raise SystemExit("Input changed: "+filename)\n')
    batch = recovery / 'Model1C_position_retry.sh'
    with batch.open('x') as handle:
        handle.write('#!/bin/bash\nset -eo pipefail\n')
        handle.write('export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
                     'export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE\n')
        handle.write('cd ' + shlex.quote(str(SOURCE.parent)) + '\n')
        handle.write('date --iso-8601=seconds\n')
        handle.write(shlex.join([str(PYTHON), '-B', '-c', check_code,
                                str(recovery / 'input_hashes.json')]) + '\n')
        handle.write(shlex.join([str(PYTHON), '-u', str(SOURCE)] + training_args) + '\n')
        handle.write('date --iso-8601=seconds\n')

    # Serialize registration; submit held, so it cannot write output before mapping.
    job = None
    with connect(readonly=False) as con:
        con.execute('BEGIN IMMEDIATE')
        require(model_row(con) == row, 'Model state changed during preflight.')
        assert_no_ml_jobs()
        require(not archived.exists(), 'Archive target already exists.')
        POSITION.rename(archived)
        print(f'Original position attempt moved, not deleted: {archived}', flush=True)
        try:
            output = command(['sbatch', '--parsable', '--hold', '--nodes=1',
                '--ntasks=1', '--cpus-per-task=1', '--time=08:00:00',
                '--account=tuttle-rmss', '--partition=standard', '--export=ALL',
                '--job-name=ML_Model1C_position_retry',
                f'--chdir={SOURCE.parent}',
                f'--output={recovery}/Model1C-position-%j.out',
                f'--error={recovery}/Model1C-position-%j.err', batch])
            job = output.split(';')[0]
            require(job.isdigit(), f'Unrecognized sbatch response: {output}')
            write_json(recovery / 'submission.json', {
                'job_id': job, 'old_job_id': OLD_JOB, 'run_id': RUN_ID,
                'archive': str(archived), 'execution_threads': 1,
                'same_run_id_reason': 'Preserve pairing with completed global representation',
                'submitted_held': True})
            jobs['position_resolved'] = job
            con.execute('UPDATE models SET training_job_id=?, training_job_ids_json=?, '
                        'status=?, updated_at=? WHERE model_id=?',
                        (job, json.dumps(jobs), 'training_submitted',
                         dt.datetime.now().isoformat(), MODEL))
            con.commit()
        except BaseException:
            print('STOP: no release requested. Any submitted retry remains held. '
                  f'Inspect {recovery}; do not submit another retry.', file=sys.stderr)
            # No automatic restoration: submission/registration may be ambiguous.
            raise
    require(all(sha(Path(p)) == h for p, h in global_hashes.items()),
            'Global files changed; retry remains held for review.')
    command(['scontrol', 'release', job])
    write_json(recovery / 'release_requested.json', {'job_id': job,
               'time': dt.datetime.now().isoformat()})
    print(f'SUBMITTED AND RELEASED: {job}')
    print('Global results, Model1C membership and other models were not reset.')
    print(f'squeue -j {job}')
    print(f'tail -n 40 {recovery}/Model1C-position-{job}.out')
    print(f'sacct -j {job} -X -n -P -o JobIDRaw,State,ExitCode,Elapsed')
    print('Do not rerun this recovery or submit other ML jobs until it finishes.')
    print('After COMPLETED: run workflow.py status to register the full comparison.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ABORT: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('No cancellation or deletion was requested. If --apply was used, '
              'inspect the printed recovery folder before any retry.', file=sys.stderr)
        sys.exit(1)
