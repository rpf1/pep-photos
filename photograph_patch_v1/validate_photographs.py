"""Explicit, isolated photograph pilot; never writes into Photos or simulations."""
import argparse
import hashlib
import json
from pathlib import Path
from photograph_validation import validate_gro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', type=Path, action='append', required=True,
                        help='Exact Boxes/2.1/{bilayer}/{run}/{molecule}_{nmol} directory; repeatable')
    parser.add_argument('--output', type=Path, required=True, help='NEW staging directory outside Photos')
    parser.add_argument('--render', action='store_true', help='Omit for read-only residue preflight')
    args = parser.parse_args()
    output = args.output.resolve()
    if any(x.startswith('Photos') for x in output.parts) or 'pfp34_v2' in output.parts:
        parser.error('Use a staging path outside Photos* and pfp34_v2')
    systems = [s.resolve(strict=True) for s in args.system]
    if len(set(systems)) != len(systems):
        parser.error('Duplicate systems supplied')
    for s in systems:
        if output == s or s in output.parents:
            parser.error('Output must not be in a simulation directory')
    reports = []
    for system in systems:
        gro, xtc = system / 'Relax.gro', system / 'Relax.xtc'
        if not xtc.is_file():
            parser.error(f'Missing {xtc}')
        report = validate_gro(gro)
        report['gro_sha256'] = hashlib.sha256(gro.read_bytes()).hexdigest()
        reports.append(report)
    print(json.dumps(reports, indent=2))
    if not args.render:
        return
    output.mkdir(parents=True, exist_ok=False)
    import Photograph as photo
    for index, system in enumerate(systems, 1):
        photo.MANUAL_MODE = True
        photo.MANUAL_SYSTEM_DIR = system
        photo.MANUAL_PHOTO_ROOT = output / f'system_{index:02d}'
        photo.main()
        panels = list(photo.MANUAL_PHOTO_ROOT.rglob('snapshot_top.png'))
        if len(panels) != 1:
            raise RuntimeError(f'Expected one rendered system; found {len(panels)}: {system}')
        from PIL import Image
        for filename, label in photo.SNAPSHOT_ORDER:
            path = panels[0].parent / filename
            with Image.open(path) as image:
                image.verify()
        reports[index-1]['ten_panels_verified'] = True
        if hashlib.sha256((system/'Relax.gro').read_bytes()).hexdigest() != reports[index-1]['gro_sha256']:
            raise RuntimeError('Input GRO changed during validation')
    (output/'validation.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(f'Pilot complete: {output}. Visual review and annotation transition still required.')

if __name__ == '__main__':
    main()
