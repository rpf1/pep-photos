"""Exact-name, fail-closed GRO validation shared by both photograph pipelines."""
from pathlib import Path
import re
from collections import Counter

REGISTRY = Path(__file__).with_name('photograph_residues.tcl')
GROUPS = {
    name: values.split()
    for name, values in re.findall(r'^set ((?:pep|pto)_\w+)\s+\{([^}]*)\}', REGISTRY.read_text(), re.M)
}
PEPTOID_RESIDUES = {k.removeprefix('pto_'): v for k, v in GROUPS.items() if k.startswith('pto_')}
PEPTOID_RESIDUES['proline_like'] = []
ENVIRONMENT = set('POPC POPS POPE POPG DPPC DTPC CHOL H2O W SW TW WF NA CL NA+ CL- ION'.split())
CATEGORY = {}
for group, names in GROUPS.items():
    for name in names:
        if name in CATEGORY:
            raise ValueError(f'Duplicate registry residue: {name}')
        CATEGORY[name] = group


def tcl_path(path):
    path = str(Path(path).resolve())
    if any(c in path for c in '{}\n\r'):
        raise ValueError(f'Unsupported Tcl path characters: {path!r}')
    return '{' + path + '}'


def validate_gro(path):
    """Every non-environment atom must belong to an explicit polymer category.

    No inference from directory names; use fixed-width GRO resnames, preserving
    case. Fail closed on unknown lipids/solvents too, pending explicit review.
    """
    path = Path(path)
    with path.open() as handle:
        handle.readline()
        count = int(handle.readline().strip())
        atoms = Counter()
        for index in range(count):
            line = handle.readline()
            if len(line.rstrip('\n')) < 44:
                raise ValueError(f'Malformed GRO atom {index + 1}: {path}')
            name = line[5:10].strip()
            if not name:
                raise ValueError(f'Empty GRO residue name: {path}')
            atoms[name] += 1
        box = handle.readline().split()
        if len(box) not in (3, 9):
            raise ValueError(f'Malformed GRO box: {path}')
        list(map(float, box))
    unknown = sorted(set(atoms) - ENVIRONMENT - CATEGORY.keys())
    if unknown:
        raise ValueError(f'FATAL unsupported GRO residues {unknown} in {path}; rendering blocked. '
                         'Nkeqm is intentionally unsupported; no residue may be omitted.')
    polymer = {n: {'category': CATEGORY[n], 'atoms': c} for n, c in atoms.items() if n in CATEGORY}
    if not polymer:
        raise ValueError(f'FATAL no recognized polymer atoms in {path}')
    return {'gro': str(path.resolve()), 'total_atoms': count,
            'polymer_atoms': sum(x['atoms'] for x in polymer.values()), 'residues': polymer}
