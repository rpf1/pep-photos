#!/usr/bin/env python3

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
)

BOX_ROOT = ROOT / "Boxes" / "2.1"

PHOTO_TCL = ROOT / "analysis" / "Photograph.tcl"
PHOTO_COMPONENTS_TCL = ROOT / "analysis" / "Photograph_components.tcl"
PHOTO_PY = ROOT / "analysis" / "Photograph.py"

# Ignore obvious non-peptoid N* residue names.
IGNORE_RESNAMES = {
    "NA",
    "NA+",
}

# Prefer one representative GRO per physical system to avoid counting
# Box.gro, Min.gro, Pull.gro, Relax.gro etc. as independent systems.
GRO_PRIORITY = [
    "Relax.gro",
    "Pull.gro",
    "Box.gro",
    "Min.gro",
]

# Useful known/possible aliases to show separately.
KNOWN_ALIAS_PAIRS = [
    ("Nfnap", "Nfnaph"),
    ("NfeC4", "Nfe4Cl"),
    ("NfeB4", "Nfe4Br"),
    ("Na", "Nx"),
]


# ============================================================
# HELPERS
# ============================================================

def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(errors="replace")


def gro_resnames(path: Path) -> set[str]:
    """
    Extract residue names using fixed-width GRO columns:
        columns 6-10 = residue name
    """
    names = set()

    with path.open(errors="replace") as handle:
        lines = handle.readlines()

    if len(lines) < 3:
        return names

    # Ignore title, atom-count line and final box line.
    for line in lines[2:-1]:
        if len(line) < 10:
            continue

        resname = line[5:10].strip()

        if resname:
            names.add(resname)

    return names


def choose_representative_gro(system_dir: Path) -> Path | None:
    for name in GRO_PRIORITY:
        p = system_dir / name
        if p.is_file():
            return p

    gros = sorted(system_dir.glob("*.gro"))
    return gros[0] if gros else None


def extract_tcl_set_members(text: str) -> set[str]:
    """
    Extract residue-like tokens from Tcl source.

    We intentionally restrict this to names beginning with uppercase N
    because the current investigation concerns peptoid residue naming.

    This is not being used to modify anything; it is a conservative audit.
    """
    return set(
        re.findall(r"\bN[A-Za-z0-9+_-]*\b", text)
    )


def extract_python_peptoid_residues(text: str) -> set[str]:
    """
    Pull quoted N* tokens from Photograph.py.

    This is deliberately broad enough to catch the PEPTOID_RESIDUES
    configuration without depending on exact formatting.
    """
    return set(
        re.findall(
            r"""["'](N[A-Za-z0-9+_-]*)["']""",
            text,
        )
    )


def system_identity(system_dir: Path) -> str:
    try:
        rel = system_dir.relative_to(BOX_ROOT)
        return str(rel)
    except ValueError:
        return str(system_dir)


def molecule_from_system_dir(system_dir: Path) -> str:
    """
    Typical folder:
        Nfnap-NfeC4-NfeC4_80

    Return:
        Nfnap-NfeC4-NfeC4
    """
    name = system_dir.name
    return re.sub(r"_\d+$", "", name)


def expected_name_tokens(system_dir: Path) -> set[str]:
    """
    Extract hyphen-separated N* names from the molecule folder name.

    This is useful because it independently describes what the system is
    supposed to contain.
    """
    molecule = molecule_from_system_dir(system_dir)

    return {
        token
        for token in molecule.split("-")
        if token.startswith("N")
    }


# ============================================================
# LOAD CURRENT PHOTOGRAPH DEFINITIONS
# ============================================================

print("=" * 78)
print("PFPEPTOIDS PHOTOGRAPH RESIDUE-NAME AUDIT")
print("=" * 78)
print()

for path in [BOX_ROOT, PHOTO_TCL, PHOTO_COMPONENTS_TCL, PHOTO_PY]:
    if not path.exists():
        raise FileNotFoundError(f"Required path does not exist:\n{path}")

photo_tcl_text = read_text(PHOTO_TCL)
components_tcl_text = read_text(PHOTO_COMPONENTS_TCL)
photo_py_text = read_text(PHOTO_PY)

photo_tcl_names = extract_tcl_set_members(photo_tcl_text)
components_tcl_names = extract_tcl_set_members(
    components_tcl_text
)
photo_py_names = extract_python_peptoid_residues(
    photo_py_text
)

recognised_anywhere = (
    photo_tcl_names
    | components_tcl_names
    | photo_py_names
)


# ============================================================
# DISCOVER PHYSICAL SYSTEM DIRECTORIES
# ============================================================

system_dirs = []

for path in BOX_ROOT.rglob("*"):
    if not path.is_dir():
        continue

    if choose_representative_gro(path) is not None:
        # System directories should normally end with "_<nmol>".
        if re.search(r"_\d+$", path.name):
            system_dirs.append(path)

system_dirs = sorted(set(system_dirs))


# ============================================================
# AUDIT
# ============================================================

actual_residue_counts = Counter()
folder_token_counts = Counter()

actual_systems = defaultdict(set)
folder_systems = defaultdict(set)

system_records = []

for system_dir in system_dirs:

    gro = choose_representative_gro(system_dir)

    if gro is None:
        continue

    identity = system_identity(system_dir)

    actual = {
        x
        for x in gro_resnames(gro)
        if x.startswith("N") and x not in IGNORE_RESNAMES
    }

    expected = {
        x
        for x in expected_name_tokens(system_dir)
        if x not in IGNORE_RESNAMES
    }

    for name in actual:
        actual_residue_counts[name] += 1
        actual_systems[name].add(identity)

    for name in expected:
        folder_token_counts[name] += 1
        folder_systems[name].add(identity)

    system_records.append(
        {
            "system": identity,
            "gro": str(gro),
            "actual": actual,
            "expected": expected,
        }
    )


actual_names = set(actual_residue_counts)
folder_names = set(folder_token_counts)

all_dataset_names = actual_names | folder_names


# ============================================================
# SUMMARY
# ============================================================

print(f"Systems inspected: {len(system_records)}")
print()

print("Recognised N* names:")
print(f"  Photograph.tcl            : {len(photo_tcl_names)}")
print(f"  Photograph_components.tcl : {len(components_tcl_names)}")
print(f"  Photograph.py             : {len(photo_py_names)}")
print()

print("Dataset N* names:")
print(f"  Actually present in GROs  : {len(actual_names)}")
print(f"  Present in molecule names : {len(folder_names)}")
print()


# ============================================================
# ACTUAL GRO NAMES
# ============================================================

print("=" * 78)
print("1. ACTUAL N* RESIDUE NAMES PRESENT IN REPRESENTATIVE GRO FILES")
print("=" * 78)

for name in sorted(actual_names):
    print(
        f"{name:<12} "
        f"{actual_residue_counts[name]:>4} systems"
    )


# ============================================================
# MAIN FAILURE LIST
# ============================================================

missing_from_standard_tcl = (
    actual_names - photo_tcl_names
)

print()
print("=" * 78)
print("2. ACTUAL GRO NAMES NOT RECOGNISED BY Photograph.tcl")
print("=" * 78)

if not missing_from_standard_tcl:
    print("NONE")
else:
    for name in sorted(missing_from_standard_tcl):
        print()
        print(
            f"{name}  "
            f"({actual_residue_counts[name]} systems)"
        )

        for system in sorted(actual_systems[name]):
            print(f"    {system}")


# ============================================================
# COMPONENT TCL DIFFERENCES
# ============================================================

missing_from_components_tcl = (
    actual_names - components_tcl_names
)

print()
print("=" * 78)
print(
    "3. ACTUAL GRO NAMES NOT RECOGNISED BY "
    "Photograph_components.tcl"
)
print("=" * 78)

if not missing_from_components_tcl:
    print("NONE")
else:
    for name in sorted(missing_from_components_tcl):
        print()
        print(
            f"{name}  "
            f"({actual_residue_counts[name]} systems)"
        )

        for system in sorted(actual_systems[name]):
            print(f"    {system}")


# ============================================================
# PYTHON CONFIG DIFFERENCES
# ============================================================

missing_from_python = actual_names - photo_py_names

print()
print("=" * 78)
print("4. ACTUAL GRO NAMES NOT PRESENT IN Photograph.py CONFIG")
print("=" * 78)

if not missing_from_python:
    print("NONE")
else:
    for name in sorted(missing_from_python):
        print(
            f"{name:<12} "
            f"{actual_residue_counts[name]:>4} systems"
        )


# ============================================================
# FOLDER NAME VS GRO NAME
# ============================================================

print()
print("=" * 78)
print("5. SYSTEMS WHERE MOLECULE FOLDER NAMES AND GRO RESNAMES DIFFER")
print("=" * 78)

n_name_mismatch = 0

for record in system_records:

    expected = record["expected"]
    actual = record["actual"]

    # Only compare expected N* molecule tokens.
    missing_expected = expected - actual

    if not missing_expected:
        continue

    n_name_mismatch += 1

    print()
    print(record["system"])
    print(f"  GRO: {record['gro']}")
    print(
        "  molecule-name tokens : "
        + " ".join(sorted(expected))
    )
    print(
        "  actual GRO N* names  : "
        + " ".join(sorted(actual))
    )
    print(
        "  expected but absent  : "
        + " ".join(sorted(missing_expected))
    )

print()
print(
    f"Systems with folder/GRO naming differences: "
    f"{n_name_mismatch}"
)


# ============================================================
# KNOWN ALIAS CHECKS
# ============================================================

print()
print("=" * 78)
print("6. KNOWN / SUSPECTED ALIAS CHECKS")
print("=" * 78)

for dataset_name, render_name in KNOWN_ALIAS_PAIRS:

    print()
    print(f"{dataset_name}  <->  {render_name}")

    print(
        f"  dataset GRO contains {dataset_name}: "
        f"{dataset_name in actual_names}"
    )

    print(
        f"  dataset GRO contains {render_name}: "
        f"{render_name in actual_names}"
    )

    print(
        f"  Photograph.tcl contains {dataset_name}: "
        f"{dataset_name in photo_tcl_names}"
    )

    print(
        f"  Photograph.tcl contains {render_name}: "
        f"{render_name in photo_tcl_names}"
    )


# ============================================================
# NAMES ONLY IN SOURCE
# ============================================================

source_only = photo_tcl_names - actual_names

print()
print("=" * 78)
print("7. N* NAMES IN Photograph.tcl BUT NEVER SEEN IN DATASET GROs")
print("=" * 78)

for name in sorted(source_only):
    print(name)


# ============================================================
# HIGH-PRIORITY IMPACT SUMMARY
# ============================================================

print()
print("=" * 78)
print("8. HIGH-PRIORITY PHOTOGRAPH.tcl MISMATCH SUMMARY")
print("=" * 78)

if not missing_from_standard_tcl:
    print("No actual dataset residue names are missing from Photograph.tcl.")
else:
    ranked = sorted(
        missing_from_standard_tcl,
        key=lambda x: (
            -actual_residue_counts[x],
            x,
        ),
    )

    print(
        f"{'Residue':<14}"
        f"{'Affected systems':>18}"
    )
    print("-" * 32)

    for name in ranked:
        print(
            f"{name:<14}"
            f"{actual_residue_counts[name]:>18}"
        )


# ============================================================
# TARGET SYSTEM
# ============================================================

TARGET = "Nfnap-NfeC4-NfeC4"

print()
print("=" * 78)
print(f"9. TARGET SYSTEM CHECK: {TARGET}")
print("=" * 78)

found_target = False

for record in system_records:

    if molecule_from_system_dir(
        BOX_ROOT / record["system"]
    ) != TARGET:
        continue

    found_target = True

    print()
    print(record["system"])
    print("  Actual GRO names:")
    print(
        "    " + " ".join(sorted(record["actual"]))
    )

    unrecognised = (
        record["actual"] - photo_tcl_names
    )

    print("  Unrecognised by Photograph.tcl:")
    if unrecognised:
        print(
            "    " + " ".join(sorted(unrecognised))
        )
    else:
        print("    NONE")

if not found_target:
    print("Target system was not found.")


# ============================================================
# DONE
# ============================================================

print()
print("=" * 78)
print("AUDIT COMPLETE — READ ONLY")
print("=" * 78)
print()
print("No project files were modified.")
