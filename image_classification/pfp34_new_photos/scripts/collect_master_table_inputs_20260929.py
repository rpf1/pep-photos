#!/usr/bin/env python3
"""
Read-only PFPeptoids structured-data collector for the next master-table step.

Creates ONE ZIP containing:
- canonical annotation exports
- canonical APL master table
- morphology/local-lipid structured results
- other small structured analysis tables that may contain useful scalar metrics
- an inventory with original path, size, rows, columns and SHA256

No project source file is modified, moved, renamed, or deleted.
"""

from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required.")

ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids")
ANALYSIS = ROOT / "analysis"
MORPH = Path("/users/pfb19164/pfp_morphology_pilot")
SESSION = ANALYSIS / "image_classification" / "pfp34_new_photos" / "session"
CANON = SESSION / "canonical_export_20260929"

OUT = SESSION / "master_table_inputs_20260929.zip"

# Structured files <= 12 MB are small enough to include directly.
MAX_INCLUDE = 12 * 1024 * 1024

# Roots likely to contain useful simulation-derived scalar data.
SCAN_ROOTS = [
    ("analysis_results", ANALYSIS / "results"),
    ("bilayer_output", ANALYSIS / "bilayer_output"),
    ("aplanalysis_runs", ANALYSIS / "aplanalysis_runs"),
    ("photograph_validation", ANALYSIS / "photograph_validation"),
    ("morphology_results", MORPH / "results"),
]

STRUCTURED_EXTS = {".csv", ".tsv", ".json"}

# Avoid obvious raw/generated/image-heavy or provenance-only trees.
SKIP_DIR_NAMES = {
    ".git", "__pycache__", "figures", "images", "photos",
    "reference_photos", "balanced_panel_reference_photos",
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def table_shape_and_columns(path):
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path, low_memory=False)
            return len(df), list(df.columns), ""
        if ext == ".tsv":
            df = pd.read_csv(path, sep="\t", low_memory=False)
            return len(df), list(df.columns), ""
        if ext == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, list):
                if obj and isinstance(obj[0], dict):
                    return len(obj), sorted({k for r in obj if isinstance(r, dict) for k in r}), ""
                return len(obj), [], ""
            if isinstance(obj, dict):
                return 1, list(obj.keys()), ""
            return 1, [], ""
    except Exception as e:
        return None, [], f"{type(e).__name__}: {e}"
    return None, [], ""

def safe_rel(path, base):
    try:
        return str(path.relative_to(base))
    except Exception:
        return path.name

required = [
    CANON / "canonical_annotations.csv",
    CANON / "primary_stacked_labelled_annotations.csv",
    CANON / "unresolved_legacy_annotations.csv",
    CANON / "all_annotations_with_canonical_decision.csv",
    CANON / "summary.json",
    ANALYSIS / "masterdataanalysis.csv",
]

missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing required inputs:\n" + "\n".join(missing))

with tempfile.TemporaryDirectory(prefix="pfp_master_inputs_") as td:
    stage = Path(td)
    inventory = []
    copied_sources = set()

    def include(path, arcname, category, force=False):
        path = Path(path)
        if not path.is_file():
            return
        size = path.stat().st_size
        rows, cols, read_error = table_shape_and_columns(path)

        rec = {
            "category": category,
            "original_path": str(path),
            "archive_path": str(arcname),
            "bytes": size,
            "included": bool(force or size <= MAX_INCLUDE),
            "rows": rows,
            "column_count": len(cols),
            "columns": json.dumps(cols),
            "read_error": read_error,
            "sha256": sha256(path),
        }
        inventory.append(rec)

        if rec["included"] and str(path.resolve()) not in copied_sources:
            dst = stage / arcname
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied_sources.add(str(path.resolve()))

    # 1) Exact required current inputs.
    for p in required:
        if p.parent == CANON:
            arc = Path("canonical_annotations") / p.name
            cat = "canonical_annotations"
        else:
            arc = Path("apl_master") / p.name
            cat = "apl_master"
        include(p, arc, cat, force=True)

    # 2) Preserve the previous morphology-for-labels package if present.
    old_bundle = ANALYSIS / "morphology_for_labels_20260928_180130_322983.zip"
    if old_bundle.exists():
        size = old_bundle.stat().st_size
        inventory.append({
            "category": "previous_morphology_bundle",
            "original_path": str(old_bundle),
            "archive_path": "previous_bundle/morphology_for_labels_20260928_180130_322983.zip",
            "bytes": size,
            "included": True,
            "rows": None,
            "column_count": None,
            "columns": "[]",
            "read_error": "",
            "sha256": sha256(old_bundle),
        })
        dst = stage / "previous_bundle" / old_bundle.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_bundle, dst)

    # 3) Structured candidate data from likely result roots.
    seen = set()
    for root_name, root in SCAN_ROOTS:
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in SKIP_DIR_NAMES
                and not d.startswith(".")
            ]

            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in STRUCTURED_EXTS:
                    continue

                rp = str(p.resolve())
                if rp in seen or rp in copied_sources:
                    continue
                seen.add(rp)

                rel = safe_rel(p, root)
                arc = Path("candidate_structured_data") / root_name / rel
                include(p, arc, f"candidate:{root_name}", force=False)

    # 4) Inventory all canonical/master columns in a compact human-readable report.
    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv(stage / "candidate_tables_inventory.csv", index=False)

    summary = {
        "read_only_collection": True,
        "required_inputs_present": True,
        "inventory_entries": int(len(inv_df)),
        "included_files": int(inv_df["included"].sum()),
        "excluded_large_structured_files": int((~inv_df["included"]).sum()),
        "archive": str(OUT),
        "max_auto_include_bytes": MAX_INCLUDE,
        "roots_scanned": {name: str(path) for name, path in SCAN_ROOTS},
    }
    (stage / "COLLECTION_SUMMARY.json").write_text(json.dumps(summary, indent=2))

    readme = """PFPeptoids master-table input bundle
====================================

Purpose:
  Off-cluster inspection and construction of the canonical manual + simulation
  master table.

This bundle is READ-ONLY provenance collection. It does not imply that every
candidate table should be merged. candidate_tables_inventory.csv records the
source path, schema, hash and inclusion status so each metric can be assessed
before use.

Primary authoritative inputs:
  canonical_annotations/canonical_annotations.csv
  canonical_annotations/primary_stacked_labelled_annotations.csv
  canonical_annotations/unresolved_legacy_annotations.csv
  apl_master/masterdataanalysis.csv

Morphology/local-lipid candidate tables are included from the validated pilot
results where small enough. Generated figures/photos are deliberately excluded.
"""
    (stage / "README.txt").write_text(readme)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing bundle: {OUT}")

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))

print(f"Created: {OUT}")
print("Upload this ONE ZIP to ChatGPT.")

