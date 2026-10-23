#!/usr/bin/env python3
from pathlib import Path
import csv
import json
import re

ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids")
ANALYSIS = ROOT / "analysis"
SESSION = ANALYSIS / "image_classification" / "pfp34_new_photos" / "session"
ANNOTATIONS = SESSION / "annotations.csv"

SEARCH_ROOTS = [
    ANALYSIS / "photograph_regeneration",
    ANALYSIS / "Photos",
    ANALYSIS / "photograph_validation",
]

OUT = SESSION / "legacy_photo_source_audit.csv"

def is_tripeptoid(name):
    parts = name.split("-")
    return len(parts) == 3 and all(p.startswith("N") for p in parts)

def evidence_kind(image_paths):
    try:
        paths = json.loads(image_paths or "[]")
    except Exception:
        paths = []
    if not paths:
        return "NONE"
    names = " ".join(paths)
    if "NEW10__" in names:
        return "STACKED"
    if "LEGACY__" in names:
        return "LEGACY"
    return "OTHER"

if not ANNOTATIONS.exists():
    raise SystemExit(f"Missing: {ANNOTATIONS}")

with ANNOTATIONS.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

targets = [
    r for r in rows
    if is_tripeptoid(r["molecule"])
    and evidence_kind(r.get("image_paths", "")) != "STACKED"
]

# Index candidate PNGs once.
pngs = []
for root in SEARCH_ROOTS:
    if root.exists():
        for p in root.rglob("*.png"):
            pngs.append(p)

out_rows = []
for r in targets:
    molecule = r["molecule"]
    bilayer = r["bilayer_type"]
    run = r["run"]

    candidates = []
    for p in pngs:
        s = str(p)
        n = p.name
        # Require exact molecule and bilayer somewhere in the path/name.
        if molecule not in s or bilayer not in s:
            continue
        # Prefer exact run when it is represented in output paths/names.
        run_match = run in s
        # Likely combined/stacked annotation image.
        combined_hint = (
            "ALL_BILAYERS" in n
            or "__nmol_" in n
            or "stack" in s.lower()
            or "photograph_regeneration" in s
        )
        candidates.append((not run_match, not combined_hint, s))

    candidates.sort()
    candidate_paths = [x[2] for x in candidates]

    out_rows.append({
        "system_key": r["system_key"],
        "molecule": molecule,
        "bilayer_type": bilayer,
        "run": run,
        "annotation_status": r["annotation_status"],
        "label": r["label"],
        "current_evidence": evidence_kind(r.get("image_paths", "")),
        "current_image_paths": r.get("image_paths", ""),
        "candidate_count": len(candidate_paths),
        "candidate_paths": json.dumps(candidate_paths),
        "has_candidate_outside_gui": "YES" if candidate_paths else "NO",
    })

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_rows[0].keys() if out_rows else [
        "system_key","molecule","bilayer_type","run","annotation_status","label",
        "current_evidence","current_image_paths","candidate_count","candidate_paths",
        "has_candidate_outside_gui"
    ])
    w.writeheader()
    w.writerows(out_rows)

print(f"Targets checked: {len(out_rows)}")
print(f"With candidate external images: {sum(r['has_candidate_outside_gui']=='YES' for r in out_rows)}")
print(f"Without candidate external images: {sum(r['has_candidate_outside_gui']=='NO' for r in out_rows)}")
print(f"Saved: {OUT}")
