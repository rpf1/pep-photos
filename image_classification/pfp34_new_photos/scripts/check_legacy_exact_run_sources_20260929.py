#!/usr/bin/env python3
from pathlib import Path
import csv
import json

ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids")
ANALYSIS = ROOT / "analysis"
BOXES = ROOT / "Boxes"
SESSION = ANALYSIS / "image_classification" / "pfp34_new_photos" / "session"

INFILE = SESSION / "legacy_photo_source_audit.csv"
OUTFILE = SESSION / "legacy_exact_run_source_audit.csv"

if not INFILE.exists():
    raise SystemExit(f"Missing input: {INFILE}")

with INFILE.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

out = []

for row in rows:
    molecule = row["molecule"]
    bilayer = row["bilayer_type"]
    run = row["run"]

    # PFPeptoids Boxes layout is normally:
    # Boxes/<box-version>/<bilayer>/<run>/<molecule>_<nmol>
    matches = sorted(
        p for p in BOXES.glob(f"*/{bilayer}/{run}/{molecule}_*")
        if p.is_dir()
    )

    # Record files relevant to re-rendering without changing anything.
    match_info = []
    for d in matches:
        files = {
            name: (d / name).exists()
            for name in [
                "Box.gro",
                "Pull.gro",
                "Pull.tpr",
                "Pull.xtc",
                "Relax.gro",
                "Relax.tpr",
                "Relax.xtc",
            ]
        }
        match_info.append({
            "system_dir": str(d),
            **files,
        })

    has_relax_gro = any(x["Relax.gro"] for x in match_info)
    has_box_gro = any(x["Box.gro"] for x in match_info)
    render_source_available = bool(matches) and (has_relax_gro or has_box_gro)

    out.append({
        "system_key": row["system_key"],
        "molecule": molecule,
        "bilayer_type": bilayer,
        "run": run,
        "annotation_status": row["annotation_status"],
        "label": row["label"],
        "current_evidence": row["current_evidence"],
        "exact_system_dir_count": len(matches),
        "exact_system_dirs": json.dumps([str(p) for p in matches]),
        "render_source_available": "YES" if render_source_available else "NO",
        "source_file_details": json.dumps(match_info),
    })

with OUTFILE.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys())
    w.writeheader()
    w.writerows(out)

print(f"Targets checked: {len(out)}")
print(f"Exact simulation source available: {sum(r['render_source_available']=='YES' for r in out)}")
print(f"Exact simulation source missing: {sum(r['render_source_available']=='NO' for r in out)}")
print(f"Saved: {OUTFILE}")
