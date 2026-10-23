#!/usr/bin/env python3
"""
PFPeptoids read-only reconciliation audit:
- ADA historical run -> run_0 mapping
- duplicate physical annotation rows
- remaining tri-peptoid LEGACY/NONE image evidence
- current simulation-source availability
- regenerated/STACKED image availability
- stale ADA references in masterdata / active-learning DB / project audit text

This script NEVER modifies annotations, images, simulations, databases, or Git state.
It only writes a new audit folder + ZIP under the current annotation session.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys

try:
    import pandas as pd
except ImportError:
    raise SystemExit("This audit requires pandas, which is already used by the PFPeptoids analysis environment.")

ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids")
ANALYSIS = ROOT / "analysis"
BOXES = ROOT / "Boxes"
ML = ANALYSIS / "machinelearning"
SESSION = ANALYSIS / "image_classification" / "pfp34_new_photos" / "session"

ANNOTATIONS = SESSION / "annotations.csv"
SYSTEM_IMAGES = SESSION / "system_images.csv"
STATE_DB = SESSION / "state.sqlite3"

ADA_RUN = "run_First_predictions-20260813_190947_583893_ada"
CANONICAL_RUN = "run_0"

OUTDIR = SESSION / "run_alias_photo_audit_20260929"
ZIPBASE = SESSION / "run_alias_photo_audit_20260929"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def is_tripeptoid(name):
    parts = str(name).split("-")
    return len(parts) == 3 and all(p.startswith("N") for p in parts)

def parse_image_paths(value):
    if pd.isna(value):
        return []
    try:
        x = json.loads(value)
        if isinstance(x, list):
            return [str(v) for v in x]
    except Exception:
        pass
    return [str(value)]

def evidence_kind(value):
    paths = parse_image_paths(value)
    if not paths:
        return "NONE"
    joined = " ".join(paths)
    kinds = []
    if "NEW10__" in joined:
        kinds.append("STACKED")
    if "LEGACY__" in joined:
        kinds.append("LEGACY")
    if not kinds:
        kinds.append("OTHER")
    return "+".join(kinds)

def source_file_flags(d):
    names = [
        "Box.gro", "Min.gro", "Pull.gro", "Pull.tpr", "Pull.xtc",
        "Relax.gro", "Relax.tpr", "Relax.xtc",
    ]
    return {n: (d / n).exists() for n in names}

def has_render_source(d):
    f = source_file_flags(d)
    return bool(f["Relax.gro"] or f["Box.gro"])

def exact_system_dirs(molecule, bilayer, run):
    hits = []
    if not BOXES.exists():
        return hits
    for p in BOXES.glob(f"*/{bilayer}/{run}/{molecule}_*"):
        if p.is_dir():
            hits.append(p)
    return sorted(set(hits))

def archived_duplicate_dirs(molecule, bilayer):
    hits = []
    if not BOXES.exists():
        return hits
    patterns = [
        f"*/{bilayer}/accidental_duplicates/{molecule}_*",
        f"*/{bilayer}/**/accidental_duplicates/**/{molecule}_*",
    ]
    for pat in patterns:
        for p in BOXES.glob(pat):
            if p.is_dir():
                hits.append(p)
    return sorted(set(hits))

def safe_json(obj):
    return json.dumps(obj, sort_keys=True)

def details_for_dirs(paths):
    out = []
    for p in paths:
        flags = source_file_flags(p)
        out.append({
            "path": str(p),
            "render_source_available": bool(flags["Relax.gro"] or flags["Box.gro"]),
            **flags,
        })
    return out

def build_png_index(roots):
    """One filesystem walk, then cheap filtering for all targets."""
    records = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Avoid hidden/git/cache trees.
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
            for fn in filenames:
                if fn.lower().endswith(".png"):
                    p = Path(dirpath) / fn
                    records.append((root.name, str(p), fn))
    return records

def find_image_candidates(index, molecule, bilayer, run):
    exact = []
    other_run = []
    for root_name, path, fn in index:
        if molecule not in path or bilayer not in path:
            continue
        is_stacked = (
            "NEW10__" in fn
            or "STACKED" in path.upper()
            or "photograph_regeneration" in path
        )
        if not is_stacked:
            continue
        rec = {"root": root_name, "path": path}
        if run in path:
            exact.append(rec)
        else:
            other_run.append(rec)
    return exact, other_run

def write_df(df, path):
    df.to_csv(path, index=False)

def scan_sqlite_for_literal(db_path, literal):
    results = []
    if not db_path.exists():
        return results
    con = sqlite3.connect(db_path)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con
        )["name"].tolist()
        for table in tables:
            safe_table = table.replace('"', '""')
            try:
                cols = pd.read_sql_query(f'PRAGMA table_info("{safe_table}")', con)
            except Exception:
                continue
            text_cols = cols.loc[
                cols["type"].astype(str).str.upper().str.contains("CHAR|TEXT|CLOB"),
                "name"
            ].tolist()
            for col in text_cols:
                safe_col = col.replace('"', '""')
                try:
                    q = (
                        f'SELECT rowid AS _rowid_, "{safe_col}" AS _value_ '
                        f'FROM "{safe_table}" '
                        f'WHERE CAST("{safe_col}" AS TEXT) LIKE ? LIMIT 200'
                    )
                    hits = pd.read_sql_query(q, con, params=[f"%{literal}%"])
                except Exception:
                    continue
                for _, r in hits.iterrows():
                    results.append({
                        "database": str(db_path),
                        "table": table,
                        "column": col,
                        "rowid": r["_rowid_"],
                        "value": r["_value_"],
                    })
    finally:
        con.close()
    return results

def scan_text_for_literal(roots, literal):
    allowed_ext = {
        ".csv", ".tsv", ".json", ".txt", ".md", ".py", ".sh", ".tcl",
        ".log", ".yaml", ".yml", ".toml"
    }
    skip_dirs = {
        ".git", "__pycache__", "Photos", "photograph_regeneration",
        "output_2", "models", "figures", "prediction_runs"
    }
    hits = []
    seen = set()

    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p in seen:
                    continue
                seen.add(p)
                if p.suffix.lower() not in allowed_ext:
                    continue
                try:
                    if p.stat().st_size > 20 * 1024 * 1024:
                        continue
                    with p.open("r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if literal in line:
                                hits.append({
                                    "file": str(p),
                                    "line": lineno,
                                    "text": line.strip()[:1000],
                                })
                                if len(hits) >= 1000:
                                    return hits
                except Exception:
                    pass
    return hits

def git_log_search(repo, literal):
    if not (repo / ".git").exists():
        return []
    commands = [
        ["git", "-C", str(repo), "log", "--all", "--oneline", "--max-count=100", f"-S{literal}"],
        ["git", "-C", str(repo), "log", "--all", "--oneline", "--max-count=100", "--grep", literal],
    ]
    lines = []
    for cmd in commands:
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if cp.stdout:
                lines.extend(x for x in cp.stdout.splitlines() if x.strip())
        except Exception as e:
            lines.append(f"SEARCH_ERROR: {e}")
    # preserve order / unique
    return list(dict.fromkeys(lines))

# ---------------------------------------------------------------------
# Preconditions / output
# ---------------------------------------------------------------------

for required in [ANNOTATIONS, SYSTEM_IMAGES, STATE_DB]:
    if not required.exists():
        raise SystemExit(f"Missing required current-session file: {required}")

OUTDIR.mkdir(parents=True, exist_ok=True)

ann = pd.read_csv(ANNOTATIONS)
system_images = pd.read_csv(SYSTEM_IMAGES)

ann["is_tripeptoid"] = ann["molecule"].map(is_tripeptoid)
ann["evidence_kind"] = ann["image_paths"].map(evidence_kind)

tri = ann[ann["is_tripeptoid"]].copy()
ada = tri[tri["run"].astype(str) == ADA_RUN].copy()
run0 = tri[tri["run"].astype(str) == CANONICAL_RUN].copy()

# ---------------------------------------------------------------------
# 1. ADA -> run_0 row-by-row reconciliation
# ---------------------------------------------------------------------

ada_rows = []

for _, a in ada.iterrows():
    same = run0[
        (run0["molecule"] == a["molecule"]) &
        (run0["bilayer_type"] == a["bilayer_type"])
    ]

    current_ada_dirs = exact_system_dirs(a["molecule"], a["bilayer_type"], ADA_RUN)
    current_run0_dirs = exact_system_dirs(a["molecule"], a["bilayer_type"], CANONICAL_RUN)
    archived_dirs = archived_duplicate_dirs(a["molecule"], a["bilayer_type"])

    if len(same) == 1:
        r0 = same.iloc[0]
        run0_system_key = r0["system_key"]
        run0_status = r0["annotation_status"]
        run0_label = r0["label"]
        run0_evidence = r0["evidence_kind"]
        run0_event = r0.get("event_id")
        same_label = (
            (pd.isna(a["label"]) and pd.isna(run0_label))
            or str(a["label"]) == str(run0_label)
        )
        same_status = str(a["annotation_status"]) == str(run0_status)
    else:
        run0_system_key = ""
        run0_status = ""
        run0_label = ""
        run0_evidence = ""
        run0_event = ""
        same_label = False
        same_status = False

    no_current_ada_dir = len(current_ada_dirs) == 0
    run0_source_ok = any(has_render_source(p) for p in current_run0_dirs)
    run0_current_row_ok = len(same) == 1
    run0_stacked = run0_evidence == "STACKED"

    if no_current_ada_dir and run0_source_ok and run0_current_row_ok and run0_stacked:
        disposition = "STRONG_ADA_TO_RUN0_ALIAS"
    elif no_current_ada_dir and run0_source_ok and run0_current_row_ok:
        disposition = "PROBABLE_ADA_TO_RUN0_ALIAS"
    elif current_ada_dirs:
        disposition = "ADA_SOURCE_STILL_EXISTS_REVIEW"
    else:
        disposition = "UNRESOLVED"

    if not same_label or not same_status:
        disposition += ";ANNOTATION_CONFLICT"

    if archived_dirs:
        disposition += ";ARCHIVED_DUPLICATE_PRESENT"

    ada_rows.append({
        "ada_system_key": a["system_key"],
        "molecule": a["molecule"],
        "bilayer_type": a["bilayer_type"],
        "ada_run": a["run"],
        "ada_status": a["annotation_status"],
        "ada_label": a["label"],
        "ada_evidence": a["evidence_kind"],
        "ada_event_id": a.get("event_id"),
        "run0_match_count_in_annotations": len(same),
        "run0_system_key": run0_system_key,
        "run0_status": run0_status,
        "run0_label": run0_label,
        "run0_evidence": run0_evidence,
        "run0_event_id": run0_event,
        "same_label": same_label,
        "same_status": same_status,
        "current_ada_source_dir_count": len(current_ada_dirs),
        "current_ada_source_dirs": safe_json(details_for_dirs(current_ada_dirs)),
        "run0_source_dir_count": len(current_run0_dirs),
        "run0_source_dirs": safe_json(details_for_dirs(current_run0_dirs)),
        "archived_duplicate_dir_count": len(archived_dirs),
        "archived_duplicate_dirs": safe_json(details_for_dirs(archived_dirs)),
        "disposition": disposition,
    })

ada_audit = pd.DataFrame(ada_rows)
write_df(ada_audit, OUTDIR / "01_ada_to_run0_alias_audit.csv")

# ---------------------------------------------------------------------
# 2. Canonical physical duplicate groups
# ---------------------------------------------------------------------

tri["canonical_run_for_audit"] = tri["run"].replace({ADA_RUN: CANONICAL_RUN})
tri["canonical_physical_key"] = (
    tri["molecule"].astype(str) + "|" +
    tri["bilayer_type"].astype(str) + "|" +
    tri["canonical_run_for_audit"].astype(str)
)

dup_counts = tri["canonical_physical_key"].value_counts()
dup_keys = set(dup_counts[dup_counts > 1].index)

dup_cols = [
    "canonical_physical_key", "system_key", "molecule", "bilayer_type",
    "run", "canonical_run_for_audit", "annotation_status", "label",
    "evidence_kind", "event_id", "annotation_timestamp", "last_modified",
    "relative_path", "image_paths"
]
physical_dups = tri[tri["canonical_physical_key"].isin(dup_keys)][dup_cols].copy()
physical_dups = physical_dups.sort_values(
    ["canonical_physical_key", "run", "event_id"], na_position="last"
)
write_df(physical_dups, OUTDIR / "02_canonical_physical_duplicate_groups.csv")

# ---------------------------------------------------------------------
# 3. Remaining non-ADA LEGACY/NONE tri-peptoids
# ---------------------------------------------------------------------

remaining = tri[
    (tri["run"].astype(str) != ADA_RUN) &
    (tri["evidence_kind"].isin(["LEGACY", "NONE"]))
].copy()

image_index = build_png_index([
    ANALYSIS / "photograph_regeneration",
    ANALYSIS / "Photos",
    ANALYSIS / "photograph_validation",
])

remaining_rows = []

for _, r in remaining.iterrows():
    exact_dirs = exact_system_dirs(r["molecule"], r["bilayer_type"], r["run"])
    run0_dirs = (
        exact_system_dirs(r["molecule"], r["bilayer_type"], CANONICAL_RUN)
        if r["run"] != CANONICAL_RUN else exact_dirs
    )

    exact_images, other_run_images = find_image_candidates(
        image_index, r["molecule"], r["bilayer_type"], r["run"]
    )

    exact_source_ok = any(has_render_source(p) for p in exact_dirs)
    run0_source_ok = any(has_render_source(p) for p in run0_dirs)

    if exact_images:
        next_action = "USE_EXISTING_EXACT_STACKED_IMAGE"
    elif exact_source_ok:
        next_action = "REGENERATE_EXACT_RUN_STACKED_IMAGE"
    elif r["run"] != CANONICAL_RUN and run0_source_ok:
        next_action = "POSSIBLE_RUN_ALIAS_REQUIRES_PROVENANCE_CHECK"
    else:
        next_action = "SOURCE_NOT_FOUND_MANUAL_REVIEW"

    remaining_rows.append({
        "system_key": r["system_key"],
        "molecule": r["molecule"],
        "bilayer_type": r["bilayer_type"],
        "run": r["run"],
        "annotation_status": r["annotation_status"],
        "label": r["label"],
        "evidence_kind": r["evidence_kind"],
        "event_id": r.get("event_id"),
        "exact_source_dir_count": len(exact_dirs),
        "exact_source_dirs": safe_json(details_for_dirs(exact_dirs)),
        "run0_source_dir_count": len(run0_dirs),
        "run0_source_dirs": safe_json(details_for_dirs(run0_dirs)),
        "exact_external_stacked_image_count": len(exact_images),
        "exact_external_stacked_images": safe_json(exact_images),
        "other_run_stacked_image_count": len(other_run_images),
        "other_run_stacked_images": safe_json(other_run_images),
        "next_action": next_action,
    })

remaining_audit = pd.DataFrame(remaining_rows)
write_df(remaining_audit, OUTDIR / "03_remaining_nonada_legacy_none_audit.csv")

# ---------------------------------------------------------------------
# 4. Current annotation / system-image consistency
# ---------------------------------------------------------------------

img_counts = system_images.groupby("system_key").size().rename("manifest_image_rows")
consistency = ann.merge(
    img_counts, how="left", left_on="system_key", right_index=True
)
consistency["manifest_image_rows"] = consistency["manifest_image_rows"].fillna(0).astype(int)
consistency["declared_image_count"] = consistency["image_count"].fillna(0).astype(int)
consistency["image_count_agrees"] = (
    consistency["manifest_image_rows"] == consistency["declared_image_count"]
)

problem_consistency = consistency[
    (~consistency["image_count_agrees"]) |
    (consistency["manifest_image_rows"] == 0)
][[
    "system_key", "molecule", "bilayer_type", "run", "annotation_status",
    "label", "evidence_kind", "declared_image_count", "manifest_image_rows",
    "image_count_agrees"
]]
write_df(problem_consistency, OUTDIR / "04_annotation_manifest_consistency_problems.csv")

# ---------------------------------------------------------------------
# 5. masterdataanalysis.csv: stale ADA rows and relevant current rows
# ---------------------------------------------------------------------

master_path = ANALYSIS / "masterdataanalysis.csv"
master_ada = pd.DataFrame()
master_relevant = pd.DataFrame()

if master_path.exists():
    try:
        master = pd.read_csv(master_path, low_memory=False)

        # Find rows containing the literal ADA run/path anywhere.
        text = master.astype(str)
        ada_mask = text.apply(
            lambda col: col.str.contains(ADA_RUN, regex=False, na=False)
        ).any(axis=1)
        master_ada = master[ada_mask].copy()

        # Relevant rows by molecule names from ADA annotation records.
        ada_molecules = set(ada["molecule"].astype(str))
        molecule_cols = [
            c for c in master.columns
            if c.lower() in {
                "molecule", "molecule_name", "token_sequence",
                "peptoid", "sequence", "id"
            }
        ]
        if molecule_cols:
            rel_mask = pd.Series(False, index=master.index)
            for c in molecule_cols:
                rel_mask |= master[c].astype(str).isin(ada_molecules)
            master_relevant = master[rel_mask].copy()

        write_df(master_ada, OUTDIR / "05_masterdata_stale_ada_rows.csv")
        write_df(master_relevant, OUTDIR / "06_masterdata_rows_for_ada_molecules.csv")
    except Exception as e:
        (OUTDIR / "05_masterdata_read_error.txt").write_text(str(e))
else:
    (OUTDIR / "05_masterdata_missing.txt").write_text(str(master_path))

# ---------------------------------------------------------------------
# 6. SQLite stale ADA references
# ---------------------------------------------------------------------

db_hits = []
for db in [
    STATE_DB,
    ML / "shared_data" / "active_learning" / "active_learning.db",
]:
    db_hits.extend(scan_sqlite_for_literal(db, ADA_RUN))

db_hits_df = pd.DataFrame(db_hits)
if db_hits_df.empty:
    db_hits_df = pd.DataFrame(
        columns=["database", "table", "column", "rowid", "value"]
    )
write_df(db_hits_df, OUTDIR / "07_sqlite_ada_literal_references.csv")

# ---------------------------------------------------------------------
# 7. Text/audit provenance references
# ---------------------------------------------------------------------

text_hits = scan_text_for_literal([
    ROOT / "17082026_cleanup",
    ANALYSIS / "campaign_state_audits",
    ANALYSIS / "d2_provenance_cleanup",
    ANALYSIS / "image_classification",
    ML / "shared_data",
], ADA_RUN)

text_hits_df = pd.DataFrame(text_hits)
if text_hits_df.empty:
    text_hits_df = pd.DataFrame(columns=["file", "line", "text"])
write_df(text_hits_df, OUTDIR / "08_textual_ada_provenance_references.csv")

# ---------------------------------------------------------------------
# 8. Git-history search for the ADA identifier
# ---------------------------------------------------------------------

git_lines = []
for repo_name, repo in [
    ("ROOT", ROOT),
    ("ANALYSIS", ANALYSIS),
    ("MACHINELEARNING", ML),
]:
    for line in git_log_search(repo, ADA_RUN):
        git_lines.append({"repository": repo_name, "line": line})

git_df = pd.DataFrame(git_lines)
if git_df.empty:
    git_df = pd.DataFrame(columns=["repository", "line"])
write_df(git_df, OUTDIR / "09_git_history_ada_search.csv")

# ---------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------

ada_dispositions = (
    Counter(ada_audit["disposition"]) if not ada_audit.empty else Counter()
)
remaining_actions = (
    Counter(remaining_audit["next_action"]) if not remaining_audit.empty else Counter()
)

summary = {
    "read_only_audit": True,
    "annotation_total_rows": int(len(ann)),
    "tripeptoid_rows_name_based_filter": int(len(tri)),
    "ada_annotation_rows": int(len(ada)),
    "ada_unique_molecules": int(ada["molecule"].nunique()),
    "ada_rows_with_exactly_one_run0_annotation_counterpart": int(
        (ada_audit["run0_match_count_in_annotations"] == 1).sum()
    ) if not ada_audit.empty else 0,
    "ada_rows_with_current_ada_source_dir": int(
        (ada_audit["current_ada_source_dir_count"] > 0).sum()
    ) if not ada_audit.empty else 0,
    "ada_rows_with_current_run0_source_dir": int(
        (ada_audit["run0_source_dir_count"] > 0).sum()
    ) if not ada_audit.empty else 0,
    "ada_rows_with_archived_duplicate_dir": int(
        (ada_audit["archived_duplicate_dir_count"] > 0).sum()
    ) if not ada_audit.empty else 0,
    "ada_rows_same_label_as_run0": int(ada_audit["same_label"].sum())
        if not ada_audit.empty else 0,
    "ada_rows_same_status_as_run0": int(ada_audit["same_status"].sum())
        if not ada_audit.empty else 0,
    "ada_dispositions": dict(ada_dispositions),
    "canonicalized_duplicate_physical_groups": int(len(dup_keys)),
    "remaining_nonada_legacy_or_none_rows": int(len(remaining)),
    "remaining_next_actions": dict(remaining_actions),
    "annotation_manifest_problem_rows": int(len(problem_consistency)),
    "masterdata_ada_literal_rows": int(len(master_ada)),
    "sqlite_ada_literal_hits_total": int(len(db_hits_df)),
    "textual_ada_reference_hits_total": int(len(text_hits_df)),
    "git_history_search_hits_total": int(len(git_df)),
}

(OUTDIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2))

lines = [
    "PFPEPTOIDS ADA/RUN_0 + LEGACY PHOTO RECONCILIATION AUDIT",
    "=" * 68,
    "",
    "READ-ONLY: no annotation/image/simulation/database/Git state was modified.",
    "",
    f"Current annotation rows: {summary['annotation_total_rows']}",
    f"Tri-peptoid rows (name-based audit filter only): {summary['tripeptoid_rows_name_based_filter']}",
    "",
    f"ADA annotation rows: {summary['ada_annotation_rows']}",
    f"ADA unique molecules: {summary['ada_unique_molecules']}",
    f"ADA rows with exactly one run_0 annotation counterpart: "
        f"{summary['ada_rows_with_exactly_one_run0_annotation_counterpart']}",
    f"ADA rows with a CURRENT ADA simulation directory: "
        f"{summary['ada_rows_with_current_ada_source_dir']}",
    f"ADA rows with a CURRENT run_0 simulation directory: "
        f"{summary['ada_rows_with_current_run0_source_dir']}",
    f"ADA rows with an archived duplicate directory: "
        f"{summary['ada_rows_with_archived_duplicate_dir']}",
    f"ADA rows with same label as run_0 counterpart: "
        f"{summary['ada_rows_same_label_as_run0']}",
    f"ADA rows with same status as run_0 counterpart: "
        f"{summary['ada_rows_same_status_as_run0']}",
    f"Canonicalized duplicate physical groups: "
        f"{summary['canonicalized_duplicate_physical_groups']}",
    "",
    "ADA dispositions:",
]
for k, v in sorted(summary["ada_dispositions"].items()):
    lines.append(f"  {k}: {v}")

lines += [
    "",
    f"Remaining NON-ADA tri-peptoid LEGACY/NONE rows: "
        f"{summary['remaining_nonada_legacy_or_none_rows']}",
    "Remaining next actions:",
]
for k, v in sorted(summary["remaining_next_actions"].items()):
    lines.append(f"  {k}: {v}")

lines += [
    "",
    f"Annotation/manifest consistency problem rows: "
        f"{summary['annotation_manifest_problem_rows']}",
    f"masterdataanalysis.csv rows still containing ADA literal: "
        f"{summary['masterdata_ada_literal_rows']}",
    f"SQLite ADA literal hits: {summary['sqlite_ada_literal_hits_total']}",
    f"Text/audit ADA reference hits: {summary['textual_ada_reference_hits_total']}",
    f"Git-history search hits: {summary['git_history_search_hits_total']}",
    "",
    "FILES:",
    "  01_ada_to_run0_alias_audit.csv",
    "  02_canonical_physical_duplicate_groups.csv",
    "  03_remaining_nonada_legacy_none_audit.csv",
    "  04_annotation_manifest_consistency_problems.csv",
    "  05_masterdata_stale_ada_rows.csv (or read/missing note)",
    "  06_masterdata_rows_for_ada_molecules.csv",
    "  07_sqlite_ada_literal_references.csv",
    "  08_textual_ada_provenance_references.csv",
    "  09_git_history_ada_search.csv",
    "  SUMMARY.json",
]

(OUTDIR / "SUMMARY.txt").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------
# 10. ZIP everything for one-upload handoff
# ---------------------------------------------------------------------

zip_path = shutil.make_archive(str(ZIPBASE), "zip", root_dir=OUTDIR)

print("\n".join(lines))
print()
print(f"ZIP FOR CHATGPT: {zip_path}")

