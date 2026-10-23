#!/usr/bin/env python3
"""
Analyse exactly two pristine/reference bilayer final structures (prod.gro):
  1) POPE75POPG25_W_WF  (bacterial-like)
  2) POPC60CHOL40_W_WF  (mammalian-like)

This is intentionally a TWO-BOX, ONE-FRAME-EACH analysis.
No trajectory is read. The validated production morphology geometry kernel is reused.

Outputs:
  baseline_core.csv
  baseline_all.csv
  baseline_long.csv
  provenance.json
  README.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import MDAnalysis as mda


DEFAULT_PROJECT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids"
)

DEFAULT_BACTERIAL = (
    DEFAULT_PROJECT
    / "Reference/2.1/bilayers/POPE75POPG25_W_WF/prod.gro"
)
DEFAULT_MAMMALIAN = (
    DEFAULT_PROJECT
    / "Reference/2.1/bilayers/POPC60CHOL40_W_WF/prod.gro"
)
DEFAULT_GEOMETRY = (
    DEFAULT_PROJECT
    / "analysis/image_classification/morph_prod/scripts/morphology_geometry.py"
)
DEFAULT_OUT = (
    DEFAULT_PROJECT
    / "analysis/image_classification/morph_prod/results/baselines"
    / "prod_20261001"
)

SYSTEMS = (
    {
        "membrane": "POPE75POPG25_W_WF",
        "role": "bacterial",
        "marker_resnames": ("POPE", "POPG"),
        "membrane_resnames": ("POPE", "POPG"),
        "expected_po4_total": 512,
        "expected_membrane_total": 512,
    },
    {
        "membrane": "POPC60CHOL40_W_WF",
        "role": "mammalian",
        "marker_resnames": ("POPC",),
        "membrane_resnames": ("POPC", "CHOL"),
        "expected_po4_total": 432,
        "expected_membrane_total": 720,
    },
)

GRID_N = 32
MIN_WAVELENGTH_NM = 4.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-box pristine reference bilayer morphology analysis."
    )
    p.add_argument("--bacterial-gro", type=Path, default=DEFAULT_BACTERIAL)
    p.add_argument("--mammalian-gro", type=Path, default=DEFAULT_MAMMALIAN)
    p.add_argument("--geometry-script", type=Path, default=DEFAULT_GEOMETRY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_geometry_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Validated geometry kernel not found: {path}")
    spec = importlib.util.spec_from_file_location("morphology_geometry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import geometry kernel: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gro_title(path: Path) -> str:
    with path.open("rt", encoding="utf-8", errors="replace") as f:
        return f.readline().strip()


def residue_counts(u):
    out = {}
    for r in u.residues:
        name = str(r.resname)
        out[name] = out.get(name, 0) + 1
    return out


def analyse_one(path: Path, spec: dict, G):
    if not path.is_file():
        raise FileNotFoundError(path)

    # GRO is a single static structure. No XTC is opened.
    u = mda.Universe(str(path), to_guess=())

    dims_A = np.asarray(u.dimensions, dtype=float)
    if dims_A.shape[0] < 3 or not np.all(np.isfinite(dims_A[:3])):
        raise ValueError(f"Invalid box dimensions in {path}: {dims_A}")

    marker_sel = "name PO4 and resname " + " ".join(spec["marker_resnames"])
    po4 = u.select_atoms(marker_sel)
    actual_po4 = len(po4)
    if actual_po4 != spec["expected_po4_total"]:
        raise ValueError(
            f"{spec['membrane']}: PO4 count {actual_po4} != "
            f"expected {spec['expected_po4_total']}"
        )
    if len(np.unique(po4.resindices)) != actual_po4:
        raise ValueError(f"{spec['membrane']}: duplicate PO4 markers per residue")

    mem_sel = "resname " + " ".join(spec["membrane_resnames"])
    actual_membrane_total = len(u.select_atoms(mem_sel).residues)
    if actual_membrane_total != spec["expected_membrane_total"]:
        raise ValueError(
            f"{spec['membrane']}: membrane residue count {actual_membrane_total} != "
            f"expected {spec['expected_membrane_total']}"
        )

    geom = G.build_surface_geometry(
        po4.positions.copy(),
        u.dimensions.copy(),
        min_wavelength_nm=MIN_WAVELENGTH_NM,
        grid_n=GRID_N,
    )
    metrics = G.geometry_metrics(geom, spec["expected_membrane_total"])

    counts = residue_counts(u)
    record = {
        "membrane": spec["membrane"],
        "membrane_role": spec["role"],
        "source_path": str(path.resolve()),
        "source_title": gro_title(path),
        "source_sha256": sha256(path),
        "source_size_bytes": path.stat().st_size,
        "source_mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "snapshot_definition": "single prod.gro final reference structure",
        "grid_n": GRID_N,
        "min_wavelength_nm": MIN_WAVELENGTH_NM,
        "box_x_nm": float(dims_A[0] / 10.0),
        "box_y_nm": float(dims_A[1] / 10.0),
        "box_z_nm": float(dims_A[2] / 10.0),
        "actual_po4_total": actual_po4,
        "actual_membrane_total": actual_membrane_total,
        "POPE_count": counts.get("POPE", 0),
        "POPG_count": counts.get("POPG", 0),
        "POPC_count": counts.get("POPC", 0),
        "CHOL_count": counts.get("CHOL", 0),
    }
    record.update(metrics)
    record["projected_APL_nm2"] = record["projected_APL_A2"] / 100.0
    record["surface_APL_proxy_nm2"] = record["surface_APL_proxy_A2"] / 100.0
    return record


def write_rows(path: Path, rows):
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()

    # Hard safety guard: this program is intentionally restricted to exactly two boxes.
    if len(SYSTEMS) != 2:
        raise RuntimeError("Internal safety guard failed: expected exactly two reference boxes.")

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {out}\n"
            "Choose a new --out path if you intentionally want another run."
        )
    out.mkdir(parents=True)

    G = load_geometry_module(args.geometry_script.resolve())

    paths = {
        "bacterial": args.bacterial_gro.resolve(),
        "mammalian": args.mammalian_gro.resolve(),
    }

    rows = []
    for spec in SYSTEMS:
        rows.append(analyse_one(paths[spec["role"]], spec, G))

    # Exactly two rows must be written.
    if len(rows) != 2:
        raise RuntimeError(f"Expected exactly 2 results, got {len(rows)}")

    full_path = out / "baseline_all.csv"
    write_rows(full_path, rows)

    core_columns = [
        "membrane",
        "membrane_role",
        "source_path",
        "projected_area_nm2",
        "projected_APL_A2",
        "midplane_surface_area_nm2",
        "midplane_excess_area_fraction",
        "midplane_height_rms_nm",
        "midplane_H_rms_nm_inv",
        "thickness_median_nm",
        "upper_surface_area_nm2",
        "lower_surface_area_nm2",
        "upper_excess_area_fraction",
        "lower_excess_area_fraction",
        "upper_height_rms_nm",
        "lower_height_rms_nm",
        "upper_H_rms_nm_inv",
        "lower_H_rms_nm_inv",
        "surface_area_abs_upper_minus_lower_nm2",
        "excess_area_abs_upper_minus_lower",
        "height_rms_abs_upper_minus_lower_nm",
        "H_rms_abs_upper_minus_lower_nm_inv",
        "thickness_valid_fraction",
        "fit_rmse_upper_nm",
        "fit_rmse_lower_nm",
        "actual_po4_total",
        "actual_membrane_total",
    ]
    core_rows = [{c: r.get(c, "") for c in core_columns} for r in rows]
    write_rows(out / "baseline_core.csv", core_rows)

    # Long-format table for direct bacterial/mammalian comparison.
    identity = {
        "membrane", "membrane_role", "source_path", "source_title",
        "source_sha256", "source_size_bytes", "source_mtime_utc",
        "snapshot_definition"
    }
    numeric_keys = []
    for k, v in rows[0].items():
        if k not in identity and isinstance(v, (int, float, np.integer, np.floating)):
            numeric_keys.append(k)

    long_rows = []
    for metric in numeric_keys:
        long_rows.append({
            "metric": metric,
            "bacterial_POPE75POPG25": rows[0].get(metric, ""),
            "mammalian_POPC60CHOL40": rows[1].get(metric, ""),
        })
    write_rows(out / "baseline_long.csv", long_rows)

    provenance = {
        "analysis_scope": "EXACTLY_TWO_REFERENCE_BOXES",
        "frame_scope": "one static final prod.gro structure per box",
        "trajectory_used": False,
        "sasa_calculated": False,
        "sasa_note": (
            "SASA is intentionally not calculated here. Static morphology uses coordinates "
            "and box vectors from prod.gro; a defensible SASA calculation should additionally "
            "use an appropriate topology/TPR and an explicitly documented selection/probe."
        ),
        "geometry_kernel": {
            "path": str(args.geometry_script.resolve()),
            "sha256": sha256(args.geometry_script.resolve()),
            "grid_n": GRID_N,
            "min_wavelength_nm": MIN_WAVELENGTH_NM,
            "method": "validated production Fourier-surface morphology kernel",
            "marker": "PO4 phospholipid geometry",
        },
        "inputs": [
            {
                "membrane": r["membrane"],
                "path": r["source_path"],
                "sha256": r["source_sha256"],
                "title": r["source_title"],
            }
            for r in rows
        ],
        "outputs": [
            "baseline_core.csv",
            "baseline_all.csv",
            "baseline_long.csv",
            "provenance.json",
            "README.md",
        ],
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# Reference-box baseline morphology

Scope: **exactly two pre-insertion reference bilayer boxes**.

1. `POPE75POPG25_W_WF` — `{args.bacterial_gro}`
2. `POPC60CHOL40_W_WF` — `{args.mammalian_gro}`

Each input is analysed **once**, using the single structure contained in `prod.gro`.
No XTC trajectory is read and there is no per-peptoid/per-run repetition.

The analysis reuses the validated production morphology kernel with:
- PO4 geometry markers
- 4 nm minimum Fourier wavelength
- 32 × 32 grid
- projected area/APL
- leaflet and midplane surface area
- excess surface area
- height RMS
- RMS mean curvature
- local-normal thickness
- upper/lower leaflet asymmetry
- geometry/QC diagnostics

`prod.gro` is sufficient for these static geometric measurements because it contains
the coordinates and periodic box vectors for the requested final reference structure.

SASA is **not** included in this output. A GRO coordinate file alone is not the
appropriate basis for a fully specified SASA calculation; if SASA is required, add
the corresponding topology/TPR and define the selection/probe explicitly.

Main files:
- `baseline_core.csv`: concise two-row scientific table
- `baseline_all.csv`: every emitted geometry/QC metric
- `baseline_long.csv`: side-by-side metric comparison
- `provenance.json`: exact inputs, hashes and method
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print("REFERENCE BASELINE ANALYSIS: COMPLETE")
    print(f"Exactly two boxes analysed: {len(rows)}")
    print(f"Output: {out}")
    for r in rows:
        print(
            f"{r['membrane']}: "
            f"APL={r['projected_APL_A2']:.6f} A^2, "
            f"height_RMS={r['midplane_height_rms_nm']:.6f} nm, "
            f"excess_area={r['midplane_excess_area_fraction']:.6f}, "
            f"H_RMS={r['midplane_H_rms_nm_inv']:.6f} nm^-1, "
            f"thickness={r['thickness_median_nm']:.6f} nm"
        )


if __name__ == "__main__":
    main()
