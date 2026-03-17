# -*- coding: utf-8 -*-
"""
Photograph.py

Created on Mon Jan  8 22:54:47 2024

@author: Orignally rkb19187 modified by pfb19164

Purpose
-------
Batch-generate VMD snapshot panels for PFPeptoids systems.

What it does
------------
1. Scans the current PFPeptoids folder layout:
   Boxes/2.1/{bilayer}/run_x/{molecule}_{nmol}

2. Chooses trajectory source with strict logic:
   - Prefer Relax.gro + Relax.xtc if they exist
   - Only fall back to Pull.gro + Pull.xtc if:
       ALLOW_PULL_FALLBACK = True
     and Relax files do not exist

3. Checks that the chosen trajectory has reached a minimum final time:
   - Relax: MIN_RELAX_TIME_PS
   - Pull : MIN_PULL_TIME_PS

4. Writes a system-specific Tcl file based on Photograph.tcl

5. Runs VMD to render:
   snapshot_top_bilayer.png
   snapshot_side_bilayer.png
   snapshot_top.png
   snapshot_side.png
   snapshot_bottom.png
   snapshot_water.png

6. Trims whitespace around those PNGs using ImageMagick `convert`
   so the panels pack tightly and consistently

7. Creates:
   a) One horizontal stitched image per system
   b) One vertical stack per molecule/run combining all bilayers

Why trimming is used
--------------------
VMD snapshots often contain large white borders/background margins.
Trimming removes empty whitespace so:
- stitched panels align better
- final images waste less space
- output looks much cleaner for reports/slides

Notes
-----
- Generated Tcl files are stored alongside each system's photo outputs
  inside the analysis/Photos tree, not inside the raw simulation folders.
- If you want Pull fallback, set ALLOW_PULL_FALLBACK = True manually.
- This script expects ImageMagick `convert` and VMD to be available.
"""

import copy
import os
import shutil
import subprocess
from pathlib import Path

import MDAnalysis as mda
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# USER SETTINGS
# =============================================================================

SCRIPT_DIR = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis")
DATA_ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1")
TCL_TEMPLATE = SCRIPT_DIR / "Photograph.tcl"
PHOTO_ROOT = SCRIPT_DIR / "Photos"

FORCEFIELD_FOLDER = "2.1"

# Strict stage preference
ALLOW_PULL_FALLBACK = False

# Completion thresholds
MIN_RELAX_TIME_PS = 100000.0
MIN_PULL_TIME_PS = 100000.0

# Rendering command
VMD_COMMAND = "vglrun vmd -e Photograph.tcl"

# Image trimming
TRIM_IMAGES = True
TRIM_FUZZ = "7%"

# Panel labels / titles
DRAW_LABELS = True
LABEL_FONT_SIZE = 28
TITLE_FONT_SIZE = 34
LABEL_COLOR = "black"
TITLE_COLOR = "black"
LABEL_PADDING = 12
TITLE_BAND_HEIGHT = 56
PANEL_LABEL_BAND_HEIGHT = 42

# Final naming convention
# system image: {molecule}__nmol_{nmol}__{bilayer}__{run}.png
# molecule stack: {molecule}__ALL_BILAYERS__{run}.png

# Sort order for stacked bilayer overview
BILAYER_PRIORITY = [
    "POPC60CHOL40_W_WF",
    "POPC80POPS20_W_WF",
    "POPE75POPG25_W_WF",
]

# Expected VMD output order
SNAPSHOT_ORDER = [
    ("snapshot_top_bilayer.png", "Top bilayer"),
    ("snapshot_side_bilayer.png", "Side bilayer"),
    ("snapshot_top.png", "Top"),
    ("snapshot_side.png", "Side"),
    ("snapshot_bottom.png", "Bottom"),
    ("snapshot_water.png", "Water"),
]


# =============================================================================
# HELPERS
# =============================================================================

def readin(fname):
    with open(fname, "r", errors="ignore") as f:
        return f.read()


def sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
    )


def split_system_folder_name(folder_name: str):
    molecule_name, nmol_str = folder_name.rsplit("_", 1)
    return molecule_name, int(nmol_str)


def ordered_bilayers(bilayers):
    bilayers = list(bilayers)
    ordered = [b for b in BILAYER_PRIORITY if b in bilayers]
    ordered.extend(sorted([b for b in bilayers if b not in BILAYER_PRIORITY]))
    return ordered


def find_existing_file(system_dir: Path, candidates):
    for candidate in candidates:
        p = system_dir / candidate
        if p.exists():
            return p
    return None


def get_stage_files(system_dir: Path):
    relax_gro = find_existing_file(system_dir, ["Relax.gro", "relax.gro"])
    relax_xtc = find_existing_file(system_dir, ["Relax.xtc", "relax.xtc"])
    pull_gro = find_existing_file(system_dir, ["Pull.gro", "pull.gro"])
    pull_xtc = find_existing_file(system_dir, ["Pull.xtc", "pull.xtc"])

    # Prefer Relax always if present
    if relax_gro is not None and relax_xtc is not None:
        return {
            "stage": "Relax",
            "gro": relax_gro,
            "xtc": relax_xtc,
            "min_time_ps": MIN_RELAX_TIME_PS,
        }

    # Pull only if explicitly allowed
    if ALLOW_PULL_FALLBACK and pull_gro is not None and pull_xtc is not None:
        return {
            "stage": "Pull",
            "gro": pull_gro,
            "xtc": pull_xtc,
            "min_time_ps": MIN_PULL_TIME_PS,
        }

    if pull_gro is not None or pull_xtc is not None:
        raise FileNotFoundError(
            f"Relax stage missing in {system_dir}. "
            f"Pull stage exists but ALLOW_PULL_FALLBACK is False."
        )

    raise FileNotFoundError(
        f"No usable Relax/Pull files found in {system_dir}."
    )


def check_trajectory_complete(gro_file: Path, xtc_file: Path, min_time_ps: float):
    u = mda.Universe(str(gro_file), str(xtc_file))
    last_frame = u.trajectory[-1]
    final_time = float(last_frame.time)
    if final_time < min_time_ps:
        raise ValueError(
            f"Incomplete trajectory: final time {final_time:.1f} ps < required {min_time_ps:.1f} ps"
        )
    return final_time


def write_system_tcl(template_text: str, gro_name: str, xtc_name: str, outpath: Path):
    tcl = copy.copy(template_text)
    tcl = tcl.replace("mol new AAAADEMC_80.psf type psf", f"mol new {gro_name} type gro")
    tcl = tcl.replace("mol addfile AAAADEMC_80.pdb type pdb waitfor all",
                      f"mol addfile {xtc_name} type xtc waitfor all")
    with open(outpath, "w") as f:
        f.write(tcl)


def trim_png_inplace(img_path: Path):
    cmd = [
        "convert",
        str(img_path),
        "-fuzz", TRIM_FUZZ,
        "-trim",
        str(img_path),
    ]
    subprocess.run(cmd, check=False)


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_title_band(img: Image.Image, title: str):
    if not DRAW_LABELS:
        return img

    width, height = img.size
    new_img = Image.new("RGB", (width, height + TITLE_BAND_HEIGHT), "white")
    new_img.paste(img, (0, TITLE_BAND_HEIGHT))

    draw = ImageDraw.Draw(new_img)
    font = load_font(TITLE_FONT_SIZE)
    draw.text((LABEL_PADDING, 10), title, fill=TITLE_COLOR, font=font)
    return new_img


def add_panel_label(img: Image.Image, label: str):
    if not DRAW_LABELS:
        return img

    width, height = img.size
    new_img = Image.new("RGB", (width, height + PANEL_LABEL_BAND_HEIGHT), "white")
    new_img.paste(img, (0, PANEL_LABEL_BAND_HEIGHT))

    draw = ImageDraw.Draw(new_img)
    font = load_font(LABEL_FONT_SIZE)
    draw.text((LABEL_PADDING, 6), label, fill=LABEL_COLOR, font=font)
    return new_img


def make_horizontal_panel(image_paths, output_path: Path, title: str):
    panels = []

    for img_path, label in image_paths:
        img = Image.open(img_path).convert("RGB")
        img = add_panel_label(img, label)
        panels.append(img)

    total_width = sum(img.size[0] for img in panels)
    max_height = max(img.size[1] for img in panels)

    stitched = Image.new("RGB", (total_width, max_height), "white")

    x = 0
    for img in panels:
        y = (max_height - img.size[1]) // 2
        stitched.paste(img, (x, y))
        x += img.size[0]

    stitched = add_title_band(stitched, title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path, "PNG")

    for img in panels:
        img.close()


def make_vertical_stack(image_paths, output_path: Path, title: str):
    rows = [Image.open(p).convert("RGB") for p in image_paths]

    max_width = max(img.size[0] for img in rows)
    total_height = sum(img.size[1] for img in rows)

    stitched = Image.new("RGB", (max_width, total_height), "white")

    y = 0
    for img in rows:
        x = (max_width - img.size[0]) // 2
        stitched.paste(img, (x, y))
        y += img.size[1]

    stitched = add_title_band(stitched, title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path, "PNG")

    for img in rows:
        img.close()


def run_vmd_in_system_dir(system_dir: Path):
    subprocess.run(VMD_COMMAND, cwd=system_dir, shell=True, check=False)


def collect_snapshot_paths(system_dir: Path):
    paths = []
    for fname, label in SNAPSHOT_ORDER:
        p = system_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Expected VMD snapshot missing: {p}")
        if TRIM_IMAGES:
            trim_png_inplace(p)
        paths.append((p, label))
    return paths


def system_image_name(molecule_name: str, nmol: int, bilayer: str, run_name: str):
    return f"{sanitize(molecule_name)}__nmol_{nmol}__{sanitize(bilayer)}__{sanitize(run_name)}.png"


def stacked_image_name(molecule_name: str, run_name: str):
    return f"{sanitize(molecule_name)}__ALL_BILAYERS__{sanitize(run_name)}.png"


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not TCL_TEMPLATE.exists():
        raise FileNotFoundError(f"Missing Tcl template: {TCL_TEMPLATE}")

    PHOTO_ROOT.mkdir(parents=True, exist_ok=True)
    template_text = readin(TCL_TEMPLATE)

    generated_system_panels = {}

    for bilayer_dir in sorted(DATA_ROOT.iterdir()):
        if not bilayer_dir.is_dir():
            continue

        bilayer = bilayer_dir.name

        for run_dir in sorted(bilayer_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            run_name = run_dir.name

            for system_dir in sorted(run_dir.iterdir()):
                if not system_dir.is_dir():
                    continue

                system_folder_name = system_dir.name
                molecule_name, nmol = split_system_folder_name(system_folder_name)

                stage_info = None
                try:
                    stage_info = get_stage_files(system_dir)
                except Exception as exc:
                    print(f"SKIP {system_dir}: {exc}")
                    continue

                gro_file = stage_info["gro"]
                xtc_file = stage_info["xtc"]
                stage = stage_info["stage"]
                min_time_ps = stage_info["min_time_ps"]

                try:
                    final_time = check_trajectory_complete(gro_file, xtc_file, min_time_ps)
                except Exception as exc:
                    print(f"SKIP {system_dir}: {exc}")
                    continue

                out_dir = PHOTO_ROOT / sanitize(molecule_name) / sanitize(bilayer) / sanitize(run_name)
                out_dir.mkdir(parents=True, exist_ok=True)

                panel_name = system_image_name(molecule_name, nmol, bilayer, run_name)
                final_panel_path = out_dir / panel_name

                if final_panel_path.exists():
                    print(f"SKIP existing {final_panel_path}")
                    generated_system_panels.setdefault((molecule_name, run_name), []).append((bilayer, final_panel_path))
                    continue

                # Save generated Tcl with analysis outputs
                tcl_out = out_dir / "Photograph_generated.tcl"
                write_system_tcl(
                    template_text=template_text,
                    gro_name=gro_file.name,
                    xtc_name=xtc_file.name,
                    outpath=tcl_out,
                )

                # Copy Tcl into system folder temporarily because VMD is run there
                system_tcl = system_dir / "Photograph.tcl"
                shutil.copy2(tcl_out, system_tcl)

                print(f"Running VMD for {molecule_name} | {bilayer} | {run_name} | stage={stage} | final_time={final_time:.1f} ps")
                run_vmd_in_system_dir(system_dir)

                try:
                    snapshot_paths = collect_snapshot_paths(system_dir)
                except Exception as exc:
                    print(f"FAILED snapshot collection {system_dir}: {exc}")
                    continue

                title = f"{molecule_name} | nmol={nmol} | {bilayer} | {run_name} | {stage}"
                make_horizontal_panel(snapshot_paths, final_panel_path, title=title)

                generated_system_panels.setdefault((molecule_name, run_name), []).append((bilayer, final_panel_path))

                # Clean up per-system raw snapshot PNGs from simulation folder
                for fname, _ in SNAPSHOT_ORDER:
                    p = system_dir / fname
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

                print(f"SAVED {final_panel_path}")

    # Additional stacked image: all bilayers for same molecule/run
    for (molecule_name, run_name), bilayer_panels in generated_system_panels.items():
        if not bilayer_panels:
            continue

        bilayer_panels_sorted = sorted(
            bilayer_panels,
            key=lambda x: (
                BILAYER_PRIORITY.index(x[0]) if x[0] in BILAYER_PRIORITY else 999,
                x[0]
            )
        )

        panel_paths = [p for _, p in bilayer_panels_sorted]
        stack_dir = PHOTO_ROOT / sanitize(molecule_name) / "STACKED"
        stack_dir.mkdir(parents=True, exist_ok=True)

        stack_path = stack_dir / stacked_image_name(molecule_name, run_name)
        title = f"{molecule_name} | {run_name} | all bilayers"
        make_vertical_stack(panel_paths, stack_path, title=title)

        print(f"SAVED STACK {stack_path}")


if __name__ == "__main__":
    main()