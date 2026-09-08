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
import textwrap

# =============================================================================
# USER SETTINGS
# =============================================================================
# OPTIONAL MANUAL INPUT OVERRIDE
# Default = False
# False -> normal full scan from DATA_ROOT
# True  -> analyze only one target you provide
#
# PROVIDE EITHER:
#   MANUAL_SYSTEM_DIR:
#       Path to one simulation folder containing Relax/Pull files
#
# OR both:
#   MANUAL_GRO_FILE
#   MANUAL_XTC_FILE
#
# Example:
# MANUAL_MODE = True
# MANUAL_SYSTEM_DIR = Path("/path/to/system_folder")
#
# or
#
# MANUAL_MODE = True
# MANUAL_GRO_FILE = Path("/path/to/Relax.gro")
# MANUAL_XTC_FILE = Path("/path/to/Relax.xtc")
# =============================================================================
MANUAL_MODE         = False       #True or False
MANUAL_SYSTEM_DIR   = None        #path or None
MANUAL_GRO_FILE     = None        #file or None
MANUAL_XTC_FILE     = None        #file or None

# Optional manual labels when using direct file paths outside normal folder layout
MANUAL_MOLECULE_NAME = "manual_system"
MANUAL_NMOL = 1
MANUAL_BILAYER = "manual_bilayer"
MANUAL_RUN_NAME = "run_manual"
MANUAL_STAGE_NAME = "Manual"
MANUAL_MIN_TIME_PS = 100000.0

SCRIPT_DIR = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis")
DATA_ROOT = Path("/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/Boxes/2.1")
TCL_TEMPLATE = SCRIPT_DIR / "Photograph.tcl"
PHOTO_ROOT = SCRIPT_DIR / "Photos"
MANUAL_PHOTO_ROOT = SCRIPT_DIR / "Photos" / "0_manually_picked"

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

# =============================================================================
# TEXT / CLEAN IMAGE SETTINGS
# =============================================================================

# Master switch.
# False = detailed figure.
# True  = cleaner figure with less text.
CLEAN_IMAGE_MODE = False

# If True, existing final panels are regenerated.
# Use this after changing label/format settings.
FORCE_REMAKE_PANELS = False

# Main row/system title.
SHOW_SYSTEM_TITLE = True

# Top title for the all-bilayers stacked image.
SHOW_STACK_TITLE = True

# Labels above each individual VMD panel:
# Top bilayer, Side bilayer, Top, Side, Bottom, Water
SHOW_PANEL_LABELS = True

# Individual title components.
SHOW_MOLECULE_IN_TITLE = True
SHOW_NMOL_IN_TITLE = True
SHOW_BILAYER_IN_TITLE = True
SHOW_RUN_IN_TITLE = True
SHOW_STAGE_IN_TITLE = True
SHOW_APL_IN_TITLE = True

# Bilayer label cleaning.
REMOVE_W_WF_FROM_BILAYER_LABEL = True
PRETTY_BILAYER_LABELS = True

# Text appearance.
LABEL_FONT_SIZE = 28
TITLE_FONT_SIZE = 34
TITLE_FONT_MIN_SIZE = 18

LABEL_COLOR = "black"
TITLE_COLOR = "black"

LABEL_PADDING = 12
TITLE_LINE_SPACING = 6
TITLE_TOP_BOTTOM_PADDING = 10
PANEL_LABEL_BAND_HEIGHT = 42

# Maximum number of title lines.
# If the title is still too long, it will be truncated with "..."
MAX_TITLE_LINES = 2

# Image alignment / scaling.
# This makes each VMD panel sit inside a fixed-size white canvas.
# It prevents different bilayers appearing shifted/scaled after trimming.
NORMALISE_PANEL_SIZES = True

# Fixed canvas sizes for the six snapshot types.
# These do not change the VMD view/colouring; they only standardise the final PNG layout.
PANEL_CANVAS_SIZES = {
    "snapshot_top_bilayer.png":  (330, 330),
    "snapshot_side_bilayer.png": (330, 250),
    "snapshot_top.png":          (330, 330),
    "snapshot_side.png":         (330, 250),
    "snapshot_bottom.png":       (330, 330),
    "snapshot_water.png":        (330, 330),
}

DEFAULT_PANEL_CANVAS_SIZE = (330, 330)

# Vertical stack row alignment.
# "left" keeps all bilayer rows aligned at the same left edge.
# "center" centres shorter rows.
STACK_ROW_ALIGNMENT = "left"


if CLEAN_IMAGE_MODE:
    SHOW_SYSTEM_TITLE = True
    SHOW_STACK_TITLE = True
    SHOW_PANEL_LABELS = False

    SHOW_MOLECULE_IN_TITLE = True
    SHOW_NMOL_IN_TITLE = False
    SHOW_BILAYER_IN_TITLE = True
    SHOW_RUN_IN_TITLE = False
    SHOW_STAGE_IN_TITLE = False
    SHOW_APL_IN_TITLE = True

    TITLE_FONT_SIZE = 30
    MAX_TITLE_LINES = 1

# Visual-only bilayer Z repositioning
SHIFT_BILAYER_VIEW = True
BILAYER_TARGET_Z_FRACTION = 0.25   # 0.25+ recommended
SHIFTED_OUTPUT_TAG = "bilayer_z025"

# Final naming convention
# system image: {molecule}__nmol_{nmol}__{bilayer}__{run}.png
# molecule stack: {molecule}__ALL_BILAYERS__{run}.png

# Sort order for stacked bilayer overview
BILAYER_PRIORITY = [
    "POPC60CHOL40_W_WF",  # POPC:CHOL
    "POPE75POPG25_W_WF",  # POPE:POPG
    "POPC80POPS20_W_WF",  # POPC:POPS
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

# Peptoid residue categories
# Edit these to match the exact residue names in your .gro files
#NOTE - the following residues are not 100% perfectly place or are middle points between bins:
#   "NmO", "Nfe[4Cl]", "Nfe[4Br]", "Nf[naph]" 

PEPTOID_RESIDUES = {
    "hydrophobic": [
        "Na", "Nab", "Ni", "Nl", "Nm", "Nv"
    ],
    "proline_like": [
        # none in current mapping
    ],
    "polar": [
        "Nn", "Nq", "Ns", "Nse", "Nt", "NmO"
    ],
    "positive": [
        "Nk", "Nke", "Nr"
    ],
    "negative": [
        "Nd", "Ne"
    ],
    "aromatic": [
        "Nf", "Nfn", "Nfe", "Nfex", "Nw", "Nwe", "Ny", "Nfe[4Cl]", "Nfe[4Br]", "Nf[naph]"
    ],
}


PHOSPHOLIPID_RESNAMES = {"POPC", "POPS", "POPE", "POPG"}
CHOL_RESNAMES = {"CHOL"}

SYSTEM_TYPE_TO_ASSUMED_PHOSPHOLIPID_FRACTION = {
    "POPC60CHOL40_W_WF": 0.60,
    "POPC80POPS20_W_WF": 1.00,
    "POPE75POPG25_W_WF": 1.00,
}


# =============================================================================
# HELPERS
# =============================================================================

def resolve_photograph_target():
    """
    Returns a list of one or more targets.
    """
    if not MANUAL_MODE:
        targets = []

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

                    try:
                        molecule_name, nmol = split_system_folder_name(system_dir.name)
                        stage_info = get_stage_files(system_dir)
                    except Exception as exc:
                        print(f"SKIP target discovery: {system_dir}: {exc}")
                        continue

                    targets.append({
                        "system_dir": system_dir,
                        "molecule_name": molecule_name,
                        "nmol": nmol,
                        "bilayer": bilayer,
                        "run_name": run_name,
                        "gro": stage_info["gro"],
                        "xtc": stage_info["xtc"],
                        "stage": stage_info["stage"],
                        "min_time_ps": stage_info["min_time_ps"],
                    })

        return targets

    if MANUAL_SYSTEM_DIR is not None:
        system_dir = Path(MANUAL_SYSTEM_DIR)

        if not system_dir.is_dir():
            raise NotADirectoryError(f"MANUAL_SYSTEM_DIR is not a directory: {system_dir}")

        stage_info = get_stage_files(system_dir)

        try:
            molecule_name, nmol = split_system_folder_name(system_dir.name)
        except Exception:
            molecule_name, nmol = MANUAL_MOLECULE_NAME, MANUAL_NMOL

        try:
            run_name = system_dir.parent.name
            bilayer = system_dir.parents[1].name
        except Exception:
            run_name = MANUAL_RUN_NAME
            bilayer = MANUAL_BILAYER

        return [{
            "system_dir": system_dir,
            "molecule_name": molecule_name,
            "nmol": nmol,
            "bilayer": bilayer,
            "run_name": run_name,
            "gro": stage_info["gro"],
            "xtc": stage_info["xtc"],
            "stage": stage_info["stage"],
            "min_time_ps": stage_info["min_time_ps"],
        }]

    if MANUAL_GRO_FILE is not None and MANUAL_XTC_FILE is not None:
        gro_file = Path(MANUAL_GRO_FILE)
        xtc_file = Path(MANUAL_XTC_FILE)

        if not gro_file.exists():
            raise FileNotFoundError(f"MANUAL_GRO_FILE not found: {gro_file}")
        if not xtc_file.exists():
            raise FileNotFoundError(f"MANUAL_XTC_FILE not found: {xtc_file}")

        return [{
            "system_dir": gro_file.parent,
            "molecule_name": MANUAL_MOLECULE_NAME,
            "nmol": MANUAL_NMOL,
            "bilayer": MANUAL_BILAYER,
            "run_name": MANUAL_RUN_NAME,
            "gro": gro_file,
            "xtc": xtc_file,
            "stage": MANUAL_STAGE_NAME,
            "min_time_ps": MANUAL_MIN_TIME_PS,
        }]

    raise ValueError(
        "MANUAL_MODE=True but no valid manual input was provided. "
        "Use MANUAL_SYSTEM_DIR or both MANUAL_GRO_FILE + MANUAL_XTC_FILE."
    )


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


def calculate_final_apl_box(gro_file: Path, xtc_file: Path, bilayer: str) -> float:
    """
    Calculates final-frame box-method APL:
        APL = box_x * box_y / lipids_per_leaflet
    """
    u = mda.Universe(str(gro_file), str(xtc_file))
    u.trajectory[-1]

    dims = u.trajectory.ts.dimensions
    if dims is None or len(dims) < 2:
        raise ValueError("Missing box dimensions.")

    box_x = float(dims[0])
    box_y = float(dims[1])

    phospholipid_sel = "resname " + " ".join(sorted(PHOSPHOLIPID_RESNAMES))
    chol_sel = "resname " + " ".join(sorted(CHOL_RESNAMES))

    phospholipids = u.select_atoms(phospholipid_sel).residues
    chol = u.select_atoms(chol_sel).residues

    n_phospholipids = phospholipids.n_residues
    n_chol = chol.n_residues

    n_membrane_molecules = n_phospholipids + n_chol

    if "CHOL" in bilayer and n_chol == 0:
        assumed_fraction = SYSTEM_TYPE_TO_ASSUMED_PHOSPHOLIPID_FRACTION.get(bilayer)
        if assumed_fraction is None:
            raise ValueError(f"No CHOL fallback fraction defined for {bilayer}")
        n_membrane_molecules = n_phospholipids / assumed_fraction

    if n_membrane_molecules <= 0:
        raise ValueError("No membrane molecules counted for APL.")

    lipids_per_leaflet = n_membrane_molecules / 2.0
    apl = (box_x * box_y) / lipids_per_leaflet

    print(
    f"APL DEBUG | {bilayer} | "
    f"box_x={box_x:.3f} box_y={box_y:.3f} | "
    f"n_phospholipids={n_phospholipids} | "
    f"n_chol={n_chol} | "
    f"n_membrane_molecules={n_membrane_molecules:.1f} | "
    f"lipids_per_leaflet={lipids_per_leaflet:.1f} | "
    f"APL={apl:.3f}"
)
    return float(apl)


def write_system_tcl(template_text: str, gro_name: str, xtc_name: str, outpath: Path):
    tcl = copy.copy(template_text)

    tcl = tcl.replace(
        "mol new AAAADEMC_80.psf type psf",
        f"mol new {gro_name} type gro"
    )

    tcl = tcl.replace(
        "mol addfile AAAADEMC_80.pdb type pdb waitfor all",
        f"mol addfile {xtc_name} type xtc waitfor all"
    )

    tcl = tcl.replace(
        "set SHIFT_BILAYER_VIEW 0",
        f"set SHIFT_BILAYER_VIEW {1 if SHIFT_BILAYER_VIEW else 0}"
    )

    tcl = tcl.replace(
        "set BILAYER_TARGET_Z_FRACTION 0.125",
        f"set BILAYER_TARGET_Z_FRACTION {BILAYER_TARGET_Z_FRACTION}"
    )

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


def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def text_height(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]


def display_bilayer_name(bilayer: str) -> str:
    """
    Convert raw folder bilayer name into a shorter display label.

    Example:
        POPC60CHOL40_W_WF -> POPC60CHOL40
    """
    out = str(bilayer)

    if REMOVE_W_WF_FROM_BILAYER_LABEL:
        out = out.replace("_W_WF", "")

    if PRETTY_BILAYER_LABELS:
        out = out.replace("POPC60CHOL40", "POPC60:CHOL40")
        out = out.replace("POPE75POPG25", "POPE75:POPG25")
        out = out.replace("POPC80POPS20", "POPC80:POPS20")

    return out


def build_system_title(molecule_name, nmol, bilayer, run_name, stage, apl_text):
    """
    Build the title shown above each horizontal system panel.
    Each part can be switched on/off in the settings section.
    """
    parts = []

    if SHOW_MOLECULE_IN_TITLE:
        parts.append(str(molecule_name))

    if SHOW_NMOL_IN_TITLE:
        parts.append(f"nmol={nmol}")

    if SHOW_BILAYER_IN_TITLE:
        parts.append(display_bilayer_name(bilayer))

    if SHOW_RUN_IN_TITLE:
        parts.append(str(run_name))

    if SHOW_STAGE_IN_TITLE:
        parts.append(str(stage))

    if SHOW_APL_IN_TITLE:
        parts.append(str(apl_text))

    return " | ".join(parts)


def build_stack_title(molecule_name, run_name):
    """
    Build the title shown above the all-bilayers stacked image.
    """
    return f"{molecule_name} | {run_name} | all bilayers"


def wrap_title_to_width(draw, title, font, max_width):
    """
    Wrap a title by logical ' | ' chunks first.
    This keeps titles cleaner than wrapping at arbitrary words.
    """
    parts = str(title).split(" | ")

    lines = []
    current = ""

    for part in parts:
        candidate = part if current == "" else current + " | " + part

        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
                current = part
            else:
                # Fallback if one individual part is too long.
                wrapped = textwrap.wrap(part, width=40)
                lines.extend(wrapped[:-1])
                current = wrapped[-1] if wrapped else part

    if current:
        lines.append(current)

    return lines


def truncate_line_to_width(draw, text, font, max_width):
    """
    Truncate a line so it cannot run past the image width.
    """
    text = str(text)

    if text_width(draw, text, font) <= max_width:
        return text

    suffix = "..."
    while text and text_width(draw, text + suffix, font) > max_width:
        text = text[:-1]

    return text + suffix


def choose_title_font_and_lines(draw, title, max_width):
    """
    Choose a font size and wrapped title that fits inside the image width.
    """
    for size in range(TITLE_FONT_SIZE, TITLE_FONT_MIN_SIZE - 1, -2):
        font = load_font(size)
        lines = wrap_title_to_width(draw, title, font, max_width)

        if len(lines) <= MAX_TITLE_LINES:
            return font, lines

    # Last-resort fallback: use minimum font and truncate.
    font = load_font(TITLE_FONT_MIN_SIZE)
    lines = wrap_title_to_width(draw, title, font, max_width)

    if len(lines) > MAX_TITLE_LINES:
        lines = lines[:MAX_TITLE_LINES]

    lines = [truncate_line_to_width(draw, line, font, max_width) for line in lines]
    return font, lines


def add_title_band(img: Image.Image, title: str, show_title: bool = True):
    """
    Add a title band that wraps/shrinks/truncates text so it cannot run off the edge.
    """
    if not show_title:
        return img

    width, height = img.size

    dummy = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(dummy)

    max_text_width = width - 2 * LABEL_PADDING
    font, lines = choose_title_font_and_lines(draw, title, max_text_width)

    line_h = text_height(draw, "Ag", font)
    band_height = (
        TITLE_TOP_BOTTOM_PADDING * 2
        + len(lines) * line_h
        + max(0, len(lines) - 1) * TITLE_LINE_SPACING
    )

    new_img = Image.new("RGB", (width, height + band_height), "white")
    new_img.paste(img, (0, band_height))

    draw = ImageDraw.Draw(new_img)

    y = TITLE_TOP_BOTTOM_PADDING
    for line in lines:
        draw.text((LABEL_PADDING, y), line, fill=TITLE_COLOR, font=font)
        y += line_h + TITLE_LINE_SPACING

    return new_img


def add_panel_label(img: Image.Image, label: str):
    """
    Add a short label above one VMD snapshot panel.
    """
    if not SHOW_PANEL_LABELS:
        return img

    width, height = img.size
    new_img = Image.new("RGB", (width, height + PANEL_LABEL_BAND_HEIGHT), "white")
    new_img.paste(img, (0, PANEL_LABEL_BAND_HEIGHT))

    draw = ImageDraw.Draw(new_img)
    font = load_font(LABEL_FONT_SIZE)

    label = truncate_line_to_width(
        draw=draw,
        text=label,
        font=font,
        max_width=width - 2 * LABEL_PADDING,
    )

    draw.text((LABEL_PADDING, 6), label, fill=LABEL_COLOR, font=font)
    return new_img


def fit_image_on_canvas(img: Image.Image, canvas_size):
    """
    Fit an image onto a fixed-size white canvas without changing aspect ratio.

    This fixes the alignment problem caused by trimming, where different bilayers
    end up with slightly different panel sizes.
    """
    if not NORMALISE_PANEL_SIZES:
        return img

    canvas_w, canvas_h = canvas_size

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    fitted = img.copy()
    fitted.thumbnail((canvas_w, canvas_h), resample_filter)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    x = (canvas_w - fitted.size[0]) // 2
    y = (canvas_h - fitted.size[1]) // 2

    canvas.paste(fitted, (x, y))
    return canvas


def make_horizontal_panel(image_paths, output_path: Path, title: str):
    """
    Create one horizontal panel for one system/bilayer.

    Each VMD snapshot is first placed on a fixed-size canvas, so all systems
    have the same final panel geometry.
    """
    panels = []

    for img_path, label in image_paths:
        img = Image.open(img_path).convert("RGB")

        canvas_size = PANEL_CANVAS_SIZES.get(
            img_path.name,
            DEFAULT_PANEL_CANVAS_SIZE,
        )

        img = fit_image_on_canvas(img, canvas_size)
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

    stitched = add_title_band(
        img=stitched,
        title=title,
        show_title=SHOW_SYSTEM_TITLE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path, "PNG")

    for img in panels:
        img.close()


def make_vertical_stack(image_paths, output_path: Path, title: str):
    """
    Stack system panels vertically.

    STACK_ROW_ALIGNMENT = "left" is recommended because it keeps all bilayer rows
    aligned at the same left edge.
    """
    rows = [Image.open(p).convert("RGB") for p in image_paths]

    max_width = max(img.size[0] for img in rows)
    total_height = sum(img.size[1] for img in rows)

    stitched = Image.new("RGB", (max_width, total_height), "white")

    y = 0
    for img in rows:
        if STACK_ROW_ALIGNMENT == "center":
            x = (max_width - img.size[0]) // 2
        else:
            x = 0

        stitched.paste(img, (x, y))
        y += img.size[1]

    stitched = add_title_band(
        img=stitched,
        title=title,
        show_title=SHOW_STACK_TITLE,
    )

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
    tag = f"__{sanitize(SHIFTED_OUTPUT_TAG)}" if SHIFT_BILAYER_VIEW else ""
    return f"{sanitize(molecule_name)}__nmol_{nmol}__{sanitize(bilayer)}__{sanitize(run_name)}{tag}.png"


def stacked_image_name(molecule_name: str, run_name: str):
    tag = f"__{sanitize(SHIFTED_OUTPUT_TAG)}" if SHIFT_BILAYER_VIEW else ""
    return f"{sanitize(molecule_name)}__ALL_BILAYERS__{sanitize(run_name)}{tag}.png"


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not TCL_TEMPLATE.exists():
        raise FileNotFoundError(f"Missing Tcl template: {TCL_TEMPLATE}")

    active_photo_root = MANUAL_PHOTO_ROOT if MANUAL_MODE else PHOTO_ROOT
    active_photo_root.mkdir(parents=True, exist_ok=True)
    template_text = readin(TCL_TEMPLATE)

    generated_system_panels = {}
    new_systems_created = []
    
    n_targets = 0
    n_existing = 0
    n_incomplete = 0
    n_vmd_attempted = 0
    n_vmd_failed = 0
    n_snapshot_failed = 0
    n_saved = 0

    try:
        targets = resolve_photograph_target()
    except Exception as exc:
        raise RuntimeError(f"Could not resolve input target(s): {exc}")

    n_targets = len(targets)
    print(f"Targets found after discovery filtering: {n_targets}")

    for target in targets:
        system_dir = target["system_dir"]
        molecule_name = target["molecule_name"]
        nmol = target["nmol"]
        bilayer = target["bilayer"]
        run_name = target["run_name"]
        gro_file = target["gro"]
        xtc_file = target["xtc"]
        stage = target["stage"]
        min_time_ps = target["min_time_ps"]

        try:
            final_time = check_trajectory_complete(gro_file, xtc_file, min_time_ps)
        except Exception as exc:
            n_incomplete += 1
            print(f"SKIP trajectory check: {gro_file.parent}: {exc}")
            continue

        out_dir = active_photo_root / sanitize(molecule_name) / sanitize(bilayer) / sanitize(run_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        panel_name = system_image_name(molecule_name, nmol, bilayer, run_name)
        final_panel_path = out_dir / panel_name

        if final_panel_path.exists() and not FORCE_REMAKE_PANELS:
            n_existing += 1
            print(f"SKIP existing {final_panel_path}")
            generated_system_panels.setdefault((molecule_name, run_name), []).append((bilayer, final_panel_path))
            continue

        tcl_out = out_dir / "Photograph_generated.tcl"
        write_system_tcl(
            template_text=template_text,
            gro_name=gro_file.name,
            xtc_name=xtc_file.name,
            outpath=tcl_out,
        )

        system_tcl = system_dir / "Photograph.tcl"
        shutil.copy2(tcl_out, system_tcl)

        print(f"Running VMD for {molecule_name} | {bilayer} | {run_name} | stage={stage} | final_time={final_time:.1f} ps")
        n_vmd_attempted += 1

        try:
            run_vmd_in_system_dir(system_dir)
        except Exception as exc:
            n_vmd_failed += 1
            print(f"FAILED VMD {system_dir}: {exc}")
            continue

        try:
            snapshot_paths = collect_snapshot_paths(system_dir)
        except Exception as exc:
            n_snapshot_failed += 1
            print(f"FAILED snapshot collection {system_dir}: {exc}")
            continue

        try:
            final_apl = calculate_final_apl_box(gro_file, xtc_file, bilayer)
            apl_text = f"APL = {final_apl:.2f}Å²"
        except Exception as exc:
            print(f"WARNING: Could not calculate APL for title: {system_dir}: {exc}")
            apl_text = "APL = NA"
        
        title = build_system_title(
            molecule_name=molecule_name,
            nmol=nmol,
            bilayer=bilayer,
            run_name=run_name,
            stage=stage,
            apl_text=apl_text,
        )
        
        make_horizontal_panel(snapshot_paths, final_panel_path, title=title)
        
        generated_system_panels.setdefault((molecule_name, run_name), []).append((bilayer, final_panel_path))

        for fname, _ in SNAPSHOT_ORDER:
            p = system_dir / fname
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        n_saved += 1
        
        new_systems_created.append({
            "molecule": molecule_name,
            "nmol": nmol,
            "bilayer": bilayer,
            "run": run_name,
            "stage": stage,
            "path": str(final_panel_path),
        })
        
        print(f"SAVED {final_panel_path}")

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
        stack_dir = active_photo_root / sanitize(molecule_name) / "STACKED"
        stack_dir.mkdir(parents=True, exist_ok=True)

        stack_path = stack_dir / stacked_image_name(molecule_name, run_name)
        
        title = build_stack_title(molecule_name, run_name)
        make_vertical_stack(panel_paths, stack_path, title=title)

        print(f"SAVED STACK {stack_path}")

    print("\nPHOTO SUMMARY")
    print(f"Targets found after discovery filtering: {n_targets}")
    print(f"Existing panels skipped: {n_existing}")
    print(f"Skipped at trajectory check: {n_incomplete}")
    print(f"VMD attempted: {n_vmd_attempted}")
    print(f"VMD failed: {n_vmd_failed}")
    print(f"Snapshot collection failed: {n_snapshot_failed}")
    print(f"New panels saved: {n_saved}")

    print("\nNEW SYSTEMS CREATED")
    if not new_systems_created:
        print("None")
    else:
        for i, entry in enumerate(new_systems_created, 1):
            print(
                f"{i:03d} | "
                f"{entry['molecule']} | nmol={entry['nmol']} | "
                f"{entry['bilayer']} | {entry['run']} | {entry['stage']}"
            )
    
    output_list_file = active_photo_root / "new_systems_created.txt"
    
    with open(output_list_file, "w") as f:
        for entry in new_systems_created:
            f.write(
                f"{entry['molecule']} | nmol={entry['nmol']} | "
                f"{entry['bilayer']} | {entry['run']} | {entry['stage']} | "
                f"{entry['path']}\n"
            )
        
        
if __name__ == "__main__":
    main()