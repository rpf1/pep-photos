#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bilayer_only_apl_box_stages_colours.py

Calculate bilayer-only area per lipid (APL) over time for GROMACS systems.

APL method used:
    APL = box_x * box_y / lipids_per_leaflet

This script uses ONLY the box X-by-Y method.
It does NOT use the PO4 method.
It does NOT select PO4 atoms.

This version supports multiple simulation stages, for example:
    4_nvt
    5_npt
    6_production

It can join stages continuously so that NPT starts after NVT and Production starts
after NPT, rather than every stage starting from 0 ns.

Important colour behaviour:
    Colours are assigned to the SYSTEM + STAGE combination.
    This means POPC:CHOL NPT and POPE:POPG NPT are different colours.

Outputs per stage:
    APL_time_series_BOX.csv / .xlsx
    APL_binned_5percent_BOX.csv / .xlsx
    APL_over_time_BOX.png
    APL_binned_5percent_BOX.png

Outputs per bilayer/system:
    continuous_stages/APL_continuous_stages_BOX.csv / .xlsx
    continuous_stages/APL_continuous_stages_BOX.png
    continuous_stages/APL_continuous_stages_binned_BOX.csv / .xlsx
    continuous_stages/APL_continuous_stages_binned_BOX.png

Global outputs:
    bilayer_only_APL_BOX_summary.csv / .xlsx
    comparison_APL_over_time_BOX.png
    comparison_APL_binned_5percent_BOX.png
    comparison_continuous_stages_BOX.png
    comparison_continuous_stages_binned_BOX.png
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import MDAnalysis as mda


# =============================================================================
# USER SETTINGS
# =============================================================================

TARGET_DIRS = [
    "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPC-6_CHOL-4(127Axy-197Az)/5_npt",
    "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPE-3_POPG-1(123Axy-206Az)/5_npt",
    "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPC-6_CHOL-4(127Axy-197Az)/6_production",
    "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPE-3_POPG-1(123Axy-206Az)/6_production",
]

# =============================================================================
# TARGET_DIRS = [
#     "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPC-6_CHOL-4(127Axy-197Az)/4_nvt",
#     "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPE-3_POPG-1(123Axy-206Az)/4_nvt",
#     "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPC-6_CHOL-4(127Axy-197Az)/5_npt",
#     "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPE-3_POPG-1(123Axy-206Az)/5_npt",
#     "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPC-6_CHOL-4(127Axy-197Az)/6_production",
#     "/users/pfb19164/Desktop/post_medical_leave/first_simulations/bilayer/20ATall/POPE-3_POPG-1(123Axy-206Az)/6_production",
# ]
# =============================================================================

OUTPUT_ROOT = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/bilayer_output"
)

# Residues counted as membrane molecules.
# lipids_per_leaflet = number of these membrane residues / 2
MEMBRANE_RESNAMES = {
    "POPC",
    "POPE",
    "POPG",
    "POPS",
    "CHOL",
    "DPPC",
    "DTPC",
}

# 20 bins = 5% bins across the trajectory.
N_PERCENT_BINS = 20


# =============================================================================
# STAGE, LABEL, AND COLOUR SETTINGS
# =============================================================================

STAGE_CFG = {
    # Stages are plotted in this order when they are present.
    # Missing stages are ignored automatically.
    "stage_order": ["nvt", "npt", "production"],

    # Stage detection from folder names.
    # Examples:
    #   4_nvt        -> nvt
    #   5_npt        -> npt
    #   6_production -> production
    "stage_name_contains": {
        "nvt": ["nvt"],
        "npt": ["npt"],
        "production": ["production", "prod"],
    },

    # Stage names used in legends and output tables.
    "stage_labels": {
        "nvt": "NVT",
        "npt": "NPT",
        "production": "Production",
        "unknown": "Unknown stage",
    },

    # Clean system labels used in legends and plot titles.
    # Add more mappings here if you add more bilayers.
    "system_labels": {
        "POPC-6_CHOL-4(127Axy-197Az)": "POPC:CHOL",
        "POPE-3_POPG-1(123Axy-206Az)": "POPE:POPG",
        "POPC80POPS20_W_WF": "POPC:POPS",
        "POPC60CHOL40_W_WF": "POPC:CHOL",
        "POPE75POPG25_W_WF": "POPE:POPG",
    },

    # Colour assignment mode.
    # "system_stage" means every bilayer+stage combination gets a different colour.
    # This is the safest option for combined plots.
    "colour_mode": "system_stage",

    # These are used in order for each unique system+stage combination.
    # There are more than 9 colours here, so 3 bilayers x 3 stages is covered.
    # You can edit these manually if you want exact journal colours.
    "combination_colours": [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#003f5c",  # dark blue
        "#ffa600",  # gold
        "#665191",  # indigo
        "#a05195",  # magenta-purple
        "#f95d6a",  # coral
    ],

    # Optional exact manual override.
    # Keys must be "System label | Stage label".
    # If a key is present here, this colour overrides automatic assignment.
    "manual_colour_overrides": {
        # "POPC:CHOL | NVT": "#1f77b4",
        # "POPC:CHOL | NPT": "#ff7f0e",
        # "POPC:CHOL | Production": "#2ca02c",
        # "POPE:POPG | NVT": "#d62728",
        # "POPE:POPG | NPT": "#9467bd",
        # "POPE:POPG | Production": "#8c564b",
    },
}


# =============================================================================
# PLOT SETTINGS
# =============================================================================

PLOT_CFG = {
    # Figure size/output
    "figsize": (8.5, 5.8),
    "dpi": 300,

    # -------------------------------------------------------------------------
    # Axis limits
    # -------------------------------------------------------------------------
    # These now behave strictly.
    # If you set (0, 12), the axis line is forced from exactly 0 to exactly 12.
    # The endpoint tick label is also forced to appear when a major tick is set.
    "apl_ylim": (40, 85),

    # Raw per-stage plots and non-continuous comparison plots.
    "time_xlim": (0, 12),

    # Continuous joined-stage plots.
    # Use "same_as_time_xlim" if you want continuous plots to use time_xlim.
    # Use None for automatic continuous scaling.
    # Use a tuple such as (0, 24) for a separate continuous x range.
    "continuous_time_xlim": "same_as_time_xlim",

    # Binned percentage plots.
    "percent_xlim": (0, 100),

    # Force matplotlib not to add padding below/above the requested x-axis range.
    "force_exact_xlim": True,

    # Tick spacing.
    # For time_xlim = (0, 12), time_major_tick = 2 gives 0, 2, 4, 6, 8, 10, 12.
    # Set any of these to None to let matplotlib choose automatically.
    "time_major_tick": 2.0,
    "time_minor_tick": 1.0,
    "continuous_time_major_tick": 2.0,
    "continuous_time_minor_tick": 1.0,
    "percent_major_tick": 10.0,
    "percent_minor_tick": 5.0,

    # -------------------------------------------------------------------------
    # Titles
    # -------------------------------------------------------------------------
    # Available placeholders: {system_name}, {system_label}, {stage_name}, {stage_label}
    "title_apl_over_time": "{system_label} | {stage_label} | APL over time",
    "title_binned_apl": "{system_label} | {stage_label} | 5% binned APL",
    "title_continuous_stages": "{system_label} | continuous APL across simulation stages",
    "title_continuous_stages_binned": "{system_label} | continuous binned APL across simulation stages",

    "comparison_title_apl_over_time": "Bilayer-only APL over time",
    "comparison_title_binned_apl": "Bilayer-only 5% binned APL",
    "comparison_title_continuous_stages": "Bilayer-only continuous APL across stages",
    "comparison_title_continuous_stages_binned": "Bilayer-only continuous binned APL across stages",

    # Axis titles
    "x_label_time": "Simulation time (ns)",
    "x_label_continuous_time": "Continuous simulation time (ns)",
    "x_label_percent": "Trajectory progress (%)",
    "y_label_apl": "Area per lipid (A^2)",
    "y_label_binned_apl": "Mean area per lipid (A^2)",

    # Font sizes
    "title_fontsize": 14,
    "axis_label_fontsize": 13,
    "tick_label_fontsize": 11,
    "legend_fontsize": 9,
    "stage_boundary_label_fontsize": 9,

    # Manual text spacing
    "title_pad": 32,
    "title_y": 1.06,
    "x_labelpad": 14,
    "y_labelpad": 18,
    "tick_pad": 6,

    # Manual plot margins.
    "subplots_left": 0.18,
    "subplots_right": 0.97,
    "subplots_bottom": 0.17,
    "subplots_top": 0.74,

    # Line/marker settings
    "line_width": 2.0,
    "marker_size": 5.0,

    # Grid
    "show_grid": True,
    "grid_alpha": 0.3,

    # Title wrapping
    "wrap_title": True,
    "wrap_width": 68,

    # Legend and comparison plots
    "make_comparison_plots": True,
    "make_continuous_stage_plots": True,
    "show_legend": True,
    "legend_ncols": 1,
    "legend_location": "best",

    # Stage boundary markers on continuous plots
    "show_stage_boundaries": True,
    "stage_boundary_color": "black",
    "stage_boundary_linestyle": "--",
    "stage_boundary_linewidth": 0.8,
    "show_stage_boundary_labels": True,
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def safe_name(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def pretty_system_name(folder):
    folder = Path(folder)
    return folder.parent.name


def display_system_label(system_name):
    return STAGE_CFG["system_labels"].get(str(system_name), str(system_name))


def detect_stage_key(stage_name):
    text = str(stage_name).lower()

    for stage_key, patterns in STAGE_CFG["stage_name_contains"].items():
        for pattern in patterns:
            if str(pattern).lower() in text:
                return stage_key

    return "unknown"


def display_stage_label(stage_key):
    return STAGE_CFG["stage_labels"].get(str(stage_key), str(stage_key))


def stage_sort_index(stage_key):
    order = STAGE_CFG["stage_order"]
    if stage_key in order:
        return order.index(stage_key)
    return len(order) + 999


def format_plot_title(title):
    if not PLOT_CFG["wrap_title"]:
        return str(title)

    return "\n".join(
        textwrap.wrap(
            str(title),
            width=int(PLOT_CFG["wrap_width"]),
            break_long_words=False,
        )
    )


def choose_file(files, preferred_terms):
    if not files:
        return None

    files = sorted(files)

    for term in preferred_terms:
        term = term.lower()
        for f in files:
            if term in f.name.lower():
                return f

    return files[0]


def glob_many(folder, patterns):
    files = []
    folder = Path(folder)
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted(set(files))


def find_structure_and_trajectory(folder):
    folder = Path(folder)

    gro_files = glob_many(folder, ["*.gro", "*.GRO"])
    xtc_files = glob_many(folder, ["*.xtc", "*.XTC"])
    trr_files = glob_many(folder, ["*.trr", "*.TRR"])

    trajectory = choose_file(xtc_files, ["npt", "nvt", "prod", "production", "md", "traj"])
    if trajectory is None:
        trajectory = choose_file(trr_files, ["npt", "nvt", "prod", "production", "md", "traj"])

    if trajectory is None:
        raise FileNotFoundError(f"No .xtc or .trr trajectory found in {folder}")

    same_stem_gro = folder / f"{trajectory.stem}.gro"

    if same_stem_gro.exists():
        structure = same_stem_gro
    else:
        structure = choose_file(gro_files, ["npt", "nvt", "prod", "production", "confout", "final", "md", "box"])

    if structure is None:
        raise FileNotFoundError(f"No .gro structure found in {folder}")

    return structure, trajectory


def safe_write_table(df, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    try:
        df.to_excel(csv_path.with_suffix(".xlsx"), index=False)
    except Exception as exc:
        print(f"Excel output skipped for {csv_path.name}: {exc}")


# =============================================================================
# COLOUR HELPERS
# =============================================================================

def combination_key(system_label, stage_label):
    return f"{system_label} | {stage_label}"


def collect_colour_keys(stage_results_by_system):
    keys = []

    for system_name in sorted(stage_results_by_system.keys()):
        items = stage_results_by_system[system_name]
        items = sorted(items, key=lambda x: (stage_sort_index(x["stage_key"]), x["stage_name"]))

        for item in items:
            key = combination_key(item["system_label"], item["stage_label"])
            if key not in keys:
                keys.append(key)

    return keys


def build_colour_map(stage_results_by_system):
    keys = collect_colour_keys(stage_results_by_system)

    colours = STAGE_CFG["combination_colours"]
    overrides = STAGE_CFG.get("manual_colour_overrides", {})

    colour_map = {}

    for i, key in enumerate(keys):
        if key in overrides:
            colour_map[key] = overrides[key]
        else:
            colour_map[key] = colours[i % len(colours)]

    return colour_map


def get_colour_for_item(item, colour_map):
    key = combination_key(item["system_label"], item["stage_label"])
    return colour_map.get(key, "black")


# =============================================================================
# BOX-AREA APL ONLY
# =============================================================================

def count_membrane_molecules(universe):
    selection = "resname " + " ".join(sorted(MEMBRANE_RESNAMES))
    membrane_residues = universe.select_atoms(selection).residues

    n_membrane_molecules = membrane_residues.n_residues

    if n_membrane_molecules <= 0:
        raise ValueError(
            f"No membrane residues found. Selection used: {selection}"
        )

    return float(n_membrane_molecules)


def calculate_apl_box_timeseries(structure, trajectory):
    u = mda.Universe(str(structure), str(trajectory), in_memory=False)

    n_membrane_molecules = count_membrane_molecules(u)
    lipids_per_leaflet = n_membrane_molecules / 2.0

    rows = []

    for frame_i, ts in enumerate(u.trajectory):
        dims = ts.dimensions

        if dims is None or len(dims) < 3:
            raise ValueError(f"Missing box dimensions at frame {frame_i}")

        box_x = float(dims[0])
        box_y = float(dims[1])
        box_z = float(dims[2])

        if not np.all(np.isfinite([box_x, box_y, box_z])):
            raise ValueError(f"Non-finite box dimensions at frame {frame_i}: {dims}")

        if min(box_x, box_y, box_z) <= 0:
            raise ValueError(f"Invalid box dimensions at frame {frame_i}: {dims}")

        membrane_area_xy = box_x * box_y
        apl_box = membrane_area_xy / lipids_per_leaflet

        time_ps = float(ts.time)
        time_ns = time_ps / 1000.0

        rows.append({
            "frame": int(frame_i),
            "time_ps": time_ps,
            "time_ns": time_ns,
            "APL": float(apl_box),
            "APL_BOX": float(apl_box),
            "APL_method": "BOX_XY",
            "box_x": box_x,
            "box_y": box_y,
            "box_z": box_z,
            "membrane_area_xy": membrane_area_xy,
            "n_membrane_molecules": n_membrane_molecules,
            "lipids_per_leaflet": lipids_per_leaflet,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No trajectory frames were read.")

    return df


# =============================================================================
# 5% BINNING
# =============================================================================

def bin_by_5_percent(df):
    out = df.copy()

    n_frames = len(out)

    if n_frames == 1:
        out["progress_percent"] = 100.0
    else:
        out["progress_percent"] = out["frame"] / float(n_frames - 1) * 100.0

    out["bin_index"] = np.floor(out["progress_percent"] / 5.0).astype(int)
    out.loc[out["bin_index"] >= N_PERCENT_BINS, "bin_index"] = N_PERCENT_BINS - 1

    out["bin_start_percent"] = out["bin_index"] * 5.0
    out["bin_end_percent"] = out["bin_start_percent"] + 5.0

    grouped = (
        out.groupby("bin_index", as_index=False)
        .agg(
            bin_start_percent=("bin_start_percent", "first"),
            bin_end_percent=("bin_end_percent", "first"),
            n_frames=("APL", "size"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            time_start_ns=("time_ns", "min"),
            time_end_ns=("time_ns", "max"),
            APL_mean=("APL", "mean"),
            APL_std=("APL", "std"),
            APL_min=("APL", "min"),
            APL_max=("APL", "max"),
            box_x_mean=("box_x", "mean"),
            box_y_mean=("box_y", "mean"),
            box_z_mean=("box_z", "mean"),
            membrane_area_xy_mean=("membrane_area_xy", "mean"),
        )
        .sort_values("bin_index")
        .reset_index(drop=True)
    )

    grouped["APL_std"] = grouped["APL_std"].fillna(0.0)
    grouped["x_percent"] = 0.5 * (
        grouped["bin_start_percent"] + grouped["bin_end_percent"]
    )

    return grouped


# =============================================================================
# CONTINUOUS STAGE DATA
# =============================================================================

def build_continuous_stage_data(stage_items):
    stage_items = sorted(stage_items, key=lambda x: (stage_sort_index(x["stage_key"]), x["stage_name"]))

    continuous_dfs = []
    continuous_binned_dfs = []
    boundaries = []
    offset_ns = 0.0

    for item in stage_items:
        df = item["df"].copy()
        binned = item["binned"].copy()

        stage_start = float(df["time_ns"].min())
        df["stage_time_ns"] = df["time_ns"] - stage_start
        df["time_ns_continuous"] = df["stage_time_ns"] + offset_ns

        stage_duration = float(df["stage_time_ns"].max())
        stage_end_continuous = offset_ns + stage_duration

        df["system_name"] = item["system_name"]
        df["system_label"] = item["system_label"]
        df["stage_name"] = item["stage_name"]
        df["stage_key"] = item["stage_key"]
        df["stage_label"] = item["stage_label"]
        df["legend_label"] = combination_key(item["system_label"], item["stage_label"])
        df["stage_offset_ns"] = offset_ns

        binned_stage_start = float(binned["time_start_ns"].min())
        binned["stage_time_mid_ns"] = (
            0.5 * (binned["time_start_ns"] + binned["time_end_ns"])
            - binned_stage_start
        )
        binned["time_ns_continuous"] = binned["stage_time_mid_ns"] + offset_ns
        binned["system_name"] = item["system_name"]
        binned["system_label"] = item["system_label"]
        binned["stage_name"] = item["stage_name"]
        binned["stage_key"] = item["stage_key"]
        binned["stage_label"] = item["stage_label"]
        binned["legend_label"] = combination_key(item["system_label"], item["stage_label"])
        binned["stage_offset_ns"] = offset_ns

        continuous_dfs.append(df)
        continuous_binned_dfs.append(binned)

        boundaries.append({
            "stage_key": item["stage_key"],
            "stage_label": item["stage_label"],
            "start_ns": offset_ns,
            "end_ns": stage_end_continuous,
            "duration_ns": stage_duration,
        })

        offset_ns = stage_end_continuous

    continuous_df = pd.concat(continuous_dfs, ignore_index=True) if continuous_dfs else pd.DataFrame()
    continuous_binned_df = pd.concat(continuous_binned_dfs, ignore_index=True) if continuous_binned_dfs else pd.DataFrame()
    boundaries_df = pd.DataFrame(boundaries)

    return continuous_df, continuous_binned_df, boundaries_df


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def get_resolved_xlim(axis_kind):
    """
    Resolve x-axis limits for different plot types.

    axis_kind options:
        "time"       -> raw per-stage time plots
        "continuous" -> joined-stage continuous time plots
        "percent"    -> 5% binned trajectory-progress plots
    """
    if axis_kind == "time":
        return PLOT_CFG["time_xlim"]

    if axis_kind == "continuous":
        value = PLOT_CFG["continuous_time_xlim"]
        if value == "same_as_time_xlim":
            return PLOT_CFG["time_xlim"]
        return value

    if axis_kind == "percent":
        return PLOT_CFG["percent_xlim"]

    raise ValueError(f"Unknown axis_kind: {axis_kind}")


def get_major_minor_ticks(axis_kind):
    if axis_kind == "time":
        return PLOT_CFG["time_major_tick"], PLOT_CFG["time_minor_tick"]

    if axis_kind == "continuous":
        return PLOT_CFG["continuous_time_major_tick"], PLOT_CFG["continuous_time_minor_tick"]

    if axis_kind == "percent":
        return PLOT_CFG["percent_major_tick"], PLOT_CFG["percent_minor_tick"]

    raise ValueError(f"Unknown axis_kind: {axis_kind}")


def ticks_including_endpoints(xlim, step):
    """
    Build ticks that include the exact x-axis minimum and maximum.

    This avoids the annoying case where xlim=(0, 12) but matplotlib labels
    only 0, 2, 4, 6, 8, 10 and leaves 12 unlabelled.
    """
    if xlim is None or step is None:
        return None

    left, right = float(xlim[0]), float(xlim[1])
    step = float(step)

    if step <= 0:
        return None

    eps = step * 1e-9
    ticks = list(np.arange(left, right + 0.5 * step, step))
    ticks = [x for x in ticks if left - eps <= x <= right + eps]

    if not ticks or abs(ticks[0] - left) > eps:
        ticks.insert(0, left)

    if abs(ticks[-1] - right) > eps:
        ticks.append(right)

    # Avoid tiny floating-point display noise.
    ticks = [round(float(x), 10) for x in ticks]
    return ticks


def apply_exact_x_axis(ax, axis_kind):
    """
    Apply exact x-axis limits and tick labels.

    This is the important part that makes:
        "time_xlim": (0, 12)
    actually show an axis from exactly 0 to exactly 12.
    """
    xlim = get_resolved_xlim(axis_kind)
    major_tick, minor_tick = get_major_minor_ticks(axis_kind)

    # Remove matplotlib's default x-padding.
    ax.margins(x=0)

    if xlim is None:
        return

    left, right = float(xlim[0]), float(xlim[1])

    if right <= left:
        raise ValueError(f"Invalid x-axis limits for {axis_kind}: {xlim}")

    ax.set_xlim(left, right)
    ax.set_xbound(left, right)

    if major_tick is not None:
        major_ticks = ticks_including_endpoints((left, right), major_tick)
        if major_ticks is not None:
            ax.set_xticks(major_ticks)
        else:
            ax.xaxis.set_major_locator(MultipleLocator(float(major_tick)))

    if minor_tick is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(float(minor_tick)))

    # Apply again after setting ticks. This prevents locators from expanding the view.
    ax.set_xlim(left, right)
    ax.set_xbound(left, right)


def apply_common_axis_formatting(ax, title, x_label, y_label):
    ax.set_title(
        format_plot_title(title),
        fontsize=PLOT_CFG["title_fontsize"],
        pad=PLOT_CFG["title_pad"],
        y=PLOT_CFG["title_y"],
    )

    ax.set_xlabel(
        x_label,
        fontsize=PLOT_CFG["axis_label_fontsize"],
        labelpad=PLOT_CFG["x_labelpad"],
    )

    ax.set_ylabel(
        y_label,
        fontsize=PLOT_CFG["axis_label_fontsize"],
        labelpad=PLOT_CFG["y_labelpad"],
    )

    ax.tick_params(
        axis="both",
        labelsize=PLOT_CFG["tick_label_fontsize"],
        pad=PLOT_CFG["tick_pad"],
    )

    if PLOT_CFG["apl_ylim"] is not None:
        ax.set_ylim(*PLOT_CFG["apl_ylim"])

    if PLOT_CFG["show_grid"]:
        ax.grid(True, alpha=PLOT_CFG["grid_alpha"])


def apply_manual_margins(fig):
    fig.subplots_adjust(
        left=PLOT_CFG["subplots_left"],
        right=PLOT_CFG["subplots_right"],
        bottom=PLOT_CFG["subplots_bottom"],
        top=PLOT_CFG["subplots_top"],
    )


def add_stage_boundaries(ax, boundaries_df):
    if boundaries_df.empty:
        return

    if not PLOT_CFG["show_stage_boundaries"]:
        return

    for _, row in boundaries_df.iterrows():
        start_ns = float(row["start_ns"])
        end_ns = float(row["end_ns"])
        stage_label = str(row["stage_label"])

        if start_ns > 0:
            ax.axvline(
                start_ns,
                color=PLOT_CFG["stage_boundary_color"],
                linestyle=PLOT_CFG["stage_boundary_linestyle"],
                linewidth=PLOT_CFG["stage_boundary_linewidth"],
                alpha=0.7,
            )

        if PLOT_CFG["show_stage_boundary_labels"]:
            x_mid = 0.5 * (start_ns + end_ns)
            ymin, ymax = ax.get_ylim()
            y_pos = ymax - 0.04 * (ymax - ymin)
            ax.text(
                x_mid,
                y_pos,
                stage_label,
                fontsize=PLOT_CFG["stage_boundary_label_fontsize"],
                ha="center",
                va="top",
                alpha=0.85,
            )


def save_plot(fig, output_png):
    apply_manual_margins(fig)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def plot_apl_over_time(df, output_png, title, line_colour=None):
    fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], dpi=PLOT_CFG["dpi"])

    ax.plot(
        df["time_ns"],
        df["APL"],
        linewidth=PLOT_CFG["line_width"],
        color=line_colour,
    )

    apply_common_axis_formatting(
        ax=ax,
        title=title,
        x_label=PLOT_CFG["x_label_time"],
        y_label=PLOT_CFG["y_label_apl"],
    )

    apply_exact_x_axis(ax, axis_kind="time")
    save_plot(fig, output_png)


def plot_binned_apl(binned_df, output_png, title, line_colour=None):
    fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], dpi=PLOT_CFG["dpi"])

    ax.plot(
        binned_df["x_percent"],
        binned_df["APL_mean"],
        linewidth=PLOT_CFG["line_width"],
        marker="o",
        markersize=PLOT_CFG["marker_size"],
        color=line_colour,
    )

    apply_common_axis_formatting(
        ax=ax,
        title=title,
        x_label=PLOT_CFG["x_label_percent"],
        y_label=PLOT_CFG["y_label_binned_apl"],
    )

    apply_exact_x_axis(ax, axis_kind="percent")
    save_plot(fig, output_png)


def plot_continuous_stages(continuous_df, boundaries_df, output_png, title, colour_map):
    fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], dpi=PLOT_CFG["dpi"])

    for legend_label, sub in continuous_df.groupby("legend_label", sort=False):
        colour = colour_map.get(legend_label, "black")
        ax.plot(
            sub["time_ns_continuous"],
            sub["APL"],
            linewidth=PLOT_CFG["line_width"],
            color=colour,
            label=legend_label,
        )

    apply_common_axis_formatting(
        ax=ax,
        title=title,
        x_label=PLOT_CFG["x_label_continuous_time"],
        y_label=PLOT_CFG["y_label_apl"],
    )

    add_stage_boundaries(ax, boundaries_df)
    apply_exact_x_axis(ax, axis_kind="continuous")

    if PLOT_CFG["show_legend"]:
        ax.legend(
            fontsize=PLOT_CFG["legend_fontsize"],
            ncols=PLOT_CFG["legend_ncols"],
            loc=PLOT_CFG["legend_location"],
        )

    save_plot(fig, output_png)


def plot_continuous_stages_binned(continuous_binned_df, boundaries_df, output_png, title, colour_map):
    fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], dpi=PLOT_CFG["dpi"])

    for legend_label, sub in continuous_binned_df.groupby("legend_label", sort=False):
        colour = colour_map.get(legend_label, "black")
        ax.plot(
            sub["time_ns_continuous"],
            sub["APL_mean"],
            linewidth=PLOT_CFG["line_width"],
            marker="o",
            markersize=PLOT_CFG["marker_size"],
            color=colour,
            label=legend_label,
        )

    apply_common_axis_formatting(
        ax=ax,
        title=title,
        x_label=PLOT_CFG["x_label_continuous_time"],
        y_label=PLOT_CFG["y_label_binned_apl"],
    )

    add_stage_boundaries(ax, boundaries_df)
    apply_exact_x_axis(ax, axis_kind="continuous")

    if PLOT_CFG["show_legend"]:
        ax.legend(
            fontsize=PLOT_CFG["legend_fontsize"],
            ncols=PLOT_CFG["legend_ncols"],
            loc=PLOT_CFG["legend_location"],
        )

    save_plot(fig, output_png)


def plot_simple_comparison(stage_items, output_png, mode, colour_map):
    fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], dpi=PLOT_CFG["dpi"])

    for item in stage_items:
        legend_label = combination_key(item["system_label"], item["stage_label"])
        colour = colour_map.get(legend_label, "black")

        if mode == "raw":
            x = item["df"]["time_ns"]
            y = item["df"]["APL"]
            title = PLOT_CFG["comparison_title_apl_over_time"]
            x_label = PLOT_CFG["x_label_time"]
            y_label = PLOT_CFG["y_label_apl"]
            marker = None
            axis_kind = "time"
        else:
            x = item["binned"]["x_percent"]
            y = item["binned"]["APL_mean"]
            title = PLOT_CFG["comparison_title_binned_apl"]
            x_label = PLOT_CFG["x_label_percent"]
            y_label = PLOT_CFG["y_label_binned_apl"]
            marker = "o"
            axis_kind = "percent"

        ax.plot(
            x,
            y,
            linewidth=PLOT_CFG["line_width"],
            marker=marker,
            markersize=PLOT_CFG["marker_size"],
            color=colour,
            label=legend_label,
        )

    apply_common_axis_formatting(ax, title, x_label, y_label)
    apply_exact_x_axis(ax, axis_kind=axis_kind)

    if PLOT_CFG["show_legend"]:
        ax.legend(
            fontsize=PLOT_CFG["legend_fontsize"],
            ncols=PLOT_CFG["legend_ncols"],
            loc=PLOT_CFG["legend_location"],
        )

    save_plot(fig, output_png)


# =============================================================================
# PER-FOLDER ANALYSIS
# =============================================================================

def analyze_bilayer_folder(folder):
    folder = Path(folder)

    structure, trajectory = find_structure_and_trajectory(folder)

    system_name = pretty_system_name(folder)
    system_label = display_system_label(system_name)

    stage_name = folder.name
    stage_key = detect_stage_key(stage_name)
    stage_label = display_stage_label(stage_key)

    output_dir = OUTPUT_ROOT / safe_name(system_name) / safe_name(stage_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Analyzing bilayer-only system: {system_name}")
    print(f"System label: {system_label}")
    print(f"Stage folder: {stage_name}")
    print(f"Stage label:  {stage_label}")
    print(f"Folder:       {folder}")
    print(f"Structure:    {structure.name}")
    print(f"Trajectory:   {trajectory.name}")
    print("APL method:   BOX_XY only")

    df = calculate_apl_box_timeseries(structure, trajectory)
    binned = bin_by_5_percent(df)

    safe_write_table(df, output_dir / "APL_time_series_BOX.csv")
    safe_write_table(binned, output_dir / "APL_binned_5percent_BOX.csv")

    title_over_time = PLOT_CFG["title_apl_over_time"].format(
        system_name=system_name,
        system_label=system_label,
        stage_name=stage_name,
        stage_label=stage_label,
    )

    title_binned = PLOT_CFG["title_binned_apl"].format(
        system_name=system_name,
        system_label=system_label,
        stage_name=stage_name,
        stage_label=stage_label,
    )

    first = df.iloc[0]
    last = df.iloc[-1]

    summary = {
        "status": "complete",
        "reason": "",
        "system_name": system_name,
        "system_label": system_label,
        "stage_name": stage_name,
        "stage_key": stage_key,
        "stage_label": stage_label,
        "folder": str(folder),
        "structure_file": structure.name,
        "trajectory_file": trajectory.name,
        "structure_path": str(structure),
        "trajectory_path": str(trajectory),
        "APL_method": "BOX_XY",
        "APL_formula": "box_x * box_y / lipids_per_leaflet",
        "n_frames": int(len(df)),
        "first_frame": int(first["frame"]),
        "last_frame": int(last["frame"]),
        "first_time_ns": float(first["time_ns"]),
        "last_time_ns": float(last["time_ns"]),
        "stage_duration_ns": float(df["time_ns"].max() - df["time_ns"].min()),
        "first_APL": float(first["APL"]),
        "last_APL": float(last["APL"]),
        "delta_APL": float(last["APL"] - first["APL"]),
        "percent_change_APL": (
            float((last["APL"] - first["APL"]) / first["APL"] * 100.0)
            if first["APL"] != 0 else np.nan
        ),
        "mean_APL": float(df["APL"].mean()),
        "std_APL": float(df["APL"].std(ddof=0)),
        "min_APL": float(df["APL"].min()),
        "max_APL": float(df["APL"].max()),
        "first_box_x": float(first["box_x"]),
        "first_box_y": float(first["box_y"]),
        "first_box_z": float(first["box_z"]),
        "last_box_x": float(last["box_x"]),
        "last_box_y": float(last["box_y"]),
        "last_box_z": float(last["box_z"]),
        "first_membrane_area_xy": float(first["membrane_area_xy"]),
        "last_membrane_area_xy": float(last["membrane_area_xy"]),
        "n_membrane_molecules": float(first["n_membrane_molecules"]),
        "lipids_per_leaflet": float(first["lipids_per_leaflet"]),
        "output_dir": str(output_dir),
    }

    item = {
        "summary": summary,
        "df": df,
        "binned": binned,
        "system_name": system_name,
        "system_label": system_label,
        "stage_name": stage_name,
        "stage_key": stage_key,
        "stage_label": stage_label,
        "output_dir": output_dir,
        "title_over_time": title_over_time,
        "title_binned": title_binned,
    }

    return item


# =============================================================================
# GROUPED / GLOBAL OUTPUTS
# =============================================================================

def write_individual_stage_plots(stage_items, colour_map):
    for item in stage_items:
        legend_label = combination_key(item["system_label"], item["stage_label"])
        colour = colour_map.get(legend_label, "black")

        plot_apl_over_time(
            df=item["df"],
            output_png=item["output_dir"] / "APL_over_time_BOX.png",
            title=item["title_over_time"],
            line_colour=colour,
        )

        plot_binned_apl(
            binned_df=item["binned"],
            output_png=item["output_dir"] / "APL_binned_5percent_BOX.png",
            title=item["title_binned"],
            line_colour=colour,
        )


def write_continuous_outputs(stage_results_by_system, colour_map):
    all_continuous = []
    all_continuous_binned = []

    for system_name, stage_items in stage_results_by_system.items():
        system_label = display_system_label(system_name)
        continuous_df, continuous_binned_df, boundaries_df = build_continuous_stage_data(stage_items)

        if continuous_df.empty:
            continue

        out_dir = OUTPUT_ROOT / safe_name(system_name) / "continuous_stages"
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_write_table(continuous_df, out_dir / "APL_continuous_stages_BOX.csv")
        safe_write_table(continuous_binned_df, out_dir / "APL_continuous_stages_binned_BOX.csv")
        safe_write_table(boundaries_df, out_dir / "stage_boundaries.csv")

        title = PLOT_CFG["title_continuous_stages"].format(
            system_name=system_name,
            system_label=system_label,
        )
        title_binned = PLOT_CFG["title_continuous_stages_binned"].format(
            system_name=system_name,
            system_label=system_label,
        )

        plot_continuous_stages(
            continuous_df=continuous_df,
            boundaries_df=boundaries_df,
            output_png=out_dir / "APL_continuous_stages_BOX.png",
            title=title,
            colour_map=colour_map,
        )

        plot_continuous_stages_binned(
            continuous_binned_df=continuous_binned_df,
            boundaries_df=boundaries_df,
            output_png=out_dir / "APL_continuous_stages_binned_BOX.png",
            title=title_binned,
            colour_map=colour_map,
        )

        all_continuous.append(continuous_df)
        all_continuous_binned.append(continuous_binned_df)

    if all_continuous:
        global_continuous = pd.concat(all_continuous, ignore_index=True)
        safe_write_table(global_continuous, OUTPUT_ROOT / "all_systems_continuous_stages_BOX.csv")

        plot_continuous_stages(
            continuous_df=global_continuous,
            boundaries_df=pd.DataFrame(),
            output_png=OUTPUT_ROOT / "comparison_continuous_stages_BOX.png",
            title=PLOT_CFG["comparison_title_continuous_stages"],
            colour_map=colour_map,
        )

    if all_continuous_binned:
        global_binned = pd.concat(all_continuous_binned, ignore_index=True)
        safe_write_table(global_binned, OUTPUT_ROOT / "all_systems_continuous_stages_binned_BOX.csv")

        plot_continuous_stages_binned(
            continuous_binned_df=global_binned,
            boundaries_df=pd.DataFrame(),
            output_png=OUTPUT_ROOT / "comparison_continuous_stages_binned_BOX.png",
            title=PLOT_CFG["comparison_title_continuous_stages_binned"],
            colour_map=colour_map,
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    failed_summaries = []
    stage_items = []
    stage_results_by_system = {}

    for folder in TARGET_DIRS:
        folder = Path(folder)

        try:
            item = analyze_bilayer_folder(folder)
            stage_items.append(item)
            summaries.append(item["summary"])

            system_name = item["system_name"]
            if system_name not in stage_results_by_system:
                stage_results_by_system[system_name] = []
            stage_results_by_system[system_name].append(item)

        except Exception as exc:
            print("=" * 80)
            print(f"FAILED: {folder}")
            print(f"Reason: {type(exc).__name__}: {exc}")

            failed = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "system_name": folder.parent.name,
                "system_label": display_system_label(folder.parent.name),
                "stage_name": folder.name,
                "stage_key": detect_stage_key(folder.name),
                "stage_label": display_stage_label(detect_stage_key(folder.name)),
                "folder": str(folder),
                "APL_method": "BOX_XY",
            }
            summaries.append(failed)
            failed_summaries.append(failed)

    colour_map = build_colour_map(stage_results_by_system)

    # Write a colour map so you can check exactly what each colour means.
    colour_rows = [
        {"legend_label": key, "colour": value}
        for key, value in colour_map.items()
    ]
    safe_write_table(pd.DataFrame(colour_rows), OUTPUT_ROOT / "colour_map.csv")

    write_individual_stage_plots(stage_items, colour_map)

    if PLOT_CFG["make_comparison_plots"] and stage_items:
        plot_simple_comparison(
            stage_items=stage_items,
            output_png=OUTPUT_ROOT / "comparison_APL_over_time_BOX.png",
            mode="raw",
            colour_map=colour_map,
        )

        plot_simple_comparison(
            stage_items=stage_items,
            output_png=OUTPUT_ROOT / "comparison_APL_binned_5percent_BOX.png",
            mode="binned",
            colour_map=colour_map,
        )

    if PLOT_CFG["make_continuous_stage_plots"]:
        write_continuous_outputs(stage_results_by_system, colour_map)

    summary_df = pd.DataFrame(summaries)
    safe_write_table(summary_df, OUTPUT_ROOT / "bilayer_only_APL_BOX_summary.csv")

    if failed_summaries:
        safe_write_table(pd.DataFrame(failed_summaries), OUTPUT_ROOT / "bilayer_only_APL_BOX_failed.csv")

    print("=" * 80)
    print("Done.")
    print(f"Summary:    {OUTPUT_ROOT / 'bilayer_only_APL_BOX_summary.csv'}")
    print(f"Colour map: {OUTPUT_ROOT / 'colour_map.csv'}")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
