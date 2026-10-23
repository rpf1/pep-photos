#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photograph_components.py

Selective component-coloured membrane photography for PFPeptoids.

Purpose
-------
Create the same standard VMD views as Photograph.py, but render each
bilayer component as a separate coloured QuickSurf surface.

This is an OPTIONAL diagnostic/inspection workflow.

It does NOT replace Photograph.py.

Key behaviour
-------------
- Only explicitly requested TARGETS are processed.
- Reuses Photograph.py helpers for:
    * Relax/Pull resolution
    * trajectory completion checks
    * APL calculation
    * trimming
    * panel sizing
    * titles
    * horizontal panels
    * vertical stacks
- Does not write anything into simulation directories.
- All diagnostic outputs are written to:
      analysis/Photos_component_coloured/
- Existing outputs are not overwritten unless FORCE_REMAKE = True.
"""

from pathlib import Path
import subprocess

import MDAnalysis as mda


# =============================================================================
# IMPORT EXISTING PHOTOGRAPH WORKFLOW
# =============================================================================

try:
    import Photograph as photo
except ImportError:
    import photograph as photo


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_ROOT = Path(photo.DATA_ROOT)

TCL_TEMPLATE = SCRIPT_DIR / "Photograph_components.tcl"

OUTPUT_ROOT = SCRIPT_DIR / "Photos_component_coloured"


# =============================================================================
# USER SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# EXPLICIT TARGETS ONLY
# -------------------------------------------------------------------------
#
# bilayers:
#     "all"
#
# OR:
#
#     [
#         "POPC60CHOL40_W_WF",
#         "POPE75POPG25_W_WF",
#     ]
#
# nmol:
#     integer -> exact system, e.g. molecule_80
#     None    -> find every matching nmol for that molecule/run/bilayer
#
# Add/remove entries here whenever you want to inspect a particular set.
# -------------------------------------------------------------------------

TARGETS = [
    {
        "molecule": ["Nfnap-Nfnap-Nfex", "Nd-Nw-Nwe", "Nk-Nm-Nke", "Nwe-Nn-Na"],
        "run": ["run_ML_cycle_01", "run_ML_cycle_02"],
        "nmol": 80,
        "bilayers": ["POPC60CHOL40_W_WF", "POPE75POPG25_W_WF"],
    },

    # Example:
    #
    # {
    #     "molecule": "Nf-Nw-Nl",
    #     "run": "run_ML_cycle_01",
    #     "nmol": 80,
    #     "bilayers": [
    #         "POPC60CHOL40_W_WF",
    #         "POPE75POPG25_W_WF",
    #     ],
    # },
]


# Never replace existing diagnostic outputs unless explicitly enabled.
FORCE_REMAKE = False


# Add lipid colour identities to each row title.
#
# Example:
# POPC=blue | CHOL=orange
SHOW_COMPONENT_KEY_IN_TITLE = True


# =============================================================================
# BILAYER DEFINITIONS
# =============================================================================

# Exact component identity expected for each bilayer directory.
#
# These are deliberately explicit so an incorrectly named/malformed system
# is not silently rendered.

BILAYER_COMPONENTS = {
    "POPC60CHOL40_W_WF": ("POPC", "CHOL"),
    "POPE75POPG25_W_WF": ("POPE", "POPG"),
    "POPC80POPS20_W_WF": ("POPC", "POPS"),
}


# =============================================================================
# COMPONENT COLOURS
# =============================================================================

# Dedicated VMD ColorIDs are used so these colours do not modify the
# standard colours used by Photograph.tcl.
#
# RGB values are 0-1, as expected by VMD:
#     color change rgb <ColorID> R G B
#
# Colours remain chemically consistent across bilayer types.

COMPONENT_STYLES = {
    "POPC": {
        "color_id": 18,
        "rgb": (0.16, 0.48, 0.78),
        "name": "blue",
    },

    "CHOL": {
        "color_id": 19,
        "rgb": (0.95, 0.46, 0.12),
        "name": "orange",
    },

    "POPE": {
        "color_id": 20,
        "rgb": (0.20, 0.66, 0.38),
        "name": "green",
    },

    "POPG": {
        "color_id": 21,
        "rgb": (0.82, 0.22, 0.43),
        "name": "red",
    },

    "POPS": {
        "color_id": 22,
        "rgb": (0.55, 0.34, 0.72),
        "name": "purple",
    },
}


# =============================================================================
# VMD
# =============================================================================

# Same basic VMD launch mechanism used by Photograph.py.
#
# We pass the absolute generated Tcl path, so nothing needs to be copied
# into the raw simulation directory.

VMD_COMMAND_PREFIX = [
    "vglrun",
    "vmd",
    "-e",
]


# =============================================================================
# HELPERS
# =============================================================================

def component_style(resname):
    if resname not in COMPONENT_STYLES:
        raise KeyError(
            f"No COMPONENT_STYLES entry defined for membrane component "
            f"{resname!r}"
        )

    return COMPONENT_STYLES[resname]


def resolve_requested_bilayers(spec):
    requested = spec.get("bilayers", "all")

    if requested == "all":
        available = list(BILAYER_COMPONENTS.keys())
        return photo.ordered_bilayers(available)

    if isinstance(requested, str):
        requested = [requested]

    requested = list(requested)

    unknown = [
        bilayer
        for bilayer in requested
        if bilayer not in BILAYER_COMPONENTS
    ]

    if unknown:
        raise ValueError(
            "Unsupported bilayer(s) requested:\n  "
            + "\n  ".join(unknown)
        )

    return photo.ordered_bilayers(requested)


def find_matching_system_dirs(
    bilayer,
    run_name,
    molecule_name,
    nmol,
):
    """
    Locate the requested physical simulation directory/directories.

    If nmol is specified:
        requires exact folder:
            {molecule}_{nmol}

    If nmol=None:
        finds every folder whose parsed molecule name exactly matches.
    """

    run_dir = DATA_ROOT / bilayer / run_name

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist:\n{run_dir}"
        )

    if nmol is not None:
        system_dir = run_dir / f"{molecule_name}_{int(nmol)}"

        if not system_dir.is_dir():
            raise FileNotFoundError(
                f"Requested system does not exist:\n{system_dir}"
            )

        return [system_dir]

    matches = []

    for candidate in sorted(run_dir.iterdir()):
        if not candidate.is_dir():
            continue

        try:
            parsed_molecule, parsed_nmol = (
                photo.split_system_folder_name(candidate.name)
            )
        except Exception:
            continue

        if parsed_molecule == molecule_name:
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            f"No systems found for molecule {molecule_name!r} in:\n"
            f"{run_dir}"
        )

    return matches


def resolve_targets():
    """
    Resolve TARGETS into concrete simulation systems.

    Supports single values or lists for:
        - molecule
        - run

    Missing molecule/run/bilayer combinations are reported and skipped
    rather than terminating the whole script.

    Duplicate physical systems are automatically collapsed.
    """

    resolved = []
    seen = set()

    n_requested = 0
    n_missing = 0
    n_stage_unavailable = 0

    for spec in TARGETS:

        # ---------------------------------------------------------------------
        # MOLECULES
        # ---------------------------------------------------------------------

        molecules = spec["molecule"]

        if isinstance(molecules, str):
            molecules = [molecules]

        # ---------------------------------------------------------------------
        # RUNS
        # ---------------------------------------------------------------------

        runs = spec["run"]

        if isinstance(runs, str):
            runs = [runs]

        # ---------------------------------------------------------------------
        # OTHER SETTINGS
        # ---------------------------------------------------------------------

        nmol = spec.get("nmol")
        bilayers = resolve_requested_bilayers(spec)

        # ---------------------------------------------------------------------
        # RESOLVE ALL REQUESTED COMBINATIONS
        # ---------------------------------------------------------------------

        for molecule_name in molecules:

            for run_name in runs:

                for bilayer in bilayers:

                    n_requested += 1

                    try:
                        system_dirs = find_matching_system_dirs(
                            bilayer=bilayer,
                            run_name=run_name,
                            molecule_name=molecule_name,
                            nmol=nmol,
                        )

                    except FileNotFoundError:
                        n_missing += 1

                        if nmol is None:
                            system_label = molecule_name
                        else:
                            system_label = f"{molecule_name}_{nmol}"

                        print(
                            f"SKIP missing | "
                            f"{system_label} | "
                            f"{bilayer} | "
                            f"{run_name}"
                        )

                        continue

                    # ---------------------------------------------------------
                    # PROCESS FOUND SYSTEMS
                    # ---------------------------------------------------------

                    for system_dir in system_dirs:

                        actual_molecule, actual_nmol = (
                            photo.split_system_folder_name(system_dir.name)
                        )

                        try:
                            stage_info = photo.get_stage_files(system_dir)

                        except FileNotFoundError as exc:
                            n_stage_unavailable += 1

                            print(
                                f"SKIP no usable trajectory | "
                                f"{actual_molecule}_{actual_nmol} | "
                                f"{bilayer} | "
                                f"{run_name} | "
                                f"{exc}"
                            )

                            continue

                        unique_key = (
                            str(system_dir.resolve()),
                            stage_info["stage"],
                        )

                        if unique_key in seen:
                            print(
                                f"SKIP duplicate target | "
                                f"{actual_molecule}_{actual_nmol} | "
                                f"{bilayer} | "
                                f"{run_name}"
                            )
                            continue

                        seen.add(unique_key)

                        resolved.append({
                            "system_dir": system_dir,
                            "molecule_name": actual_molecule,
                            "nmol": actual_nmol,
                            "bilayer": bilayer,
                            "run_name": run_name,
                            "gro": stage_info["gro"],
                            "xtc": stage_info["xtc"],
                            "stage": stage_info["stage"],
                            "min_time_ps": stage_info["min_time_ps"],
                        })

    # -------------------------------------------------------------------------
    # DISCOVERY SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("TARGET DISCOVERY SUMMARY")
    print(f"Requested combinations: {n_requested}")
    print(f"Resolved systems: {len(resolved)}")
    print(f"Missing systems: {n_missing}")
    print(f"No usable trajectory: {n_stage_unavailable}")
    print()

    return resolved


def validate_bilayer_components(
    gro_file,
    xtc_file,
    bilayer,
):
    """
    Check that both expected membrane components actually exist in
    the trajectory.

    Returns:
        {
            "POPC": 120,
            "CHOL": 80,
        }
    """

    if bilayer not in BILAYER_COMPONENTS:
        raise KeyError(
            f"No component definition exists for bilayer {bilayer!r}"
        )

    components = BILAYER_COMPONENTS[bilayer]

    universe = mda.Universe(
        str(gro_file),
        str(xtc_file),
    )

    universe.trajectory[-1]

    counts = {}

    for component in components:
        atoms = universe.select_atoms(
            f"resname {component}"
        )

        residue_count = atoms.residues.n_residues

        counts[component] = int(residue_count)

        if residue_count == 0:
            raise ValueError(
                f"Bilayer validation failed for {bilayer}.\n"
                f"Expected component {component!r}, but zero residues "
                f"were found in:\n{gro_file}"
            )

    return counts


def rgb_string(rgb):
    return " ".join(
        f"{float(value):.4f}"
        for value in rgb
    )


def build_generated_tcl(
    gro_file,
    xtc_file,
    bilayer,
):
    """
    Fill Photograph_components.tcl placeholders for one system.
    """

    photo.validate_gro(gro_file)
    template = TCL_TEMPLATE.read_text()
    template = template.replace(
        'source [file join [file dirname [info script]] photograph_residues.tcl]',
        "source " + photo.tcl_path(SCRIPT_DIR / "photograph_residues.tcl"),
    )

    component_1, component_2 = BILAYER_COMPONENTS[bilayer]

    style_1 = component_style(component_1)
    style_2 = component_style(component_2)

    replacements = {
        "__GRO_FILE__":
            str(Path(gro_file).resolve()),

        "__XTC_FILE__":
            str(Path(xtc_file).resolve()),

        "__SHIFT_BILAYER_VIEW__":
            "1" if photo.SHIFT_BILAYER_VIEW else "0",

        "__BILAYER_TARGET_Z_FRACTION__":
            str(photo.BILAYER_TARGET_Z_FRACTION),

        "__COMPONENT_1_RESNAME__":
            component_1,

        "__COMPONENT_2_RESNAME__":
            component_2,

        "__COMPONENT_1_COLOR_ID__":
            str(style_1["color_id"]),

        "__COMPONENT_2_COLOR_ID__":
            str(style_2["color_id"]),

        "__COMPONENT_1_RGB__":
            rgb_string(style_1["rgb"]),

        "__COMPONENT_2_RGB__":
            rgb_string(style_2["rgb"]),
    }

    output = template

    for placeholder, value in replacements.items():

        if placeholder not in output:
            raise ValueError(
                f"Required Tcl placeholder missing: {placeholder}"
            )

        output = output.replace(
            placeholder,
            value,
        )

    return output


def safe_write_generated_file(
    path,
    content,
):
    """
    Do not silently replace an existing generated file with different
    contents unless FORCE_REMAKE=True.
    """

    path = Path(path)

    if path.exists():

        old_content = path.read_text()

        if old_content == content:
            return

        if not FORCE_REMAKE:
            raise FileExistsError(
                f"Generated file already exists with different contents:\n"
                f"{path}\n\n"
                f"Set FORCE_REMAKE=True if replacement is intended."
            )

    path.write_text(content)


def run_vmd(
    tcl_file,
    working_directory,
):
    command = [
        *VMD_COMMAND_PREFIX,
        str(Path(tcl_file).resolve()),
    ]

    subprocess.run(
        command,
        cwd=str(working_directory),
        check=True,
    )


def diagnostic_snapshot_paths(out_dir):
    return [
        out_dir / filename
        for filename, _ in photo.SNAPSHOT_ORDER
    ]


def check_existing_raw_snapshots(out_dir):
    existing = [
        path
        for path in diagnostic_snapshot_paths(out_dir)
        if path.exists()
    ]

    if existing and not FORCE_REMAKE:
        paths = "\n".join(
            f"  {path}"
            for path in existing
        )

        raise FileExistsError(
            "Diagnostic raw snapshots already exist and would be "
            "overwritten by VMD:\n"
            f"{paths}\n\n"
            "Either inspect/remove them manually, or explicitly set "
            "FORCE_REMAKE=True."
        )


def collect_snapshot_paths(out_dir):
    """
    Validate the nine expected VMD snapshots and optionally trim them.

    Raw snapshots are RETAINED after panel generation.
    """

    output = []

    for filename, label in photo.SNAPSHOT_ORDER:

        path = out_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Expected VMD snapshot missing:\n{path}"
            )

        if photo.TRIM_IMAGES:
            photo.trim_png_inplace(path)

        output.append(
            (path, label)
        )

    return output


def component_panel_name(
    molecule_name,
    nmol,
    bilayer,
    run_name,
):
    tags = [
        "component_coloured",
    ]

    if photo.SHIFT_BILAYER_VIEW:
        tags.append(
            photo.sanitize(
                photo.SHIFTED_OUTPUT_TAG
            )
        )

    suffix = "__".join(tags)

    return (
        f"{photo.sanitize(molecule_name)}"
        f"__nmol_{nmol}"
        f"__{photo.sanitize(bilayer)}"
        f"__{photo.sanitize(run_name)}"
        f"__{suffix}.png"
    )


def component_stack_name(
    molecule_name,
    run_name,
):
    tags = [
        "component_coloured",
    ]

    if photo.SHIFT_BILAYER_VIEW:
        tags.append(
            photo.sanitize(
                photo.SHIFTED_OUTPUT_TAG
            )
        )

    suffix = "__".join(tags)

    return (
        f"{photo.sanitize(molecule_name)}"
        f"__ALL_BILAYERS"
        f"__{photo.sanitize(run_name)}"
        f"__{suffix}.png"
    )


def build_component_key(
    bilayer,
):
    component_1, component_2 = (
        BILAYER_COMPONENTS[bilayer]
    )

    style_1 = component_style(component_1)
    style_2 = component_style(component_2)

    return (
        f"{component_1}={style_1['name']} | "
        f"{component_2}={style_2['name']}"
    )


def write_component_information(
    out_dir,
    bilayer,
    counts,
):
    component_1, component_2 = (
        BILAYER_COMPONENTS[bilayer]
    )

    style_1 = component_style(component_1)
    style_2 = component_style(component_2)

    text = (
        f"Bilayer: {bilayer}\n"
        f"\n"
        f"Component 1\n"
        f"  resname: {component_1}\n"
        f"  residues: {counts[component_1]}\n"
        f"  VMD ColorID: {style_1['color_id']}\n"
        f"  RGB: {style_1['rgb']}\n"
        f"  colour name: {style_1['name']}\n"
        f"\n"
        f"Component 2\n"
        f"  resname: {component_2}\n"
        f"  residues: {counts[component_2]}\n"
        f"  VMD ColorID: {style_2['color_id']}\n"
        f"  RGB: {style_2['rgb']}\n"
        f"  colour name: {style_2['name']}\n"
    )

    path = out_dir / "component_key.txt"

    safe_write_generated_file(
        path=path,
        content=text,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Fresh diagnostic staging directory required: {OUTPUT_ROOT}")


    if not TCL_TEMPLATE.exists():
        raise FileNotFoundError(
            f"Missing Tcl template:\n{TCL_TEMPLATE}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = resolve_targets()

    if not targets:
        print("No component-coloured targets resolved.")
        return

    print()
    print("=" * 80)
    print("COMPONENT-COLOURED PHOTOGRAPHY")
    print("=" * 80)
    print(f"Resolved targets: {len(targets)}")
    print(f"Output root: {OUTPUT_ROOT}")
    print()

    generated_system_panels = {}

    n_existing = 0
    n_rendered = 0
    n_failed = 0

    for index, target in enumerate(
        targets,
        start=1,
    ):

        system_dir = target["system_dir"]
        molecule_name = target["molecule_name"]
        nmol = target["nmol"]
        bilayer = target["bilayer"]
        run_name = target["run_name"]
        gro_file = target["gro"]
        xtc_file = target["xtc"]
        stage = target["stage"]
        min_time_ps = target["min_time_ps"]

        component_1, component_2 = (
            BILAYER_COMPONENTS[bilayer]
        )

        print()
        print("-" * 80)
        print(
            f"[{index}/{len(targets)}] "
            f"{molecule_name} | "
            f"nmol={nmol} | "
            f"{bilayer} | "
            f"{run_name}"
        )
        print(
            f"Components: "
            f"{component_1} + {component_2}"
        )

        out_dir = (
            OUTPUT_ROOT
            / photo.sanitize(molecule_name)
            / photo.sanitize(bilayer)
            / photo.sanitize(run_name)
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        panel_path = (
            out_dir
            / component_panel_name(
                molecule_name=molecule_name,
                nmol=nmol,
                bilayer=bilayer,
                run_name=run_name,
            )
        )

        group_key = (
            molecule_name,
            run_name,
        )

        if (
            panel_path.exists()
            and not FORCE_REMAKE
        ):
            print(
                f"SKIP existing panel:\n"
                f"  {panel_path}"
            )

            n_existing += 1

            generated_system_panels.setdefault(
                group_key,
                [],
            ).append(
                (bilayer, panel_path)
            )

            continue

        try:
            final_time = (
                photo.check_trajectory_complete(
                    gro_file=gro_file,
                    xtc_file=xtc_file,
                    min_time_ps=min_time_ps,
                )
            )

            counts = validate_bilayer_components(
                gro_file=gro_file,
                xtc_file=xtc_file,
                bilayer=bilayer,
            )

            print(
                f"Stage: {stage}"
            )

            print(
                f"Final trajectory time: "
                f"{final_time:.1f} ps"
            )

            print(
                "Validated components:"
            )

            for component, count in counts.items():
                print(
                    f"  {component}: "
                    f"{count} residues"
                )

            check_existing_raw_snapshots(
                out_dir
            )

            tcl_text = build_generated_tcl(
                gro_file=gro_file,
                xtc_file=xtc_file,
                bilayer=bilayer,
            )

            generated_tcl = (
                out_dir
                / "Photograph_components_generated.tcl"
            )

            safe_write_generated_file(
                path=generated_tcl,
                content=tcl_text,
            )

            write_component_information(
                out_dir=out_dir,
                bilayer=bilayer,
                counts=counts,
            )

            print(
                "Running VMD..."
            )

            run_vmd(
                tcl_file=generated_tcl,
                working_directory=out_dir,
            )

            snapshots = collect_snapshot_paths(
                out_dir
            )

            try:
                final_apl = (
                    photo.calculate_final_apl_box(
                        gro_file,
                        xtc_file,
                        bilayer,
                    )
                )

                apl_text = (
                    f"APL = {final_apl:.2f}Å²"
                )

            except Exception as exc:
                print(
                    "WARNING: Could not calculate "
                    f"APL: {exc}"
                )

                apl_text = "APL = NA"

            title = photo.build_system_title(
                molecule_name=molecule_name,
                nmol=nmol,
                bilayer=bilayer,
                run_name=run_name,
                stage=stage,
                apl_text=apl_text,
            )

            if SHOW_COMPONENT_KEY_IN_TITLE:
                title = (
                    f"{title} | "
                    f"{build_component_key(bilayer)}"
                )

            photo.make_horizontal_panel(
                image_paths=snapshots,
                output_path=panel_path,
                title=title,
            )

            generated_system_panels.setdefault(
                group_key,
                [],
            ).append(
                (bilayer, panel_path)
            )

            n_rendered += 1

            print(
                f"SAVED:\n  {panel_path}"
            )

        except Exception as exc:
            n_failed += 1

            print(
                f"FAILED:\n"
                f"  {system_dir}\n"
                f"  {exc}"
            )


    # =========================================================================
    # STACK SELECTED BILAYERS
    # =========================================================================

    for (
        molecule_name,
        run_name,
    ), bilayer_panels in generated_system_panels.items():

        if not bilayer_panels:
            continue

        bilayer_panels = sorted(
            bilayer_panels,
            key=lambda item: (
                photo.BILAYER_PRIORITY.index(item[0])
                if item[0] in photo.BILAYER_PRIORITY
                else 999,
                item[0],
            ),
        )

        panel_paths = [
            panel_path
            for _, panel_path in bilayer_panels
        ]

        stack_dir = (
            OUTPUT_ROOT
            / photo.sanitize(molecule_name)
            / "STACKED"
        )

        stack_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        stack_path = (
            stack_dir
            / component_stack_name(
                molecule_name=molecule_name,
                run_name=run_name,
            )
        )

        if (
            stack_path.exists()
            and not FORCE_REMAKE
        ):
            print()
            print(
                f"SKIP existing stack:\n"
                f"  {stack_path}"
            )

            continue

        stack_title = (
            f"{photo.build_stack_title(molecule_name, run_name)} "
            f"| component coloured"
        )

        photo.make_vertical_stack(
            image_paths=panel_paths,
            output_path=stack_path,
            title=stack_title,
        )

        print()
        print(
            f"SAVED STACK:\n"
            f"  {stack_path}"
        )


    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("COMPONENT-COLOURED PHOTO SUMMARY")
    print("=" * 80)
    print(
        f"Resolved targets: {len(targets)}"
    )
    print(
        f"Existing panels skipped: {n_existing}"
    )
    print(
        f"New panels rendered: {n_rendered}"
    )
    print(
        f"Failed targets: {n_failed}"
    )
    print(
        f"Output root: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
