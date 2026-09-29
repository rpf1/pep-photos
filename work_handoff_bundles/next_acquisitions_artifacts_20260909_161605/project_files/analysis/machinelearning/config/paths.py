#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Central filesystem map for PFPeptoids machine-learning workflows.

from pathlib import Path

MACHINELEARNING_ROOT = Path(__file__).resolve().parents[1]
PFPEPTOIDS_ROOT = MACHINELEARNING_ROOT.parents[1]

CONFIG_DIR = MACHINELEARNING_ROOT / "config"
SHARED_DATA_DIR = MACHINELEARNING_ROOT / "shared_data"
GLOBAL_MOLECULAR_DIR = MACHINELEARNING_ROOT / "global_molecular"
POSITION_RESOLVED_DIR = MACHINELEARNING_ROOT / "position_resolved"
LEGACY_DIR = MACHINELEARNING_ROOT / "legacy"
SLURM_OUT_DIR = MACHINELEARNING_ROOT / "slurm_out"
CLEANUP_DIR = MACHINELEARNING_ROOT / "17082026_cleanup"

MOLECULES_DIR = SHARED_DATA_DIR / "molecules"
SIMULATION_RESULTS_DIR = SHARED_DATA_DIR / "simulation_results"
DATASETS_DIR = SHARED_DATA_DIR / "datasets"
SPLITS_DIR = SHARED_DATA_DIR / "splits"
DESCRIPTORS_DIR = SHARED_DATA_DIR / "descriptors"
GLOBAL_MOLECULAR_DESCRIPTORS_DIR = DESCRIPTORS_DIR / "global_molecular"
POSITION_RESOLVED_DESCRIPTORS_DIR = DESCRIPTORS_DIR / "position_resolved"
MONOMER_DESCRIPTORS_DIR = DESCRIPTORS_DIR / "monomers"

VALID_WORKFLOWS = {
    "global_molecular": GLOBAL_MOLECULAR_DIR,
    "position_resolved": POSITION_RESOLVED_DIR,
}


def workflow_dir(workflow: str) -> Path:
    try:
        return VALID_WORKFLOWS[workflow]
    except KeyError as exc:
        raise ValueError(
            f"Unknown workflow {workflow!r}. Expected one of: "
            f"{sorted(VALID_WORKFLOWS)}"
        ) from exc


def scripts_dir(workflow: str) -> Path:
    return workflow_dir(workflow) / "scripts"


def training_runs_dir(workflow: str) -> Path:
    return workflow_dir(workflow) / "training_runs"


def prediction_runs_dir(workflow: str) -> Path:
    return workflow_dir(workflow) / "prediction_runs"


def analysis_runs_dir(workflow: str) -> Path:
    return workflow_dir(workflow) / "analysis_runs"


def descriptor_runs_dir(workflow: str) -> Path:
    return workflow_dir(workflow) / "descriptor_runs"


def descriptor_version_dir(workflow: str, version: str) -> Path:
    version = str(version)
    if workflow == "global_molecular":
        return GLOBAL_MOLECULAR_DESCRIPTORS_DIR / version
    if workflow == "position_resolved":
        return POSITION_RESOLVED_DESCRIPTORS_DIR / version
    raise ValueError(f"Unknown workflow {workflow!r}.")


def monomer_descriptor_version_dir(version: str) -> Path:
    return MONOMER_DESCRIPTORS_DIR / str(version)


def split_path(version: str) -> Path:
    version = str(version)
    if version.startswith("split_assignments_"):
        filename = version
        if not filename.endswith(".csv"):
            filename += ".csv"
    else:
        filename = f"split_assignments_{version}.csv"
    return SPLITS_DIR / filename


def dataset_path(dataset_name: str) -> Path:
    dataset_name = str(dataset_name)
    if not dataset_name.endswith(".csv"):
        dataset_name += ".csv"
    return DATASETS_DIR / dataset_name


def simulation_results_path(
    condition: str = "Neutral",
    filename: str = "Smiles.csv",
) -> Path:
    return SIMULATION_RESULTS_DIR / str(condition) / str(filename)


def training_run_dir(workflow: str, run_id: str) -> Path:
    return training_runs_dir(workflow) / str(run_id)


def prediction_run_dir(workflow: str, run_id: str) -> Path:
    return prediction_runs_dir(workflow) / str(run_id)


def analysis_run_dir(
    workflow: str,
    analysis_type: str,
    run_id: str,
) -> Path:
    return analysis_runs_dir(workflow) / str(analysis_type) / str(run_id)


def descriptor_run_dir(workflow: str, run_id: str) -> Path:
    return descriptor_runs_dir(workflow) / str(run_id)


RUN_REGISTRY_CSV = SHARED_DATA_DIR / "run_registry.csv"
