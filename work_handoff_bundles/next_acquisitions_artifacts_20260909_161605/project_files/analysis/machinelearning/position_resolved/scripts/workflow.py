#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PFPeptoids active-learning controller.

PFPEPTOIDS_ACTIVE_LEARNING_V5_PARTIAL_SELECTION: representation-aware, midpoint-corrected, pooled/self-feedback fork, branch-isolated partial selection, physical-simulation deduplication, per-model configurable search.

This is the normal user-facing entry point for the position-resolved
PFPeptoids machine-learning / active-learning workflow.

Key rules enforced here:
- real CGMD-derived targets only;
- fixed 32,768-sequence descriptor universe and fixed 323-feature schema;
- explicit model dataset membership (never silently add new IDs);
- Model 0 = 40 unique real peptoids -> 32 train / 8 fixed benchmark;
- branch-specific A-E membership; pooled/self-feedback F; pooled/no-self-feedback G;
- Model1F is the shared pooled A-E seed; the F/G fork begins at Model2;
- canonical A-E acquisition first, then pooled F/G acquisition after their models are trained/predicted;
- 15 new valid acquisitions per active strategy/cycle;
- C/E use the prediction-range midpoint; historical zero-target C/E are preserved as inactive X/Y;
- same-cycle overlap across strategies allowed, physical queue deduplicated across cycles;
- branch eligibility is defined by the source model's own dataset/history;
- already-real/queued molecules may be independently rediscovered and reused;
- monomer availability restrictions apply only to genuinely new physical simulations;
- predictions and later real DDAPL values are linked for residual analysis;
- one SQLite database is the workflow state authority;
- Excel output is a regenerated human-readable view of that database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT IMPORTS / PATHS
# =============================================================================

_MACHINELEARNING_ROOT = Path(__file__).resolve().parents[2]
if str(_MACHINELEARNING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MACHINELEARNING_ROOT))

from config.paths import (  # noqa: E402
    DATASETS_DIR,
    POSITION_RESOLVED_DIR,
    RUN_REGISTRY_CSV,
    SHARED_DATA_DIR,
    SLURM_OUT_DIR,
    SPLITS_DIR,
    descriptor_version_dir,
    prediction_run_dir,
    scripts_dir,
    training_run_dir,
)

PFPEPTOIDS_ROOT = _MACHINELEARNING_ROOT.parents[1]


# =============================================================================
# 1. PROJECT / STORAGE SETTINGS
# =============================================================================

MASTERDATAANALYSIS_CSV = (
    PFPEPTOIDS_ROOT / "analysis" / "masterdataanalysis.csv"
)

ACTIVE_LEARNING_DIR = SHARED_DATA_DIR / "active_learning"
ACTIVE_LEARNING_DB = ACTIVE_LEARNING_DIR / "active_learning.db"
ACTIVE_LEARNING_XLSX = ACTIVE_LEARNING_DIR / "active_learning_overview.xlsx"
TRAINING_DATASET_DIR = ACTIVE_LEARNING_DIR / "training_datasets"
SELECTION_EXPORT_DIR = ACTIVE_LEARNING_DIR / "selections"
SIMULATION_QUEUE_DIR = ACTIVE_LEARNING_DIR / "simulation_queues"

TRAINING_SCRIPT = scripts_dir("position_resolved") / "peptoidmachinelearning.py"
PREDICTION_SCRIPT = scripts_dir("position_resolved") / "tripeptoid_predictions.py"
TRAINING_SLURM_SCRIPT = scripts_dir("position_resolved") / "machinelearningsubmission.sh"
GLOBAL_COMPARISON_SLURM_SCRIPT = (
    scripts_dir("global_molecular") / "machinelearningsubmission.sh"
)
PREDICTION_SLURM_SCRIPT = scripts_dir("position_resolved") / "tripeptoid_predictions.sh"

PYTHON_EXECUTABLE = Path("/users/pfb19164/.conda/envs/my_env_2025/bin/python")


# =============================================================================
# 2. SIMULATION / TARGET SETTINGS
# =============================================================================

TARGET_NAME = "DDAPL"
BACTERIAL_SYSTEM_TOKEN = "POPE75POPG25_W_WF"
MAMMALIAN_SYSTEM_TOKEN = "POPC60CHOL40_W_WF"
REQUIRED_RELAX_NS = 100.0

# A duplicate/rerun must never be silently resolved. Explicitly nominate the
# preferred source run for known duplicated IDs. Any new ambiguous duplicate
# without an override causes sync to stop.
PREFERRED_OBSERVATION_SOURCE_RUN_OVERRIDES = {
    "Nfe-Nfnap-Nfe": "run_0",
}


# =============================================================================
# 3. DESCRIPTOR SETTINGS
# =============================================================================

DESCRIPTOR_VERSION = "v1"
EXPECTED_SEQUENCE_COUNT = 32768
EXPECTED_FEATURE_COUNT = 323

FULL_CLEANED_DESCRIPTOR_CSV = (
    descriptor_version_dir("position_resolved", DESCRIPTOR_VERSION)
    / "all_sequence_rdkit_descriptors_with_targets_cleaned.csv"
)

PROTECTED_COLUMNS = [
    "ID",
    "SMILES",
    "DAPL_bacteria",
    "DAPL_mammalian",
    "DDAPL",
]

KNOWN_MONOMERS = {
    "Na", "Nab", "Nc", "Nd", "Ne", "Nf", "Nfe", "NfeB4",
    "NfeC4", "Nfex", "Nfn", "Nfnap", "Ng", "Nh", "Ni", "Nk",
    "Nke", "Nkeqm", "Nl", "Nm", "NmO", "Nn", "Np", "Nq", "Nr",
    "Ns", "Nse", "Nt", "Nv", "Nw", "Nwe", "Ny",
}


# =============================================================================
# 4. MODEL-0 / BENCHMARK SETTINGS
# =============================================================================

EXPECTED_MODEL0_UNIQUE_PEPTOIDS = 40
MODEL0_BENCHMARK_SIZE = 8
MODEL0_TRAIN_SIZE = 32
MODEL0_SPLIT_VERSION = "active_model0_v1"
MODEL0_SPLIT_RANDOM_SEED = 42
BENCHMARK_POLICY = "fixed"  # Designed so a future growing policy can be added.
MODEL_SELECTION_METRIC = "r2"  # Highest benchmark R2; RMSE is tie-break/secondary.


# =============================================================================
# 5. MACHINE-LEARNING SETTINGS
# =============================================================================

ALL_MODEL_CANDIDATES = [
    "rf",
    "lgbm",
    "xgb",
    "gbrt",
    "mlp",
    "ada",
    "svr_gaussian",
    "knn",
]

N_OPTUNA_TRIALS = 50
N_CV_FOLDS = 5
N_JOBS = 40
OPTUNA_RANDOM_SEED = 42
CV_RANDOM_SEED = 42

# PFPEPTOIDS_ACTIVE_LEARNING_V2
# Candidate algorithms and representations are configured PER LOGICAL MODEL
# in config/model_search_config.json. A prepared model freezes those choices
# into SQLite, so later config edits cannot silently rewrite its experiment.
MODEL_SEARCH_CONFIG_PATH = _MACHINELEARNING_ROOT / "config" / "model_search_config.json"
VALID_REPRESENTATIONS = ["global_molecular", "position_resolved"]
MODEL_COMPARISON_DIR = ACTIVE_LEARNING_DIR / "model_comparisons"

# "slurm" or "direct". Keep user-editable here rather than scattered through
# subordinate scripts.
TRAIN_EXECUTION_MODE = "slurm"
PREDICTION_EXECUTION_MODE = "slurm"


# =============================================================================
# 6. ACTIVE-LEARNING STRATEGY SETTINGS
# =============================================================================

# Canonical branch-selection phase. These strategies may be selected together
# or in reviewed subsets (for example A/B/D while C/E are still waiting).
# Each branch is isolated by its own source-model dataset and acquisition history.
ACTIVE_STRATEGIES = ["A", "B", "C", "D", "E"]

# Pooled branch definitions.
#
# Model1F is the single shared pooled A-E seed.
#
# From cycle 2 onward the pooled experiment forks:
#   F = cumulative canonical A-E acquisitions through the current cycle
#       + all PRIOR canonical F acquisitions.  F therefore learns from its own
#       discoveries and is the self-feedback pooled active-learning branch.
#   G = cumulative canonical A-E acquisitions through the current cycle ONLY.
#       G never trains on its own prior selections and is the no-self-feedback
#       pooled control.  There is intentionally no Model1G because it would be
#       identical to Model1F and would waste training resources.
F_POOL_STRATEGIES = ["A", "B", "C", "D", "E"]
ALL_CANONICAL_STRATEGIES = ACTIVE_STRATEGIES + ["F", "G"]

# H is deliberately left unused so a genuinely new future acquisition strategy
# can be introduced without another historical relabelling exercise.
RESERVED_STRATEGIES = ["H"]

# X/Y preserve the original failed closest-to-zero C/E methods.  They remain
# queryable and plottable for retrospective comparison, but normal workflow
# commands must never generate new X/Y acquisitions.  Their methodology remains
# implemented below because closest-to-zero may become scientifically useful if
# future models predict a genuine negative-DDAPL region.
HISTORICAL_INACTIVE_STRATEGIES = ["X", "Y"]

# The original Cycle-1 F selection was made directly from Model0 before the
# corrected pooled Model1F existed.  Preserve it under an explicit non-canonical
# label; it is not allowed into canonical F/G lineages.
LEGACY_F_STRATEGY = "F_legacy"
LEGACY_F_MODEL_ID = "Model1F_legacy_pre_midpoint"

# Historical/non-canonical models are preserved only for audit/provenance.
# They are never allowed to participate in active training, prediction,
# selection, status summaries, or normal active-workflow exports.
ARCHIVED_MODEL_REASONS = {
    "Model1X": "historical inactive old-C closest-to-zero strategy",
    "Model1Y": "historical inactive old-E closest-to-zero strategy",
    LEGACY_F_MODEL_ID: (
        "premature pre-midpoint pooled F trained before the corrected canonical "
        "A-E Cycle-1 cohort existed"
    ),
}
HISTORICAL_SELECTION_STRATEGIES = set(HISTORICAL_INACTIVE_STRATEGIES) | {LEGACY_F_STRATEGY}

ACQUISITION_SIZE = 15

STRATEGY_COUNTS = {
    "A": {"top": 15},
    "B": {"even_range": 15},
    "C": {"top": 5, "midpoint": 5, "bottom": 5},
    "D": {"top": 10, "bottom": 5},
    "E": {"top": 7, "midpoint": 4, "bottom": 4},
    "F": {"top": 15},
    "G": {"top": 15},
    # Historical/inactive methods retained for reproducibility/investigation.
    "X": {"top": 5, "zero": 5, "bottom": 5},
    "Y": {"top": 7, "zero": 4, "bottom": 4},
}


# =============================================================================
# 7. MONOMER / SIMULATION AVAILABILITY SETTINGS
# =============================================================================

# These restrictions apply ONLY to NEW physical simulation selection.
# They do NOT remove molecules from the 32,768 ML descriptor/prediction space,
# and they do NOT invalidate historical real training observations.

MONOMER_UNAVAILABLE_REASONS = {
    "Nh": "monomer not available to Martinoid",
    "Nc": "monomer not available to Martinoid",
    "Ng": "monomer not available to Martinoid",
    "Np": "monomer not available to Martinoid",
    "Nr": "monomer not available to synthetic lab",
    "Ns": "monomer not available to synthetic lab",
    "Nt": "monomer not available to synthetic lab",
    "Nq": "monomer not available to synthetic lab",
    "Ny": "monomer not available to synthetic lab",
    "Nkeqm": "monomer not available to synthetic lab",
    "Nab": "monomer unavailable",
}

POSITION_UNAVAILABLE_MONOMERS = {
    1: {"Nw"},  # N-terminus / first token
    2: set(),
    3: {"Nw"},  # C-terminus / third token
}

# Nkeqm was intentionally available, however, this is no longer the case with the lab not having it - unsure if / when that will need changed again and unsure why it was special in the first place - before not being available.


# =============================================================================
# 8. DATABASE SCHEMA
# =============================================================================

DB_SCHEMA_VERSION = "1.2"


# =============================================================================
# BASIC HELPERS
# =============================================================================


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_true(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def bool_false(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"false", "0", "no", "n"})
    )


def ensure_directories() -> None:
    for path in [
        ACTIVE_LEARNING_DIR,
        TRAINING_DATASET_DIR,
        SELECTION_EXPORT_DIR,
        SIMULATION_QUEUE_DIR,
        MODEL_COMPARISON_DIR,
        SLURM_OUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def connect_db() -> sqlite3.Connection:
    ensure_directories()
    connection = sqlite3.connect(ACTIVE_LEARNING_DB)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialise_database() -> None:
    with connect_db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS peptoids (
                id TEXT PRIMARY KEY,
                smiles TEXT,
                m1 TEXT NOT NULL,
                m2 TEXT NOT NULL,
                m3 TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_key TEXT UNIQUE NOT NULL,
                id TEXT NOT NULL REFERENCES peptoids(id),
                source_run TEXT NOT NULL,
                dapl_bacteria REAL NOT NULL,
                dapl_mammalian REAL NOT NULL,
                ddapl REAL NOT NULL,
                bacteria_system_dir TEXT NOT NULL,
                mammalian_system_dir TEXT NOT NULL,
                relax_complete INTEGER NOT NULL,
                analysis_complete INTEGER NOT NULL,
                preferred INTEGER NOT NULL DEFAULT 0,
                imported_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS splits (
                split_version TEXT PRIMARY KEY,
                policy TEXT NOT NULL,
                target TEXT NOT NULL,
                random_seed INTEGER,
                locked INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS split_membership (
                split_version TEXT NOT NULL REFERENCES splits(split_version),
                id TEXT NOT NULL REFERENCES peptoids(id),
                role TEXT NOT NULL,
                target_at_split REAL NOT NULL,
                target_rank INTEGER NOT NULL,
                rank_block INTEGER NOT NULL,
                PRIMARY KEY (split_version, id)
            );

            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                cycle INTEGER NOT NULL,
                strategy TEXT,
                parent_model_id TEXT,
                dataset_name TEXT,
                split_version TEXT,
                descriptor_version TEXT NOT NULL,
                target TEXT NOT NULL,
                candidate_models_json TEXT NOT NULL,
                training_run_id TEXT,
                prediction_run_id TEXT,
                chosen_algorithm TEXT,
                benchmark_r2 REAL,
                benchmark_rmse REAL,
                benchmark_mae REAL,
                n_train INTEGER,
                n_benchmark INTEGER,
                training_job_id TEXT,
                prediction_job_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_membership (
                model_id TEXT NOT NULL REFERENCES models(model_id),
                id TEXT NOT NULL REFERENCES peptoids(id),
                role TEXT NOT NULL,
                acquisition_cycle INTEGER,
                acquisition_strategy TEXT,
                PRIMARY KEY (model_id, id)
            );

            CREATE TABLE IF NOT EXISTS prediction_runs (
                prediction_run_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL REFERENCES models(model_id),
                training_run_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS selection_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                source_model_id TEXT NOT NULL REFERENCES models(model_id),
                prediction_run_id TEXT NOT NULL,
                id TEXT NOT NULL REFERENCES peptoids(id),
                predicted_ddapl REAL NOT NULL,
                raw_prediction_rank INTEGER NOT NULL,
                selection_group TEXT NOT NULL,
                group_rank INTEGER,
                target_value REAL,
                decision_order INTEGER NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                reason_detail TEXT,
                created_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_selection_decision_unique
            ON selection_decisions(cycle, strategy, selection_group, decision_order);

            CREATE TABLE IF NOT EXISTS simulation_queue (
                cycle INTEGER NOT NULL,
                id TEXT NOT NULL REFERENCES peptoids(id),
                requested_by TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (cycle, id)
            );

            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                decision_id INTEGER PRIMARY KEY REFERENCES selection_decisions(decision_id),
                observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
                predicted_ddapl REAL NOT NULL,
                actual_ddapl REAL NOT NULL,
                residual REAL NOT NULL,
                absolute_error REAL NOT NULL,
                resolved_at TEXT NOT NULL
            );
            """
        )
        # Schema v1.2: representation-aware tracking plus explicit archival
        # state for historical/non-canonical models.
        def ensure_column(table: str, column: str, declaration: str) -> None:
            existing = {
                str(row[1])
                for row in con.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

        ensure_column("models", "candidate_representations_json", "TEXT")
        ensure_column("models", "chosen_representation", "TEXT")
        ensure_column("models", "training_job_ids_json", "TEXT")
        ensure_column("models", "archived", "INTEGER NOT NULL DEFAULT 0")
        ensure_column("models", "archive_reason", "TEXT")
        ensure_column("simulation_queue", "archived", "INTEGER NOT NULL DEFAULT 0")
        ensure_column("simulation_queue", "archive_reason", "TEXT")
        ensure_column("prediction_runs", "representation", "TEXT")

        # Known historical/non-canonical models are deterministically quarantined
        # from the active workflow. Their scientific records remain in SQLite for
        # provenance, but normal controller actions cannot use them.
        for model_id, reason in ARCHIVED_MODEL_REASONS.items():
            con.execute(
                """
                UPDATE models
                SET archived=1, archive_reason=?
                WHERE model_id=?
                """,
                (reason, model_id),
            )

        con.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("schema_version", DB_SCHEMA_VERSION),
        )


# =============================================================================
# VALIDATION / DESCRIPTOR UNIVERSE
# =============================================================================


def validate_monomer_configuration() -> None:
    configured_unavailable = set(MONOMER_UNAVAILABLE_REASONS)
    position_only = set().union(*POSITION_UNAVAILABLE_MONOMERS.values())
    unknown = (configured_unavailable | position_only) - KNOWN_MONOMERS
    if unknown:
        raise ValueError(f"Unknown monomers in restriction configuration: {sorted(unknown)}")


def load_and_validate_descriptor_universe() -> tuple[pd.DataFrame, list[str]]:
    if not FULL_CLEANED_DESCRIPTOR_CSV.is_file():
        raise FileNotFoundError(
            f"Authoritative cleaned descriptor file not found:\n{FULL_CLEANED_DESCRIPTOR_CSV}"
        )

    df = pd.read_csv(FULL_CLEANED_DESCRIPTOR_CSV)

    if len(df) != EXPECTED_SEQUENCE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SEQUENCE_COUNT} descriptor rows; found {len(df)}."
        )

    missing = [c for c in PROTECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Descriptor file missing protected columns: {missing}")

    if df["ID"].astype(str).duplicated().any():
        raise ValueError("Authoritative descriptor universe contains duplicate IDs.")

    feature_columns = [c for c in df.columns if c not in PROTECTED_COLUMNS]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features; found {len(feature_columns)}."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate feature names detected.")

    parsed = df["ID"].astype(str).str.split("-")
    bad = df.loc[parsed.map(len) != 3, "ID"].astype(str).tolist()
    if bad:
        raise ValueError(f"IDs are not valid three-token sequences: {bad[:20]}")

    monomers = set(token for parts in parsed for token in parts)
    if monomers != KNOWN_MONOMERS:
        raise ValueError(
            "Descriptor monomer universe mismatch.\n"
            f"Expected: {sorted(KNOWN_MONOMERS)}\n"
            f"Found:    {sorted(monomers)}"
        )

    return df, feature_columns


def sync_peptoids(con: sqlite3.Connection, descriptor_df: pd.DataFrame) -> None:
    rows = []
    for row in descriptor_df[["ID", "SMILES"]].itertuples(index=False):
        parts = str(row.ID).split("-")
        rows.append((str(row.ID), str(row.SMILES), parts[0], parts[1], parts[2]))

    con.executemany(
        """
        INSERT INTO peptoids(id, smiles, m1, m2, m3)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            smiles=excluded.smiles,
            m1=excluded.m1,
            m2=excluded.m2,
            m3=excluded.m3
        """,
        rows,
    )


# =============================================================================
# SIMULATION SYNCHRONISATION
# =============================================================================


def source_run_from_system_dir(system_dir: str) -> str:
    path = Path(str(system_dir))
    if len(path.parts) < 2:
        raise ValueError(f"Cannot determine source run from system_dir: {system_dir}")
    return path.parent.name


def observation_key(id_: str, source_run: str, bacteria_dir: str, mammalian_dir: str) -> str:
    raw = "|".join([id_, source_run, bacteria_dir, mammalian_dir])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_complete_observation_pairs() -> list[dict]:
    if not MASTERDATAANALYSIS_CSV.is_file():
        raise FileNotFoundError(f"masterdataanalysis.csv not found:\n{MASTERDATAANALYSIS_CSV}")

    df = pd.read_csv(MASTERDATAANALYSIS_CSV)
    required = {
        "molecule_type",
        "length",
        "contains_mixture",
        "token_sequence",
        "system_dir",
        "APL_change",
        "analysis_complete",
        "reached_100ns",
        "final_relax_time_ns",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"masterdataanalysis.csv missing columns: {missing}")

    mask = (
        df["molecule_type"].astype(str).str.strip().str.lower().eq("peptoid")
        & pd.to_numeric(df["length"], errors="coerce").eq(3)
        & bool_false(df["contains_mixture"])
    )
    work = df.loc[mask].copy()
    work["token_sequence"] = work["token_sequence"].astype(str).str.strip()
    work["system_dir"] = work["system_dir"].astype(str)

    work["membrane"] = np.select(
        [
            work["system_dir"].str.contains(BACTERIAL_SYSTEM_TOKEN, regex=False, na=False),
            work["system_dir"].str.contains(MAMMALIAN_SYSTEM_TOKEN, regex=False, na=False),
        ],
        ["bacteria", "mammalian"],
        default="other",
    )
    work = work[work["membrane"].isin(["bacteria", "mammalian"])].copy()

    work["analysis_complete_bool"] = bool_true(work["analysis_complete"])
    work["reached_100ns_bool"] = bool_true(work["reached_100ns"])
    work["relax_ns"] = pd.to_numeric(work["final_relax_time_ns"], errors="coerce")
    work["apl"] = pd.to_numeric(work["APL_change"], errors="coerce")

    work = work[
        work["analysis_complete_bool"]
        & work["reached_100ns_bool"]
        & work["relax_ns"].ge(REQUIRED_RELAX_NS)
        & work["apl"].notna()
    ].copy()

    work["source_run"] = work["system_dir"].map(source_run_from_system_dir)

    pairs = []
    for (id_, source_run), group in work.groupby(["token_sequence", "source_run"]):
        counts = group["membrane"].value_counts().to_dict()
        bacteria_n = int(counts.get("bacteria", 0))
        mammalian_n = int(counts.get("mammalian", 0))

        if bacteria_n == 0 or mammalian_n == 0:
            continue
        if bacteria_n != 1 or mammalian_n != 1:
            raise ValueError(
                "Ambiguous duplicate rows within the same paired simulation run:\n"
                f"ID={id_}, source_run={source_run}, "
                f"bacteria={bacteria_n}, mammalian={mammalian_n}"
            )

        b = group.loc[group["membrane"] == "bacteria"].iloc[0]
        m = group.loc[group["membrane"] == "mammalian"].iloc[0]

        dapl_bacteria = float(b["apl"])
        dapl_mammalian = float(m["apl"])
        ddapl = dapl_bacteria - dapl_mammalian

        pairs.append(
            {
                "id": str(id_),
                "source_run": str(source_run),
                "dapl_bacteria": dapl_bacteria,
                "dapl_mammalian": dapl_mammalian,
                "ddapl": ddapl,
                "bacteria_system_dir": str(b["system_dir"]),
                "mammalian_system_dir": str(m["system_dir"]),
                "relax_complete": 1,
                "analysis_complete": 1,
            }
        )

    return pairs


def choose_preferred_observations(pairs: list[dict]) -> dict[str, str]:
    by_id: dict[str, list[dict]] = {}
    for pair in pairs:
        by_id.setdefault(pair["id"], []).append(pair)

    preferred_source: dict[str, str] = {}
    for id_, id_pairs in by_id.items():
        if len(id_pairs) == 1:
            preferred_source[id_] = id_pairs[0]["source_run"]
            continue

        override = PREFERRED_OBSERVATION_SOURCE_RUN_OVERRIDES.get(id_)
        if override is None:
            runs = [p["source_run"] for p in id_pairs]
            raise ValueError(
                "Multiple complete simulation observations exist and no explicit "
                "preferred source-run override is configured.\n"
                f"ID={id_}\nsource_runs={runs}"
            )

        matches = [p for p in id_pairs if p["source_run"] == override]
        if len(matches) != 1:
            raise ValueError(
                f"Preferred source-run override for {id_} is {override!r}, "
                f"but matching complete pairs found={len(matches)}."
            )
        preferred_source[id_] = override

    return preferred_source


def sync_observations(con: sqlite3.Connection) -> tuple[int, int]:
    pairs = extract_complete_observation_pairs()
    preferred_by_id = choose_preferred_observations(pairs)
    now = now_iso()

    for pair in pairs:
        key = observation_key(
            pair["id"],
            pair["source_run"],
            pair["bacteria_system_dir"],
            pair["mammalian_system_dir"],
        )

        existing = con.execute(
            "SELECT * FROM observations WHERE observation_key = ?",
            (key,),
        ).fetchone()

        if existing is not None:
            checks = {
                "dapl_bacteria": pair["dapl_bacteria"],
                "dapl_mammalian": pair["dapl_mammalian"],
                "ddapl": pair["ddapl"],
            }
            for field, value in checks.items():
                if not np.isclose(float(existing[field]), float(value), rtol=0, atol=1e-12):
                    raise ValueError(
                        "A known simulation source changed scientific values. "
                        "The workflow will not silently overwrite it.\n"
                        f"ID={pair['id']} source_run={pair['source_run']} "
                        f"field={field} old={existing[field]} new={value}"
                    )
            con.execute(
                "UPDATE observations SET last_seen_at = ? WHERE observation_key = ?",
                (now, key),
            )
        else:
            con.execute(
                """
                INSERT INTO observations(
                    observation_key, id, source_run,
                    dapl_bacteria, dapl_mammalian, ddapl,
                    bacteria_system_dir, mammalian_system_dir,
                    relax_complete, analysis_complete,
                    preferred, imported_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    key,
                    pair["id"],
                    pair["source_run"],
                    pair["dapl_bacteria"],
                    pair["dapl_mammalian"],
                    pair["ddapl"],
                    pair["bacteria_system_dir"],
                    pair["mammalian_system_dir"],
                    pair["relax_complete"],
                    pair["analysis_complete"],
                    now,
                    now,
                ),
            )

    con.execute("UPDATE observations SET preferred = 0")
    for id_, source_run in preferred_by_id.items():
        updated = con.execute(
            """
            UPDATE observations
            SET preferred = 1
            WHERE id = ? AND source_run = ?
            """,
            (id_, source_run),
        ).rowcount
        if updated != 1:
            raise RuntimeError(
                f"Expected one preferred observation row for {id_}/{source_run}; updated {updated}."
            )

    preferred_count = con.execute(
        "SELECT COUNT(*) FROM observations WHERE preferred = 1"
    ).fetchone()[0]

    return len(pairs), int(preferred_count)


def preferred_observation_df(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT observation_id, id,
               dapl_bacteria AS DAPL_bacteria,
               dapl_mammalian AS DAPL_mammalian,
               ddapl AS DDAPL,
               source_run,
               bacteria_system_dir,
               mammalian_system_dir
        FROM observations
        WHERE preferred = 1
        ORDER BY DDAPL DESC, id ASC
        """,
        con,
    )


# =============================================================================
# PREDICTION OUTCOMES / QUEUE STATE
# =============================================================================


def refresh_prediction_outcomes(con: sqlite3.Connection) -> int:
    rows = con.execute(
        """
        SELECT d.decision_id, d.predicted_ddapl,
               o.observation_id, o.ddapl
        FROM selection_decisions d
        JOIN observations o ON o.id = d.id AND o.preferred = 1
        """
    ).fetchall()

    now = now_iso()
    for row in rows:
        predicted = float(row["predicted_ddapl"])
        actual = float(row["ddapl"])
        residual = actual - predicted
        con.execute(
            """
            INSERT INTO prediction_outcomes(
                decision_id, observation_id, predicted_ddapl,
                actual_ddapl, residual, absolute_error, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(decision_id) DO UPDATE SET
                observation_id=excluded.observation_id,
                predicted_ddapl=excluded.predicted_ddapl,
                actual_ddapl=excluded.actual_ddapl,
                residual=excluded.residual,
                absolute_error=excluded.absolute_error,
                resolved_at=excluded.resolved_at
            """,
            (
                row["decision_id"],
                row["observation_id"],
                predicted,
                actual,
                residual,
                abs(residual),
                now,
            ),
        )

    con.execute(
        """
        UPDATE simulation_queue
        SET status = 'analysis_ready', updated_at = ?
        WHERE id IN (SELECT id FROM observations WHERE preferred = 1)
          AND COALESCE(archived,0)=0
        """,
        (now,),
    )

    return len(rows)


# =============================================================================
# MODEL-0 SPLIT
# =============================================================================


def create_model0_split(con: sqlite3.Connection) -> None:
    existing = con.execute(
        "SELECT 1 FROM splits WHERE split_version = ?",
        (MODEL0_SPLIT_VERSION,),
    ).fetchone()
    if existing:
        print(f"Model-0 split already exists and is locked: {MODEL0_SPLIT_VERSION}")
        return

    obs = preferred_observation_df(con)
    if len(obs) != EXPECTED_MODEL0_UNIQUE_PEPTOIDS:
        raise ValueError(
            f"Model 0 requires exactly {EXPECTED_MODEL0_UNIQUE_PEPTOIDS} unique "
            f"preferred real peptoids; currently found {len(obs)}."
        )

    if MODEL0_TRAIN_SIZE + MODEL0_BENCHMARK_SIZE != EXPECTED_MODEL0_UNIQUE_PEPTOIDS:
        raise ValueError("Model-0 configured train + benchmark sizes do not equal baseline total.")

    if EXPECTED_MODEL0_UNIQUE_PEPTOIDS % MODEL0_BENCHMARK_SIZE != 0:
        raise ValueError(
            "Current Model-0 split method requires equal rank blocks. "
            "Baseline total must divide exactly by benchmark size."
        )

    ranked = obs.sort_values([TARGET_NAME, "id"], ascending=[True, True]).reset_index(drop=True)
    ranked["target_rank"] = np.arange(1, len(ranked) + 1)
    block_size = EXPECTED_MODEL0_UNIQUE_PEPTOIDS // MODEL0_BENCHMARK_SIZE
    ranked["rank_block"] = ((ranked["target_rank"] - 1) // block_size) + 1

    rng = np.random.default_rng(MODEL0_SPLIT_RANDOM_SEED)
    benchmark_ids = []
    for block in range(1, MODEL0_BENCHMARK_SIZE + 1):
        block_ids = (
            ranked.loc[ranked["rank_block"] == block, "id"]
            .astype(str)
            .sort_values()
            .tolist()
        )
        if len(block_ids) != block_size:
            raise RuntimeError(f"Rank block {block} has {len(block_ids)} IDs; expected {block_size}.")
        benchmark_ids.append(str(rng.choice(block_ids)))

    ranked["role"] = np.where(ranked["id"].isin(benchmark_ids), "benchmark", "train")

    if int((ranked["role"] == "benchmark").sum()) != MODEL0_BENCHMARK_SIZE:
        raise RuntimeError("Wrong number of benchmark molecules selected.")
    if int((ranked["role"] == "train").sum()) != MODEL0_TRAIN_SIZE:
        raise RuntimeError("Wrong number of training molecules selected.")

    con.execute(
        """
        INSERT INTO splits(split_version, policy, target, random_seed, locked, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (
            MODEL0_SPLIT_VERSION,
            "DDAPL_rank_stratified_equal_blocks",
            TARGET_NAME,
            MODEL0_SPLIT_RANDOM_SEED,
            now_iso(),
        ),
    )

    con.executemany(
        """
        INSERT INTO split_membership(
            split_version, id, role, target_at_split, target_rank, rank_block
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                MODEL0_SPLIT_VERSION,
                str(row.id),
                str(row.role),
                float(getattr(row, TARGET_NAME)),
                int(row.target_rank),
                int(row.rank_block),
            )
            for row in ranked.itertuples(index=False)
        ],
    )

    split_export = SPLITS_DIR / f"split_assignments_{MODEL0_SPLIT_VERSION}.csv"
    export_df = ranked[["id", TARGET_NAME, "target_rank", "rank_block", "role"]].copy()
    export_df = export_df.rename(columns={"id": "ID", "role": "split"})
    export_df.insert(1, "split_version", MODEL0_SPLIT_VERSION)
    export_df.to_csv(split_export, index=False)

    print("Created and locked Model-0 benchmark split.")
    print(f"Split: {MODEL0_SPLIT_VERSION}")
    print(f"Export: {split_export}")
    print("Benchmark IDs:")
    for id_ in benchmark_ids:
        ddapl = float(ranked.loc[ranked["id"] == id_, TARGET_NAME].iloc[0])
        block = int(ranked.loc[ranked["id"] == id_, "rank_block"].iloc[0])
        print(f"    block {block}: {id_}  DDAPL={ddapl:.6f}")


# =============================================================================
# LOGICAL MODEL / DATASET MEMBERSHIP
# =============================================================================


def active_model_row(
    con: sqlite3.Connection,
    model_id: str,
    *,
    purpose: str = "use",
) -> sqlite3.Row:
    """Return a model row only if it is part of the canonical active workflow."""
    row = con.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown model {model_id}")
    archived = int(row["archived"] or 0) if "archived" in row.keys() else 0
    if archived or model_id in ARCHIVED_MODEL_REASONS:
        reason = row["archive_reason"] if "archive_reason" in row.keys() else None
        reason = reason or ARCHIVED_MODEL_REASONS.get(model_id, "historical/non-canonical model")
        raise ValueError(
            f"{model_id} is archived and cannot be used for active workflow {purpose}. "
            f"Reason: {reason}."
        )
    return row


def parse_model_id(model_id: str) -> tuple[int, str | None]:
    if model_id == "Model0":
        return 0, None
    match = re.fullmatch(r"Model(\d+)([A-HXY])", model_id)
    if not match:
        raise ValueError(
            "Model ID must be Model0 or Model<cycle><strategy>, e.g. Model2F. "
            "Canonical active strategies are A-G; H is reserved; X/Y are "
            "historical inactive zero-target branches."
        )
    return int(match.group(1)), match.group(2)


def previous_model_id(cycle: int, strategy: str | None) -> str | None:
    if cycle == 0:
        return None
    if cycle == 1:
        return "Model0"
    if strategy == "G" and cycle == 2:
        # Model1F is the shared pooled A-E seed; Model1G is intentionally absent.
        return "Model1F"
    return f"Model{cycle - 1}{strategy}"


def load_model_search_config() -> dict:
    if not MODEL_SEARCH_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "Per-model search configuration is missing:\n"
            f"{MODEL_SEARCH_CONFIG_PATH}"
        )
    with MODEL_SEARCH_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "default" not in config or not isinstance(config["default"], dict):
        raise ValueError("model_search_config.json requires a 'default' object.")
    return config


def resolve_model_search_policy(
    con: sqlite3.Connection,
    model_id: str,
    cycle: int,
    strategy: str | None,
) -> tuple[list[str], list[str]]:
    config = load_model_search_config()
    policy = dict(config.get("default", {}))
    specific = config.get(model_id)
    if specific is not None:
        if not isinstance(specific, dict):
            raise ValueError(f"Configuration for {model_id} must be a JSON object.")
        policy.update(specific)

    model_spec = policy.get("models", "all")
    if model_spec == "all":
        candidates = list(ALL_MODEL_CANDIDATES)
    elif model_spec == "parent_winner":
        parent = previous_model_id(cycle, strategy)
        if parent is None:
            raise ValueError("Model0 cannot use models='parent_winner'.")
        row = con.execute(
            "SELECT chosen_algorithm FROM models WHERE model_id = ?", (parent,)
        ).fetchone()
        if row is None or not row["chosen_algorithm"]:
            raise ValueError(
                f"Cannot prepare {model_id}: parent model {parent} has no chosen algorithm."
            )
        candidates = [str(row["chosen_algorithm"])]
    elif isinstance(model_spec, list):
        candidates = [str(x) for x in model_spec]
    else:
        raise ValueError(
            f"{model_id} models must be 'all', 'parent_winner', or a list; got {model_spec!r}."
        )

    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError(f"{model_id} candidate model list is empty or contains duplicates: {candidates}")
    unknown = sorted(set(candidates) - set(ALL_MODEL_CANDIDATES))
    if unknown:
        raise ValueError(f"{model_id} contains unsupported candidate algorithms: {unknown}")

    representations = policy.get(
        "representations", ["global_molecular", "position_resolved"]
    )
    if not isinstance(representations, list):
        raise ValueError(f"{model_id} representations must be a JSON list.")
    representations = [str(x) for x in representations]
    if not representations or len(representations) != len(set(representations)):
        raise ValueError(
            f"{model_id} representation list is empty or contains duplicates: {representations}"
        )
    unknown_rep = sorted(set(representations) - set(VALID_REPRESENTATIONS))
    if unknown_rep:
        raise ValueError(f"{model_id} contains unsupported representations: {unknown_rep}")
    return candidates, representations


def candidate_models_for(
    con: sqlite3.Connection,
    model_id: str,
    cycle: int,
    strategy: str | None,
) -> list[str]:
    return resolve_model_search_policy(con, model_id, cycle, strategy)[0]


def candidate_representations_for(
    con: sqlite3.Connection,
    model_id: str,
    cycle: int,
    strategy: str | None,
) -> list[str]:
    return resolve_model_search_policy(con, model_id, cycle, strategy)[1]

def get_benchmark_ids(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        """
        SELECT id FROM split_membership
        WHERE split_version = ? AND role = 'benchmark'
        ORDER BY target_rank
        """,
        (MODEL0_SPLIT_VERSION,),
    ).fetchall()
    return [str(row["id"]) for row in rows]


def get_baseline_train_ids(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        """
        SELECT id FROM split_membership
        WHERE split_version = ? AND role = 'train'
        ORDER BY target_rank
        """,
        (MODEL0_SPLIT_VERSION,),
    ).fetchall()
    return [str(row["id"]) for row in rows]


def selected_ids_for_lineage(
    con: sqlite3.Connection,
    cycle: int,
    strategy: str,
) -> dict[str, tuple[int, str]]:
    """Return the exact real-acquisition lineage for a logical model.

    A-E
        Strict isolated cumulative branch lineages.

    F
        Self-feedback pooled branch.  Model1F uses canonical A-E Cycle-1 data.
        Model<n>F (n >= 2) uses canonical A-E acquisitions through cycle n plus
        canonical F acquisitions from cycles < n.  The current cycle F selection
        cannot enter its own training because it is generated only after Model<n>F
        has trained and predicted.

    G
        Pooled no-self-feedback control, beginning at Model2G.  Uses canonical
        A-E acquisitions through the current cycle and never G acquisitions.

    X/Y
        Historical inactive isolated branches retained only for provenance.
    """
    if strategy in ACTIVE_STRATEGIES or strategy in HISTORICAL_INACTIVE_STRATEGIES:
        rows = con.execute(
            """
            SELECT cycle, strategy, id
            FROM selection_decisions
            WHERE decision='selected' AND cycle<=? AND strategy=?
            ORDER BY cycle, decision_order
            """,
            (cycle, strategy),
        ).fetchall()
    elif strategy == "F":
        placeholders = ",".join("?" for _ in F_POOL_STRATEGIES)
        rows = con.execute(
            f"""
            SELECT cycle, strategy, id
            FROM selection_decisions
            WHERE decision='selected'
              AND (
                    (cycle<=? AND strategy IN ({placeholders}))
                 OR (cycle<?  AND strategy='F')
              )
            ORDER BY cycle, strategy, decision_order
            """,
            (cycle, *F_POOL_STRATEGIES, cycle),
        ).fetchall()
    elif strategy == "G":
        if cycle < 2:
            raise ValueError(
                "Model1G is intentionally absent: Model1F is the shared pooled A-E seed. "
                "The no-self-feedback control begins at Model2G."
            )
        placeholders = ",".join("?" for _ in F_POOL_STRATEGIES)
        rows = con.execute(
            f"""
            SELECT cycle, strategy, id
            FROM selection_decisions
            WHERE decision='selected'
              AND cycle<=?
              AND strategy IN ({placeholders})
            ORDER BY cycle, strategy, decision_order
            """,
            (cycle, *F_POOL_STRATEGIES),
        ).fetchall()
    else:
        raise ValueError(f"Strategy {strategy!r} has no training-lineage definition.")

    # Preserve one membership row per physical ID while retaining all qualifying
    # strategy provenance tokens.  The acquisition cycle is the earliest eligible
    # acquisition of that ID in this logical lineage.
    first_cycle: dict[str, int] = {}
    strategy_tokens: dict[str, set[str]] = {}
    ordered_ids: list[str] = []
    for row in rows:
        id_ = str(row["id"])
        cyc = int(row["cycle"])
        strat = str(row["strategy"])
        if id_ not in first_cycle:
            first_cycle[id_] = cyc
            ordered_ids.append(id_)
        else:
            first_cycle[id_] = min(first_cycle[id_], cyc)
        strategy_tokens.setdefault(id_, set()).add(strat)

    provenance: dict[str, tuple[int, str]] = {}
    for id_ in ordered_ids:
        provenance[id_] = (
            first_cycle[id_],
            ",".join(sorted(strategy_tokens[id_])),
        )
    return provenance


def _assert_prior_f_outputs_real(con: sqlite3.Connection, cycle: int, *, requester: str) -> None:
    """Keep the F/G fork synchronized: every earlier F output must be complete."""
    if cycle <= 1:
        return
    preferred = {
        str(row["id"])
        for row in con.execute("SELECT id FROM observations WHERE preferred=1").fetchall()
    }
    for c in range(1, cycle):
        rows = con.execute(
            """
            SELECT DISTINCT id FROM selection_decisions
            WHERE cycle=? AND strategy='F' AND decision='selected'
            """,
            (c,),
        ).fetchall()
        ids = {str(row["id"]) for row in rows}
        if len(ids) != ACQUISITION_SIZE:
            raise ValueError(
                f"Cannot prepare {requester}: prior Cycle-{c} F has {len(ids)} selected IDs; "
                f"expected {ACQUISITION_SIZE}."
            )
        missing = sorted(ids - preferred)
        if missing:
            raise ValueError(
                f"Cannot prepare {requester}: prior Cycle-{c} F simulations are not yet "
                "real/analysis-ready:\n" + "\n".join(missing)
            )


def validate_cycle_acquisitions(
    con: sqlite3.Connection,
    cycle: int,
    strategy: str,
) -> None:
    """Validate the selection history required before building a logical model."""
    if strategy in HISTORICAL_INACTIVE_STRATEGIES:
        raise ValueError(
            f"Strategy {strategy} is historical/inactive and cannot be prepared as a new model."
        )
    if strategy in RESERVED_STRATEGIES:
        raise ValueError(
            f"Strategy {strategy} is reserved for future work and has no active definition."
        )
    if strategy == "G" and cycle < 2:
        raise ValueError(
            "Model1G must not be created: Model1F is the shared pooled A-E seed. "
            "Start the pooled control at Model2G."
        )

    if strategy in ACTIVE_STRATEGIES:
        required = [(c, strategy) for c in range(1, cycle + 1)]
    elif strategy in {"F", "G"}:
        required = [
            (c, s)
            for c in range(1, cycle + 1)
            for s in F_POOL_STRATEGIES
        ]
    else:
        raise ValueError(f"No active acquisition validation rule for strategy {strategy!r}.")

    for c, s in required:
        n = con.execute(
            """
            SELECT COUNT(DISTINCT id)
            FROM selection_decisions
            WHERE cycle=? AND strategy=? AND decision='selected'
            """,
            (c, s),
        ).fetchone()[0]
        if int(n) != ACQUISITION_SIZE:
            raise ValueError(
                f"Cannot prepare Model{cycle}{strategy}: cycle {c}{s} has {n} selected IDs; "
                f"expected {ACQUISITION_SIZE}."
            )

    if strategy in {"F", "G"}:
        # F needs prior F acquisitions as training data; G does not train on them,
        # but the scientific fork is intentionally synchronized after those F
        # discovery simulations are complete.
        _assert_prior_f_outputs_real(
            con, cycle, requester=f"Model{cycle}{strategy}"
        )


def prepare_logical_model(con: sqlite3.Connection, model_id: str) -> None:
    if model_id in ARCHIVED_MODEL_REASONS:
        raise ValueError(
            f"{model_id} is historical/non-canonical and cannot be prepared as an active model."
        )
    cycle, strategy = parse_model_id(model_id)

    if strategy in HISTORICAL_INACTIVE_STRATEGIES:
        raise ValueError(
            f"{model_id} is a preserved historical X/Y model identity and cannot be rebuilt."
        )
    if strategy in RESERVED_STRATEGIES:
        raise ValueError(
            f"{model_id} uses reserved strategy H, which intentionally has no active definition."
        )
    if strategy == "G" and cycle < 2:
        raise ValueError(
            "Model1G is intentionally omitted because Model1F is the shared pooled A-E seed."
        )

    if cycle == 0:
        create_model0_split(con)
    else:
        if not con.execute(
            "SELECT 1 FROM models WHERE model_id = 'Model0'"
        ).fetchone():
            raise ValueError("Model0 must be prepared before later models.")
        validate_cycle_acquisitions(con, cycle, strategy)

    benchmark_ids = get_benchmark_ids(con)
    baseline_train = get_baseline_train_ids(con)

    if len(benchmark_ids) != MODEL0_BENCHMARK_SIZE or len(baseline_train) != MODEL0_TRAIN_SIZE:
        raise RuntimeError("Locked Model-0 split has incorrect membership counts.")

    acquisition_provenance: dict[str, tuple[int, str]] = {}
    if cycle > 0:
        acquisition_provenance = selected_ids_for_lineage(con, cycle, strategy)

    benchmark_set = set(benchmark_ids)
    acquired_ids = [id_ for id_ in acquisition_provenance if id_ not in benchmark_set]
    train_ids = list(dict.fromkeys(baseline_train + acquired_ids))

    # Every training ID must already have a preferred real observation.
    preferred_ids = {
        str(row["id"])
        for row in con.execute("SELECT id FROM observations WHERE preferred = 1").fetchall()
    }
    missing_real = sorted(set(train_ids) - preferred_ids)
    if missing_real:
        raise ValueError(
            f"Cannot prepare {model_id}: selected acquisitions are not yet real/analysis-ready:\n"
            + "\n".join(missing_real)
        )

    candidates, representations = resolve_model_search_policy(
        con, model_id, cycle, strategy
    )
    parent = previous_model_id(cycle, strategy)
    dataset_name = f"dataset_{model_id}"
    split_version = f"split_{model_id}"
    now = now_iso()

    existing = con.execute(
        "SELECT status FROM models WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    if existing and existing["status"] not in {"prepared", "failed", "training_failed"}:
        raise ValueError(
            f"{model_id} already exists with status={existing['status']!r}; "
            "refusing to silently rebuild its membership."
        )

    con.execute(
        """
        INSERT INTO models(
            model_id, cycle, strategy, parent_model_id,
            dataset_name, split_version, descriptor_version, target,
            candidate_models_json, candidate_representations_json,
            n_train, n_benchmark,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
        ON CONFLICT(model_id) DO UPDATE SET
            parent_model_id=excluded.parent_model_id,
            dataset_name=excluded.dataset_name,
            split_version=excluded.split_version,
            descriptor_version=excluded.descriptor_version,
            target=excluded.target,
            candidate_models_json=excluded.candidate_models_json,
            candidate_representations_json=excluded.candidate_representations_json,
            n_train=excluded.n_train,
            n_benchmark=excluded.n_benchmark,
            status='prepared',
            updated_at=excluded.updated_at
        """,
        (
            model_id,
            cycle,
            strategy,
            parent,
            dataset_name,
            split_version,
            DESCRIPTOR_VERSION,
            TARGET_NAME,
            json.dumps(candidates),
            json.dumps(representations),
            len(train_ids),
            len(benchmark_ids),
            now,
            now,
        ),
    )

    con.execute("DELETE FROM dataset_membership WHERE model_id = ?", (model_id,))

    membership_rows = []
    for id_ in baseline_train:
        membership_rows.append((model_id, id_, "train", 0, "baseline"))
    for id_ in acquired_ids:
        acq_cycle, acq_strategy = acquisition_provenance[id_]
        membership_rows.append((model_id, id_, "train", acq_cycle, acq_strategy))
    for id_ in benchmark_ids:
        membership_rows.append((model_id, id_, "benchmark", 0, "benchmark"))

    con.executemany(
        """
        INSERT INTO dataset_membership(
            model_id, id, role, acquisition_cycle, acquisition_strategy
        ) VALUES (?, ?, ?, ?, ?)
        """,
        membership_rows,
    )

    write_model_training_files(con, model_id)

    print(f"Prepared {model_id}")
    print(f"    train:     {len(train_ids)}")
    print(f"    benchmark: {len(benchmark_ids)}")
    print(f"    models:    {candidates}")
    print(f"    representations: {representations}")


def validate_model_lineage(con: sqlite3.Connection, model_id: str) -> None:
    """Fail closed if stored membership differs from the defined branch lineage."""
    model = active_model_row(con, model_id, purpose="lineage validation")

    cycle = int(model["cycle"])
    strategy = model["strategy"]
    benchmark_ids = set(get_benchmark_ids(con))
    baseline_ids = set(get_baseline_train_ids(con))
    rows = con.execute(
        "SELECT id,role,acquisition_cycle,acquisition_strategy "
        "FROM dataset_membership WHERE model_id=?",
        (model_id,),
    ).fetchall()
    train_rows = [r for r in rows if str(r["role"]) == "train"]
    train_ids = {str(r["id"]) for r in train_rows}

    benchmark_in_train = sorted(train_ids & benchmark_ids)
    if benchmark_in_train:
        raise RuntimeError(
            f"{model_id} LINEAGE VIOLATION: benchmark IDs entered training: {benchmark_in_train}"
        )

    acquired = train_ids - baseline_ids
    if cycle == 0:
        if acquired:
            raise RuntimeError(
                f"Model0 LINEAGE VIOLATION: non-baseline training IDs: {sorted(acquired)}"
            )
        return

    if strategy in RESERVED_STRATEGIES:
        raise RuntimeError(f"{model_id} uses reserved strategy H.")

    expected_provenance = selected_ids_for_lineage(con, cycle, str(strategy))
    expected_acquired = set(expected_provenance) - baseline_ids - benchmark_ids

    illegal = sorted(acquired - expected_acquired)
    missing = sorted(expected_acquired - acquired)
    if illegal or missing:
        raise RuntimeError(
            f"{model_id} LINEAGE VIOLATION: membership does not match defined lineage. "
            f"illegal={illegal}; missing={missing}"
        )

    provenance_errors: list[str] = []
    for row in train_rows:
        id_ = str(row["id"])
        if id_ in baseline_ids:
            continue
        expected_cycle, expected_strategy = expected_provenance[id_]
        stored_cycle = int(row["acquisition_cycle"] or 0)
        stored_tokens = {
            token.strip()
            for token in str(row["acquisition_strategy"] or "").split(",")
            if token.strip()
        }
        expected_tokens = {
            token.strip()
            for token in str(expected_strategy).split(",")
            if token.strip()
        }
        if stored_cycle != expected_cycle or stored_tokens != expected_tokens:
            provenance_errors.append(
                f"{id_}: stored=({stored_cycle},{sorted(stored_tokens)}) "
                f"expected=({expected_cycle},{sorted(expected_tokens)})"
            )

    if provenance_errors:
        raise RuntimeError(
            f"{model_id} PROVENANCE VIOLATION:\n" + "\n".join(provenance_errors)
        )


def write_model_training_files(con: sqlite3.Connection, model_id: str) -> tuple[Path, Path, Path]:
    validate_model_lineage(con, model_id)
    model = con.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
    if model is None:
        raise ValueError(f"Unknown model {model_id}")

    membership = pd.read_sql_query(
        """
        SELECT id AS ID, role, acquisition_cycle, acquisition_strategy
        FROM dataset_membership
        WHERE model_id = ?
        ORDER BY CASE role WHEN 'train' THEN 0 ELSE 1 END,
                 acquisition_cycle, ID
        """,
        con,
        params=(model_id,),
    )
    if membership.empty:
        raise ValueError(f"{model_id} has no dataset membership.")

    if membership["ID"].duplicated().any():
        raise RuntimeError(f"{model_id} membership contains duplicate IDs.")

    descriptor_df, feature_columns = load_and_validate_descriptor_universe()
    source = descriptor_df.copy()
    source["ID"] = source["ID"].astype(str)

    ids = membership["ID"].astype(str).tolist()
    missing = sorted(set(ids) - set(source["ID"]))
    if missing:
        raise ValueError(f"Training IDs absent from descriptor universe: {missing}")

    cohort = source.set_index("ID").loc[ids].reset_index()

    obs = preferred_observation_df(con).set_index("id")
    missing_obs = sorted(set(ids) - set(obs.index.astype(str)))
    if missing_obs:
        raise ValueError(
            "Every train/benchmark row must have a preferred real target. Missing:\n"
            + "\n".join(missing_obs)
        )

    cohort["DAPL_bacteria"] = [float(obs.loc[id_, "DAPL_bacteria"]) for id_ in ids]
    cohort["DAPL_mammalian"] = [float(obs.loc[id_, "DAPL_mammalian"]) for id_ in ids]
    cohort["DDAPL"] = [float(obs.loc[id_, "DDAPL"]) for id_ in ids]
    cohort = cohort[PROTECTED_COLUMNS + feature_columns]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError("Feature schema changed unexpectedly.")
    if cohort[feature_columns].isna().any().any():
        raise ValueError(f"NaN descriptor values found while preparing {model_id}.")

    dataset_csv = TRAINING_DATASET_DIR / f"{model_id}_training_data.csv"
    membership_csv = DATASETS_DIR / f"dataset_{model_id}.csv"
    split_csv = SPLITS_DIR / f"split_assignments_{model_id}.csv"

    cohort.to_csv(dataset_csv, index=False)
    membership.to_csv(membership_csv, index=False)

    split_df = membership[["ID", "role"]].copy()
    split_df.insert(1, "split_version", f"split_{model_id}")
    split_df["split"] = split_df["role"].map({"train": "train", "benchmark": "test"})
    split_df = split_df[["ID", "split_version", "split"]]
    split_df.to_csv(split_csv, index=False)

    return dataset_csv, membership_csv, split_csv


# =============================================================================
# TRAINING / PREDICTION RUN SYNCHRONISATION
# =============================================================================


def candidate_representations_from_row(model: sqlite3.Row) -> list[str]:
    raw = model["candidate_representations_json"]
    if raw:
        representations = json.loads(raw)
    else:
        representations = ["position_resolved"]
    return [str(x) for x in representations]


def collect_representation_results(
    run_id: str,
    candidate_models: list[str],
    representations: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames: list[pd.DataFrame] = []
    incomplete: list[str] = []
    failed: list[str] = []
    for representation in representations:
        run_dir = training_run_dir(representation, run_id)
        manifest_path = run_dir / "run_manifest.json"
        results_path = run_dir / "results.csv"
        if not manifest_path.is_file():
            incomplete.append(representation)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = str(manifest.get("status", ""))
        if status == "failed":
            failed.append(representation)
            continue
        if status != "complete":
            incomplete.append(representation)
            continue
        if not results_path.is_file():
            raise FileNotFoundError(f"Completed run lacks results.csv: {run_dir}")
        df = pd.read_csv(results_path)
        df["model"] = df["model"].astype(str)
        df = df[df["model"].isin(candidate_models)].copy()
        missing_models = sorted(set(candidate_models) - set(df["model"]))
        if missing_models:
            raise ValueError(
                f"{representation} training results missing candidate model rows: {missing_models}"
            )
        df.insert(1, "representation", representation)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, incomplete, failed


def select_training_winner_across_representations(combined: pd.DataFrame) -> dict:
    if combined.empty:
        raise ValueError("No completed representation results are available for winner selection.")
    for column in ["r2", "rmse", "mae"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    if combined["r2"].isna().all():
        raise ValueError("All benchmark R2 values are NaN; cannot choose a model/representation.")
    winner = combined.sort_values(
        ["r2", "rmse", "mae", "representation", "model"],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    return {
        "model": str(winner["model"]),
        "representation": str(winner["representation"]),
        "r2": float(winner["r2"]),
        "rmse": float(winner["rmse"]),
        "mae": float(winner["mae"]),
    }


def write_model_comparison(model_id: str, run_id: str, combined: pd.DataFrame) -> Path:
    MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_COMPARISON_DIR / f"{model_id}_training-{run_id}_comparison.csv"
    ordered = combined.sort_values(
        ["r2", "rmse", "mae", "representation", "model"],
        ascending=[False, True, True, True, True],
    )
    ordered.to_csv(path, index=False)
    return path


def sync_run_artifacts(con: sqlite3.Connection) -> None:
    models = con.execute("SELECT * FROM models WHERE COALESCE(archived,0)=0").fetchall()
    now = now_iso()
    for original_model in models:
        model_id = str(original_model["model_id"])
        training_run_id = original_model["training_run_id"]
        if training_run_id:
            candidates = json.loads(original_model["candidate_models_json"])
            representations = candidate_representations_from_row(original_model)
            combined, incomplete, failed = collect_representation_results(
                str(training_run_id), candidates, representations
            )
            if failed:
                con.execute(
                    "UPDATE models SET status='training_failed', updated_at=? WHERE model_id=?",
                    (now, model_id),
                )
            elif not incomplete and not combined.empty:
                winner = select_training_winner_across_representations(combined)
                comparison_path = write_model_comparison(model_id, str(training_run_id), combined)
                con.execute(
                    """
                    UPDATE models
                    SET chosen_algorithm=?, chosen_representation=?,
                        benchmark_r2=?, benchmark_rmse=?, benchmark_mae=?,
                        status=CASE WHEN prediction_run_id IS NULL THEN 'trained' ELSE status END,
                        updated_at=?
                    WHERE model_id=?
                    """,
                    (
                        winner["model"], winner["representation"], winner["r2"],
                        winner["rmse"], winner["mae"], now, model_id,
                    ),
                )
                print(
                    f"Winner {model_id}: {winner['representation']} + {winner['model']} "
                    f"(R2={winner['r2']:.4f}, RMSE={winner['rmse']:.4f}, MAE={winner['mae']:.4f})"
                )
                print(f"    comparison: {comparison_path}")

        model = con.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
        prediction_run_id = model["prediction_run_id"]
        if prediction_run_id:
            representation = str(model["chosen_representation"] or "position_resolved")
            pred_dir = prediction_run_dir(representation, str(prediction_run_id))
            manifest_path = pred_dir / "run_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if str(manifest.get("status", "")) == "complete":
                    con.execute(
                        "UPDATE models SET status='predicted', updated_at=? WHERE model_id=?",
                        (now, model_id),
                    )
                    con.execute(
                        """
                        INSERT INTO prediction_runs(
                            prediction_run_id, model_id, training_run_id,
                            algorithm, representation, path, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'complete', ?, ?)
                        ON CONFLICT(prediction_run_id) DO UPDATE SET
                            representation=excluded.representation,
                            path=excluded.path,
                            status='complete', updated_at=excluded.updated_at
                        """,
                        (
                            str(prediction_run_id), model_id, str(model["training_run_id"]),
                            str(model["chosen_algorithm"]), representation,
                            str(pred_dir), now, now,
                        ),
                    )


def build_training_command(con: sqlite3.Connection, model_id: str, run_id: str) -> list[str]:
    model = active_model_row(con, model_id, purpose="training")
    if model["status"] != "prepared":
        raise ValueError(
            f"{model_id} must have status='prepared' before training; found {model['status']!r}."
        )
    dataset_csv, membership_csv, split_csv = write_model_training_files(con, model_id)
    candidates = json.loads(model["candidate_models_json"])
    return [
        "--run-id", run_id,
        "--logical-model-id", model_id,
        "--descriptor-version", DESCRIPTOR_VERSION,
        "--dataset-name", f"dataset_{model_id}",
        "--split-version", f"split_{model_id}",
        "--target", TARGET_NAME,
        "--models", ",".join(candidates),
        "--n-trials", str(N_OPTUNA_TRIALS),
        "--n-jobs", str(N_JOBS),
        "--n-cv-folds", str(N_CV_FOLDS),
        "--cv-seed", str(CV_RANDOM_SEED),
        "--optuna-seed", str(OPTUNA_RANDOM_SEED),
        "--purpose", f"{model_id} active-learning training",
        "--dataset-membership-path", str(membership_csv),
        "--descriptor-path", str(dataset_csv),
        "--reference-cleaned-descriptor-path", str(FULL_CLEANED_DESCRIPTOR_CSV),
        "--split-assignment-path", str(split_csv),
    ]


def submit_training(con: sqlite3.Connection, model_id: str) -> None:
    model = active_model_row(con, model_id, purpose="training submission")
    representations = candidate_representations_from_row(model)
    run_id = timestamp_id()
    args = build_training_command(con, model_id, run_id)
    job_ids: dict[str, str | None] = {}
    try:
        if TRAIN_EXECUTION_MODE == "slurm":
            for representation in representations:
                if representation == "position_resolved":
                    command = ["sbatch", "--parsable", str(TRAINING_SLURM_SCRIPT)] + args + [
                        "--workflow-name", "position_resolved"
                    ]
                elif representation == "global_molecular":
                    command = ["sbatch", "--parsable", str(GLOBAL_COMPARISON_SLURM_SCRIPT)] + args
                else:
                    raise ValueError(f"Unsupported representation {representation!r}")
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                job_id = result.stdout.strip().split(";")[0]
                job_ids[representation] = job_id
                print(f"Submitted {model_id} {representation} training as SLURM job {job_id}")
        elif TRAIN_EXECUTION_MODE == "direct":
            for representation in representations:
                if representation == "position_resolved":
                    command = [str(PYTHON_EXECUTABLE), "-u", str(TRAINING_SCRIPT)] + args + [
                        "--workflow-name", "position_resolved"
                    ]
                elif representation == "global_molecular":
                    command = [
                        str(PYTHON_EXECUTABLE), "-u",
                        str(scripts_dir("global_molecular") / "run_global_comparison.py"),
                    ] + args
                else:
                    raise ValueError(f"Unsupported representation {representation!r}")
                subprocess.run(command, check=True)
                job_ids[representation] = None
                print(f"Completed direct {representation} training for {model_id}")
        else:
            raise ValueError(f"Unknown TRAIN_EXECUTION_MODE={TRAIN_EXECUTION_MODE!r}")
    except Exception:
        con.execute(
            """
            UPDATE models
            SET training_run_id=?, training_job_ids_json=?, status='training_partial_submission', updated_at=?
            WHERE model_id=?
            """,
            (run_id, json.dumps(job_ids), now_iso(), model_id),
        )
        raise

    compatibility_job = (
        job_ids.get("position_resolved")
        or next((value for value in job_ids.values() if value is not None), None)
    )
    con.execute(
        """
        UPDATE models
        SET training_run_id=?, training_job_id=?, training_job_ids_json=?,
            status='training_submitted', updated_at=?
        WHERE model_id=?
        """,
        (run_id, compatibility_job, json.dumps(job_ids), now_iso(), model_id),
    )
    if TRAIN_EXECUTION_MODE == "direct":
        sync_run_artifacts(con)


def build_prediction_args(con: sqlite3.Connection, model_id: str, prediction_run_id: str) -> list[str]:
    model = active_model_row(con, model_id, purpose="prediction")
    if model["status"] not in {"trained", "predicted"}:
        raise ValueError(
            f"{model_id} must be fully trained before prediction; current status={model['status']!r}."
        )
    if not model["training_run_id"] or not model["chosen_algorithm"] or not model["chosen_representation"]:
        raise ValueError(f"{model_id} lacks training run, chosen algorithm, or chosen representation.")
    return [
        "--training-run-id", str(model["training_run_id"]),
        "--model-name", str(model["chosen_algorithm"]),
        "--workflow-name", str(model["chosen_representation"]),
        "--descriptor-version", DESCRIPTOR_VERSION,
        "--prediction-run-id", prediction_run_id,
        "--logical-model-id", model_id,
        "--purpose", f"{model_id} full 32,768-sequence prediction",
    ]


def submit_prediction(con: sqlite3.Connection, model_id: str) -> None:
    model = active_model_row(con, model_id, purpose="prediction submission")
    if model["prediction_run_id"]:
        raise ValueError(f"{model_id} already has prediction_run_id={model['prediction_run_id']}")
    if not model["chosen_representation"]:
        raise ValueError(f"{model_id} has no chosen representation; cannot predict safely.")
    representation = str(model["chosen_representation"])
    prediction_run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_training-{model['training_run_id']}_{representation}_{model['chosen_algorithm']}"
    )
    args = build_prediction_args(con, model_id, prediction_run_id)
    if PREDICTION_EXECUTION_MODE == "slurm":
        command = ["sbatch", "--parsable", str(PREDICTION_SLURM_SCRIPT)] + args
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";")[0]
        print(
            f"Submitted {model_id} {representation}+{model['chosen_algorithm']} prediction as SLURM job {job_id}"
        )
    elif PREDICTION_EXECUTION_MODE == "direct":
        command = [str(PYTHON_EXECUTABLE), "-u", str(PREDICTION_SCRIPT)] + args
        subprocess.run(command, check=True)
        job_id = None
        print(f"Completed direct prediction command for {model_id}")
    else:
        raise ValueError(f"Unknown PREDICTION_EXECUTION_MODE={PREDICTION_EXECUTION_MODE!r}")
    now = now_iso()
    con.execute(
        """
        UPDATE models
        SET prediction_run_id=?, prediction_job_id=?, status='prediction_submitted', updated_at=?
        WHERE model_id=?
        """,
        (prediction_run_id, job_id, now, model_id),
    )
    con.execute(
        """
        INSERT INTO prediction_runs(
            prediction_run_id, model_id, training_run_id, algorithm,
            representation, path, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'submitted', ?, ?)
        """,
        (
            prediction_run_id, model_id, str(model["training_run_id"]),
            str(model["chosen_algorithm"]), representation,
            str(prediction_run_dir(representation, prediction_run_id)), now, now,
        ),
    )
    if PREDICTION_EXECUTION_MODE == "direct":
        sync_run_artifacts(con)


# =============================================================================
# ACTIVE-LEARNING SELECTION
# =============================================================================


def source_model_known_ids(con: sqlite3.Connection, model_id: str) -> set[str]:
    """Return every train/benchmark ID already known to a source model."""
    rows = con.execute(
        "SELECT id FROM dataset_membership WHERE model_id=?",
        (model_id,),
    ).fetchall()
    if not rows:
        raise ValueError(
            f"Source model {model_id} has no dataset_membership rows; "
            "cannot define branch-specific acquisition eligibility safely."
        )
    return {str(row["id"]) for row in rows}


def prior_strategy_selected_ids(
    con: sqlite3.Connection,
    cycle: int,
    strategy: str,
) -> set[str]:
    """IDs already acquired by this same canonical strategy in earlier cycles."""
    rows = con.execute(
        """
        SELECT DISTINCT id
        FROM selection_decisions
        WHERE strategy=? AND decision='selected' AND cycle<?
        """,
        (strategy, cycle),
    ).fetchall()
    return {str(row["id"]) for row in rows}


def preferred_real_ids(con: sqlite3.Connection) -> set[str]:
    return {
        str(row["id"])
        for row in con.execute(
            "SELECT id FROM observations WHERE preferred=1"
        ).fetchall()
    }


def pending_physical_ids(con: sqlite3.Connection) -> set[str]:
    """Physical IDs already represented by an unresolved simulation-queue entry.

    The controller does not submit CGMD itself, so any non-analysis-ready queue
    entry is treated as an already-accounted-for physical simulation.  This is
    used only to prevent duplicate physical work; it does not make the molecule
    part of another branch's training lineage.
    """
    real = preferred_real_ids(con)
    rows = con.execute(
        """
        SELECT DISTINCT id
        FROM simulation_queue
        WHERE status <> 'analysis_ready'
          AND COALESCE(archived,0)=0
        """
    ).fetchall()
    return {str(row["id"]) for row in rows} - real


def candidate_physical_restriction_reason(id_: str) -> tuple[str | None, str | None]:
    """Restrictions that apply only when a NEW physical simulation is required."""
    parts = id_.split("-")
    if len(parts) != 3:
        return "invalid sequence", "ID is not a three-token tripeptoid sequence"

    for position, monomer in enumerate(parts, start=1):
        if monomer in MONOMER_UNAVAILABLE_REASONS:
            return MONOMER_UNAVAILABLE_REASONS[monomer], f"{monomer} at position {position}"
        if monomer in POSITION_UNAVAILABLE_MONOMERS.get(position, set()):
            return "monomer not available in that position", f"{monomer} at position {position}"

    return None, None


def candidate_restriction_reason(
    id_: str,
    benchmark_ids: set[str],
    source_model_ids: set[str],
    prior_strategy_ids: set[str],
    reusable_physical_ids: set[str],
) -> tuple[str | None, str | None]:
    """Return why a candidate is ineligible for a new logical acquisition.

    Branch isolation is defined by the source model's own frozen dataset.
    A valid molecule simulated by another branch may still be selected here and
    its physical result reused.

    CURRENT acquisition restrictions apply even when physical data are reusable.
    Historical observations remain preserved in the database and may remain in
    already-trained historical/baseline datasets, but a currently unavailable
    monomer must not enter a NEW A-G acquisition decision merely because that
    molecule was simulated previously.
    """
    if id_ in benchmark_ids:
        return "benchmark molecule", "fixed benchmark molecules cannot be selected"

    if id_ in source_model_ids:
        return (
            "already known to source model",
            "candidate is already present in the source model train/benchmark dataset",
        )

    if id_ in prior_strategy_ids:
        return (
            "already acquired by this strategy",
            "candidate was selected by the same strategy in an earlier cycle",
        )

    parts = id_.split("-")
    if len(parts) != 3:
        return "invalid sequence", "ID is not a three-token tripeptoid sequence"

    # Apply CURRENT monomer/position restrictions before considering physical reuse.
    physical_reason, physical_detail = candidate_physical_restriction_reason(id_)
    if physical_reason is not None:
        return physical_reason, physical_detail

    # Valid existing physical data may still be reused; logical overlap remains allowed.
    if id_ in reusable_physical_ids:
        return None, None

    return None, None



def selection_eligibility_context(
    con: sqlite3.Connection,
    *,
    cycle: int,
    strategy: str,
    source_model_id: str,
) -> tuple[set[str], set[str], set[str], set[str]]:
    benchmark_ids = set(get_benchmark_ids(con))
    source_ids = source_model_known_ids(con, source_model_id)
    prior_ids = prior_strategy_selected_ids(con, cycle, strategy)
    reusable_ids = preferred_real_ids(con) | pending_physical_ids(con)
    return benchmark_ids, source_ids, prior_ids, reusable_ids


def load_source_prediction_df(con: sqlite3.Connection, model_id: str) -> tuple[str, pd.DataFrame]:
    model = active_model_row(con, model_id, purpose="prediction loading/selection")
    if model["status"] != "predicted":
        raise ValueError(f"Source model {model_id} is not prediction-complete; status={model['status']!r}")
    if not model["chosen_representation"]:
        raise ValueError(f"Source model {model_id} has no chosen representation.")
    prediction_run_id = str(model["prediction_run_id"])
    representation = str(model["chosen_representation"])
    path = prediction_run_dir(representation, prediction_run_id) / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV missing for {model_id}: {path}")
    df = pd.read_csv(path)
    if "ID" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Prediction CSV must contain ID and y_pred: {path}")
    if len(df) != EXPECTED_SEQUENCE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SEQUENCE_COUNT} full-space predictions; found {len(df)}")
    if df["ID"].astype(str).duplicated().any():
        raise ValueError("Prediction CSV contains duplicate IDs.")
    df = df[["ID", "y_pred"]].copy()
    df["ID"] = df["ID"].astype(str)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="raise")
    df = df.sort_values(["y_pred", "ID"], ascending=[False, True]).reset_index(drop=True)
    df["raw_prediction_rank"] = np.arange(1, len(df) + 1)
    return prediction_run_id, df


def record_candidate_decision(
    con: sqlite3.Connection,
    *,
    cycle: int,
    strategy: str,
    source_model_id: str,
    prediction_run_id: str,
    row: pd.Series,
    selection_group: str,
    group_rank: int | None,
    target_value: float | None,
    decision_order: int,
    decision: str,
    reason: str,
    reason_detail: str | None,
) -> int:
    cursor = con.execute(
        """
        INSERT INTO selection_decisions(
            cycle, strategy, source_model_id, prediction_run_id,
            id, predicted_ddapl, raw_prediction_rank,
            selection_group, group_rank, target_value,
            decision_order, decision, reason, reason_detail, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cycle,
            strategy,
            source_model_id,
            prediction_run_id,
            str(row["ID"]),
            float(row["y_pred"]),
            int(row["raw_prediction_rank"]),
            selection_group,
            group_rank,
            target_value,
            decision_order,
            decision,
            reason,
            reason_detail,
            now_iso(),
        ),
    )
    return int(cursor.lastrowid)


def select_from_ordered_candidates(
    con: sqlite3.Connection,
    *,
    cycle: int,
    strategy: str,
    source_model_id: str,
    prediction_run_id: str,
    ordered_df: pd.DataFrame,
    selection_group: str,
    n_select: int,
    benchmark_ids: set[str],
    source_model_ids: set[str],
    prior_strategy_ids: set[str],
    reusable_physical_ids: set[str],
    selected_within_strategy: set[str],
    target_value: float | None = None,
    starting_decision_order: int = 1,
) -> tuple[list[str], int]:
    selected = []
    decision_order = starting_decision_order

    for group_rank, (_, row) in enumerate(ordered_df.iterrows(), start=1):
        id_ = str(row["ID"])

        if id_ in selected_within_strategy:
            reason = "already selected by this strategy"
            detail = "candidate was already selected in another group for the same strategy"
            decision = "skipped"
        else:
            reason, detail = candidate_restriction_reason(
                id_,
                benchmark_ids,
                source_model_ids,
                prior_strategy_ids,
                reusable_physical_ids,
            )
            if reason is None:
                decision = "selected"
                reason = f"selected_{selection_group}"
            else:
                decision = "skipped"

        record_candidate_decision(
            con,
            cycle=cycle,
            strategy=strategy,
            source_model_id=source_model_id,
            prediction_run_id=prediction_run_id,
            row=row,
            selection_group=selection_group,
            group_rank=group_rank,
            target_value=target_value,
            decision_order=decision_order,
            decision=decision,
            reason=reason,
            reason_detail=detail,
        )
        decision_order += 1

        if decision == "selected":
            selected.append(id_)
            selected_within_strategy.add(id_)
            if len(selected) == n_select:
                break

    if len(selected) != n_select:
        raise ValueError(
            f"Strategy {strategy}/{selection_group} could only find {len(selected)} "
            f"valid candidates; required {n_select}."
        )

    return selected, decision_order


def eligible_prediction_midpoint(
    predictions: pd.DataFrame,
    benchmark_ids: set[str],
    source_model_ids: set[str],
    prior_strategy_ids: set[str],
    reusable_physical_ids: set[str],
) -> tuple[float, float, float, int]:
    """Return min/max/midpoint across candidates this branch can actually select."""
    eligible_values: list[float] = []
    for row in predictions.itertuples(index=False):
        id_ = str(row.ID)
        reason, _ = candidate_restriction_reason(
            id_,
            benchmark_ids,
            source_model_ids,
            prior_strategy_ids,
            reusable_physical_ids,
        )
        if reason is None:
            eligible_values.append(float(row.y_pred))

    if not eligible_values:
        raise ValueError("No eligible candidates remain for prediction-range calculation.")

    minimum = float(min(eligible_values))
    maximum = float(max(eligible_values))
    midpoint = float((minimum + maximum) / 2.0)
    return minimum, maximum, midpoint, len(eligible_values)


def generate_strategy_selection(
    con: sqlite3.Connection,
    *,
    cycle: int,
    strategy: str,
    source_model_id: str,
    prediction_run_id: str,
    predictions: pd.DataFrame,
    benchmark_ids: set[str],
    source_model_ids: set[str],
    prior_strategy_ids: set[str],
    reusable_physical_ids: set[str],
) -> list[str]:
    selected_within_strategy: set[str] = set()
    all_selected: list[str] = []
    decision_order = 1

    def do_group(
        name: str,
        n: int,
        ordered: pd.DataFrame,
        target: float | None = None,
    ) -> None:
        nonlocal decision_order
        chosen, decision_order = select_from_ordered_candidates(
            con,
            cycle=cycle,
            strategy=strategy,
            source_model_id=source_model_id,
            prediction_run_id=prediction_run_id,
            ordered_df=ordered,
            selection_group=name,
            n_select=n,
            benchmark_ids=benchmark_ids,
            source_model_ids=source_model_ids,
            prior_strategy_ids=prior_strategy_ids,
            reusable_physical_ids=reusable_physical_ids,
            selected_within_strategy=selected_within_strategy,
            target_value=target,
            starting_decision_order=decision_order,
        )
        all_selected.extend(chosen)

    if strategy in {"A", "F", "G"}:
        do_group("top", ACQUISITION_SIZE, predictions)

    elif strategy == "B":
        minimum, maximum, _, _ = eligible_prediction_midpoint(
            predictions,
            benchmark_ids,
            source_model_ids,
            prior_strategy_ids,
            reusable_physical_ids,
        )
        edges = np.linspace(minimum, maximum, ACQUISITION_SIZE + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        for i, centre in enumerate(centres, start=1):
            ordered = predictions.assign(
                _distance=(predictions["y_pred"] - centre).abs()
            ).sort_values(["_distance", "raw_prediction_rank", "ID"])
            do_group(f"even_bin_{i:02d}", 1, ordered, float(centre))

    elif strategy == "C":
        _, _, midpoint, _ = eligible_prediction_midpoint(
            predictions,
            benchmark_ids,
            source_model_ids,
            prior_strategy_ids,
            reusable_physical_ids,
        )
        do_group("top", 5, predictions)
        midpoint_order = predictions.assign(
            _distance=(predictions["y_pred"] - midpoint).abs()
        ).sort_values(["_distance", "raw_prediction_rank", "ID"])
        do_group("midpoint", 5, midpoint_order, midpoint)
        bottom_order = predictions.sort_values(
            ["y_pred", "ID"], ascending=[True, True]
        )
        do_group("bottom", 5, bottom_order)

    elif strategy == "D":
        do_group("top", 10, predictions)
        bottom_order = predictions.sort_values(
            ["y_pred", "ID"], ascending=[True, True]
        )
        do_group("bottom", 5, bottom_order)

    elif strategy == "E":
        _, _, midpoint, _ = eligible_prediction_midpoint(
            predictions,
            benchmark_ids,
            source_model_ids,
            prior_strategy_ids,
            reusable_physical_ids,
        )
        do_group("top", 7, predictions)
        midpoint_order = predictions.assign(
            _distance=(predictions["y_pred"] - midpoint).abs()
        ).sort_values(["_distance", "raw_prediction_rank", "ID"])
        do_group("midpoint", 4, midpoint_order, midpoint)
        bottom_order = predictions.sort_values(
            ["y_pred", "ID"], ascending=[True, True]
        )
        do_group("bottom", 4, bottom_order)

    elif strategy == "X":
        # HISTORICAL / INACTIVE METHOD: original Cycle-1 C zero-target method.
        # Retained for diagnostic use if future landscapes develop a meaningful
        # negative-DDAPL region where closest-to-zero is scientifically distinct.
        do_group("top", 5, predictions)
        zero_order = predictions.assign(
            _distance=predictions["y_pred"].abs()
        ).sort_values(["_distance", "raw_prediction_rank", "ID"])
        do_group("zero", 5, zero_order, 0.0)
        bottom_order = predictions.sort_values(
            ["y_pred", "ID"], ascending=[True, True]
        )
        do_group("bottom", 5, bottom_order)

    elif strategy == "Y":
        # HISTORICAL / INACTIVE METHOD: original Cycle-1 E zero-target method.
        do_group("top", 7, predictions)
        zero_order = predictions.assign(
            _distance=predictions["y_pred"].abs()
        ).sort_values(["_distance", "raw_prediction_rank", "ID"])
        do_group("zero", 4, zero_order, 0.0)
        bottom_order = predictions.sort_values(
            ["y_pred", "ID"], ascending=[True, True]
        )
        do_group("bottom", 4, bottom_order)

    elif strategy == "H":
        raise ValueError(
            "Strategy H is deliberately reserved and currently has no selection method."
        )

    else:
        raise ValueError(f"Unknown strategy {strategy}")

    if len(all_selected) != ACQUISITION_SIZE or len(set(all_selected)) != ACQUISITION_SIZE:
        raise RuntimeError(
            f"Strategy {strategy} did not produce exactly {ACQUISITION_SIZE} unique selections."
        )
    return all_selected


def validate_strategy_selection(
    con: sqlite3.Connection,
    *,
    cycle: int,
    strategy: str,
    source_model_id: str,
    prediction_run_id: str,
    selected_ids: list[str],
    benchmark_ids: set[str],
    source_model_ids: set[str],
    prior_strategy_ids: set[str],
    reusable_physical_ids: set[str],
) -> None:
    """Fail closed before queue updates if a strategy selection is inconsistent."""
    if len(selected_ids) != ACQUISITION_SIZE or len(set(selected_ids)) != ACQUISITION_SIZE:
        raise RuntimeError(
            f"Cycle {cycle}{strategy} must contain exactly {ACQUISITION_SIZE} unique selections."
        )

    rows = con.execute(
        """
        SELECT id, source_model_id, prediction_run_id
        FROM selection_decisions
        WHERE cycle=? AND strategy=? AND decision='selected'
        """,
        (cycle, strategy),
    ).fetchall()
    stored_ids = {str(row["id"]) for row in rows}
    if len(rows) != ACQUISITION_SIZE or stored_ids != set(selected_ids):
        raise RuntimeError(
            f"Cycle {cycle}{strategy} stored selection history does not match generated IDs."
        )

    wrong_source = [
        str(row["id"])
        for row in rows
        if str(row["source_model_id"]) != source_model_id
        or str(row["prediction_run_id"]) != prediction_run_id
    ]
    if wrong_source:
        raise RuntimeError(
            f"Cycle {cycle}{strategy} contains decisions from the wrong source model/run: "
            f"{wrong_source}"
        )

    illegal_known = sorted(set(selected_ids) & source_model_ids)
    illegal_benchmark = sorted(set(selected_ids) & benchmark_ids)
    repeated_strategy = sorted(set(selected_ids) & prior_strategy_ids)
    if illegal_known or illegal_benchmark or repeated_strategy:
        raise RuntimeError(
            f"Cycle {cycle}{strategy} eligibility violation: "
            f"source_known={illegal_known}; benchmark={illegal_benchmark}; "
            f"previous_same_strategy={repeated_strategy}"
        )

    physical_errors: list[str] = []
    for id_ in selected_ids:
        if id_ in reusable_physical_ids:
            continue
        reason, detail = candidate_physical_restriction_reason(id_)
        if reason is not None:
            physical_errors.append(f"{id_}: {reason} ({detail})")
    if physical_errors:
        raise RuntimeError(
            f"Cycle {cycle}{strategy} selected physically unavailable NEW candidates:\n"
            + "\n".join(physical_errors)
        )


def selection_source_model_id(cycle: int, strategy: str) -> str:
    """Return the prediction-complete model that makes an acquisition decision."""
    if strategy == "F":
        return f"Model{cycle}F"
    if strategy == "G":
        if cycle < 2:
            raise ValueError(
                "There is no Cycle-1 G acquisition: Model1F is the shared pooled seed."
            )
        return f"Model{cycle}G"
    if strategy in ACTIVE_STRATEGIES:
        if cycle == 1:
            return "Model0"
        return f"Model{cycle - 1}{strategy}"
    raise ValueError(f"Strategy {strategy!r} is not an active acquisition strategy.")


def _queue_request_for_selected_id(
    con: sqlite3.Connection,
    *,
    cycle: int,
    id_: str,
    strategy: str,
) -> tuple[str, bool]:
    """Merge queue provenance and return (status, new_physical_requirement).

    A single cycle/id queue row represents one physical simulation regardless of
    how many strategies select it.  If the molecule is already real, the new
    logical acquisition is immediately analysis_ready.  If an unresolved queue
    entry exists in another cycle, the current cycle waits on that existing
    physical simulation rather than requesting a duplicate.
    """
    now = now_iso()
    is_real = con.execute(
        "SELECT 1 FROM observations WHERE id=? AND preferred=1",
        (id_,),
    ).fetchone() is not None

    current = con.execute(
        "SELECT requested_by,status,archived FROM simulation_queue WHERE cycle=? AND id=?",
        (cycle, id_),
    ).fetchone()

    if current is not None:
        requesting = {
            token.strip()
            for token in str(current["requested_by"]).split(",")
            if token.strip() in ALL_CANONICAL_STRATEGIES
        }
        requesting.add(strategy)
        status = "analysis_ready" if is_real else str(current["status"])
        if status == "archived_historical":
            status = "selected"
        con.execute(
            """
            UPDATE simulation_queue
            SET requested_by=?, status=?, archived=0, archive_reason=NULL, updated_at=?
            WHERE cycle=? AND id=?
            """,
            (",".join(sorted(requesting)), status, now, cycle, id_),
        )
        return status, False

    if is_real:
        status = "analysis_ready"
        new_physical = False
    else:
        pending_elsewhere = con.execute(
            """
            SELECT 1 FROM simulation_queue
            WHERE id=? AND cycle<>? AND status<>'analysis_ready'
              AND COALESCE(archived,0)=0
            LIMIT 1
            """,
            (id_, cycle),
        ).fetchone() is not None
        if pending_elsewhere:
            status = "waiting_existing_simulation"
            new_physical = False
        else:
            status = "selected"
            new_physical = True

    con.execute(
        """
        INSERT INTO simulation_queue(cycle,id,requested_by,status,created_at,updated_at)
        VALUES (?,?,?,?,?,?)
        """,
        (cycle, id_, strategy, status, now, now),
    )
    return status, new_physical


def generate_cycle_selections(
    con: sqlite3.Connection,
    cycle: int,
    strategies: list[str] | None = None,
) -> None:
    """Generate one or more canonical A-E acquisition strategies for a cycle.

    ``select --cycle N`` retains the historical all-A-E behaviour.
    ``select --cycle N --strategies A B D`` allows independent branches to move
    ahead while other source models are still waiting.  Previously recorded
    strategies are immutable and are never regenerated or overwritten.
    """
    if cycle < 1:
        raise ValueError("Active-learning acquisition cycles start at 1.")

    requested = list(ACTIVE_STRATEGIES if strategies is None else strategies)
    if not requested:
        raise ValueError("At least one A-E strategy must be requested.")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate strategies requested: {requested}")
    unknown = sorted(set(requested) - set(ACTIVE_STRATEGIES))
    if unknown:
        raise ValueError(f"Only canonical A-E strategies may be selected here: {unknown}")

    # Preserve canonical order regardless of command-line order.
    requested = [strategy for strategy in ACTIVE_STRATEGIES if strategy in requested]

    # Preflight every requested strategy before recording any new decisions.
    prepared: dict[str, dict] = {}
    for strategy in requested:
        existing = int(con.execute(
            "SELECT COUNT(*) FROM selection_decisions WHERE cycle=? AND strategy=?",
            (cycle, strategy),
        ).fetchone()[0])
        if existing > 0:
            raise ValueError(
                f"Cycle {cycle} already contains {strategy} selection decisions. "
                "Refusing to overwrite scientific selection history."
            )

        source_model = selection_source_model_id(cycle, strategy)
        prediction_run_id, predictions = load_source_prediction_df(con, source_model)
        benchmark_ids, source_ids, prior_ids, reusable_ids = selection_eligibility_context(
            con,
            cycle=cycle,
            strategy=strategy,
            source_model_id=source_model,
        )
        prepared[strategy] = {
            "source_model": source_model,
            "prediction_run_id": prediction_run_id,
            "predictions": predictions,
            "benchmark_ids": benchmark_ids,
            "source_ids": source_ids,
            "prior_ids": prior_ids,
            "reusable_ids": reusable_ids,
        }

    selected_by_strategy: dict[str, list[str]] = {}
    for strategy in requested:
        info = prepared[strategy]
        selected = generate_strategy_selection(
            con,
            cycle=cycle,
            strategy=strategy,
            source_model_id=info["source_model"],
            prediction_run_id=info["prediction_run_id"],
            predictions=info["predictions"],
            benchmark_ids=info["benchmark_ids"],
            source_model_ids=info["source_ids"],
            prior_strategy_ids=info["prior_ids"],
            reusable_physical_ids=info["reusable_ids"],
        )
        validate_strategy_selection(
            con,
            cycle=cycle,
            strategy=strategy,
            source_model_id=info["source_model"],
            prediction_run_id=info["prediction_run_id"],
            selected_ids=selected,
            benchmark_ids=info["benchmark_ids"],
            source_model_ids=info["source_ids"],
            prior_strategy_ids=info["prior_ids"],
            reusable_physical_ids=info["reusable_ids"],
        )
        selected_by_strategy[strategy] = selected

    unique_ids = sorted({id_ for ids in selected_by_strategy.values() for id_ in ids})
    new_physical_ids: set[str] = set()
    status_by_id: dict[str, str] = {}
    for id_ in unique_ids:
        requesting = [
            strategy for strategy in requested if id_ in selected_by_strategy[strategy]
        ]
        status = ""
        for strategy in requesting:
            status, newly_required = _queue_request_for_selected_id(
                con,
                cycle=cycle,
                id_=id_,
                strategy=strategy,
            )
            if newly_required:
                new_physical_ids.add(id_)
        status_by_id[id_] = status

    refresh_prediction_outcomes(con)
    export_cycle_selection_files(con, cycle)

    print(f"Cycle {cycle} canonical selection update complete.")
    for strategy in requested:
        print(f"    {strategy}: {len(selected_by_strategy[strategy])} logical selections")
    print(f"    unique IDs in this selection update: {len(unique_ids)}")
    print(f"    NEW physical simulations required: {len(new_physical_ids)}")
    print(
        "    already real/reused: "
        f"{sum(status == 'analysis_ready' for status in status_by_id.values())}"
    )
    print(
        "    waiting on an existing queued simulation: "
        f"{sum(status == 'waiting_existing_simulation' for status in status_by_id.values())}"
    )
    print(f"    queue: {SIMULATION_QUEUE_DIR / f'cycle_{cycle}_simulation_queue.csv'}")

    remaining = [
        strategy
        for strategy in ACTIVE_STRATEGIES
        if int(con.execute(
            "SELECT COUNT(*) FROM selection_decisions WHERE cycle=? AND strategy=?",
            (cycle, strategy),
        ).fetchone()[0]) == 0
    ]
    if remaining:
        print(f"    canonical A-E strategies still awaiting selection: {','.join(remaining)}")
    else:
        print("    canonical A-E selection phase is complete for this cycle.")
        print(
            "    F is selected later from the trained/predicted self-feedback pooled "
            f"Model{cycle}F via: select-f --cycle {cycle}"
        )
        if cycle >= 2:
            print(
                "    G is selected separately from the trained/predicted no-self-feedback "
                f"Model{cycle}G via: select-g --cycle {cycle}"
            )


def _generate_pooled_selection(
    con: sqlite3.Connection,
    cycle: int,
    strategy: str,
) -> None:
    """Select 15 top candidates from a prediction-complete pooled F or G model."""
    if strategy not in {"F", "G"}:
        raise ValueError("Pooled selection is defined only for F or G.")
    if cycle < 1:
        raise ValueError("Active-learning acquisition cycles start at 1.")
    if strategy == "G" and cycle < 2:
        raise ValueError(
            "Cycle-1 G is intentionally absent because Model1F is the shared pooled seed."
        )

    existing = con.execute(
        "SELECT COUNT(*) FROM selection_decisions WHERE cycle=? AND strategy=?",
        (cycle, strategy),
    ).fetchone()[0]
    if int(existing) > 0:
        raise ValueError(
            f"Cycle {cycle} already contains canonical {strategy} selection decisions. "
            "Refusing to overwrite scientific selection history."
        )

    source_model = selection_source_model_id(cycle, strategy)
    prediction_run_id, predictions = load_source_prediction_df(con, source_model)
    benchmark_ids, source_ids, prior_ids, reusable_ids = selection_eligibility_context(
        con,
        cycle=cycle,
        strategy=strategy,
        source_model_id=source_model,
    )

    selected = generate_strategy_selection(
        con,
        cycle=cycle,
        strategy=strategy,
        source_model_id=source_model,
        prediction_run_id=prediction_run_id,
        predictions=predictions,
        benchmark_ids=benchmark_ids,
        source_model_ids=source_ids,
        prior_strategy_ids=prior_ids,
        reusable_physical_ids=reusable_ids,
    )
    validate_strategy_selection(
        con,
        cycle=cycle,
        strategy=strategy,
        source_model_id=source_model,
        prediction_run_id=prediction_run_id,
        selected_ids=selected,
        benchmark_ids=benchmark_ids,
        source_model_ids=source_ids,
        prior_strategy_ids=prior_ids,
        reusable_physical_ids=reusable_ids,
    )

    new_physical_ids: set[str] = set()
    statuses: list[str] = []
    for id_ in selected:
        status, newly_required = _queue_request_for_selected_id(
            con,
            cycle=cycle,
            id_=id_,
            strategy=strategy,
        )
        statuses.append(status)
        if newly_required:
            new_physical_ids.add(id_)

    refresh_prediction_outcomes(con)
    export_cycle_selection_files(con, cycle)
    print(f"Cycle {cycle} {strategy} selection complete.")
    print(f"    source model: {source_model}")
    print(f"    logical selections: {len(selected)}")
    print(f"    NEW physical simulations required: {len(new_physical_ids)}")
    print(f"    already real/reused: {sum(status == 'analysis_ready' for status in statuses)}")
    print(
        "    waiting on an existing queued simulation: "
        f"{sum(status == 'waiting_existing_simulation' for status in statuses)}"
    )
    print(f"    queue: {SIMULATION_QUEUE_DIR / f'cycle_{cycle}_simulation_queue.csv'}")


def generate_f_selection(con: sqlite3.Connection, cycle: int) -> None:
    """Select F discovery candidates; these feed later F models, not the current one."""
    _generate_pooled_selection(con, cycle, "F")


def generate_g_selection(con: sqlite3.Connection, cycle: int) -> None:
    """Select G control candidates; G selections never feed later G training."""
    _generate_pooled_selection(con, cycle, "G")


def export_cycle_selection_files(con: sqlite3.Connection, cycle: int) -> None:
    # Normal selection export contains canonical active strategies only. Historical
    # X/Y/F_legacy records are written separately so they cannot be mistaken for
    # current active-learning decisions.
    selection_df = pd.read_sql_query(
        """
        SELECT d.*, p.smiles, p.m1, p.m2, p.m3,
               o.actual_ddapl, o.residual, o.absolute_error
        FROM selection_decisions d
        LEFT JOIN peptoids p ON p.id = d.id
        LEFT JOIN prediction_outcomes o ON o.decision_id = d.decision_id
        WHERE d.cycle = ?
          AND d.strategy IN ('A','B','C','D','E','F','G')
        ORDER BY d.strategy, d.decision_order
        """,
        con,
        params=(cycle,),
    )
    selection_df.to_csv(
        SELECTION_EXPORT_DIR / f"cycle_{cycle}_selection_decisions.csv",
        index=False,
    )

    historical_df = pd.read_sql_query(
        """
        SELECT d.*, p.smiles, p.m1, p.m2, p.m3,
               o.actual_ddapl, o.residual, o.absolute_error
        FROM selection_decisions d
        LEFT JOIN peptoids p ON p.id = d.id
        LEFT JOIN prediction_outcomes o ON o.decision_id = d.decision_id
        WHERE d.cycle = ?
          AND d.strategy IN ('X','Y','F_legacy')
        ORDER BY d.strategy, d.decision_order
        """,
        con,
        params=(cycle,),
    )
    historical_path = SELECTION_EXPORT_DIR / f"cycle_{cycle}_historical_selection_decisions.csv"
    if not historical_df.empty:
        historical_df.to_csv(historical_path, index=False)
    elif historical_path.exists():
        historical_path.unlink()

    queue_df = pd.read_sql_query(
        """
        SELECT q.cycle, q.id AS ID, p.smiles, p.m1, p.m2, p.m3,
               q.requested_by, q.status, q.created_at, q.updated_at
        FROM simulation_queue q
        LEFT JOIN peptoids p ON p.id = q.id
        WHERE q.cycle = ?
          AND COALESCE(q.archived,0)=0
        ORDER BY q.id
        """,
        con,
        params=(cycle,),
    )
    queue_df.to_csv(
        SIMULATION_QUEUE_DIR / f"cycle_{cycle}_simulation_queue.csv",
        index=False,
    )


def compare_model_generation(
    con: sqlite3.Connection,
    generation: int,
) -> Path:
    """
    Compare all prediction-complete canonical models within one model generation.

    Examples
    --------
    generation=1:
        Compare available Model1A, Model1B, Model1C, Model1D, Model1E, Model1F.

    generation>=2:
        Compare available A-G models, including the F/G pooled fork.

    This is a READ-ONLY scientific analysis of:
        - full 32,768-sequence prediction landscapes;
        - Pearson prediction correlation;
        - Spearman/rank correlation;
        - top/bottom prediction-region overlap;
        - actual acquisition-set overlap;
        - how every selected molecule ranks/predicts under every other model;
        - selection-group behaviour (top, bottom, midpoint, even-bin, etc.).

    It does NOT modify selections, models, observations, datasets, or queues.
    Existing analysis files for the same generation are regenerated because they
    are derived reports from immutable prediction/selection records.
    """

    if generation < 1:
        raise ValueError("Model-generation comparisons start at generation 1.")

    if generation == 1:
        expected_strategies = ["A", "B", "C", "D", "E", "F"]
    else:
        expected_strategies = ["A", "B", "C", "D", "E", "F", "G"]

    placeholders = ",".join("?" for _ in expected_strategies)

    model_rows = con.execute(
        f"""
        SELECT
            model_id,
            cycle,
            strategy,
            status,
            prediction_run_id,
            chosen_algorithm,
            chosen_representation,
            benchmark_r2,
            benchmark_rmse,
            benchmark_mae,
            n_train,
            n_benchmark
        FROM models
        WHERE cycle = ?
          AND strategy IN ({placeholders})
        ORDER BY
            CASE strategy
                WHEN 'A' THEN 1
                WHEN 'B' THEN 2
                WHEN 'C' THEN 3
                WHEN 'D' THEN 4
                WHEN 'E' THEN 5
                WHEN 'F' THEN 6
                WHEN 'G' THEN 7
                ELSE 99
            END
        """,
        (generation, *expected_strategies),
    ).fetchall()

    if not model_rows:
        raise ValueError(
            f"No canonical Model{generation} branch models are recorded."
        )

    predicted_rows = [
        row
        for row in model_rows
        if str(row["status"]) == "predicted"
        and row["prediction_run_id"] is not None
    ]

    if len(predicted_rows) < 2:
        available = [
            f"{row['model_id']}={row['status']}"
            for row in model_rows
        ]
        raise ValueError(
            f"Generation {generation} requires at least two prediction-complete "
            "canonical models for comparison.\n"
            + "\n".join(available)
        )

    model_ids = [str(row["model_id"]) for row in predicted_rows]

    output_dir = (
        ACTIVE_LEARNING_DIR
        / "model_agreement"
        / f"generation_{generation}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)


    # ---------------------------------------------------------------------
    # LOAD AND VALIDATE ALL FULL-SPACE PREDICTIONS
    # ---------------------------------------------------------------------

    prediction_series: dict[str, pd.Series] = {}
    rank_series: dict[str, pd.Series] = {}
    prediction_run_ids: dict[str, str] = {}
    prediction_paths: dict[str, str] = {}

    reference_ids: set[str] | None = None

    for model_id in model_ids:

        prediction_run_id, prediction_df = load_source_prediction_df(
            con,
            model_id,
        )

        values = prediction_df["y_pred"].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{model_id} prediction file contains NaN or infinite values."
            )

        ids = set(prediction_df["ID"].astype(str))

        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            missing = sorted(reference_ids - ids)
            extra = sorted(ids - reference_ids)

            raise RuntimeError(
                "Prediction-universe mismatch between canonical models.\n"
                f"Model: {model_id}\n"
                f"Missing IDs: {missing[:20]}\n"
                f"Extra IDs:   {extra[:20]}"
            )

        indexed = prediction_df.set_index("ID")

        prediction_series[model_id] = indexed["y_pred"].astype(float)
        rank_series[model_id] = indexed["raw_prediction_rank"].astype(int)

        prediction_run_ids[model_id] = prediction_run_id

        model_row = con.execute(
            "SELECT chosen_representation FROM models WHERE model_id=?",
            (model_id,),
        ).fetchone()

        representation = str(model_row["chosen_representation"])

        prediction_paths[model_id] = str(
            prediction_run_dir(
                representation,
                prediction_run_id,
            )
            / "predictions.csv"
        )

    if reference_ids is None:
        raise RuntimeError("No prediction IDs were loaded.")

    if len(reference_ids) != EXPECTED_SEQUENCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SEQUENCE_COUNT:,} common prediction IDs; "
            f"found {len(reference_ids):,}."
        )

    ordered_ids = sorted(reference_ids)

    prediction_matrix = pd.DataFrame(
        {
            model_id: prediction_series[model_id].reindex(ordered_ids)
            for model_id in model_ids
        },
        index=ordered_ids,
    )

    rank_matrix = pd.DataFrame(
        {
            model_id: rank_series[model_id].reindex(ordered_ids)
            for model_id in model_ids
        },
        index=ordered_ids,
    )

    if prediction_matrix.isna().any().any():
        raise RuntimeError(
            "Prediction matrix contains missing values after model alignment."
        )

    if rank_matrix.isna().any().any():
        raise RuntimeError(
            "Rank matrix contains missing values after model alignment."
        )

    # ---------------------------------------------------------------------
    # PREDICTION DISTRIBUTION SUMMARY
    # ---------------------------------------------------------------------

    prediction_summary_rows: list[dict] = []

    row_by_model = {
        str(row["model_id"]): row
        for row in predicted_rows
    }

    for model_id in model_ids:

        values = prediction_matrix[model_id]

        model = row_by_model[model_id]

        prediction_summary_rows.append(
            {
                "model_id": model_id,
                "strategy": model["strategy"],
                "chosen_algorithm": model["chosen_algorithm"],
                "chosen_representation": model["chosen_representation"],
                "benchmark_r2": model["benchmark_r2"],
                "benchmark_rmse": model["benchmark_rmse"],
                "benchmark_mae": model["benchmark_mae"],
                "n_train": model["n_train"],
                "n_benchmark": model["n_benchmark"],
                "prediction_run_id": prediction_run_ids[model_id],
                "n_predictions": len(values),
                "prediction_min": float(values.min()),
                "prediction_q05": float(values.quantile(0.05)),
                "prediction_q25": float(values.quantile(0.25)),
                "prediction_median": float(values.median()),
                "prediction_mean": float(values.mean()),
                "prediction_q75": float(values.quantile(0.75)),
                "prediction_q95": float(values.quantile(0.95)),
                "prediction_max": float(values.max()),
                "prediction_std": float(values.std()),
            }
        )

    prediction_summary_df = pd.DataFrame(
        prediction_summary_rows
    )

    # ---------------------------------------------------------------------
    # LOAD ACTUAL SELECTIONS MADE FROM THESE MODELS
    #
    # Query by source_model_id rather than assuming acquisition cycle:
    #
    #   Model1A -> Cycle-2 A
    #   Model1F -> Cycle-1 F
    #
    # This therefore also remains correct for the F/G architecture.
    # ---------------------------------------------------------------------

    model_placeholders = ",".join("?" for _ in model_ids)

    selection_df = pd.read_sql_query(
        f"""
        SELECT
            decision_id,
            cycle AS selection_cycle,
            strategy AS selected_by_strategy,
            source_model_id,
            prediction_run_id,
            id AS ID,
            predicted_ddapl AS stored_source_prediction,
            raw_prediction_rank AS stored_source_rank,
            selection_group,
            group_rank,
            target_value,
            decision_order
        FROM selection_decisions
        WHERE decision='selected'
          AND source_model_id IN ({model_placeholders})
        ORDER BY
            source_model_id,
            decision_order
        """,
        con,
        params=tuple(model_ids),
    )

    # Verify that stored selection values are exactly traceable back to the
    # prediction file from which they were selected.
    if not selection_df.empty:

        for row in selection_df.itertuples(index=False):

            source_model = str(row.source_model_id)
            id_ = str(row.ID)

            if id_ not in prediction_matrix.index:
                raise RuntimeError(
                    f"Selected ID {id_} from {source_model} is absent from "
                    "the aligned prediction universe."
                )

            expected_prediction = float(
                prediction_matrix.loc[id_, source_model]
            )

            expected_rank = int(
                rank_matrix.loc[id_, source_model]
            )

            if not np.isclose(
                float(row.stored_source_prediction),
                expected_prediction,
                rtol=0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "Stored acquisition prediction differs from the source "
                    "prediction file.\n"
                    f"Model={source_model}\n"
                    f"ID={id_}\n"
                    f"stored={row.stored_source_prediction}\n"
                    f"source={expected_prediction}"
                )

            if int(row.stored_source_rank) != expected_rank:
                raise RuntimeError(
                    "Stored acquisition rank differs from the source "
                    "prediction file.\n"
                    f"Model={source_model}\n"
                    f"ID={id_}\n"
                    f"stored={row.stored_source_rank}\n"
                    f"source={expected_rank}"
                )

        # Every completed canonical strategy acquisition should have 15 IDs.
        for source_model, group in selection_df.groupby(
            "source_model_id"
        ):
            n_unique = int(group["ID"].nunique())

            if n_unique != ACQUISITION_SIZE:
                raise RuntimeError(
                    f"{source_model} has {n_unique} selected unique IDs in its "
                    f"recorded acquisition; expected {ACQUISITION_SIZE}."
                )

    # ---------------------------------------------------------------------
    # PAIRWISE GLOBAL CORRELATIONS + ACTUAL SELECTION OVERLAP
    # ---------------------------------------------------------------------

    pairwise_rows: list[dict] = []

    for i, model_1 in enumerate(model_ids):
        for model_2 in model_ids[i + 1:]:

            pearson = float(
                prediction_matrix[model_1].corr(
                    prediction_matrix[model_2],
                    method="pearson",
                )
            )

            spearman = float(
                prediction_matrix[model_1].corr(
                    prediction_matrix[model_2],
                    method="spearman",
                )
            )

            selected_1 = set(
                selection_df.loc[
                    selection_df["source_model_id"].eq(model_1),
                    "ID",
                ].astype(str)
            )

            selected_2 = set(
                selection_df.loc[
                    selection_df["source_model_id"].eq(model_2),
                    "ID",
                ].astype(str)
            )

            if selected_1 and selected_2:

                overlap = selected_1 & selected_2
                union = selected_1 | selected_2

                selected_overlap_n: int | float = len(overlap)
                selected_jaccard: float = (
                    len(overlap) / len(union)
                    if union
                    else np.nan
                )

                overlap_ids = ";".join(
                    sorted(overlap)
                )

            else:
                selected_overlap_n = np.nan
                selected_jaccard = np.nan
                overlap_ids = ""

            pairwise_rows.append(
                {
                    "model_1": model_1,
                    "model_2": model_2,
                    "pearson_prediction_r": pearson,
                    "spearman_rank_rho": spearman,
                    "model_1_selection_n": len(selected_1),
                    "model_2_selection_n": len(selected_2),
                    "actual_selection_overlap_n": selected_overlap_n,
                    "actual_selection_jaccard": selected_jaccard,
                    "actual_selection_overlap_IDs": overlap_ids,
                }
            )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    # ---------------------------------------------------------------------
    # RAW FULL-SPACE TOP/BOTTOM OVERLAP
    #
    # This compares prediction landscapes themselves, not branch eligibility.
    # Actual acquisition overlap is already recorded separately above.
    # ---------------------------------------------------------------------

    overlap_rows: list[dict] = []

    k_values = [5, 10, 15, 25, 50, 100]

    for i, model_1 in enumerate(model_ids):
        for model_2 in model_ids[i + 1:]:

            for region in ("top", "bottom"):

                for k in k_values:

                    if region == "top":

                        ids_1 = set(
                            rank_matrix[model_1]
                            .nsmallest(k)
                            .index.astype(str)
                        )

                        ids_2 = set(
                            rank_matrix[model_2]
                            .nsmallest(k)
                            .index.astype(str)
                        )

                    else:

                        ids_1 = set(
                            rank_matrix[model_1]
                            .nlargest(k)
                            .index.astype(str)
                        )

                        ids_2 = set(
                            rank_matrix[model_2]
                            .nlargest(k)
                            .index.astype(str)
                        )

                    overlap = ids_1 & ids_2
                    union = ids_1 | ids_2

                    overlap_rows.append(
                        {
                            "model_1": model_1,
                            "model_2": model_2,
                            "region": region,
                            "k": k,
                            "overlap_n": len(overlap),
                            "overlap_fraction_of_k": len(overlap) / k,
                            "jaccard": (
                                len(overlap) / len(union)
                                if union
                                else np.nan
                            ),
                            "overlap_IDs": ";".join(
                                sorted(overlap)
                            ),
                        }
                    )

    rank_overlap_df = pd.DataFrame(
        overlap_rows
    )

    # ---------------------------------------------------------------------
    # CROSS-MODEL RANKING OF EVERY ACTUAL SELECTED MOLECULE
    #
    # THIS is the main file for questions such as:
    #
    #   "Where do D's bottom five rank according to A?"
    #   "What does A predict for D's highest ten?"
    #   "Does B view A's selections as extreme or ordinary?"
    # ---------------------------------------------------------------------

    cross_rows: list[dict] = []

    if not selection_df.empty:

        for selection in selection_df.itertuples(
            index=False
        ):

            id_ = str(selection.ID)

            output_row = {
                "selection_cycle": int(
                    selection.selection_cycle
                ),
                "selected_by_strategy": str(
                    selection.selected_by_strategy
                ),
                "source_model_id": str(
                    selection.source_model_id
                ),
                "ID": id_,
                "selection_group": str(
                    selection.selection_group
                ),
                "group_rank": selection.group_rank,
                "target_value": selection.target_value,
                "decision_order": int(
                    selection.decision_order
                ),
                "source_prediction_run_id": str(
                    selection.prediction_run_id
                ),
                "source_predicted_ddapl": float(
                    selection.stored_source_prediction
                ),
                "source_raw_prediction_rank": int(
                    selection.stored_source_rank
                ),
            }

            for model_id in model_ids:

                predicted = float(
                    prediction_matrix.loc[
                        id_,
                        model_id,
                    ]
                )

                rank = int(
                    rank_matrix.loc[
                        id_,
                        model_id,
                    ]
                )

                # 100 = very highest prediction;
                # ~0 = very lowest prediction.
                high_percentile = (
                    100.0
                    * (
                        EXPECTED_SEQUENCE_COUNT
                        - rank
                    )
                    / (
                        EXPECTED_SEQUENCE_COUNT
                        - 1
                    )
                )

                output_row[
                    f"{model_id}_predicted_DDAPL"
                ] = predicted

                output_row[
                    f"{model_id}_rank_high_to_low"
                ] = rank

                output_row[
                    f"{model_id}_high_percentile"
                ] = high_percentile

            cross_rows.append(
                output_row
            )

    selection_cross_df = pd.DataFrame(
        cross_rows
    )

    # ---------------------------------------------------------------------
    # SUMMARISE EACH ACQUISITION GROUP UNDER EVERY MODEL
    #
    # Example:
    #
    # selected_by=D, group=bottom, evaluated_by=Model1A
    #
    # gives the mean/min/max Model1A prediction and rank for D's bottom five.
    # ---------------------------------------------------------------------

    selection_group_rows: list[dict] = []

    if not selection_cross_df.empty:

        grouped = selection_cross_df.groupby(
            [
                "source_model_id",
                "selected_by_strategy",
                "selection_group",
            ],
            dropna=False,
        )

        for (
            source_model,
            strategy,
            selection_group,
        ), group in grouped:

            for evaluated_model in model_ids:

                prediction_column = (
                    f"{evaluated_model}_predicted_DDAPL"
                )

                rank_column = (
                    f"{evaluated_model}_rank_high_to_low"
                )

                percentile_column = (
                    f"{evaluated_model}_high_percentile"
                )

                selection_group_rows.append(
                    {
                        "source_model_id": source_model,
                        "selected_by_strategy": strategy,
                        "selection_group": selection_group,
                        "selection_n": len(group),
                        "evaluated_by_model": evaluated_model,
                        "predicted_DDAPL_mean": float(
                            group[prediction_column].mean()
                        ),
                        "predicted_DDAPL_min": float(
                            group[prediction_column].min()
                        ),
                        "predicted_DDAPL_max": float(
                            group[prediction_column].max()
                        ),
                        "rank_mean": float(
                            group[rank_column].mean()
                        ),
                        "rank_median": float(
                            group[rank_column].median()
                        ),
                        "rank_best": int(
                            group[rank_column].min()
                        ),
                        "rank_worst": int(
                            group[rank_column].max()
                        ),
                        "high_percentile_mean": float(
                            group[percentile_column].mean()
                        ),
                    }
                )

    selection_group_df = pd.DataFrame(
        selection_group_rows
    )

    # ---------------------------------------------------------------------
    # WRITE OUTPUTS
    # ---------------------------------------------------------------------

    prediction_summary_path = (
        output_dir
        / "prediction_summary.csv"
    )

    pairwise_path = (
        output_dir
        / "pairwise_model_agreement.csv"
    )

    overlap_path = (
        output_dir
        / "full_space_rank_overlap.csv"
    )

    cross_path = (
        output_dir
        / "selection_cross_ranking.csv"
    )

    group_path = (
        output_dir
        / "selection_group_cross_summary.csv"
    )

    manifest_path = (
        output_dir
        / "comparison_manifest.json"
    )

    summary_path = (
        output_dir
        / "model_agreement_summary.txt"
    )

    prediction_summary_df.to_csv(
        prediction_summary_path,
        index=False,
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )

    rank_overlap_df.to_csv(
        overlap_path,
        index=False,
    )

    selection_cross_df.to_csv(
        cross_path,
        index=False,
    )

    selection_group_df.to_csv(
        group_path,
        index=False,
    )

    manifest = {
        "generated_at": now_iso(),
        "generation": generation,
        "expected_strategies": expected_strategies,
        "prediction_complete_models": model_ids,
        "models_not_yet_prediction_complete": [
            {
                "model_id": str(row["model_id"]),
                "strategy": str(row["strategy"]),
                "status": str(row["status"]),
            }
            for row in model_rows
            if str(row["model_id"]) not in model_ids
        ],
        "prediction_run_ids": prediction_run_ids,
        "prediction_paths": prediction_paths,
        "prediction_universe_n": len(reference_ids),
        "rank_overlap_k_values": k_values,
        "notes": [
            "Pearson compares predicted DDAPL values across all 32,768 sequences.",
            "Spearman compares complete sequence rankings across all 32,768 sequences.",
            "Full-space top/bottom overlap is raw prediction-landscape agreement and is not branch-eligibility filtered.",
            "Actual selection overlap compares the acquisition decisions genuinely made by each prediction model.",
            "selection_cross_ranking.csv evaluates every selected molecule under every available model.",
            "selection_group_cross_summary.csv preserves top/bottom/midpoint/even-bin acquisition-group meaning.",
            "This command is analysis-only and never alters active-learning selections.",
        ],
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # HUMAN-READABLE SUMMARY
    # ---------------------------------------------------------------------

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(
        f"PFPEPTOIDS MODEL-AGREEMENT ANALYSIS — GENERATION {generation}"
    )
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        "Prediction-complete models: "
        + ", ".join(model_ids)
    )

    missing_models = [
        f"{row['model_id']} ({row['status']})"
        for row in model_rows
        if str(row["model_id"]) not in model_ids
    ]

    if missing_models:
        lines.append(
            "Not yet included: "
            + ", ".join(missing_models)
        )

    lines.append("")
    lines.append(
        f"Common prediction universe: {len(reference_ids):,} IDs"
    )

    lines.append("")
    lines.append("PAIRWISE AGREEMENT")
    lines.append("-" * 80)

    if pairwise_df.empty:

        lines.append(
            "No pairwise model comparison available."
        )

    else:

        for row in pairwise_df.itertuples(
            index=False
        ):

            selection_overlap = (
                "-"
                if pd.isna(
                    row.actual_selection_overlap_n
                )
                else str(
                    int(
                        row.actual_selection_overlap_n
                    )
                )
            )

            lines.append(
                f"{row.model_1} vs {row.model_2}: "
                f"Pearson={row.pearson_prediction_r:.4f}, "
                f"Spearman={row.spearman_rank_rho:.4f}, "
                f"actual-selection overlap={selection_overlap}"
            )

    if not selection_group_df.empty:

        lines.append("")
        lines.append(
            "SELECTION-GROUP CROSS-RANKING"
        )
        lines.append("-" * 80)

        for row in selection_group_df.itertuples(
            index=False
        ):

            lines.append(
                f"{row.source_model_id} "
                f"{row.selection_group} "
                f"(n={row.selection_n}) "
                f"evaluated by {row.evaluated_by_model}: "
                f"mean_pred={row.predicted_DDAPL_mean:.4f}, "
                f"mean_rank={row.rank_mean:.1f}, "
                f"rank_range={row.rank_best}-{row.rank_worst}, "
                f"mean_high_percentile={row.high_percentile_mean:.2f}%"
            )

    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("-" * 80)

    for path in [
        prediction_summary_path,
        pairwise_path,
        overlap_path,
        cross_path,
        group_path,
        manifest_path,
    ]:
        lines.append(str(path))

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # TERMINAL SUMMARY
    # ---------------------------------------------------------------------

    print("=" * 80)
    print(
        f"MODEL-AGREEMENT ANALYSIS COMPLETE — GENERATION {generation}"
    )
    print("=" * 80)

    print(
        "Compared: "
        + ", ".join(model_ids)
    )

    if missing_models:
        print(
            "Not yet prediction-complete: "
            + ", ".join(missing_models)
        )

    print()

    for row in pairwise_df.itertuples(
        index=False
    ):

        overlap = (
            "-"
            if pd.isna(
                row.actual_selection_overlap_n
            )
            else str(
                int(
                    row.actual_selection_overlap_n
                )
            )
        )

        print(
            f"{row.model_1} vs {row.model_2}: "
            f"Pearson={row.pearson_prediction_r:.4f}, "
            f"Spearman={row.spearman_rank_rho:.4f}, "
            f"selection overlap={overlap}"
        )

    print()
    print(f"Output directory: {output_dir}")
    print(
        "Main detailed file: "
        f"{cross_path}"
    )
    print(
        "Human summary:      "
        f"{summary_path}"
    )

    return output_dir

# =============================================================================
# MIDPOINT/FORK STRATEGY MIGRATION (old C/E -> X/Y; corrected C/E; F/G fork)
# =============================================================================

MIDPOINT_MIGRATION_KEY = "midpoint_strategy_migration_v2"
MIDPOINT_MIGRATION_ROOT = ACTIVE_LEARNING_DIR / "migrations"


def _model_row_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def _insert_model_row(con: sqlite3.Connection, data: dict) -> None:
    columns = list(data)
    placeholders = ",".join("?" for _ in columns)
    con.execute(
        f"INSERT INTO models ({','.join(columns)}) VALUES ({placeholders})",
        tuple(data[column] for column in columns),
    )


def _rename_model_record(
    con: sqlite3.Connection,
    *,
    old_model_id: str,
    new_model_id: str,
    new_strategy: str,
) -> dict:
    """Move a model identity while preserving all run IDs, metrics and child records."""
    old = con.execute(
        "SELECT * FROM models WHERE model_id=?", (old_model_id,)
    ).fetchone()
    if old is None:
        raise ValueError(f"Migration expected model {old_model_id}, but it is absent.")
    if con.execute(
        "SELECT 1 FROM models WHERE model_id=?", (new_model_id,)
    ).fetchone():
        raise ValueError(f"Migration target model {new_model_id} already exists.")

    data = _model_row_dict(old)
    data["model_id"] = new_model_id
    data["strategy"] = new_strategy
    data["dataset_name"] = f"dataset_{new_model_id}"
    data["split_version"] = f"split_{new_model_id}"
    if new_model_id in ARCHIVED_MODEL_REASONS:
        data["archived"] = 1
        data["archive_reason"] = ARCHIVED_MODEL_REASONS[new_model_id]
    data["updated_at"] = now_iso()
    _insert_model_row(con, data)

    # New parent exists before moving children because foreign_keys=ON.
    con.execute(
        "UPDATE dataset_membership SET model_id=? WHERE model_id=?",
        (new_model_id, old_model_id),
    )
    con.execute(
        "UPDATE prediction_runs SET model_id=? WHERE model_id=?",
        (new_model_id, old_model_id),
    )
    con.execute(
        "UPDATE selection_decisions SET source_model_id=? WHERE source_model_id=?",
        (new_model_id, old_model_id),
    )
    con.execute(
        "UPDATE models SET parent_model_id=? WHERE parent_model_id=?",
        (new_model_id, old_model_id),
    )
    con.execute("DELETE FROM models WHERE model_id=?", (old_model_id,))
    return data


def _replace_acquisition_strategy_tokens(value: str | None) -> str | None:
    if value is None:
        return None
    mapping = {"C": "X", "E": "Y", "F": LEGACY_F_STRATEGY}
    tokens = [x.strip() for x in str(value).split(",") if x.strip()]
    return ",".join(mapping.get(token, token) for token in tokens)


def _plan_midpoint_replacement_selections(con: sqlite3.Connection) -> dict:
    """
    Recreate corrected Cycle-1 C/E in an isolated in-memory copy.

    The frozen simulation set is exactly the original locked Model0 cohort
    (32 train + 8 benchmark), not the current 79 preferred observations.
    """
    marker_row = con.execute(
        "SELECT value FROM metadata WHERE key=?", (MIDPOINT_MIGRATION_KEY,)
    ).fetchone()
    if marker_row is not None:
        raise ValueError("The midpoint migration is already recorded; refusing to rerun it.")

    temp = sqlite3.connect(":memory:")
    temp.row_factory = sqlite3.Row
    con.backup(temp)
    try:
        # Free C/E/F labels in the temporary planner only.
        temp.execute("UPDATE selection_decisions SET strategy='X' WHERE cycle=1 AND strategy='C'")
        temp.execute("UPDATE selection_decisions SET strategy='Y' WHERE cycle=1 AND strategy='E'")
        temp.execute(
            "UPDATE selection_decisions SET strategy=? WHERE cycle=1 AND strategy='F'",
            (LEGACY_F_STRATEGY,),
        )

        benchmark_ids = set(get_benchmark_ids(temp))
        frozen_model0_ids = {
            str(row["id"])
            for row in temp.execute(
                "SELECT id FROM split_membership WHERE split_version=?",
                (MODEL0_SPLIT_VERSION,),
            ).fetchall()
        }
        if len(frozen_model0_ids) != EXPECTED_MODEL0_UNIQUE_PEPTOIDS:
            raise RuntimeError(
                "Migration requires the locked original Model0 40-peptoid cohort; "
                f"found {len(frozen_model0_ids)}."
            )

        prediction_run_id, predictions = load_source_prediction_df(temp, "Model0")
        # Preserve the reviewed historical migration behaviour exactly: the
        # original locked Model0 cohort is excluded, while current later-cycle
        # real/queued observations do not alter the reconstructed Cycle-1 plan.
        pmin, pmax, midpoint, eligible_n = eligible_prediction_midpoint(
            predictions,
            benchmark_ids,
            frozen_model0_ids,
            set(),
            set(),
        )

        new_selected: dict[str, list[dict]] = {}
        for strategy in ("C", "E"):
            generate_strategy_selection(
                temp,
                cycle=1,
                strategy=strategy,
                source_model_id="Model0",
                prediction_run_id=prediction_run_id,
                predictions=predictions,
                benchmark_ids=benchmark_ids,
                source_model_ids=frozen_model0_ids,
                prior_strategy_ids=set(),
                reusable_physical_ids=set(),
            )
            rows = temp.execute(
                """
                SELECT id, predicted_ddapl, raw_prediction_rank,
                       selection_group, target_value, decision_order
                FROM selection_decisions
                WHERE cycle=1 AND strategy=? AND decision='selected'
                ORDER BY decision_order
                """,
                (strategy,),
            ).fetchall()
            if len(rows) != ACQUISITION_SIZE:
                raise RuntimeError(
                    f"Planned {strategy} produced {len(rows)} selections; expected 15."
                )
            new_selected[strategy] = [
                {
                    "ID": str(row["id"]),
                    "predicted_ddapl": float(row["predicted_ddapl"]),
                    "raw_prediction_rank": int(row["raw_prediction_rank"]),
                    "selection_group": str(row["selection_group"]),
                    "target_value": None if row["target_value"] is None else float(row["target_value"]),
                    "decision_order": int(row["decision_order"]),
                }
                for row in rows
            ]

        preferred_now = {
            str(row["id"])
            for row in con.execute("SELECT id FROM observations WHERE preferred=1").fetchall()
        }
        old_ce = {
            str(row["id"])
            for row in con.execute(
                """
                SELECT DISTINCT id FROM selection_decisions
                WHERE cycle=1 AND strategy IN ('C','E') AND decision='selected'
                """
            ).fetchall()
        }
        old_abd = {
            str(row["id"])
            for row in con.execute(
                """
                SELECT DISTINCT id FROM selection_decisions
                WHERE cycle=1 AND strategy IN ('A','B','D') AND decision='selected'
                """
            ).fetchall()
        }
        old_ce_only = sorted(old_ce - old_abd)

        c_ids = {row["ID"] for row in new_selected["C"]}
        e_ids = {row["ID"] for row in new_selected["E"]}
        ce_union = c_ids | e_ids
        canonical_ae_union = old_abd | ce_union
        new_physical = sorted(ce_union - preferred_now)
        reused = sorted(ce_union & preferred_now)
        recovered = sorted(set(old_ce_only) & ce_union)
        missed = sorted(set(old_ce_only) - ce_union)

        return {
            "migration": MIDPOINT_MIGRATION_KEY,
            "source_model": "Model0",
            "source_prediction_run_id": prediction_run_id,
            "frozen_model0_ids_n": len(frozen_model0_ids),
            "eligible_candidate_count": eligible_n,
            "eligible_prediction_min": pmin,
            "eligible_prediction_max": pmax,
            "prediction_range_midpoint": midpoint,
            "new_C": new_selected["C"],
            "new_E": new_selected["E"],
            "new_C_E_unique_n": len(ce_union),
            "reused_existing_real_ids": reused,
            "reused_existing_real_n": len(reused),
            "new_physical_simulation_ids": new_physical,
            "new_physical_simulation_n": len(new_physical),
            "old_C_E_only_ids": old_ce_only,
            "old_C_E_only_recovered_by_new_C_E": recovered,
            "old_C_E_only_not_recovered": missed,
            "canonical_A_E_unique_acquisitions_n": len(canonical_ae_union),
            "expected_replacement_Model1F_train_n": MODEL0_TRAIN_SIZE + len(canonical_ae_union),
            "notes": [
                "Old Cycle-1 C becomes historical X.",
                "Old Cycle-1 E becomes historical Y.",
                "Old premature Model0-based F selection becomes F_legacy.",
                "New C/E use the original frozen Model0 40-peptoid state.",
                "Canonical Model1F is the shared pooled A-E Cycle-1 seed and must be retrained after corrected C/E are real.",
                "From Model2 onward F adds prior F acquisitions (self-feedback), while G pools A-E only (control).",
                "There is intentionally no Model1G; H remains reserved for future work.",
            ],
        }
    finally:
        temp.close()


def _print_midpoint_migration_plan(plan: dict) -> None:
    print("=" * 80)
    print("MIDPOINT STRATEGY MIGRATION PLAN")
    print("=" * 80)
    print(f"Source model:                     {plan['source_model']}")
    print(f"Frozen Model0 cohort:             {plan['frozen_model0_ids_n']}")
    print(f"Eligible prediction candidates:   {plan['eligible_candidate_count']}")
    print(
        "Eligible predicted range:        "
        f"{plan['eligible_prediction_min']:.6f} -> {plan['eligible_prediction_max']:.6f}"
    )
    print(f"Prediction-range midpoint:        {plan['prediction_range_midpoint']:.6f}")
    print()
    for strategy in ("C", "E"):
        print(f"NEW {strategy}:")
        for row in plan[f"new_{strategy}"]:
            target = "-" if row["target_value"] is None else f"{row['target_value']:.6f}"
            print(
                f"    {row['selection_group']:<9s} {row['ID']:<22s} "
                f"pred={row['predicted_ddapl']:.6f} "
                f"rank={row['raw_prediction_rank']:<6d} target={target}"
            )
        print()

    print(f"Unique new C/E logical selections: {plan['new_C_E_unique_n']}")
    print(f"Already have real CGMD values:      {plan['reused_existing_real_n']}")
    print(f"NEW physical simulations required:  {plan['new_physical_simulation_n']}")
    for id_ in plan["new_physical_simulation_ids"]:
        print(f"    NEW: {id_}")

    print()
    print("Old C/E-only peptoids:")
    for id_ in plan["old_C_E_only_ids"]:
        marker = "RECOVERED" if id_ in plan["old_C_E_only_recovered_by_new_C_E"] else "NOT RECOVERED"
        print(f"    {id_:<22s} {marker}")
    print()
    print(f"Canonical A-E unique acquisitions:  {plan['canonical_A_E_unique_acquisitions_n']}")
    print(
        f"Expected replacement Model1F train: {plan['expected_replacement_Model1F_train_n']} "
        f"+ {MODEL0_BENCHMARK_SIZE} benchmark"
    )


def _backup_file_for_migration(path: Path, snapshot_root: Path) -> Path | None:
    path = Path(path)
    if not path.is_file():
        return None
    relative = path.resolve().relative_to(_MACHINELEARNING_ROOT.resolve())
    destination = snapshot_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite the first pre-migration snapshot with a later, already
    # modified version of the same file (notably run_registry.csv).
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def _rewrite_text_file(
    path: Path,
    replacements: dict[str, str],
    snapshot_root: Path,
    audit_rows: list[dict],
) -> None:
    path = Path(path)
    if not path.is_file():
        return
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    updated = original
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated == original:
        return
    backup = _backup_file_for_migration(path, snapshot_root)
    before_hash = sha256_file(path)
    path.write_text(updated, encoding="utf-8")
    audit_rows.append(
        {
            "action": "rewrite_text",
            "path": str(path),
            "backup": None if backup is None else str(backup),
            "sha256_before": before_hash,
            "sha256_after": sha256_file(path),
        }
    )


def _rename_file_for_migration(
    old_path: Path,
    new_path: Path,
    snapshot_root: Path,
    audit_rows: list[dict],
) -> None:
    old_path = Path(old_path)
    new_path = Path(new_path)
    if not old_path.exists():
        return
    if new_path.exists():
        raise FileExistsError(f"Migration destination already exists:\n{new_path}")
    backup = _backup_file_for_migration(old_path, snapshot_root)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.rename(new_path)
    audit_rows.append(
        {
            "action": "rename",
            "old_path": str(old_path),
            "new_path": str(new_path),
            "backup": None if backup is None else str(backup),
        }
    )


def _rewrite_membership_provenance_csv(
    path: Path,
    snapshot_root: Path,
    audit_rows: list[dict],
) -> None:
    """Update migrated dataset-membership provenance labels C/E/F -> X/Y/F_legacy.

    The SQLite rows are migrated with the same mapping. This keeps the canonical
    membership CSV consistent with the database while the untouched pre-migration
    file remains preserved in the migration snapshot.
    """
    path = Path(path)
    if not path.is_file():
        return

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "acquisition_strategy" not in df.columns:
        raise ValueError(
            "Migrated dataset membership file has no acquisition_strategy column:\n"
            f"{path}"
        )

    original_values = df["acquisition_strategy"].astype(str).tolist()
    df["acquisition_strategy"] = [
        _replace_acquisition_strategy_tokens(value) or ""
        for value in df["acquisition_strategy"].tolist()
    ]
    updated_values = df["acquisition_strategy"].astype(str).tolist()

    if updated_values == original_values:
        return

    backup = _backup_file_for_migration(path, snapshot_root)
    before_hash = sha256_file(path)
    df.to_csv(path, index=False)
    audit_rows.append(
        {
            "action": "rewrite_membership_provenance",
            "path": str(path),
            "backup": None if backup is None else str(backup),
            "sha256_before": before_hash,
            "sha256_after": sha256_file(path),
        }
    )


def _refresh_json_path_records(
    path: Path,
    snapshot_root: Path,
    audit_rows: list[dict],
) -> None:
    """Refresh hashes/sizes for path-record dictionaries in migrated JSON.

    Training manifests record hashes for dataset membership and split/input files.
    Renaming C/E/F identities and updating membership provenance changes those
    canonical files, so stale hashes would make the migrated provenance internally
    inconsistent. The original JSON is retained in the pre-migration snapshot.
    """
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".json":
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return

    changed = False

    def refresh(node):
        nonlocal changed
        if isinstance(node, dict):
            if "resolved_path" in node and node.get("resolved_path"):
                resolved = Path(str(node["resolved_path"])).expanduser()
                exists = resolved.exists()

                if node.get("exists") != exists:
                    node["exists"] = exists
                    changed = True

                try:
                    relative = str(
                        resolved.resolve().relative_to(_MACHINELEARNING_ROOT.resolve())
                    )
                except (ValueError, OSError):
                    relative = None
                if node.get("project_relative_path") != relative:
                    node["project_relative_path"] = relative
                    changed = True

                if resolved.is_file():
                    digest = sha256_file(resolved)
                    size = resolved.stat().st_size
                else:
                    digest = None
                    size = None

                if node.get("sha256") != digest:
                    node["sha256"] = digest
                    changed = True
                if node.get("size_bytes") != size:
                    node["size_bytes"] = size
                    changed = True

            for value in node.values():
                refresh(value)
        elif isinstance(node, list):
            for value in node:
                refresh(value)

    refresh(payload)
    if not changed:
        return

    backup = _backup_file_for_migration(path, snapshot_root)
    before_hash = sha256_file(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    audit_rows.append(
        {
            "action": "refresh_json_path_records",
            "path": str(path),
            "backup": None if backup is None else str(backup),
            "sha256_before": before_hash,
            "sha256_after": sha256_file(path),
        }
    )


def _rollback_migration_files(audit_rows: list[dict]) -> list[str]:
    """Best-effort filesystem rollback after a failed migration apply.

    Database mutations are handled by SQLite rollback. This reverses only the
    canonical file rewrites/renames that were already performed. No historical
    backup is deleted.
    """
    warnings: list[str] = []
    for action in reversed(audit_rows):
        kind = str(action.get("action", ""))
        try:
            if kind in {
                "rewrite_text",
                "rewrite_membership_provenance",
                "refresh_json_path_records",
            }:
                path = Path(str(action["path"]))
                backup_value = action.get("backup")
                if backup_value:
                    backup = Path(str(backup_value))
                    if backup.is_file():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup, path)

            elif kind == "rename":
                old_path = Path(str(action["old_path"]))
                new_path = Path(str(action["new_path"]))
                if new_path.exists() and not old_path.exists():
                    old_path.parent.mkdir(parents=True, exist_ok=True)
                    new_path.rename(old_path)
                elif new_path.exists() and old_path.exists():
                    warnings.append(
                        "Could not automatically reverse rename because both paths exist: "
                        f"{old_path} | {new_path}"
                    )
                elif not old_path.exists():
                    backup_value = action.get("backup")
                    if backup_value and Path(str(backup_value)).is_file():
                        old_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(Path(str(backup_value)), old_path)
                    else:
                        warnings.append(
                            f"Could not restore missing migration source: {old_path}"
                        )
        except Exception as exc:
            warnings.append(f"Rollback warning for {action}: {exc}")
    return warnings


def _rewrite_run_artifacts_for_model_migration(
    *,
    run_id: str,
    old_model_id: str,
    new_model_id: str,
    snapshot_root: Path,
    audit_rows: list[dict],
) -> None:
    roots = [
        training_run_dir("position_resolved", run_id),
        training_run_dir("global_molecular", run_id),
        ACTIVE_LEARNING_DIR / "global_comparison_inputs" / run_id,
    ]
    allowed_suffixes = {".json", ".csv", ".txt"}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_suffixes:
                _rewrite_text_file(
                    path, {old_model_id: new_model_id}, snapshot_root, audit_rows
                )

    # Model-ID replacement can change resolved paths inside run manifests.
    # Refresh any path-record hashes/sizes only after all textual rewrites.
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            _refresh_json_path_records(path, snapshot_root, audit_rows)


def _migrate_named_model_files(
    *,
    old_model_id: str,
    new_model_id: str,
    training_run_id: str,
    snapshot_root: Path,
    audit_rows: list[dict],
) -> None:
    pairs = [
        (
            TRAINING_DATASET_DIR / f"{old_model_id}_training_data.csv",
            TRAINING_DATASET_DIR / f"{new_model_id}_training_data.csv",
        ),
        (
            DATASETS_DIR / f"dataset_{old_model_id}.csv",
            DATASETS_DIR / f"dataset_{new_model_id}.csv",
        ),
        (
            SPLITS_DIR / f"split_assignments_{old_model_id}.csv",
            SPLITS_DIR / f"split_assignments_{new_model_id}.csv",
        ),
        (
            MODEL_COMPARISON_DIR / f"{old_model_id}_training-{training_run_id}_comparison.csv",
            MODEL_COMPARISON_DIR / f"{new_model_id}_training-{training_run_id}_comparison.csv",
        ),
    ]
    for old_path, new_path in pairs:
        _rename_file_for_migration(old_path, new_path, snapshot_root, audit_rows)
        if new_path.is_file():
            _rewrite_text_file(
                new_path, {old_model_id: new_model_id}, snapshot_root, audit_rows
            )

            # dataset_<Model>.csv is the canonical logical membership export.
            # Its acquisition_strategy column must match the migrated DB labels.
            if new_path.parent == DATASETS_DIR and new_path.name.startswith("dataset_"):
                _rewrite_membership_provenance_csv(
                    new_path, snapshot_root, audit_rows
                )


def _rebuild_cycle_queue_from_selection_history(con: sqlite3.Connection, cycle: int) -> None:
    # Simulation queue is operational/canonical state only. Historical X/Y and
    # F_legacy selection records remain in selection_decisions but must never
    # create or retain physical-work queue entries by themselves.
    selected_rows = con.execute(
        """
        SELECT DISTINCT strategy, id FROM selection_decisions
        WHERE cycle=? AND decision='selected'
          AND strategy IN ('A','B','C','D','E','F','G')
        ORDER BY strategy, id
        """,
        (cycle,),
    ).fetchall()
    by_id: dict[str, set[str]] = {}
    for row in selected_rows:
        by_id.setdefault(str(row["id"]), set()).add(str(row["strategy"]))

    canonical_ids = set(by_id)
    existing_queue_ids = {
        str(row["id"])
        for row in con.execute(
            "SELECT id FROM simulation_queue WHERE cycle=?", (cycle,)
        ).fetchall()
    }
    historical_only = sorted(existing_queue_ids - canonical_ids)
    if historical_only:
        con.executemany(
            """
            UPDATE simulation_queue
            SET archived=1,
                archive_reason='historical-only X/Y/F_legacy queue provenance; not canonical physical work',
                updated_at=?
            WHERE cycle=? AND id=?
            """,
            [(now_iso(), cycle, id_) for id_ in historical_only],
        )

    preferred = {
        str(row["id"])
        for row in con.execute("SELECT id FROM observations WHERE preferred=1").fetchall()
    }
    now = now_iso()
    for id_, strategies in sorted(by_id.items()):
        requested_by = ",".join(sorted(strategies))
        existing = con.execute(
            "SELECT status FROM simulation_queue WHERE cycle=? AND id=?", (cycle, id_)
        ).fetchone()
        if existing is None:
            status = "analysis_ready" if id_ in preferred else "selected"
            con.execute(
                """
                INSERT INTO simulation_queue(cycle,id,requested_by,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?)
                """,
                (cycle, id_, requested_by, status, now, now),
            )
        else:
            status = str(existing["status"])
            if id_ in preferred and status == "selected":
                status = "analysis_ready"
            con.execute(
                """
                UPDATE simulation_queue
                SET requested_by=?,status=?,archived=0,archive_reason=NULL,updated_at=?
                WHERE cycle=? AND id=?
                """,
                (requested_by, status, now, cycle, id_),
            )


def rebuild_all_canonical_simulation_queues(con: sqlite3.Connection) -> list[int]:
    """Reconcile every queue from canonical selection history only."""
    cycles = [
        int(row["cycle"])
        for row in con.execute(
            "SELECT DISTINCT cycle FROM selection_decisions ORDER BY cycle"
        ).fetchall()
    ]
    for cycle in cycles:
        _rebuild_cycle_queue_from_selection_history(con, cycle)
    return cycles


def migrate_midpoint_strategy_v2(con: sqlite3.Connection, *, apply: bool) -> dict:
    """Controlled first-cycle correction and pooled F/G fork migration.

    Dry run (default)
        Exact planning only; database is opened read-only by main().

    Apply
        * old Cycle-1 C -> X and old Model1C -> Model1X;
        * old Cycle-1 E -> Y and old Model1E -> Model1Y;
        * premature old Model0-based F selection -> F_legacy;
        * old trained Model1F -> Model1F_legacy_pre_midpoint;
        * insert corrected midpoint Cycle-1 C/E decisions from the original
          frozen Model0 40-peptoid state;
        * preserve completed run artifacts under migrated identities.

    It does NOT submit simulations, training or predictions.  Corrected Model1F
    is rebuilt only after the reviewed new C/E acquisitions are real.  Model1F
    then acts as the shared seed for F/G; Model2G is the first G model.
    """
    if apply:
        con.commit()

    plan = _plan_midpoint_replacement_selections(con)
    _print_midpoint_migration_plan(plan)

    MIDPOINT_MIGRATION_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_path = MIDPOINT_MIGRATION_ROOT / f"{stamp}_midpoint_v2_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print()
    print(f"Plan JSON: {plan_path}")

    if not apply:
        print()
        print("DRY RUN ONLY — database and canonical project artifacts were not changed.")
        print("Review the exact C/E selections and X/Y-only recovery check before --apply.")
        return plan

    required_models = {"Model1C": "trained", "Model1E": "trained", "Model1F": "trained"}
    old_rows: dict[str, sqlite3.Row] = {}
    for model_id, status in required_models.items():
        row = con.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
        if row is None or str(row["status"]) != status:
            found = None if row is None else row["status"]
            raise ValueError(
                f"Migration requires {model_id} status={status!r}; found {found!r}."
            )
        old_rows[model_id] = row

    for target in ("Model1X", "Model1Y", LEGACY_F_MODEL_ID):
        if con.execute("SELECT 1 FROM models WHERE model_id=?", (target,)).fetchone():
            raise ValueError(f"Migration target {target} already exists.")

    migrations = {
        "Model1C": ("Model1X", "X"),
        "Model1E": ("Model1Y", "Y"),
        "Model1F": (LEGACY_F_MODEL_ID, LEGACY_F_STRATEGY),
    }

    # Fail before touching DB/files if an expected canonical source is absent or
    # a destination is already occupied.
    for old_model_id, (new_model_id, _) in migrations.items():
        run_id = str(old_rows[old_model_id]["training_run_id"])
        required_pairs = [
            (TRAINING_DATASET_DIR / f"{old_model_id}_training_data.csv",
             TRAINING_DATASET_DIR / f"{new_model_id}_training_data.csv"),
            (DATASETS_DIR / f"dataset_{old_model_id}.csv",
             DATASETS_DIR / f"dataset_{new_model_id}.csv"),
            (SPLITS_DIR / f"split_assignments_{old_model_id}.csv",
             SPLITS_DIR / f"split_assignments_{new_model_id}.csv"),
            (MODEL_COMPARISON_DIR / f"{old_model_id}_training-{run_id}_comparison.csv",
             MODEL_COMPARISON_DIR / f"{new_model_id}_training-{run_id}_comparison.csv"),
        ]
        for old_path, new_path in required_pairs:
            if not old_path.is_file():
                raise FileNotFoundError(
                    f"Migration expected canonical source file but it is missing:\n{old_path}"
                )
            if new_path.exists():
                raise FileExistsError(
                    f"Migration destination already exists; refusing to overwrite:\n{new_path}"
                )
        for representation in ("position_resolved", "global_molecular"):
            run_dir = training_run_dir(representation, run_id)
            if not run_dir.is_dir() or not (run_dir / "run_manifest.json").is_file():
                raise FileNotFoundError(
                    f"Migration expected completed {representation} run with manifest:\n{run_dir}"
                )

    if not RUN_REGISTRY_CSV.is_file():
        raise FileNotFoundError(f"Run registry is missing:\n{RUN_REGISTRY_CSV}")

    migration_dir = MIDPOINT_MIGRATION_ROOT / f"{stamp}_midpoint_v2_APPLIED"
    snapshot_root = migration_dir / "pre_migration_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)

    db_backup = snapshot_root / "active_learning.db"
    backup_con = sqlite3.connect(db_backup)
    try:
        con.backup(backup_con)
    finally:
        backup_con.close()

    audit_rows: list[dict] = [{
        "action": "database_backup",
        "source": str(ACTIVE_LEARNING_DB),
        "backup": str(db_backup),
    }]
    _backup_file_for_migration(RUN_REGISTRY_CSV, snapshot_root)
    model_records: list[dict] = []

    try:
        # Historical identities: old C/E -> X/Y; premature F -> F_legacy.
        con.execute("UPDATE selection_decisions SET strategy='X' WHERE cycle=1 AND strategy='C'")
        con.execute("UPDATE selection_decisions SET strategy='Y' WHERE cycle=1 AND strategy='E'")
        con.execute(
            "UPDATE selection_decisions SET strategy=? WHERE cycle=1 AND strategy='F'",
            (LEGACY_F_STRATEGY,),
        )

        for old_model_id, (new_model_id, new_strategy) in migrations.items():
            old = old_rows[old_model_id]
            run_id = str(old["training_run_id"])
            _rename_model_record(
                con,
                old_model_id=old_model_id,
                new_model_id=new_model_id,
                new_strategy=new_strategy,
            )
            model_records.append({
                "old_model_id": old_model_id,
                "new_model_id": new_model_id,
                "training_run_id": run_id,
                "chosen_algorithm": old["chosen_algorithm"],
                "chosen_representation": old["chosen_representation"],
                "benchmark_r2": old["benchmark_r2"],
                "benchmark_rmse": old["benchmark_rmse"],
                "benchmark_mae": old["benchmark_mae"],
            })

        # Keep stored provenance labels consistent with the migrated identities.
        rows = con.execute(
            "SELECT model_id,id,acquisition_strategy FROM dataset_membership "
            "WHERE acquisition_strategy IS NOT NULL"
        ).fetchall()
        for row in rows:
            old_value = row["acquisition_strategy"]
            new_value = _replace_acquisition_strategy_tokens(old_value)
            if new_value != old_value:
                con.execute(
                    "UPDATE dataset_membership SET acquisition_strategy=? "
                    "WHERE model_id=? AND id=?",
                    (new_value, str(row["model_id"]), str(row["id"])),
                )

        # Insert corrected midpoint C/E from the ORIGINAL frozen Model0 cohort.
        benchmark_ids = set(get_benchmark_ids(con))
        frozen_model0_ids = {
            str(row["id"])
            for row in con.execute(
                "SELECT id FROM split_membership WHERE split_version=?",
                (MODEL0_SPLIT_VERSION,),
            ).fetchall()
        }
        if len(frozen_model0_ids) != EXPECTED_MODEL0_UNIQUE_PEPTOIDS:
            raise RuntimeError(
                "Locked Model0 cohort changed during migration planning; refusing to continue."
            )
        prediction_run_id, predictions = load_source_prediction_df(con, "Model0")
        for strategy in ("C", "E"):
            generate_strategy_selection(
                con,
                cycle=1,
                strategy=strategy,
                source_model_id="Model0",
                prediction_run_id=prediction_run_id,
                predictions=predictions,
                benchmark_ids=benchmark_ids,
                source_model_ids=frozen_model0_ids,
                prior_strategy_ids=set(),
                reusable_physical_ids=set(),
            )

        _rebuild_cycle_queue_from_selection_history(con, 1)
        refresh_prediction_outcomes(con)

        for old_id in ("Model1C", "Model1E", "Model1F"):
            if con.execute("SELECT 1 FROM models WHERE model_id=?", (old_id,)).fetchone():
                raise RuntimeError(f"Stale model {old_id} remains after migration.")
        for new_id in ("Model1X", "Model1Y", LEGACY_F_MODEL_ID):
            if not con.execute("SELECT 1 FROM models WHERE model_id=?", (new_id,)).fetchone():
                raise RuntimeError(f"Migrated model {new_id} is missing.")
        if con.execute("SELECT 1 FROM models WHERE model_id='Model1G'").fetchone():
            raise RuntimeError("Model1G must not exist; Model1F is the shared pooled seed.")

        expected_counts = {
            "C": 15,
            "E": 15,
            "X": 15,
            "Y": 15,
            LEGACY_F_STRATEGY: 15,
            "F": 0,
            "G": 0,
        }
        for strategy, expected in expected_counts.items():
            n = int(con.execute(
                """
                SELECT COUNT(DISTINCT id) FROM selection_decisions
                WHERE cycle=1 AND strategy=? AND decision='selected'
                """,
                (strategy,),
            ).fetchone()[0])
            if n != expected:
                raise RuntimeError(
                    f"Post-migration selection count {strategy}: {n}, expected {expected}."
                )

        for strategy in ("C", "E"):
            planned_ids = [row["ID"] for row in plan[f"new_{strategy}"]]
            actual_ids = [
                str(row["id"])
                for row in con.execute(
                    """
                    SELECT id FROM selection_decisions
                    WHERE cycle=1 AND strategy=? AND decision='selected'
                    ORDER BY decision_order
                    """,
                    (strategy,),
                ).fetchall()
            ]
            if actual_ids != planned_ids:
                raise RuntimeError(
                    f"Applied {strategy} selections differ from the reviewed dry-run plan."
                )

        # Rename/rewrite canonical model artifacts; immutable historical reports,
        # rendered figures and raw scheduler logs are not rewritten.
        for record in model_records:
            old_id = record["old_model_id"]
            new_id = record["new_model_id"]
            run_id = record["training_run_id"]
            _migrate_named_model_files(
                old_model_id=old_id,
                new_model_id=new_id,
                training_run_id=run_id,
                snapshot_root=snapshot_root,
                audit_rows=audit_rows,
            )
            _rewrite_run_artifacts_for_model_migration(
                run_id=run_id,
                old_model_id=old_id,
                new_model_id=new_id,
                snapshot_root=snapshot_root,
                audit_rows=audit_rows,
            )
            _rewrite_text_file(
                RUN_REGISTRY_CSV,
                {old_id: new_id},
                snapshot_root,
                audit_rows,
            )

        record = {
            "applied_at": now_iso(),
            "plan": plan,
            "model_migrations": model_records,
            "filesystem_actions": audit_rows,
            "canonical_F_training_rule": (
                "Model1F = pooled canonical A-E Cycle-1 acquisitions; "
                "Model<n>F (n>=2) = pooled canonical A-E through n plus canonical F selections from cycles < n"
            ),
            "canonical_G_training_rule": (
                "Model<n>G (n>=2) = pooled canonical A-E through n only; G selections never feed back"
            ),
            "pooled_selection_rule": "F and G each select 15 highest predicted DDAPL candidates",
            "shared_seed_rule": "Model1F is shared; Model1G is intentionally absent; F/G fork starts at cycle 2",
            "reserved_strategy": "H",
            "historical_inactive_methods": {
                "X": "old C: 5 top + 5 closest-to-zero + 5 bottom",
                "Y": "old E: 7 top + 4 closest-to-zero + 4 bottom",
            },
            "immutable_historical_exceptions": [
                "pre-migration inspection reports",
                "already-rendered pre-migration figures",
                "raw scheduler/log evidence outside canonical run metadata",
            ],
        }
        audit_path = migration_dir / "migration_audit.json"
        audit_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        con.execute(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            (MIDPOINT_MIGRATION_KEY,
             json.dumps({"applied_at": record["applied_at"], "audit_path": str(audit_path)})),
        )
        con.commit()

    except Exception:
        con.rollback()
        rollback_warnings = _rollback_migration_files(audit_rows)
        print()
        print("MIGRATION APPLY FAILED — database transaction rolled back.")
        print(f"Pre-migration database backup remains at: {db_backup}")
        for warning in rollback_warnings:
            print(f"FILESYSTEM ROLLBACK WARNING: {warning}")
        raise

    try:
        export_cycle_selection_files(con, 1)
        export_excel(con)
    except Exception as exc:
        print(f"WARNING: migration applied but a derived export failed: {exc}")

    print()
    print("=" * 80)
    print("MIDPOINT/FORK STRATEGY MIGRATION APPLIED")
    print("=" * 80)
    print("Old Model1C -> Model1X (historical zero-target training preserved)")
    print("Old Model1E -> Model1Y (historical zero-target training preserved)")
    print(f"Old Model1F -> {LEGACY_F_MODEL_ID} (premature training preserved as non-canonical)")
    print("New Cycle-1 C/E midpoint selections inserted.")
    print("Canonical Model1F is intentionally absent and must be rebuilt after corrected C/E are real.")
    print("Model1F will be the shared pooled seed; Model2F/Model2G are the first forked models.")
    print("Strategy H remains reserved.")
    print(f"Migration audit: {audit_path}")
    print(f"Database backup: {db_backup}")
    print("NO simulations were submitted.")
    print("NO new C/E/F/G models were prepared or trained.")
    return record


def export_excel(con: sqlite3.Connection) -> None:
    """
    Export a human-readable Excel view of the active-learning database.

    IMPORTANT:
    active_learning.db remains the authoritative source of truth.
    This workbook is a visual/read-only representation of database state.
    """

    ensure_directories()

    # =====================================================================
    # SUMMARY INFORMATION
    # =====================================================================

    preferred_observations = con.execute(
        """
        SELECT COUNT(*)
        FROM observations
        WHERE preferred = 1
        """
    ).fetchone()[0]

    total_observations = con.execute(
        """
        SELECT COUNT(*)
        FROM observations
        """
    ).fetchone()[0]

    duplicate_observations = (
        total_observations
        - preferred_observations
    )

    model_count = con.execute(
        """
        SELECT COUNT(*)
        FROM models
        WHERE COALESCE(archived,0)=0
        """
    ).fetchone()[0]

    archived_model_count = con.execute(
        """
        SELECT COUNT(*)
        FROM models
        WHERE COALESCE(archived,0)=1
        """
    ).fetchone()[0]

    prediction_run_count = con.execute(
        """
        SELECT COUNT(*)
        FROM prediction_runs pr
        JOIN models m ON m.model_id=pr.model_id
        WHERE COALESCE(m.archived,0)=0
        """
    ).fetchone()[0]

    selection_count = con.execute(
        """
        SELECT COUNT(*)
        FROM selection_decisions
        WHERE strategy IN ('A','B','C','D','E','F','G')
        """
    ).fetchone()[0]

    selected_count = con.execute(
        """
        SELECT COUNT(*)
        FROM selection_decisions
        WHERE decision = 'selected'
          AND strategy IN ('A','B','C','D','E','F','G')
        """
    ).fetchone()[0]

    skipped_count = con.execute(
        """
        SELECT COUNT(*)
        FROM selection_decisions
        WHERE decision = 'skipped'
          AND strategy IN ('A','B','C','D','E','F','G')
        """
    ).fetchone()[0]

    historical_selection_count = con.execute(
        """
        SELECT COUNT(*)
        FROM selection_decisions
        WHERE strategy IN ('X','Y','F_legacy')
        """
    ).fetchone()[0]

    prediction_outcome_count = con.execute(
        """
        SELECT COUNT(*)
        FROM prediction_outcomes po
        JOIN selection_decisions d ON d.decision_id=po.decision_id
        WHERE d.strategy IN ('A','B','C','D','E','F','G')
        """
    ).fetchone()[0]

    simulation_queue_count = con.execute(
        """
        SELECT COUNT(*)
        FROM simulation_queue
        WHERE COALESCE(archived,0)=0
        """
    ).fetchone()[0]

    benchmark_count = con.execute(
        """
        SELECT COUNT(*)
        FROM split_membership
        WHERE split_version = 'active_model0_v1'
          AND role = 'benchmark'
        """
    ).fetchone()[0]

    model0_train_count = con.execute(
        """
        SELECT COUNT(*)
        FROM split_membership
        WHERE split_version = 'active_model0_v1'
          AND role = 'train'
        """
    ).fetchone()[0]

    # =====================================================================
    # CURRENT CYCLE
    # =====================================================================

    current_cycle_row = con.execute(
        """
        SELECT MAX(cycle)
        FROM models
        WHERE COALESCE(archived,0)=0
        """
    ).fetchone()

    current_cycle = (
        current_cycle_row[0]
        if (
            current_cycle_row is not None
            and current_cycle_row[0] is not None
        )
        else 0
    )

    # =====================================================================
    # LATEST MODEL
    # =====================================================================

    latest_model = con.execute(
        """
        SELECT
            model_id,
            status,
            chosen_algorithm,
            chosen_representation,
            training_run_id,
            prediction_run_id,
            benchmark_r2,
            benchmark_rmse,
            benchmark_mae
        FROM models
        WHERE COALESCE(archived,0)=0
        ORDER BY
            cycle DESC,
            updated_at DESC
        LIMIT 1
        """
    ).fetchone()

    if latest_model is None:

        latest_model_id = "-"
        latest_model_status = "-"
        latest_algorithm = "-"
        latest_representation = "-"
        latest_training_run = "-"
        latest_prediction_run = "-"
        latest_r2 = "-"
        latest_rmse = "-"
        latest_mae = "-"

    else:

        latest_model_id = (
            latest_model["model_id"]
            if latest_model["model_id"] is not None
            else "-"
        )

        latest_model_status = (
            latest_model["status"]
            if latest_model["status"] is not None
            else "-"
        )

        latest_algorithm = (
            latest_model["chosen_algorithm"]
            if latest_model["chosen_algorithm"] is not None
            else "-"
        )

        latest_representation = (
            latest_model["chosen_representation"]
            if latest_model["chosen_representation"] is not None
            else "-"
        )

        latest_training_run = (
            latest_model["training_run_id"]
            if latest_model["training_run_id"] is not None
            else "-"
        )

        latest_prediction_run = (
            latest_model["prediction_run_id"]
            if latest_model["prediction_run_id"] is not None
            else "-"
        )

        latest_r2 = (
            latest_model["benchmark_r2"]
            if latest_model["benchmark_r2"] is not None
            else "-"
        )

        latest_rmse = (
            latest_model["benchmark_rmse"]
            if latest_model["benchmark_rmse"] is not None
            else "-"
        )

        latest_mae = (
            latest_model["benchmark_mae"]
            if latest_model["benchmark_mae"] is not None
            else "-"
        )

    # =====================================================================
    # DETERMINE CURRENT WORKFLOW STATE
    # =====================================================================

    model0_exists = con.execute(
        """
        SELECT COUNT(*)
        FROM models
        WHERE model_id = 'Model0'
        """
    ).fetchone()[0]

    if model0_exists:

        model0_status_row = con.execute(
            """
            SELECT status
            FROM models
            WHERE model_id = 'Model0'
            """
        ).fetchone()

        model0_status = (
            model0_status_row["status"]
            if model0_status_row is not None
            else "prepared"
        )

        workflow_state = (
            f"Model0 exists: {model0_status}"
        )

        next_action = (
            "See Model0 status / continue model workflow"
        )

    elif (
        preferred_observations
        < EXPECTED_MODEL0_UNIQUE_PEPTOIDS
    ):

        missing = (
            EXPECTED_MODEL0_UNIQUE_PEPTOIDS
            - preferred_observations
        )

        workflow_state = (
            "Waiting for Model-0 baseline simulations"
        )

        next_action = (
            f"Complete {missing} additional unique peptoid "
            "simulation(s), then run workflow.py sync"
        )

    elif (
        preferred_observations
        == EXPECTED_MODEL0_UNIQUE_PEPTOIDS
    ):

        workflow_state = (
            "Model-0 baseline complete"
        )

        next_action = (
            "Run workflow.py prepare-model0"
        )

    else:

        workflow_state = (
            "More real observations exist than configured "
            "Model-0 baseline"
        )

        next_action = (
            "Review before preparing Model0"
        )

    # =====================================================================
    # BUILD SUMMARY SHEET
    # =====================================================================

    summary_rows = [
        {
            "Section": "Workflow",
            "Metric": "Current state",
            "Value": workflow_state,
        },
        {
            "Section": "Workflow",
            "Metric": "Next action",
            "Value": next_action,
        },
        {
            "Section": "Workflow",
            "Metric": "Current cycle",
            "Value": current_cycle,
        },

        {
            "Section": "Model-0 baseline",
            "Metric": "Required unique real peptoids",
            "Value": EXPECTED_MODEL0_UNIQUE_PEPTOIDS,
        },
        {
            "Section": "Model-0 baseline",
            "Metric": "Preferred unique real peptoids",
            "Value": preferred_observations,
        },
        {
            "Section": "Model-0 baseline",
            "Metric": "Remaining unique peptoids",
            "Value": max(
                0,
                EXPECTED_MODEL0_UNIQUE_PEPTOIDS
                - preferred_observations,
            ),
        },
        {
            "Section": "Model-0 baseline",
            "Metric": "Benchmark molecules",
            "Value": benchmark_count,
        },
        {
            "Section": "Model-0 baseline",
            "Metric": "Training molecules",
            "Value": model0_train_count,
        },

        {
            "Section": "Simulation data",
            "Metric": "Total complete observation pairs",
            "Value": total_observations,
        },
        {
            "Section": "Simulation data",
            "Metric": "Preferred observation pairs",
            "Value": preferred_observations,
        },
        {
            "Section": "Simulation data",
            "Metric": "Historical / duplicate observations",
            "Value": duplicate_observations,
        },

        {
            "Section": "Descriptors",
            "Metric": "Complete sequence universe",
            "Value": EXPECTED_SEQUENCE_COUNT,
        },
        {
            "Section": "Descriptors",
            "Metric": "Fixed ML feature count",
            "Value": EXPECTED_FEATURE_COUNT,
        },
        {
            "Section": "Descriptors",
            "Metric": "Descriptor version",
            "Value": DESCRIPTOR_VERSION,
        },

        {
            "Section": "Models",
            "Metric": "Active logical models",
            "Value": model_count,
        },
        {
            "Section": "Archive",
            "Metric": "Historical models quarantined",
            "Value": archived_model_count,
        },
        {
            "Section": "Archive",
            "Metric": "Historical selection records retained",
            "Value": historical_selection_count,
        },
        {
            "Section": "Models",
            "Metric": "Latest logical model",
            "Value": latest_model_id,
        },
        {
            "Section": "Models",
            "Metric": "Latest model status",
            "Value": latest_model_status,
        },
        {
            "Section": "Models",
            "Metric": "Latest chosen algorithm",
            "Value": latest_algorithm,
        },
        {
            "Section": "Models",
            "Metric": "Latest chosen representation",
            "Value": latest_representation,
        },
        {
            "Section": "Models",
            "Metric": "Latest training run",
            "Value": latest_training_run,
        },
        {
            "Section": "Models",
            "Metric": "Latest prediction run",
            "Value": latest_prediction_run,
        },

        {
            "Section": "Latest benchmark",
            "Metric": "R2",
            "Value": latest_r2,
        },
        {
            "Section": "Latest benchmark",
            "Metric": "RMSE",
            "Value": latest_rmse,
        },
        {
            "Section": "Latest benchmark",
            "Metric": "MAE",
            "Value": latest_mae,
        },

        {
            "Section": "Active learning",
            "Metric": "Prediction runs",
            "Value": prediction_run_count,
        },
        {
            "Section": "Active learning",
            "Metric": "Selection decisions recorded",
            "Value": selection_count,
        },
        {
            "Section": "Active learning",
            "Metric": "Selected decisions",
            "Value": selected_count,
        },
        {
            "Section": "Active learning",
            "Metric": "Skipped decisions",
            "Value": skipped_count,
        },
        {
            "Section": "Active learning",
            "Metric": "Prediction-vs-real outcomes",
            "Value": prediction_outcome_count,
        },
        {
            "Section": "Active learning",
            "Metric": "Simulation queue entries",
            "Value": simulation_queue_count,
        },

        {
            "Section": "Files",
            "Metric": "SQLite database",
            "Value": str(ACTIVE_LEARNING_DB),
        },
        {
            "Section": "Files",
            "Metric": "Excel overview",
            "Value": str(ACTIVE_LEARNING_XLSX),
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    # =====================================================================
    # DATABASE TABLES / HUMAN-READABLE VIEWS
    # =====================================================================

    queries = {

        "Real_Observations": """
            SELECT
                o.observation_id,
                o.id AS ID,
                p.smiles AS SMILES,
                p.m1 AS M1,
                p.m2 AS M2,
                p.m3 AS M3,
                o.dapl_bacteria AS DAPL_bacteria,
                o.dapl_mammalian AS DAPL_mammalian,
                o.ddapl AS DDAPL,
                o.preferred AS Preferred,
                o.source_run AS Source_Run,
                o.relax_complete AS Relax_Complete,
                o.analysis_complete AS Analysis_Complete,
                o.bacteria_system_dir AS Bacteria_System_Dir,
                o.mammalian_system_dir AS Mammalian_System_Dir,
                o.imported_at AS Imported_At,
                o.last_seen_at AS Last_Seen_At
            FROM observations o
            LEFT JOIN peptoids p
                ON p.id = o.id
            ORDER BY
                o.preferred DESC,
                o.ddapl DESC,
                o.id,
                o.source_run
        """,

        "Models": """
            SELECT *
            FROM models
            WHERE COALESCE(archived,0)=0
            ORDER BY
                cycle,
                strategy,
                model_id
        """,

        "Historical_Models": """
            SELECT *
            FROM models
            WHERE COALESCE(archived,0)=1
            ORDER BY
                cycle,
                strategy,
                model_id
        """,

        "Model_Datasets": """
            SELECT
                dm.model_id,
                m.cycle,
                m.strategy,
                dm.id AS ID,
                p.smiles AS SMILES,
                p.m1 AS M1,
                p.m2 AS M2,
                p.m3 AS M3,
                dm.role,
                dm.acquisition_cycle,
                dm.acquisition_strategy,
                o.ddapl AS Actual_DDAPL
            FROM dataset_membership dm
            LEFT JOIN models m
                ON m.model_id = dm.model_id
            LEFT JOIN peptoids p
                ON p.id = dm.id
            LEFT JOIN observations o
                ON o.id = dm.id
               AND o.preferred = 1
            WHERE COALESCE(m.archived,0)=0
            ORDER BY
                m.cycle,
                m.strategy,
                dm.role,
                o.ddapl DESC,
                dm.id
        """,

        "Benchmark": """
            SELECT
                sm.split_version,
                sm.id AS ID,
                p.smiles AS SMILES,
                p.m1 AS M1,
                p.m2 AS M2,
                p.m3 AS M3,
                sm.role,
                sm.target_at_split AS DDAPL_at_Split,
                sm.target_rank AS DDAPL_Rank,
                sm.rank_block AS Rank_Block
            FROM split_membership sm
            LEFT JOIN peptoids p
                ON p.id = sm.id
            WHERE
                sm.split_version = 'active_model0_v1'
            ORDER BY
                sm.target_rank
        """,

        "Selection_History": """
            SELECT
                d.cycle AS Cycle,
                d.strategy AS Strategy,
                d.source_model_id AS Source_Model,
                d.id AS ID,
                p.smiles AS SMILES,
                p.m1 AS M1,
                p.m2 AS M2,
                p.m3 AS M3,
                d.predicted_ddapl AS Predicted_DDAPL,
                po.actual_ddapl AS Actual_DDAPL,
                po.residual AS Residual,
                po.absolute_error AS Absolute_Error,
                d.raw_prediction_rank AS Prediction_Rank,
                d.selection_group AS Selection_Group,
                d.group_rank AS Group_Rank,
                d.target_value AS Target_Value,
                d.decision_order AS Decision_Order,
                d.decision AS Decision,
                d.reason AS Reason,
                d.reason_detail AS Reason_Detail,
                d.prediction_run_id AS Prediction_Run_ID,
                d.created_at AS Created_At
            FROM selection_decisions d
            LEFT JOIN peptoids p
                ON p.id = d.id
            LEFT JOIN prediction_outcomes po
                ON po.decision_id = d.decision_id
            WHERE d.strategy IN ('A','B','C','D','E','F','G')
            ORDER BY
                d.cycle,
                d.strategy,
                d.decision_order
        """,

        "Historical_Selections": """
            SELECT
                d.cycle AS Cycle,
                d.strategy AS Strategy,
                d.source_model_id AS Source_Model,
                d.id AS ID,
                p.smiles AS SMILES,
                d.predicted_ddapl AS Predicted_DDAPL,
                po.actual_ddapl AS Actual_DDAPL,
                po.residual AS Residual,
                po.absolute_error AS Absolute_Error,
                d.raw_prediction_rank AS Prediction_Rank,
                d.selection_group AS Selection_Group,
                d.decision AS Decision,
                d.reason AS Reason,
                d.reason_detail AS Reason_Detail,
                d.prediction_run_id AS Prediction_Run_ID,
                d.created_at AS Created_At
            FROM selection_decisions d
            LEFT JOIN peptoids p ON p.id=d.id
            LEFT JOIN prediction_outcomes po ON po.decision_id=d.decision_id
            WHERE d.strategy IN ('X','Y','F_legacy')
            ORDER BY d.cycle,d.strategy,d.decision_order
        """,

        "Prediction_Outcomes": """
            SELECT
                d.cycle AS Cycle,
                d.strategy AS Strategy,
                d.source_model_id AS Source_Model,
                d.id AS ID,
                p.smiles AS SMILES,
                po.predicted_ddapl AS Predicted_DDAPL,
                po.actual_ddapl AS Actual_DDAPL,
                po.residual AS Residual,
                po.absolute_error AS Absolute_Error,
                d.raw_prediction_rank AS Prediction_Rank,
                d.selection_group AS Selection_Group,
                d.decision AS Decision,
                d.reason AS Reason,
                d.prediction_run_id AS Prediction_Run_ID,
                po.observation_id AS Observation_ID,
                po.resolved_at AS Resolved_At
            FROM prediction_outcomes po
            JOIN selection_decisions d
                ON d.decision_id = po.decision_id
            LEFT JOIN peptoids p
                ON p.id = d.id
            WHERE d.strategy IN ('A','B','C','D','E','F','G')
            ORDER BY
                d.cycle,
                d.strategy,
                d.decision_order
        """,

        "Simulation_Status": """
            SELECT
                sq.cycle AS Cycle,
                sq.id AS ID,
                p.smiles AS SMILES,
                p.m1 AS M1,
                p.m2 AS M2,
                p.m3 AS M3,
                sq.requested_by AS Requested_By,
                sq.status AS Status,
                o.ddapl AS Actual_DDAPL,
                sq.created_at AS Created_At,
                sq.updated_at AS Updated_At
            FROM simulation_queue sq
            LEFT JOIN peptoids p
                ON p.id = sq.id
            LEFT JOIN observations o
                ON o.id = sq.id
               AND o.preferred = 1
            WHERE COALESCE(sq.archived,0)=0
            ORDER BY
                sq.cycle,
                sq.status,
                o.ddapl DESC,
                sq.id
        """,

        "Historical_Simulation_Status": """
            SELECT
                sq.cycle AS Cycle,
                sq.id AS ID,
                p.smiles AS SMILES,
                sq.requested_by AS Original_Requested_By,
                sq.status AS Status,
                sq.archive_reason AS Archive_Reason,
                o.ddapl AS Actual_DDAPL,
                sq.created_at AS Created_At,
                sq.updated_at AS Updated_At
            FROM simulation_queue sq
            LEFT JOIN peptoids p ON p.id=sq.id
            LEFT JOIN observations o ON o.id=sq.id AND o.preferred=1
            WHERE COALESCE(sq.archived,0)=1
            ORDER BY sq.cycle,sq.id
        """,

        "Prediction_Runs": """
            SELECT pr.*
            FROM prediction_runs pr
            JOIN models m ON m.model_id=pr.model_id
            WHERE COALESCE(m.archived,0)=0
            ORDER BY pr.created_at
        """,
    }

    # =====================================================================
    # WRITE WORKBOOK
    # =====================================================================

    with pd.ExcelWriter(
        ACTIVE_LEARNING_XLSX,
        engine="openpyxl",
    ) as writer:

        # Summary is deliberately first.
        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        for sheet_name, query in queries.items():

            df = pd.read_sql_query(
                query,
                con,
            )

            df.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False,
            )

        # =================================================================
        # BASIC EXCEL FORMATTING
        # =================================================================

        workbook = writer.book

        for worksheet in workbook.worksheets:

            worksheet.freeze_panes = "A2"

            if worksheet.max_row >= 1:
                worksheet.auto_filter.ref = (
                    worksheet.dimensions
                )

            # Make columns readable without making very long path/SMILES
            # columns absurdly wide.
            for column_cells in worksheet.columns:

                column_letter = (
                    column_cells[0].column_letter
                )

                maximum_length = 0

                for cell in column_cells[:200]:

                    if cell.value is None:
                        continue

                    maximum_length = max(
                        maximum_length,
                        len(str(cell.value)),
                    )

                adjusted_width = min(
                    max(maximum_length + 2, 12),
                    45,
                )

                worksheet.column_dimensions[
                    column_letter
                ].width = adjusted_width

    print(
        f"Excel overview updated: "
        f"{ACTIVE_LEARNING_XLSX}"
    )


def print_status(con: sqlite3.Connection) -> None:
    # main() has already synchronized run artifacts/outcomes before status.
    preferred = con.execute(
        "SELECT COUNT(*) FROM observations WHERE preferred = 1"
    ).fetchone()[0]
    total_observations = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    models = con.execute(
        "SELECT * FROM models WHERE COALESCE(archived,0)=0 ORDER BY cycle, strategy"
    ).fetchall()
    archived_models = int(con.execute(
        "SELECT COUNT(*) FROM models WHERE COALESCE(archived,0)=1"
    ).fetchone()[0])

    print("=" * 80)
    print("PFPEPTOIDS ACTIVE-LEARNING STATUS")
    print("=" * 80)
    print(f"Real preferred unique peptoids: {preferred}")
    print(f"Historical/duplicate observation pairs: {total_observations}")
    print(f"Model-0 target baseline: {EXPECTED_MODEL0_UNIQUE_PEPTOIDS}")
    if archived_models:
        print(
            f"Historical models quarantined: {archived_models} "
            "(excluded from active workflow; see Historical_Models in Excel)"
        )
    print()

    if not models:
        if int(preferred) < EXPECTED_MODEL0_UNIQUE_PEPTOIDS:
            print(
                f"Next action: complete/sync the remaining Model-0 simulation(s) "
                f"({preferred}/{EXPECTED_MODEL0_UNIQUE_PEPTOIDS} currently ready)."
            )
            print("Then run: python workflow.py sync")
            print("Then:     python workflow.py prepare-model0")
        elif int(preferred) == EXPECTED_MODEL0_UNIQUE_PEPTOIDS:
            print("Next action: python workflow.py prepare-model0")
        else:
            print(
                "More than the configured Model-0 baseline exists before Model0 was locked. "
                "Review before preparing Model0."
            )
        return

    print("Models:")
    for model in models:
        print(
            f"    {model['model_id']:<8} status={model['status']:<20} "
            f"train={model['n_train']} benchmark={model['n_benchmark']} "
            f"winner={(model['chosen_representation'] or '-') + '+' + (model['chosen_algorithm'] or '-')} "
            f"R2={model['benchmark_r2'] if model['benchmark_r2'] is not None else '-'}"
        )

    queue = pd.read_sql_query(
        "SELECT cycle, status, COUNT(*) AS n FROM simulation_queue "
        "WHERE COALESCE(archived,0)=0 GROUP BY cycle, status ORDER BY cycle, status",
        con,
    )
    if not queue.empty:
        print()
        print("Simulation queue:")
        print(queue.to_string(index=False))


# =============================================================================
# HIGH-LEVEL SYNC / VALIDATE
# =============================================================================


def sync_everything() -> None:
    initialise_database()
    validate_monomer_configuration()
    descriptor_df, feature_columns = load_and_validate_descriptor_universe()

    with connect_db() as con:
        sync_peptoids(con, descriptor_df)
        total_pairs, preferred_count = sync_observations(con)
        sync_run_artifacts(con)
        outcomes = refresh_prediction_outcomes(con)
        queue_cycles = rebuild_all_canonical_simulation_queues(con)
        for cycle in queue_cycles:
            export_cycle_selection_files(con, cycle)
        export_excel(con)

    print("=" * 80)
    print("SYNC COMPLETE")
    print("=" * 80)
    print(f"Descriptor universe: {len(descriptor_df):,} sequences")
    print(f"Features:            {len(feature_columns)}")
    print(f"Complete pair records: {total_pairs}")
    print(f"Preferred unique real peptoids: {preferred_count}")
    print(f"Prediction outcomes currently resolvable: {outcomes}")
    print(f"Canonical simulation queues reconciled: {len(queue_cycles)} cycle(s)")
    if preferred_count < EXPECTED_MODEL0_UNIQUE_PEPTOIDS:
        print(
            f"Model 0 is waiting for {EXPECTED_MODEL0_UNIQUE_PEPTOIDS - preferred_count} "
            "additional unique real peptoid(s)."
        )


def validate_controller_state() -> None:
    initialise_database()
    validate_monomer_configuration()
    df, features = load_and_validate_descriptor_universe()

    required_files = [
        MASTERDATAANALYSIS_CSV,
        TRAINING_SCRIPT,
        PREDICTION_SCRIPT,
        TRAINING_SLURM_SCRIPT,
        PREDICTION_SLURM_SCRIPT,
        PYTHON_EXECUTABLE,
    ]
    missing = [str(p) for p in required_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Required workflow files missing:\n" + "\n".join(missing))

    print("CONTROLLER VALIDATION PASS")
    print(f"    descriptors: {len(df):,}")
    print(f"    features:    {len(features)}")
    print(f"    monomers:    {len(KNOWN_MONOMERS)}")
    print(f"    database:    {ACTIVE_LEARNING_DB}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PFPeptoids position-resolved active-learning controller"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate")
    sub.add_parser("sync")
    sub.add_parser("status")
    sub.add_parser("export")
    sub.add_parser("prepare-model0")
    sub.add_parser("train-model0")
    sub.add_parser("predict-model0")

    p_prepare = sub.add_parser("prepare-model")
    p_prepare.add_argument("--model", required=True)

    p_train = sub.add_parser("train-model")
    p_train.add_argument("--model", required=True)

    p_predict = sub.add_parser("predict-model")
    p_predict.add_argument("--model", required=True)

    p_select = sub.add_parser(
        "select",
        help=(
            "Generate canonical A-E selections for a cycle. By default all A-E "
            "are selected; use --strategies to advance only a reviewed subset."
        ),
    )
    p_select.add_argument("--cycle", type=int, required=True)
    p_select.add_argument(
        "--strategies",
        nargs="+",
        choices=ACTIVE_STRATEGIES,
        default=None,
        help="Optional subset, e.g. --strategies A B D. Existing strategy history is never overwritten.",
    )

    p_select_f = sub.add_parser(
        "select-f",
        help=(
            "Generate 15 F selections from prediction-complete Model<cycle>F. "
            "Later F models learn from these selections."
        ),
    )
    p_select_f.add_argument("--cycle", type=int, required=True)

    p_select_g = sub.add_parser(
        "select-g",
        help=(
            "Generate 15 G control selections from prediction-complete Model<cycle>G. "
            "G begins at cycle 2 and its own selections never feed back into G training."
        ),
    )
    p_select_g.add_argument("--cycle", type=int, required=True)

    p_queue = sub.add_parser("simulation-queue")
    p_queue.add_argument("--cycle", type=int, required=True)

    p_config = sub.add_parser("model-config")
    p_config.add_argument("--model", required=True)

    p_migrate = sub.add_parser(
        "migrate-midpoint-v2",
        help=(
            "Plan/apply the controlled Cycle-1 migration: old C/E -> X/Y, "
            "corrected midpoint C/E, legacy premature F preservation, and F/G fork rules."
        ),
    )
    p_migrate.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the migration. Without --apply the command only plans and "
            "writes a plan JSON."
        ),
    )

    p_compare = sub.add_parser(
        "compare-models",
        help=(
            "Compare prediction landscapes and selections across all "
            "prediction-complete canonical models in one generation."
        ),
    )
    p_compare.add_argument(
        "--generation",
        type=int,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    # A migration dry-run is deliberately database read-only. It may write only
    # its timestamped plan JSON under active_learning/migrations/.
    if args.command == "migrate-midpoint-v2" and not bool(args.apply):
        if not ACTIVE_LEARNING_DB.is_file():
            raise FileNotFoundError(
                f"Active-learning database missing: {ACTIVE_LEARNING_DB}"
            )

        con = sqlite3.connect(
            f"file:{ACTIVE_LEARNING_DB}?mode=ro",
            uri=True,
        )
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")

        try:
            migrate_midpoint_strategy_v2(
                con,
                apply=False,
            )
        finally:
            con.close()

        return

    # Model-agreement analysis is also deliberately database read-only.
    # It may regenerate derived analysis files, but must not modify workflow state.
    if args.command == "compare-models":
        if not ACTIVE_LEARNING_DB.is_file():
            raise FileNotFoundError(
                f"Active-learning database missing: {ACTIVE_LEARNING_DB}"
            )

        con = sqlite3.connect(
            f"file:{ACTIVE_LEARNING_DB}?mode=ro",
            uri=True,
        )
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")

        try:
            compare_model_generation(
                con,
                args.generation,
            )
        finally:
            con.close()

        return

    initialise_database()

    if args.command == "validate":
        validate_controller_state()
        return

    if args.command == "sync":
        sync_everything()
        return

    with connect_db() as con:
        sync_run_artifacts(con)
        refresh_prediction_outcomes(con)

        if args.command == "status":
            print_status(con)

        elif args.command == "export":
            export_excel(con)

        elif args.command == "model-config":
            cycle, strategy = parse_model_id(args.model)
            candidates, representations = resolve_model_search_policy(
                con,
                args.model,
                cycle,
                strategy,
            )
            print(f"{args.model} resolved configuration")
            print(f"    models: {candidates}")
            print(f"    representations: {representations}")
            print(f"    config file: {MODEL_SEARCH_CONFIG_PATH}")

        elif args.command == "prepare-model0":
            descriptor_df, _ = load_and_validate_descriptor_universe()
            sync_peptoids(con, descriptor_df)
            sync_observations(con)
            prepare_logical_model(con, "Model0")
            export_excel(con)

        elif args.command == "train-model0":
            submit_training(con, "Model0")
            export_excel(con)

        elif args.command == "predict-model0":
            submit_prediction(con, "Model0")
            export_excel(con)

        elif args.command == "prepare-model":
            descriptor_df, _ = load_and_validate_descriptor_universe()
            sync_peptoids(con, descriptor_df)
            sync_observations(con)
            refresh_prediction_outcomes(con)
            prepare_logical_model(con, args.model)
            export_excel(con)

        elif args.command == "train-model":
            submit_training(con, args.model)
            export_excel(con)

        elif args.command == "predict-model":
            submit_prediction(con, args.model)
            export_excel(con)

        elif args.command == "select":
            generate_cycle_selections(
                con,
                args.cycle,
                args.strategies,
            )
            export_excel(con)

        elif args.command == "select-f":
            generate_f_selection(
                con,
                args.cycle,
            )
            export_excel(con)

        elif args.command == "select-g":
            generate_g_selection(
                con,
                args.cycle,
            )
            export_excel(con)

        elif args.command == "simulation-queue":
            export_cycle_selection_files(
                con,
                args.cycle,
            )
            path = (
                SIMULATION_QUEUE_DIR
                / f"cycle_{args.cycle}_simulation_queue.csv"
            )
            print(path)

        elif args.command == "migrate-midpoint-v2":
            migrate_midpoint_strategy_v2(
                con,
                apply=bool(args.apply),
            )

        else:
            raise ValueError(
                f"Unhandled command {args.command}"
            )


if __name__ == "__main__":
    main()
