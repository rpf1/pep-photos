#!/usr/bin/env python3
"""
PFPeptoids manual image annotation.

Human classification only. No ML, simulation, image generation or source writes.

Commands
--------
python annotate.py --package image_classification
python image_classification/scripts/annotate.py --discover
python image_classification/scripts/annotate.py --limit 3
python image_classification/scripts/annotate.py

Other commands
--------------
python annotate.py --init
python annotate.py --self-test
python annotate.py --config /path/annotation_config.json

Dependencies
------------
Pillow; Python's tkinter module for the GUI.
The storage/discovery tests do not require a graphical display.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import socket
import sqlite3
import sys
import tempfile
import time
import traceback
import uuid
import zipfile

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


VERSION = "2.1"
SCHEMA_VERSION = 2

DEFAULT_CONFIG = {
    "image_root": (
        "/users/pfb19164/Desktop/post_medical_leave/"
        "PFPeptoids/PFPeptoids/analysis/Photos"
    ),
    # Relative paths resolve against the configuration file's directory.
    "output_dir": "membrane_session",
    "classes": [
        {
            "name": "curved",
            "key": "1",
            "description": "Apply the agreed manual classification protocol.",
        },
        {
            "name": "flat_interaction",
            "key": "2",
            "description": "Apply the agreed manual classification protocol.",
        },
        {
            "name": "flat_no_interaction",
            "key": "3",
            "description": "Apply the agreed manual classification protocol.",
        },
    ],
    "targets": ["system"],  # Fixed internal target; never composite.
    "included_bilayers": ["POPC60CHOL40_W_WF", "POPC80POPS20_W_WF", "POPE75POPG25_W_WF"],
    "bilayer_to_membrane_role": {
        "POPC60CHOL40_W_WF": "mammalian_like",
        "POPC80POPS20_W_WF": "POPC80POPS20",
        "POPE75POPG25_W_WF": "bacterial_like",
    },
    "supported_image_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
    "keyboard_shortcuts": {"previous_picture": "bracketleft", "next_picture": "bracketright"},
    "order_mode": "ordered",
    "random_seed": 34,
    "window_size": [1400, 950],
    "default_view": "fit",
    "show_notes": True,
    "backup_every_decisions": 10,
    "backup_before_edit": True,
    "confirm_relabel": True,
    "decision_cooldown_ms": 300,
    "annotation_protocol": (
        "One label per molecule × bilayer_type × run. Inspect the available views. "
        "Use uncertain when evidence is ambiguous. Membranes are independent."
    ),
}

CSV_FIELDS = [
    "system_key", "molecule", "molecule_type", "bilayer_type", "membrane_role", "run",
    "label", "annotation_status", "note", "annotation_timestamp", "last_modified",
    "source_directory", "relative_path", "image_count", "image_paths",
    "evidence_sha256", "config_sha256", "event_id", "target", "members",
]
ORDER_FIELDS = ["position", "system_key", "molecule", "bilayer_type", "run", "relative_path"]
DISCOVERY_FIELDS = ["system_key", "molecule", "bilayer_type", "membrane_role", "run",
                    "relative_path", "image_count", "image_paths", "evidence_sha256",
                    "discovery_status", "problem"]
IMAGE_FIELDS = ["system_key", "relative_path", "sha256", "width", "height", "status", "problem"]

STATUSES = {"unlabelled", "labelled", "uncertain", "skipped"}
RESERVED_KEYS = {"u", "s", "n", "p", "f", "z", "r", "0", "+", "-", "="}
FILTERS = ["All", "Unlabelled", "Labelled", "Uncertain", "Skipped"]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def system_key(relative_path):
    # Root-independent identity: a copied Photos tree retains the same IDs.
    # Content hash is checked separately; identical images at distinct paths
    # remain distinct records.
    return digest_bytes(relative_path.encode("utf-8"))


def fsync_directory(directory):
    if os.name != "posix":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some network/filesystem implementations reject directory fsync.
        # Files are still fsynced before atomic replacement.
        pass


def atomic_write(path, data):
    """Write a complete file in the same directory, then atomically replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def csv_bytes(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required: python -m pip install Pillow"
        ) from exc
    return Image


def resolve_path(value, base):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def check_separate_paths(root, output):
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError(
            "Image root and annotation output must be separate directory "
            "trees; neither may contain the other."
        )


def validate_config(config):
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration settings: {sorted(unknown)}")

    if config["order_mode"] not in {"ordered", "random"}:
        raise ValueError("order_mode must be ordered or random.")
    if config["default_view"] not in {"fit", "actual"}:
        raise ValueError("default_view must be fit or actual.")

    if config["targets"] != ["system"]:
        raise ValueError("Legacy composite configuration rejected. Use the v2 template and a separate output directory.")
    included = config["included_bilayers"]
    if (not isinstance(included, list) or not included
            or any(not isinstance(b, str) or not b or b in {".", "..", "STACKED"} or "/" in b or "\\" in b for b in included)
            or len(included) != len(set(included))):
        raise ValueError("included_bilayers must contain unique exact directory names; STACKED is not a physical bilayer.")
    mapping = config["bilayer_to_membrane_role"]
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k,v in mapping.items()):
        raise ValueError("bilayer_to_membrane_role must map exact directory names to explicit roles.")
    ext = config["supported_image_extensions"]
    if not isinstance(ext, list) or not ext or any(not isinstance(e,str) or not re.fullmatch(r"\.[A-Za-z0-9]+", e) for e in ext):
        raise ValueError("supported_image_extensions must contain dotted extensions.")
    shortcuts = config["keyboard_shortcuts"]
    if set(shortcuts) != {"previous_picture", "next_picture"}:
        raise ValueError("Specify previous_picture and next_picture shortcuts.")
    values = list(shortcuts.values())
    forbidden = RESERVED_KEYS | {"left", "right", "plus", "minus", "equal", "kp_add", "kp_subtract"}
    if len(set(values)) != 2 or any(not isinstance(k,str) or not k or k != k.lower() or k in forbidden for k in values):
        raise ValueError("Picture shortcuts must be unique lowercase Tk keysyms without control conflicts.")

    classes = config["classes"]
    names, keys = [], []
    for item in classes:
        name, key = item.get("name", ""), item.get("key", "").lower()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid class name: {name!r}")
        if name in {"uncertain", "skip", "skipped", "unlabelled"}:
            raise ValueError("Special statuses must not be scientific classes.")
        if len(key) != 1 or not key.isalnum() or key in RESERVED_KEYS or key in config["keyboard_shortcuts"].values():
            raise ValueError(f"Invalid or reserved class shortcut: {key!r}")
        names.append(name)
        keys.append(key)
    if not classes or len(names) != len(set(names)) or len(keys) != len(set(keys)):
        raise ValueError("Scientific classes and shortcuts must be unique.")

    for name in ("backup_every_decisions", "decision_cooldown_ms"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config["random_seed"], int):
        raise ValueError("random_seed must be an integer.")
    if (
        len(config["window_size"]) != 2
        or min(config["window_size"]) < 400
    ):
        raise ValueError("window_size must contain two dimensions >= 400.")


def load_config(path, root_override=None, output_override=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {path}\n"
            "Create it with --init or --package."
        )
    supplied = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config.update(supplied)
    if root_override:
        config["image_root"] = root_override
    if output_override:
        config["output_dir"] = output_override
    validate_config(config)

    config["image_root"] = str(resolve_path(config["image_root"], path.parent))
    config["output_dir"] = str(resolve_path(config["output_dir"], path.parent))
    check_separate_paths(
        Path(config["image_root"]), Path(config["output_dir"])
    )
    return config


def inspect_system(directory, root, config):
    """Read direct image members only. Nested directories are ambiguous."""
    Image = require_pillow()
    relative = directory.relative_to(root).as_posix()
    molecule, bilayer, run = directory.relative_to(root).parts
    key = digest_bytes(json.dumps([molecule, bilayer, run], ensure_ascii=False).encode())
    members, problems = [], []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.is_symlink():
            problems.append(f"Symlink not followed: {path.name}")
            continue
        if path.is_dir():
            problems.append(f"Nested directory requires review: {path.name}")
            continue
        if not path.is_file():
            problems.append(f"Nonregular file not read: {path.name}")
            continue
        if path.suffix.lower() not in {e.lower() for e in config["supported_image_extensions"]}:
            continue
        member = dict(system_key=key, relative_path=path.relative_to(root).as_posix(),
                      sha256="", width="", height="", status="ready", problem="")
        try:
            raw = path.read_bytes()
            member["sha256"] = digest_bytes(raw)
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                member["width"], member["height"] = im.size
                if getattr(im, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images require an explicit frame-selection rule.")
        except Exception as exc:
            member.update(status="unreadable", problem=str(exc))
            problems.append(f"{path.name}: {exc}")
        members.append(member)
    role = config["bilayer_to_membrane_role"].get(bilayer, "unknown")
    if role == "unknown":
        problems.append("Unknown bilayer: add an explicit role mapping before annotation.")
    if not members:
        problems.append("No supported image files in system directory.")
    evidence = [[m["relative_path"], m["sha256"]] for m in members]
    return dict(system_key=key, molecule=molecule, molecule_type="", bilayer_type=bilayer,
                membrane_role=role, run=run, relative_path=relative,
                source_directory=str(directory), image_count=len(members),
                image_paths=json.dumps([m["relative_path"] for m in members], ensure_ascii=False),
                evidence_sha256=digest_bytes(json.dumps(evidence, ensure_ascii=False).encode()),
                members=members, discovery_status="blocked" if problems else "ready",
                problem="; ".join(problems))


def discover(root, config=None):
    """Read-only audit of the supplied three-directory hierarchy; never guess panels."""
    config = config or DEFAULT_CONFIG
    root = Path(root).resolve()
    report = dict(generated_at=utc_now(), image_root=str(root), root_available=root.is_dir(),
                  audit_basis="Provided hierarchy; actual dataset must be checked on its host.",
                  images_found=None, physical_systems_found=None, readable_images=None,
                  unreadable_images=None, molecule_directories=0, issues=[], malformed_paths=[],
                  legacy_stacked=[], empty_run_directories=[], duplicate_paths=[],
                  duplicate_logical_records=[], files_in_multiple_groups=[], exact_content_duplicates=[])
    if not root.is_dir():
        report["issues"].append({"path":str(root), "problem":"Root unavailable; counts unknown."})
        return [], report
    extensions, patterns = Counter(), Counter()
    bilayers, runs, molecules = set(), set(), set()
    # Full walk audits unexpected levels and legacy content, without following links.
    for folder, dirs, files in os.walk(root, followlinks=False,
            onerror=lambda e: report["issues"].append({"path":e.filename,"problem":str(e)})):
        base = Path(folder)
        for name in list(dirs):
            p = base/name
            if p.is_symlink():
                dirs.remove(name)
                report["malformed_paths"].append({"path":str(p),"problem":"Symlink directory not followed"})
        rel = base.relative_to(root).parts
        if 0 < len(rel) < 3 and not dirs and "STACKED" not in rel:
            report["malformed_paths"].append({"path":str(base), "problem":"Directory stops before a physical-system run level"})
        if "STACKED" in rel:
            report["legacy_stacked"].append(str(base))
        for name in files:
            p=base/name
            extensions[p.suffix.lower() or "<none>"] += 1
            patterns[re.sub(r"\d+", "#", name)] += 1
            if p.is_symlink() or len(rel)!=3 or "STACKED" in rel:
                report["malformed_paths"].append({"path":str(p),"problem":"Symlink or outside physical-system hierarchy"})
        if len(rel)>3:
            report["malformed_paths"].append({"path":str(base),"problem":"Nested directory below run"})
    records=[]
    excluded_directories=[]
    excluded_systems=[]
    for mol in sorted(root.iterdir()):
        if mol.is_symlink() or not mol.is_dir(): continue
        molecules.add(mol.name)
        for bil in sorted(mol.iterdir()):
            if not bil.is_dir(): continue
            bilayers.add(bil.name)
            if bil.name not in config["included_bilayers"] or bil.is_symlink():
                excluded_directories.append({"path": str(bil), "bilayer_type": bil.name,
                    "reason": "symlink_not_followed" if bil.is_symlink() else "not_in_included_bilayers"})
            if bil.is_symlink() or bil.name == "STACKED": continue
            for run in sorted(bil.iterdir()):
                if run.is_symlink() or not run.is_dir(): continue
                runs.add(run.name)
                try: record=inspect_system(run,root,config)
                except OSError as exc:
                    report["issues"].append({"path":str(run),"problem":str(exc)})
                    continue
                if bil.name in config["included_bilayers"]:
                    records.append(record)
                else:
                    excluded_systems.append({k: record[k] for k in DISCOVERY_FIELDS})
                if not record["image_count"]: report["empty_run_directories"].append(str(run))
                if record["problem"]: report["issues"].append({"path":str(run),"problem":record["problem"]})
    members=[m for r in records for m in r["members"]]
    def duplicates(field, rows):
        groups=defaultdict(list)
        for row in rows:
            if row.get(field): groups[row[field]].append(row["relative_path"])
        return [{"key":k,"paths":v} for k,v in groups.items() if len(v)>1]
    report.update(molecule_directories=len(molecules), distinct_bilayer_names=sorted(bilayers),
        included_bilayers=sorted(bilayers & set(config["included_bilayers"])),
        excluded_bilayers=sorted(bilayers - set(config["included_bilayers"])),
        configured_included_bilayers=list(config["included_bilayers"]),
        excluded_directories=excluded_directories, excluded_systems=excluded_systems,
        excluded_systems_found=len(excluded_systems),
        distinct_run_names=sorted(runs), extensions_present=dict(extensions),
        common_filename_patterns=patterns.most_common(30), images_found=len(members),
        total_candidate_image_files=sum(n for e,n in extensions.items() if e in {x.lower() for x in config["supported_image_extensions"]}),
        physical_systems_found=len(records), readable_images=sum(m["status"]=="ready" for m in members),
        unreadable_images=sum(m["status"]!="ready" for m in members),
        unreadable_image_details=[m for m in members if m["status"]!="ready"],
        systems_by_bilayer_type=dict(Counter(r["bilayer_type"] for r in records)),
        systems_by_membrane_role=dict(Counter(r["membrane_role"] for r in records)),
        systems_by_run=dict(Counter(r["run"] for r in records)),
        images_per_system_distribution=dict(Counter(r["image_count"] for r in records)),
        unknown_bilayer_types=sorted(bilayers-set(config["bilayer_to_membrane_role"])),
        duplicate_paths=duplicates("relative_path",members),
        files_in_multiple_groups=duplicates("relative_path",members),
        duplicate_logical_records=duplicates("system_key",records),
        exact_content_duplicates=duplicates("sha256",members))
    if report["duplicate_logical_records"] or report["files_in_multiple_groups"]:
        raise RuntimeError("Ambiguous duplicated system membership; registration refused.")
    return records, report


class SessionLock:
    """OS-managed single-writer lock; automatically released if process exits."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self):
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0, os.SEEK_END)
                if self.handle.tell() == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(
                    self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                "Another annotation session holds this output directory's lock. "
                "Close that session; do not run two writers."
            ) from exc

    def release(self):
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


class SavedButExportFailed(RuntimeError):
    pass


def guard_legacy(output):
    database = output / "state.sqlite3"
    if not database.exists(): return
    conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        try:
            version = json.loads(conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0])
        except Exception:
            version = None
        if version == SCHEMA_VERSION: return
        if version != 1:
            raise RuntimeError("Unknown state schema; preserved unchanged. Use a separate output directory.")
        legacy = output / "legacy"
        legacy.mkdir(exist_ok=True)
        destination = legacy / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_composite_" + uuid.uuid4().hex[:6] + ".zip")
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite3"
            copy = sqlite3.connect(db)
            try: conn.backup(copy)
            finally: copy.close()
            with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
                archive.write(db, "state.sqlite3")
                for name in ("annotations.csv", "annotation_order.csv", "annotation_config.json"):
                    path = output/name
                    if path.is_file(): archive.write(path,name)
                archive.writestr("LEGACY.md", "Legacy image/composite labels: provenance only; not membrane-specific scientific truth. No automatic migration.\n")
        raise RuntimeError(f"Legacy v1 state backed up to {destination}. Original state retained. Use the v2 config with a NEW output directory.")
    finally:
        conn.close()


class Store:
    """
    Transactional recovery state plus immediate, atomic CSV exports.

    CSV/order files are checked before replacement. Unexpected manual edits
    cause a stop instead of being silently overwritten.
    """

    def __init__(self, config):
        self.config = config
        self.root = Path(config["image_root"])
        self.output = Path(config["output_dir"])
        check_separate_paths(self.root, self.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = SessionLock(self.output / ".annotation.lock")
        self.lock.acquire()
        self.conn = None
        self.config_text = json_text(config)
        self.config_hash = digest_bytes(self.config_text.encode("utf-8"))
        self.notices = []
        self.current_discovery = {}

        try:
            database = self.output / "state.sqlite3"
            guard_legacy(self.output)
            is_new = not database.exists()
            if is_new and any(
                (self.output / name).exists()
                for name in ("annotations.csv", "annotation_order.csv")
            ):
                raise RuntimeError(
                    "CSV/order files exist without their recovery database. "
                    "They were not overwritten. Restore the complete session "
                    "from a backup or use a new empty output directory."
                )
            self.conn = sqlite3.connect(database, timeout=5)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.execute("PRAGMA journal_mode=DELETE")
            result = self.conn.execute("PRAGMA quick_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError(f"Recovery database integrity failure: {result}")

            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS systems(
                    system_key TEXT PRIMARY KEY,
                    position INTEGER NOT NULL UNIQUE,
                    relative_path TEXT NOT NULL UNIQUE,
                    metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS annotations(
                    system_key TEXT NOT NULL REFERENCES systems(system_key),
                    target TEXT NOT NULL,
                    record TEXT NOT NULL,
                    PRIMARY KEY(system_key, target)
                );
                CREATE TABLE IF NOT EXISTS configurations(
                    sha256 TEXT PRIMARY KEY,
                    contents TEXT NOT NULL,
                    first_seen TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events(
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_key TEXT NOT NULL,
                    target TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    before_record TEXT NOT NULL,
                    after_record TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    undo_of INTEGER UNIQUE
                );
            """)

            settings = {
                "targets": config["targets"],
                "included_bilayers": sorted(config["included_bilayers"]),
                "order_mode": config["order_mode"],
                "random_seed": config["random_seed"],
                "bilayer_to_membrane_role": config["bilayer_to_membrane_role"],
                "supported_image_extensions": config["supported_image_extensions"],
            }
            if is_new:
                with self.conn:
                    self.set_meta("schema_version", SCHEMA_VERSION)
                    self.set_meta("settings", settings)
                    self.set_meta("dataset_id", str(uuid.uuid4()))
                    self.set_meta("tool_version", VERSION)
            else:
                if self.get_meta("schema_version") != SCHEMA_VERSION:
                    raise RuntimeError("Unrecognised annotation database schema.")
                if self.get_meta("settings") != settings:
                    raise RuntimeError(
                        "Session settings (including bilayer whitelist) differ. "
                        "Restore the saved settings or create a separate session; "
                        "existing labels/order were not reassigned."
                    )

            # Validate existing exported files BEFORE registration/export changes.
            self.verify_exports()
            with self.conn:
                self.conn.execute(
                    "INSERT OR IGNORE INTO configurations VALUES(?,?,?)",
                    (self.config_hash, self.config_text, utc_now()),
                )
            self.export()
        except Exception:
            self.close()
            raise

    def get_meta(self, key, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta VALUES(?,?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )

    def systems(self):
        return [
            (row["position"], json.loads(row["metadata"]))
            for row in self.conn.execute(
                "SELECT position,metadata FROM systems ORDER BY position"
            )
        ]

    def units(self):
        return [
            (record["system_key"], target)
            for _, record in self.systems()
            for target in self.config["targets"]
        ]

    def get(self, unit):
        row = self.conn.execute(
            "SELECT record FROM annotations WHERE system_key=? AND target=?", unit
        ).fetchone()
        if row is None:
            raise KeyError(unit)
        return json.loads(row["record"])

    def all_records(self):
        lookup = {
            (row["system_key"], row["target"]): json.loads(row["record"])
            for row in self.conn.execute(
                "SELECT system_key,target,record FROM annotations"
            )
        }
        return [lookup[u] for u in self.units()]

    def export_payloads(self):
        order = []
        for position, record in self.systems():
            order.append({"position": position, **record})
        return {
            "annotations.csv": csv_bytes(self.all_records(), [f for f in CSV_FIELDS if f != "members"]),
            "system_images.csv": csv_bytes([m for _, r in self.systems() for m in r["members"]], IMAGE_FIELDS),
            "annotation_order.csv": csv_bytes(order, ORDER_FIELDS),
        }

    def verify_exports(self):
        for name, expected_bytes in self.export_payloads().items():
            path = self.output / name
            previous_hash = self.get_meta("export_hash:" + name)
            current_hash = digest_bytes(expected_bytes)
            if path.exists():
                actual = file_digest(path)
                if actual not in {previous_hash, current_hash}:
                    raise RuntimeError(
                        f"{name} changed outside this tool or is corrupt. "
                        "It has NOT been overwritten. Preserve it and restore "
                        "a complete backup into a separate directory."
                    )
            elif previous_hash is not None:
                self.notices.append(
                    f"{name} was missing and was reconstructed from recovery state."
                )

    def export(self):
        self.verify_exports()
        payloads = self.export_payloads()
        for name, data in payloads.items():
            atomic_write(self.output / name, data)
        with self.conn:
            for name, data in payloads.items():
                self.set_meta("export_hash:" + name, digest_bytes(data))

    def register(self, records):
        if any(r["bilayer_type"] not in self.config["included_bilayers"] for r in records):
            raise RuntimeError("Excluded bilayer supplied to registration; nothing registered.")
        keys = [r["system_key"] for r in records]
        paths = [r["relative_path"] for r in records]
        members = [m["relative_path"] for r in records for m in r["members"]]
        if len(keys)!=len(set(keys)) or len(paths)!=len(set(paths)) or len(members)!=len(set(members)):
            raise RuntimeError("Duplicate system identity or image membership; nothing registered.")
        self.verify_exports()
        self.current_discovery = {r["system_key"]: r for r in records}
        existing = {
            row["system_key"]
            for row in self.conn.execute("SELECT system_key FROM systems")
        }
        new = sorted(
            (r for r in records if r["system_key"] not in existing),
            key=lambda r: r["relative_path"],
        )
        if self.config["order_mode"] == "random":
            rng = random.Random(self.config["random_seed"] + len(existing))
            rng.shuffle(new)

        with self.conn:
            for offset, record in enumerate(new, start=len(existing)):
                self.conn.execute(
                    "INSERT INTO systems VALUES(?,?,?,?)",
                    (
                        record["system_key"], offset, record["relative_path"],
                        json.dumps(record, ensure_ascii=False),
                    ),
                )
                for target in self.config["targets"]:
                    row = {field: "" for field in CSV_FIELDS}
                    row.update({k: record[k] for k in CSV_FIELDS if k in record})
                    row.update(
                        target=target,
                        annotation_status="unlabelled",
                        config_sha256=self.config_hash,
                    )
                    self.conn.execute(
                        "INSERT INTO annotations VALUES(?,?,?)",
                        (record["system_key"], target, json.dumps(row)),
                    )
        self.export()

    def source_problem(self, unit):
        row = self.get(unit)
        directory = self.root / row["relative_path"]
        try:
            relative = directory.relative_to(self.root)
            current = self.root
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise RuntimeError("Symlink in system path.")
            if self.root.resolve() not in directory.resolve().parents:
                raise RuntimeError("System directory escapes source root.")
            fresh = inspect_system(directory, self.root, self.config)
            if fresh["discovery_status"] != "ready": return fresh["problem"]
            if fresh["evidence_sha256"] != row["evidence_sha256"]:
                return "Image membership/content changed. Labels retained; decisions blocked. Restore evidence or start a separate session."
        except Exception as exc:
            return str(exc)
        return ""

    def verified_image_bytes(self, unit, picture_index=0):
        row=self.get(unit)
        problem=self.source_problem(unit)
        if problem: raise RuntimeError(problem)
        member=row["members"][picture_index]
        path=self.root/member["relative_path"]
        raw=path.read_bytes()
        if digest_bytes(raw)!=member["sha256"]:
            raise RuntimeError("Image changed after evidence verification.")
        return raw

    def _write_event(self, unit, before, after, kind, undo_of=None):
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO events
                   (system_key,target,kind,before_record,after_record,
                    timestamp,config_sha256,undo_of)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    *unit, kind, json.dumps(before), "{}",
                    utc_now(), self.config_hash, undo_of,
                ),
            )
            event_id = cursor.lastrowid
            after["event_id"] = str(event_id)
            self.conn.execute(
                "UPDATE annotations SET record=? WHERE system_key=? AND target=?",
                (json.dumps(after), *unit),
            )
            self.conn.execute(
                "UPDATE events SET after_record=? WHERE event_id=?",
                (json.dumps(after), event_id),
            )

        # The database transaction is already durable at this point.
        try:
            self.export()
            if event_id % self.config["backup_every_decisions"] == 0:
                self.backup("periodic")
        except Exception as exc:
            raise SavedButExportFailed(
                "The decision is saved in state.sqlite3, but CSV export or "
                f"backup failed: {exc}\nUse Save/backup before continuing."
            ) from exc
        return event_id

    def decide(self, unit, label="", status="unlabelled", note="", note_only=False):
        self.verify_exports()
        if status not in STATUSES:
            raise ValueError(f"Invalid annotation status: {status}")
        if len(note) > 20000:
            raise ValueError("Note exceeds 20,000 characters.")
        before = self.get(unit)

        if note_only:
            label, status = before["label"], before["annotation_status"]
        else:
            names = {item["name"] for item in self.config["classes"]}
            if status == "labelled":
                if label not in names:
                    raise ValueError("Label is not an active scientific class.")
                self.verified_image_bytes(unit)
            elif label:
                raise ValueError("Non-scientific statuses must have an empty label.")
            if status == "uncertain":
                self.verified_image_bytes(unit)

        if (label, status, note) == (
            before["label"], before["annotation_status"], before["note"]
        ):
            return None

        if (
            self.config["backup_before_edit"]
            and before["annotation_status"] != "unlabelled"
        ):
            self.backup("before_edit")

        now = utc_now()
        after = dict(before)
        after.update(
            label=label,
            annotation_status=status,
            note=note,
            annotation_timestamp=now,
            last_modified=now,
            source_directory=str(self.root / before["relative_path"]),
            config_sha256=self.config_hash,
        )
        return self._write_event(unit, before, after, "decision")

    def undo(self):
        self.verify_exports()
        event = self.conn.execute(
            """SELECT e.* FROM events e
               WHERE e.kind='decision' AND e.event_id > 228
                 AND NOT EXISTS(
                     SELECT 1 FROM events u WHERE u.undo_of=e.event_id
                 )
               ORDER BY e.event_id DESC LIMIT 1"""
        ).fetchone()
        if event is None:
            return None
        unit = (event["system_key"], event["target"])
        current = self.get(unit)
        expected = json.loads(event["after_record"])
        # event_id/last_modified may differ after an intervening undo.
        comparable = lambda r: {
            k: v for k, v in r.items() if k not in {"event_id", "last_modified"}
        }
        if comparable(current) != comparable(expected):
            raise RuntimeError(
                "Current record does not match the last reversible decision; "
                "undo stopped without changing it."
            )
        self.backup("before_undo")
        restored = json.loads(event["before_record"])
        restored["last_modified"] = utc_now()
        self._write_event(
            unit, current, restored, "undo", undo_of=event["event_id"]
        )
        return unit

    def backup(self, reason="manual"):
        self.export()
        directory = self.output / "backups"
        directory.mkdir(exist_ok=True)
        tag = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
        target = directory / f"{tag}_{reason}_{uuid.uuid4().hex[:6]}.zip"
        temporary_zip = directory / f".backup_{uuid.uuid4().hex}.tmp"
        with tempfile.TemporaryDirectory(dir=directory, prefix=".snapshot_") as tmp:
            database_copy = Path(tmp) / "state.sqlite3"
            destination = sqlite3.connect(database_copy)
            try:
                self.conn.backup(destination)
            finally:
                destination.close()
            try:
                with zipfile.ZipFile(
                    temporary_zip, "w", zipfile.ZIP_DEFLATED
                ) as archive:
                    archive.write(database_copy, "state.sqlite3")
                    archive.write(self.output / "system_images.csv", "system_images.csv")
                    archive.write(
                        self.output / "annotations.csv", "annotations.csv"
                    )
                    archive.write(
                        self.output / "annotation_order.csv", "annotation_order.csv"
                    )
                    archive.writestr(
                        "annotation_config.json", self.config_text
                    )
                    archive.writestr(
                        "backup_info.json",
                        json_text({
                            "timestamp": utc_now(),
                            "reason": reason,
                            "tool_version": VERSION,
                            "dataset_id": self.get_meta("dataset_id"),
                            "host": socket.gethostname(),
                        }),
                    )
                with temporary_zip.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_zip, target)
                fsync_directory(directory)
            finally:
                if temporary_zip.exists():
                    temporary_zip.unlink()
        return target

    def first_pending(self, units=None):
        for unit in self.units() if units is None else units:
            if self.get(unit)["annotation_status"] == "unlabelled":
                return unit
        return None

    def counts(self):
        return Counter(r["annotation_status"] for r in self.all_records())

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        self.lock.release()


def write_discovery(config, records, report, store=None):
    report = json.loads(json.dumps(report))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    if store is not None:
        saved = store.all_records()
    elif (output / "annotations.csv").exists():
        try:
            with (output / "annotations.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                if not (set(CSV_FIELDS) - {"members"}).issubset(reader.fieldnames or []):
                    raise ValueError("Unrecognised annotations.csv schema.")
                saved = list(reader)
            keys = [(r["system_key"], r["target"]) for r in saved]
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate annotation identities.")
        except Exception as exc:
            report["annotation_read_error"] = str(exc)
            saved = []

    report["excluded_saved_records"] = [
        {"system_key": r.get("system_key", ""), "bilayer_type": r.get("bilayer_type", ""),
         "annotation_status": r["annotation_status"]}
        for r in saved if r.get("bilayer_type") not in config["included_bilayers"]
    ]
    saved = [r for r in saved if r.get("bilayer_type") in config["included_bilayers"]]
    statuses = Counter(r["annotation_status"] for r in saved)
    current_ids = {r["system_key"] for r in records}
    report["saved_annotation_status_counts"] = dict(statuses)
    report["registered_annotation_units"] = len(saved)
    report["already_labelled_count"] = statuses["labelled"]
    report["saved_records_with_missing_sources"] = [
        {
            "relative_path": r["relative_path"],
            "target": r["target"],
            "annotation_status": r["annotation_status"],
        }
        for r in saved if r["system_key"] not in current_ids
    ]
    decided = {
        (r["system_key"], r["target"])
        for r in saved if r["annotation_status"] != "unlabelled"
    }
    report["unlabelled_current_units"] = (
        sum(
            (r["system_key"], target) not in decided
            for r in records for target in config["targets"]
        )
        if report["root_available"] else None
    )
    if store is not None:
        report["changed_source_records"] = [
            {
                "relative_path": store.get(unit)["relative_path"],
                "target": unit[1],
                "problem": store.source_problem(unit),
            }
            for unit in store.units()
            if store.source_problem(unit)
        ]
    atomic_write(
        output / "discovery_report.json", json_text(report).encode("utf-8")
    )
    atomic_write(
        output / "discovery_systems.csv", csv_bytes(records, DISCOVERY_FIELDS)
    )
    atomic_write(output / "discovery_images.csv", csv_bytes(
        [m for r in records for m in r["members"]], IMAGE_FIELDS))
    report["already_labelled_systems"] = statuses["labelled"]
    report["unlabelled_systems"] = report["unlabelled_current_units"]
    atomic_write(output / "discovery_report.json", json_text(report).encode("utf-8"))
    return report


def select_units(store, limit=None):
    units = store.units()
    return units if limit is None else units[:limit]


class AnnotationGUI:
    def __init__(self, root, store, limit=None):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import ImageTk

        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.ImageTk = ImageTk
        self.root, self.store, self.config = root, store, store.config
        self.units = select_units(store, limit)
        if not self.units:
            raise RuntimeError("No physical systems were discovered.")

        self.picture_index = 0
        self.displayed_unit = None
        self.position = 0
        self.original = None
        self.photo = None
        self.scale = 1.0
        self.fit_mode = self.config["default_view"] == "fit"
        self.busy = False
        self.export_blocked = False
        self.last_decision = 0.0
        self.pressed = set()
        self.release_jobs = {}
        self.render_job = None
        self.image_ok = False

        root.title("PFPeptoids — manual membrane annotation")
        width, height = self.config["window_size"]
        root.geometry(f"{width}x{height}")
        root.minsize(800, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.identity = tk.StringVar()
        self.progress = tk.StringVar()
        self.detail = tk.StringVar()
        self.feedback = tk.StringVar()
        self.filter_var = tk.StringVar(value="All")

        ttk.Label(
            root, textvariable=self.identity,
            font=("TkDefaultFont", 14, "bold"),
            wraplength=1300,
        ).pack(fill="x", padx=10, pady=(8, 2))
        ttk.Label(root, textvariable=self.progress).pack(fill="x", padx=10)
        ttk.Label(
            root, textvariable=self.detail, wraplength=1300
        ).pack(fill="x", padx=10, pady=(0, 5))

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", padx=8, pady=4)
        for text, command in [
            ("Previous [P]", lambda: self.navigate(-1)),
            ("Next [N]", lambda: self.navigate(1)),
            ("Next unlabelled [R]", self.next_pending),
            ("Undo [Z]", self.undo),
            ("Fit [F]", self.fit),
            ("100% [0]", lambda: self.zoom(absolute=1.0)),
            ("−", lambda: self.zoom(factor=1 / 1.25)),
            ("+", lambda: self.zoom(factor=1.25)),
            ("Save / backup", self.save_backup),
        ]:
            ttk.Button(
                toolbar, text=text, command=command, takefocus=False
            ).pack(side="left", padx=2)

        ttk.Label(toolbar, text=" Review: ").pack(side="left")
        chooser = ttk.Combobox(
            toolbar, textvariable=self.filter_var,
            values=FILTERS, state="readonly", width=12,
        )
        chooser.pack(side="left")
        chooser.bind("<<ComboboxSelected>>", lambda e: self.navigate(1))

        frame = ttk.Frame(root)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas = tk.Canvas(
            frame, background="#eeeeee", highlightthickness=0, takefocus=True
        )
        self.hscroll = ttk.Scrollbar(
            frame, orient="horizontal", command=self.scroll_x
        )
        self.vscroll = ttk.Scrollbar(
            frame, orient="vertical", command=self.scroll_y
        )
        self.canvas.configure(
            xscrollcommand=self.hscroll.set,
            yscrollcommand=self.vscroll.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.hscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", lambda e: self.on_resize())
        self.canvas.bind("<ButtonPress-1>", self.pan_start)
        self.canvas.bind("<B1-Motion>", self.pan_move)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-4>", lambda e: self.wheel_linux(-1, e))
        self.canvas.bind("<Button-5>", lambda e: self.wheel_linux(1, e))

        picture_bar = ttk.Frame(root)
        picture_bar.pack(fill="x", padx=10)
        self.picture_info = tk.StringVar()
        for text, direction in [("Previous picture", -1), ("Next picture", 1)]:
            ttk.Button(picture_bar, text=text, command=lambda d=direction: self.browse_picture(d), takefocus=False).pack(side="left")
        ttk.Label(picture_bar, textvariable=self.picture_info).pack(side="left", padx=10)

        note_frame = ttk.Frame(root)
        if self.config["show_notes"]:
            note_frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(note_frame, text="Note:").pack(side="left", anchor="n")
        self.note = tk.Text(note_frame, height=3, wrap="word", undo=True)
        self.note.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(
            note_frame, text="Save note", command=self.save_note,
            takefocus=False,
        ).pack(side="left")

        decision_frame = ttk.Frame(root)
        # Six-class layout: flat variants first, curved variants and pore second.
        decision_frame.pack(fill="x", padx=8, pady=6)
        for column in range(3):
            decision_frame.columnconfigure(column, weight=1, uniform="classes")
        self.class_buttons = []
        for index, definition in enumerate(self.config["classes"]):
            button = ttk.Button(
                decision_frame,
                text=f"{definition['key']}: {definition['name']}",
                command=lambda name=definition["name"]: self.decide(name, "labelled"),
                takefocus=False,
            )
            button.grid(row=index // 3, column=index % 3, sticky="ew", padx=3, pady=3)
            self.class_buttons.append(button)
        workflow_frame = ttk.Frame(root)
        workflow_frame.pack(fill="x", padx=8, pady=(0, 6))
        self.uncertain_button = ttk.Button(
            workflow_frame, text="U: uncertain",
            command=lambda: self.decide("", "uncertain"), takefocus=False,
        )
        self.uncertain_button.pack(side="left", padx=3)
        ttk.Button(
            workflow_frame, text="S: skip",
            command=lambda: self.decide("", "skipped"), takefocus=False,
        ).pack(side="left", padx=3)

        ttk.Label(
            root, textvariable=self.feedback, wraplength=1300
        ).pack(fill="x", padx=10, pady=(0, 6))

        root.bind("<KeyPress>", self.key_press)
        root.bind("<KeyRelease>", self.key_release)
        root.bind("<FocusOut>", lambda e: self.pressed.clear())

        pending = store.first_pending(self.units)
        if pending:
            self.position = self.units.index(pending)
        self.show()
        root.after(100, self.on_resize)

    def unit(self):
        return self.units[self.position]

    def current_note(self):
        return self.note.get("1.0", "end-1c")

    def unsaved_note(self):
        if self.current_note() == self.store.get(self.unit())["note"]:
            return True
        answer = self.messagebox.askyesnocancel(
            "Unsaved note", "Save this note before navigating?"
        )
        if answer is None:
            return False
        if answer:
            return self.save_note()
        return True

    def browse_picture(self, direction):
        if not self.unsaved_note(): return
        count = self.store.get(self.unit())["image_count"]
        if count:
            self.picture_index = (self.picture_index + direction) % count
            self.show()

    def show(self):
        from PIL import Image
        row = self.store.get(self.unit())
        if self.displayed_unit != self.unit():
            self.picture_index = 0
            self.displayed_unit = self.unit()
        self.picture_info.set(f"Images: {self.picture_index + 1 if row['image_count'] else 0}/{row['image_count']} | " + (Path(row["members"][self.picture_index]["relative_path"]).name if row["image_count"] else "no images"))
        counts = self.store.counts()
        total = sum(counts.values())
        reviewed = total - counts["unlabelled"]
        subset = (
            f" | subset view: {len(self.units)} units"
            if len(self.units) != total else ""
        )
        self.identity.set(
            f"Molecule: {row['molecule']} | Bilayer: {row['bilayer_type']}\n"
            f"Membrane role: {row['membrane_role']} | Run: {row['run']}"
        )
        self.progress.set(
            f"{counts['labelled']} / {total} labelled"
            f"  |  {reviewed} reviewed"
            f"  |  {counts['uncertain']} uncertain"
            f"  |  {counts['skipped']} skipped"
            f"  |  System {self.position + 1}/{len(self.units)}{subset}"
        )
        self.detail.set(
            f"{row['relative_path']}\n"
            f"Saved: {row['label'] or row['annotation_status']}"
        )
        self.note.delete("1.0", "end")
        self.note.insert("1.0", row["note"])
        self.original = None
        self.image_ok = False
        self.canvas.delete("all")
        try:
            raw = self.store.verified_image_bytes(self.unit(), self.picture_index)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                self.original = image.convert("RGB")
            self.image_ok = True
            self.feedback.set(
                "Click a class to save and continue. Drag image to pan; "
                "Ctrl+wheel zooms. Notes do not trigger shortcuts."
            )
        except Exception as exc:
            self.feedback.set(
                f"IMAGE BLOCKED: {exc} "
                "Use skip with a note, or fix the source/session separately."
            )
            self.canvas.create_text(
                30, 30, anchor="nw", width=700,
                text="Image unavailable or changed.\n" + str(exc),
            )
        state = (
            "normal"
            if self.image_ok and not self.export_blocked else "disabled"
        )
        for button in self.class_buttons + [self.uncertain_button]:
            button.configure(state=state)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        if self.original is not None:
            if self.fit_mode:
                self.fit()
            else:
                self.zoom(absolute=self.scale)
        self.canvas.focus_set()

    def matches_filter(self, unit):
        wanted = self.filter_var.get().lower()
        return wanted == "all" or (
            self.store.get(unit)["annotation_status"] == wanted
        )

    def navigate(self, direction, after_decision=False):
        if not after_decision and not self.unsaved_note():
            return
        for step in range(1, len(self.units) + 1):
            index = (self.position + direction * step) % len(self.units)
            unit = self.units[index]
            if after_decision and self.filter_var.get() == "All":
                match = self.store.get(unit)["annotation_status"] == "unlabelled"
            else:
                match = self.matches_filter(unit)
            if match:
                self.position = index
                self.show()
                return
        self.show()
        self.feedback.set(
            "No remaining items match this navigation mode. "
            "Use Previous/Next or change the review filter to revisit decisions."
        )

    def next_pending(self):
        if not self.unsaved_note():
            return
        self.filter_var.set("All")
        self.navigate(1, after_decision=True)

    def handle_error(self, exc):
        if isinstance(exc, SavedButExportFailed):
            self.export_blocked = True
        self.messagebox.showerror("Annotation stopped safely", str(exc))
        self.show()

    def decide(self, label, status):
        now = time.monotonic()
        if (
            self.busy or self.export_blocked
            or (now - self.last_decision) * 1000
            < self.config["decision_cooldown_ms"]
        ):
            return
        old = self.store.get(self.unit())
        note = self.current_note()
        if status == "skipped" and not self.image_ok and not note.strip():
            note = "Image unavailable/changed: " + self.store.source_problem(
                self.unit()
            )
        changed = (label, status, note) != (
            old["label"], old["annotation_status"], old["note"]
        )
        if (
            changed and old["annotation_status"] != "unlabelled"
            and self.config["confirm_relabel"]
        ):
            if not self.messagebox.askyesno(
                "Change existing decision",
                f"Change {old['label'] or old['annotation_status']} to "
                f"{label or status} for this physical system?",
            ):
                return
        self.busy = True
        try:
            self.store.decide(self.unit(), label, status, note)
            self.last_decision = time.monotonic()
            self.navigate(1, after_decision=True)
        except Exception as exc:
            self.handle_error(exc)
        finally:
            self.busy = False

    def save_note(self):
        try:
            self.store.decide(
                self.unit(), note=self.current_note(), note_only=True
            )
            self.feedback.set("Note saved.")
            return True
        except Exception as exc:
            self.handle_error(exc)
            return False

    def undo(self):
        if not self.unsaved_note():
            return
        if not self.messagebox.askyesno(
            "Undo", "Undo the most recent reversible annotation or note edit?"
        ):
            return
        try:
            unit = self.store.undo()
            if unit is None:
                self.feedback.set("No decision remains to undo.")
                return
            if unit in self.units:
                self.position = self.units.index(unit)
            self.show()
            self.feedback.set("Last decision undone; history and backup retained.")
        except Exception as exc:
            self.handle_error(exc)

    def save_backup(self):
        try:
            if self.current_note() != self.store.get(self.unit())["note"]:
                if not self.save_note():
                    return
            self.store.export()
            path = self.store.backup("manual")
            self.export_blocked = False
            self.show()
            self.feedback.set(f"Saved and backed up: {path.name}")
        except Exception as exc:
            self.handle_error(exc)

    def fit(self):
        if self.original is None:
            return
        self.fit_mode = True
        w = max(1, self.canvas.winfo_width() - 4)
        h = max(1, self.canvas.winfo_height() - 4)
        iw, ih = self.original.size
        self.scale = max(0.001, min(w / iw, h / ih, 1.0))
        self.canvas.configure(scrollregion=(0, 0, iw * self.scale, ih * self.scale))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.schedule_render()

    def zoom(self, factor=None, absolute=None):
        if self.original is None:
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        cx = self.canvas.canvasx(w / 2) / self.scale
        cy = self.canvas.canvasy(h / 2) / self.scale
        new_scale = absolute if absolute is not None else self.scale * factor
        self.scale = min(8.0, max(0.001, new_scale))
        self.fit_mode = False
        iw, ih = self.original.size
        sw, sh = iw * self.scale, ih * self.scale
        self.canvas.configure(scrollregion=(0, 0, sw, sh))
        self.canvas.xview_moveto(max(0, (cx * self.scale - w / 2) / max(sw, 1)))
        self.canvas.yview_moveto(max(0, (cy * self.scale - h / 2) / max(sh, 1)))
        self.schedule_render()

    def on_resize(self):
        if self.fit_mode:
            self.fit()
        else:
            self.schedule_render()

    def schedule_render(self):
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after_idle(self.render)

    def render(self):
        """Render only the visible tile; avoid enormous zoomed bitmap copies."""
        from PIL import Image
        self.render_job = None
        if self.original is None:
            return
        iw, ih = self.original.size
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        left = max(0, int(self.canvas.canvasx(0) / self.scale) - 2)
        top = max(0, int(self.canvas.canvasy(0) / self.scale) - 2)
        right = min(iw, int(self.canvas.canvasx(cw) / self.scale) + 3)
        bottom = min(ih, int(self.canvas.canvasy(ch) / self.scale) + 3)
        if right <= left or bottom <= top:
            return
        tile = self.original.crop((left, top, right, bottom))
        size = (
            max(1, round((right - left) * self.scale)),
            max(1, round((bottom - top) * self.scale)),
        )
        tile = tile.resize(size, Image.Resampling.LANCZOS)
        self.photo = self.ImageTk.PhotoImage(tile)
        self.canvas.delete("image")
        self.canvas.create_image(
            left * self.scale, top * self.scale,
            image=self.photo, anchor="nw", tags="image"
        )

    def scroll_x(self, *args):
        self.canvas.xview(*args)
        self.schedule_render()

    def scroll_y(self, *args):
        self.canvas.yview(*args)
        self.schedule_render()

    def pan_start(self, event):
        self.canvas.focus_set()
        self.canvas.scan_mark(event.x, event.y)

    def pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.schedule_render()

    def wheel(self, event):
        direction = -1 if event.delta > 0 else 1
        if event.state & 0x0004:
            self.zoom(factor=1.2 if direction < 0 else 1 / 1.2)
        elif event.state & 0x0001:
            self.canvas.xview_scroll(direction * 3, "units")
            self.schedule_render()
        else:
            self.canvas.yview_scroll(direction * 3, "units")
            self.schedule_render()
        return "break"

    def wheel_linux(self, direction, event):
        if event.state & 0x0004:
            self.zoom(factor=1.2 if direction < 0 else 1 / 1.2)
        else:
            self.canvas.yview_scroll(direction * 3, "units")
            self.schedule_render()
        return "break"

    def key_press(self, event):
        # Typing in notes/config widgets must never classify an image.
        if event.widget.winfo_class() in {"Text", "Entry", "TEntry", "TCombobox"}:
            return
        key = event.keysym.lower()
        pending_release = self.release_jobs.pop(key, None)
        if pending_release is not None:
            self.root.after_cancel(pending_release)
        if key in self.pressed:
            return "break"
        self.pressed.add(key)

        classes = {
            definition["key"].lower(): definition["name"]
            for definition in self.config["classes"]
        }
        if key == self.config["keyboard_shortcuts"]["previous_picture"]:
            self.browse_picture(-1)
        elif key == self.config["keyboard_shortcuts"]["next_picture"]:
            self.browse_picture(1)
        elif key in classes:
            self.decide(classes[key], "labelled")
        elif key == "u":
            self.decide("", "uncertain")
        elif key == "s":
            self.decide("", "skipped")
        elif key in {"n", "right"}:
            self.navigate(1)
        elif key in {"p", "left"}:
            self.navigate(-1)
        elif key == "r":
            self.next_pending()
        elif key == "z":
            self.undo()
        elif key == "f":
            self.fit()
        elif key == "0":
            self.zoom(absolute=1.0)
        elif key in {"plus", "equal", "kp_add"}:
            self.zoom(factor=1.25)
        elif key in {"minus", "kp_subtract"}:
            self.zoom(factor=1 / 1.25)
        else:
            return
        return "break"

    def key_release(self, event):
        key = event.keysym.lower()

        def clear():
            self.pressed.discard(key)
            self.release_jobs.pop(key, None)

        old = self.release_jobs.pop(key, None)
        if old is not None:
            self.root.after_cancel(old)
        # X11 may emit release/press pairs during autorepeat. Clearing at idle
        # lets the next press cancel this release instead of labelling again.
        self.release_jobs[key] = self.root.after_idle(clear)

    def close(self):
        if not self.unsaved_note():
            return
        try:
            self.store.backup("close")
        except Exception as exc:
            if not self.messagebox.askyesno(
                "Backup/export problem",
                f"{exc}\n\nCommitted decisions remain in state.sqlite3. "
                "Close without completing this backup?",
            ):
                return
        self.root.destroy()


README = '# PFPeptoids membrane annotation — v2.1\n\nOne scientific label belongs to **molecule × bilayer_type × run**. Multiple pictures are evidence for that one record. Mammalian and bacterial systems are independent. This is a revision of the supplied tool, retaining its transactional saves, event history, undo, backups, and image viewer. No classifier, train/test split, simulation, APL analysis, or active-learning changes are included.\n\n## Installation and commands\n\nClose the old annotation GUI. Extract the complete `pfp34_v2` folder under:\n\n`/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/`\n\nThis versioned installation preserves the old script, config and session unchanged. The revised entrypoint is `pfp34_v2/scripts/annotate.py`. It needs no helper modules. Use the supplied v2 configuration, not the pasted composite configuration. Tkinter and Pillow were reported as working on Archie; no installation is requested.\n\nRun these commands in order. The initial `cd` also makes the explicit relative output overrides resolve as shown.\n\n```bash\ncd /users/pfb19164/Desktop/post_medical_leave/PFPeptoids/PFPeptoids/analysis/image_classification/pfp34_v2\npython scripts/annotate.py --self-test\npython scripts/annotate.py --archive-legacy ..\npython scripts/annotate.py --discover\npython scripts/annotate.py --output-dir smoke_test_session --limit 3\n```\n\n`--archive-legacy ..` archives recognized v1 state under the old session\'s `legacy/` directory and intentionally stops with an explanatory error/exit code 1. It does not migrate labels or change the old database. If no legacy database is present, it reports that fact. Preserve CSV-only sessions separately; they are never imported automatically. Existing unknown/corrupt databases are refused, not repaired.\n\n**Inspect discovery before annotation.** Actual Photos contents could not be audited in the development environment. Check `membrane_session/discovery_report.json`, especially malformed paths, unknown bilayers, empty systems, unreadable pictures and duplicate content. Fix configuration or investigate layout discrepancies before proceeding; do not rename source images to fit a guess.\n\nThe three-system smoke test uses separate software-test state. Inspect both membrane types and multiple pictures; close/reopen, edit, undo, mark uncertain/skip, and verify that changing pictures does not create new labels. Ordered first-three systems may not contain both membrane types in the real tree: extend the smoke-test limit if needed, while keeping its separate output directory. Test labels must not be copied into production state.\n\nAfter checking discovery and the interactive smoke test:\n\n```bash\npython scripts/annotate.py\n```\n\nThis uses `membrane_session/`, separate from `smoke_test_session/` and all legacy state. Re-run the same command to resume. `--limit N` shows the first N **physical systems in saved order**, not N pictures or N pending systems. Discovery still scans the whole tree.\n\nFor another installation directory, adjust only the initial `cd`; the configuration is located relative to the script. Relative paths, including output overrides, resolve against the configuration file\'s directory. `--config PATH` selects a different configuration.\n\n## Discovery and grouping\n\nSupported layout:\n\n`Photos/{molecule}/{bilayer_type}/{run}/{picture}`\n\nAll direct files with configured image extensions are grouped. Run names are preserved exactly, with no whitelist or inferred cycle. Source filenames are not used to infer identities. `STACKED` trees are audited but excluded from physical systems. Only exact names in `included_bilayers` enter the queue or create annotation rows. Excluded directories remain in the audit. The default whitelist is POPC60CHOL40_W_WF, POPC80POPS20_W_WF, and POPE75POPG25_W_WF; POPC80POPS20_W_WF has the neutral role POPC80POPS20. Included systems with an unknown role remain blocked. Empty, unreadable, nested, symlinked or multi-frame evidence is reported and blocked. No arbitrary first frame or nested-view rule is invented.\n\n| Bilayer directory | Membrane role |\n|---|---|\n| POPC60CHOL40_W_WF | mammalian_like |\n| POPE75POPG25_W_WF | bacterial_like |\n| POPC80POPS20_W_WF | POPC80POPS20 |\n\nIdentical image bytes at different paths are reported, never merged. Audit counts distinguish grouped images from all candidate image files elsewhere in the tree. Ambiguous duplicate system keys or image membership cause registration to fail. Molecular type stays blank because names alone cannot reliably establish peptide versus peptoid.\n\n## Controls\n\n- `1`, `2`, `3`: curved, flat_interaction, flat_no_interaction; configurable class names/keys.\n- `[` / `]`: previous/next picture within the current system; configurable Tk keysyms.\n- `P` / `N`, or Left / Right: previous/next physical system.\n- `U`: uncertain; `S`: skipped; `R`: next unlabelled; `Z`: undo.\n- `F`: fit; `0`: actual pixels; `+` / `-`: zoom.\n- Drag to pan; wheel to scroll; Ctrl+wheel to zoom; Shift+wheel horizontal scrolling on MouseWheel platforms.\n- Save note, Save/backup, and review-status filter remain available.\n\nThe title displays molecule, exact bilayer, explicit role and run. Counters display physical-system position and picture position; the picture filename is also shown. A scientific decision commits immediately and advances only after export/required backup succeeds. Normal auto-advance chooses the next unlabelled system; review filters retain the old tool\'s filtered navigation. Notes prompt before navigating away. Shortcuts are suppressed while typing notes; held keys are suppressed.\n\nAll relevant views should be inspected before choosing a class. The GUI does not enforce a minimum number of viewed pictures. A system with blocked evidence can be skipped with a note but cannot receive a scientific or uncertain decision. This conservative policy also blocks viewing its other images through the annotation viewer; inspect the discovery report to resolve the issue.\n\n## Outputs and schema\n\nInside the configured session directory:\n\n| File | Contents |\n|---|---|\n| annotations.csv | One row per physical system; scientific/workflow label, notes and provenance |\n| system_images.csv | One row per supporting picture; system key, root-relative path, SHA-256, dimensions and decode status |\n| annotation_order.csv | Persisted system order |\n| state.sqlite3 | Authoritative transactional recovery state, membership snapshots, full edit/undo history and config snapshots |\n| discovery_report.json | Current audit counts, anomalies and saved-label counts |\n| discovery_systems.csv | Current discovered systems and evidence hashes |\n| discovery_images.csv | Current discovered picture membership |\n| backups/*.zip | Database, all three session CSVs, exact configuration and backup metadata |\n\n`annotations.csv` includes `system_key`, `molecule`, `molecule_type`, `bilayer_type`, `membrane_role`, `run`, `label`, `annotation_status`, `note`, `annotation_timestamp`, `last_modified`, `source_directory`, `relative_path`, `image_count`, `image_paths`, `evidence_sha256`, `config_sha256`, `event_id`, and a fixed internal `target=system`.\n\n`system_key` is SHA-256 of the JSON encoding of `[molecule, bilayer_type, run]`; identity is independent of the Photos root. `image_paths` is a deterministic JSON array of **root-relative** supporting paths. `source_directory` is the complete original directory; resolve current files using `image_root` plus root-relative paths when moving a closed session to another machine. `system_images.csv` holds individual image hashes. `evidence_sha256` fingerprints the ordered member paths and hashes.\n\n`label` holds a scientific class only for `annotation_status=labelled`. Uncertain, skipped and unlabelled records have an empty scientific label and an explicit status. This intentionally avoids mixing workflow states into future ML targets. Undo restores the previous decision timestamp; `last_modified` and events preserve the later undo.\n\n## Safety, resume and legacy state\n\nThe default output is the new `membrane_session/` directory. Old image/composite labels are not membrane-specific truth and are never mapped to both bilayers. A recognized v1 database encountered by v2 is backed up and refused before writable database setup. Unknown schemas are refused. The original source/config supplied for this revision are recorded in the handoff through their hashes; no production files were accessed or changed here.\n\nImage changes, additions and removals are checked against saved evidence before display and scientific decisions. Existing labels are preserved, but new decisions are blocked. There is no automatic evidence-rebaseline command: restore the original evidence or create a separate session for the changed dataset. Missing systems remain registered for provenance. The report counts saved labels and current unlabelled systems separately; these totals can differ when sources disappear.\n\nSQLite synchronous FULL and atomic, fsynced CSV replacement are retained. Unexpected manual CSV modifications stop export rather than being overwritten. Missing exports can be reconstructed from the database; a committed decision survives export failure. Use Save/backup to retry. Keep the complete session directory, not just one CSV. Backups are retained, not pruned. One process can write each session; do not synchronize live databases across machines. Durability on network filesystems depends on their guarantees.\n\nBilayer whitelist, order mode, seed, membrane mapping, target and image extensions are locked for a started session. Restore the saved configuration or use a new session to change them. Class definitions remain expandable; never repurpose an existing class meaning. The old composite protocol and `targets=["composite"]` are rejected.\n\n## Validation status\n\n`validation_report.json` records the automated fixture tests actually run. They cover independent bilayer labels, grouped pictures, limits, immediate save, restart, edit/undo/notes, uncertain/skip, backups, order, locking, external CSV protection, export failure, evidence changes and legacy preservation. Headless GUI-controller tests use widget stand-ins; they verify identity text and navigation/save-failure logic, **not rendered Tk usability**.\n\nThe real Photos tree and an interactive display were unavailable here. `dataset_availability.json` records that limitation with unknown counts, not a zero-system claim. Complete the real audit and isolated GUI smoke test on Archie before full annotation. No source images, simulations, APL tables or active-learning state were modified.\n\n## v2.1 session compatibility\n\nA pre-whitelist v2.0 session is preserved and refused because its saved selection settings differ. Use a new empty output directory if such state already exists; do not delete its database or reuse smoke-test labels. The supplied config still defaults to membrane_session, separate from smoke_test_session. Discovery counts and saved-status totals exclude non-whitelisted bilayers; excluded saved records are reported without modification.\n'

def run_self_test(example_dir=None):
    """Disposable fixtures only; never operate on production source or state."""
    Image = require_pillow()
    results=[]
    started=utc_now()
    def passed(name): results.append(dict(test=name,status="passed"))
    def expect_error(action, kind=Exception):
        try: action()
        except kind: return
        raise AssertionError("Expected operation to be rejected")
    with tempfile.TemporaryDirectory(prefix="pfp34_v2_test_") as temporary:
        base=Path(temporary)
        photos=base/"Photos"
        config=json.loads(json.dumps(DEFAULT_CONFIG))
        config.update(image_root=str(photos),output_dir=str(base/"session"),backup_every_decisions=2)
        validate_config(config)
        mammal="POPC60CHOL40_W_WF"
        bacteria="POPE75POPG25_W_WF"
        specifications=[("MoleculeA",mammal,"run_ML_cycle_02",2),
                        ("MoleculeA",bacteria,"run_ML_cycle_02",3),
                        ("MoleculeA",mammal,"novel_run",1),
                        ("MoleculeB",bacteria,"run_0",1)]
        for mol,bil,run,count in specifications:
            folder=photos/mol/bil/run
            folder.mkdir(parents=True)
            for i in range(count):
                Image.new("RGB",(80+i,60),(100+i,140,180)).save(folder/f"view_{i}.png")
        empty=photos/"Empty"/mammal/"run_0"
        empty.mkdir(parents=True)
        unknown=photos/"Unknown"/"UNMAPPED"/"run_0"
        unknown.mkdir(parents=True)
        Image.new("RGB",(40,30)).save(unknown/"view.png")
        broken=photos/"Broken"/bacteria/"run_0"
        broken.mkdir(parents=True)
        (broken/"broken.png").write_bytes(b"not an image")
        legacy=photos/"MoleculeA"/"STACKED"
        legacy.mkdir()
        Image.new("RGB",(40,30)).save(legacy/"old.png")
        source_before={p.relative_to(photos).as_posix():file_digest(p) for p in photos.rglob('*') if p.is_file()}
        records,report=discover(photos,config)
        assert len(records)==6 and report["images_found"]==8
        assert report["unreadable_images"]==1
        assert report["unknown_bilayer_types"]==["STACKED", "UNMAPPED"]
        assert report["empty_run_directories"] and report["legacy_stacked"]
        assert report["exact_content_duplicates"]
        assert "novel_run" in report["distinct_run_names"]
        passed("physical_grouping_multiple_views_both_bilayers_arbitrary_runs_and_audit_flags")
        a=next(r for r in records if r["molecule"]=="MoleculeA" and r["bilayer_type"]==mammal and r["run"]=="run_ML_cycle_02")
        b=next(r for r in records if r["molecule"]=="MoleculeA" and r["bilayer_type"]==bacteria)
        ua,ub=(a["system_key"],"system"),(b["system_key"],"system")
        store=Store(config)
        try:
            store.register(records)
            assert len(store.units())==6
            assert len(select_units(store,3))==3
            assert len(select_units(store,None))==6
            passed("limit_counts_systems_not_pictures")
            before=len(store.all_records())
            assert store.verified_image_bytes(ua,0)!=store.verified_image_bytes(ua,1)
            assert len(store.all_records())==before
            passed("browsing_distinct_views_does_not_create_rows")
            store.decide(ua,"flat_interaction","labelled")
            store.decide(ub,"curved","labelled")
            rows=list(csv.DictReader((store.output/"annotations.csv").open()))
            assert len(rows)==6
            assert next(r for r in rows if r["system_key"]==ua[0])["label"]=="flat_interaction"
            assert next(r for r in rows if r["system_key"]==ub[0])["label"]=="curved"
            assert all(r["molecule_type"]=="" for r in rows)
            passed("independent_bilayer_labels_immediate_csv_one_row_per_system")
            # Exercise real GUI controller methods with widget stand-ins; no rendered-display claim.
            from types import SimpleNamespace
            class Widget:
                def __init__(self): self.value=""
                def set(self,value): self.value=value
                def get(self,*args): return self.value
                def delete(self,*args): self.value=""
                def insert(self,index,value): self.value=value
                def configure(self,**kwargs): pass
                def focus_set(self): pass
                def xview_moveto(self,*args): pass
                def yview_moveto(self,*args): pass
                def create_text(self,*args,**kwargs): pass
            gui=AnnotationGUI.__new__(AnnotationGUI)
            gui.store=store; gui.config=config; gui.units=[ua,ub]; gui.position=0
            gui.displayed_unit=None; gui.picture_index=0; gui.export_blocked=False
            gui.picture_info=Widget(); gui.identity=Widget(); gui.progress=Widget()
            gui.detail=Widget(); gui.note=Widget(); gui.canvas=Widget(); gui.feedback=Widget()
            gui.class_buttons=[Widget()]; gui.uncertain_button=Widget()
            gui.fit_mode=True; gui.fit=lambda:None
            gui.show()
            assert mammal in gui.identity.value and "MoleculeA" in gui.identity.value and "run_ML_cycle_02" in gui.identity.value
            assert "1/2" in gui.picture_info.value
            gui.browse_picture(1)
            assert "2/2" in gui.picture_info.value and gui.unit()==ua and len(store.all_records())==6
            gui.position=1; gui.show()
            assert bacteria in gui.identity.value and "1/3" in gui.picture_info.value
            passed("headless_GUI_identity_picture_navigation_and_system_transition")
            gui.busy=False; gui.last_decision=0
            gui.messagebox=SimpleNamespace(askyesno=lambda *a:True,showerror=lambda *a:None)
            original_decide=store.decide
            def saved_error(*args,**kwargs): raise SavedButExportFailed("injected")
            store.decide=saved_error
            gui.decide("flat_interaction","labelled")
            assert gui.position==1 and gui.export_blocked
            store.decide=original_decide
            passed("headless_GUI_export_failure_blocks_advancement")
            expect_error(lambda:store.register(records+[records[0]]),RuntimeError)
            passed("duplicate_registration_refused")

            expect_error(lambda: Store(config),RuntimeError)
            passed("single_writer_lock")
            backup=store.backup("test")
            with zipfile.ZipFile(backup) as z:
                assert z.testzip() is None and "system_images.csv" in z.namelist()
            passed("complete_backup_integrity")
            restored=base/"restored_session"
            restored.mkdir()
            with zipfile.ZipFile(backup) as z: z.extractall(restored)
            restored_store=Store(dict(config,output_dir=str(restored)))
            try:
                restored_store.register(records)
                assert restored_store.get(ua)["label"]=="flat_interaction"
                assert restored_store.get(ub)["label"]=="curved"
            finally: restored_store.close()
            passed("backup_restores_independent_labels_into_separate_session")
        finally: store.close()
        store=Store(config)
        try:
            store.register(records)
            assert store.get(ua)["label"]=="flat_interaction" and store.get(ub)["label"]=="curved"
            assert store.first_pending() not in [ua,ub]
            passed("restart_resume_preserves_labels")
            store.decide(ua,"curved","labelled","edit")
            assert store.get(ua)["label"]=="curved"
            assert store.undo()==ua and store.get(ua)["label"]=="flat_interaction"
            store.decide(ua,note="note only",note_only=True)
            assert store.get(ua)["label"]=="flat_interaction"
            assert store.undo()==ua and store.get(ua)["note"]==""
            passed("edit_note_and_undo_history")
            store.decide(ua,status="uncertain")
            assert store.get(ua)["label"]=="" and store.get(ua)["annotation_status"]=="uncertain"
            store.decide(ua,status="skipped")
            assert store.get(ua)["label"]=="" and store.get(ua)["annotation_status"]=="skipped"
            passed("uncertain_and_skip_distinct_from_scientific_labels")
            # Preserve bytes while exercising add/remove/content protections.
            path=photos/a["members"][0]["relative_path"]
            original=path.read_bytes()
            path.write_bytes(b"changed")
            expect_error(lambda:store.decide(ua,"curved","labelled"),RuntimeError)
            path.write_bytes(original)
            added=path.parent/"new.png"
            added.write_bytes(original)
            expect_error(lambda:store.decide(ua,"curved","labelled"),RuntimeError)
            added.unlink()
            path.unlink()
            expect_error(lambda:store.decide(ua,"curved","labelled"),RuntimeError)
            path.write_bytes(original)
            assert not store.source_problem(ua)
            passed("changed_added_removed_evidence_blocks_decisions_without_erasing_labels")
            old_export=store.export
            config["backup_before_edit"]=False
            def fail(): raise OSError("injected export failure")
            store.export=fail
            expect_error(lambda:store.decide(ua,"curved","labelled"),SavedButExportFailed)
            assert store.get(ua)["label"]=="curved"
            store.export=old_export
            config["backup_before_edit"]=True
        finally: store.close()
        store=Store(config)
        try:
            store.register(records)
            assert store.get(ua)["label"]=="curved"
            assert next(r for r in csv.DictReader((store.output/"annotations.csv").open()) if r["system_key"]==ua[0])["label"]=="curved"
            passed("interrupted_export_recovers_committed_decision")
            store.decide(ua,"flat_interaction","labelled")
            if example_dir:
                examples=Path(example_dir)
                examples.mkdir(parents=True,exist_ok=True)
                for name in ["annotations.csv","annotation_order.csv","system_images.csv"]:
                    shutil.copyfile(store.output/name,examples/name)
                write_discovery(config,records,report,store)
                shutil.copyfile(store.output/"discovery_report.json",examples/"discovery_report.json")
        finally: store.close()
        export=Path(config["output_dir"])/"annotations.csv"
        original=export.read_bytes()
        export.write_bytes(b"external edit\n")
        expect_error(lambda:Store(config),RuntimeError)
        assert export.read_bytes()==b"external edit\n"
        export.write_bytes(original)
        passed("external_csv_edit_preserved_and_rejected")
        random_config=dict(config, output_dir=str(base/"random"),order_mode="random")
        store=Store(random_config)
        try:
            store.register(records)
            order=store.units()
        finally: store.close()
        store=Store(random_config)
        try:
            store.register(list(reversed(records)))
            assert store.units()==order
        finally: store.close()
        passed("stable_random_system_order")
        copied=base/"CopiedPhotos"
        shutil.copytree(photos,copied)
        copied_records,_=discover(copied,config)
        assert [(r["system_key"],r["evidence_sha256"]) for r in records]==[(r["system_key"],r["evidence_sha256"]) for r in copied_records]
        passed("root_independent_keys_and_evidence")
        legacy_output=base/"legacy_session"
        legacy_output.mkdir()
        db=sqlite3.connect(legacy_output/"state.sqlite3")
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)")
        db.execute("INSERT INTO meta VALUES('schema_version','1')")
        db.commit(); db.close()
        db_hash=file_digest(legacy_output/"state.sqlite3")
        expect_error(lambda:Store(dict(config,output_dir=str(legacy_output))),RuntimeError)
        assert db_hash==file_digest(legacy_output/"state.sqlite3")
        assert list((legacy_output/"legacy").glob("*.zip"))
        expect_error(lambda:validate_config(dict(config,targets=["composite"])),ValueError)
        passed("legacy_backed_up_unmodified_and_composite_config_rejected")
        source_after={p.relative_to(photos).as_posix():file_digest(p) for p in photos.rglob('*') if p.is_file()}
        assert source_before==source_after
        passed("source_fixture_hashes_unchanged")
        whitelist_root=base/"WhitelistPhotos"
        excluded=["ManuallyTaken", "POPC1_W_WF", "POPC60CHOL40_20Az",
                  "POPC80POPS20_W_WF.2", "POPE75POPG25_20Az", "STACKED"]
        for bilayer in config["included_bilayers"] + excluded:
            folder=whitelist_root/"Example"/bilayer/"run_0"
            folder.mkdir(parents=True)
            Image.new("RGB",(20,20),(20,40,60)).save(folder/"view.png")
        original_hashes={str(f):file_digest(f) for f in whitelist_root.rglob("*.png")}
        selected,audit=discover(whitelist_root,config)
        assert len(selected)==3 and audit["physical_systems_found"]==3
        assert set(audit["included_bilayers"])==set(config["included_bilayers"])
        assert set(audit["excluded_bilayers"])==set(excluded)
        assert set(audit["distinct_bilayer_names"])==set(config["included_bilayers"]+excluded)
        assert len(audit["excluded_directories"])==6
        assert next(r for r in selected if r["bilayer_type"]=="POPC80POPS20_W_WF")["membrane_role"]=="POPC80POPS20"
        passed("exact_three_bilayer_whitelist_and_neutral_role_with_complete_exclusion_audit")
        smoke_config=dict(config,image_root=str(whitelist_root),output_dir=str(base/"smoke_test_session"))
        smoke=Store(smoke_config)
        try:
            smoke.register(selected)
            smoke.decide(smoke.units()[0],status="skipped")
            smoke_files={f.name:file_digest(f) for f in smoke.output.iterdir() if f.is_file() and not f.name.startswith(".")}
        finally: smoke.close()
        real_config=dict(smoke_config,output_dir=str(base/"membrane_session"))
        real=Store(real_config)
        try:
            real.register(selected)
            assert len(select_units(real))==3 and real.counts()==Counter(unlabelled=3)
            rows=list(csv.DictReader((real.output/"annotations.csv").open()))
            assert {r["bilayer_type"] for r in rows}==set(config["included_bilayers"])
            rejected=dict(selected[0],bilayer_type="POPC80POPS20_W_WF.2")
            expect_error(lambda:real.register([rejected]),RuntimeError)
            assert len(real.all_records())==3
        finally: real.close()
        assert smoke_files=={f.name:file_digest(f) for f in Path(smoke_config["output_dir"]).iterdir() if f.is_file() and not f.name.startswith(".")}
        assert original_hashes=={str(f):file_digest(f) for f in whitelist_root.rglob("*.png")}
        passed("whitelist_queue_rows_count_and_smoke_production_isolation")

    return dict(started=started,finished=utc_now(),tool_version=VERSION,status="passed",tests=results,
                gui_test="Interactive Tk display not tested; perform isolated --limit 3 trial on Archie.",
                real_dataset_validation="Unavailable here; run --discover against the actual Photos tree before use.")


def build_package(destination):
    destination = Path(destination).expanduser().resolve()
    archive_path = destination.with_name(destination.name + ".zip")
    if destination.exists() or archive_path.exists():
        raise FileExistsError(
            "Package folder/ZIP already exists; choose a new destination."
        )
    check_separate_paths(
        Path(DEFAULT_CONFIG["image_root"]).resolve(), destination
    )
    destination.mkdir(parents=True)
    scripts = destination / "scripts"
    scripts.mkdir()
    shutil.copyfile(Path(__file__).resolve(), scripts / "annotate.py")
    atomic_write(
        destination / "annotation_config.json",
        json_text(DEFAULT_CONFIG).encode("utf-8"),
    )
    atomic_write(destination / "README.md", README.encode("utf-8"))
    atomic_write(destination / "requirements.txt", b"Pillow\n")
    atomic_write(
        destination / "examples" / "README.md",
        b"These are disposable synthetic test records, not PFPeptoids labels.\n",
    )
    try:
        validation = run_self_test(destination / "examples")
    except Exception:
        validation = {
            "status": "failed",
            "timestamp": utc_now(),
            "traceback": traceback.format_exc(),
        }
        atomic_write(
            destination / "validation_report.json",
            json_text(validation).encode("utf-8"),
        )
        raise RuntimeError(
            "Self-test failed; no release ZIP was produced. "
            f"Inspect {destination / 'validation_report.json'}"
        )

    atomic_write(
        destination / "validation_report.json",
        json_text(validation).encode("utf-8"),
    )

    # Report actual availability, not a fabricated zero-sized dataset.
    records, discovery = discover(Path(DEFAULT_CONFIG["image_root"]))
    report_config = dict(DEFAULT_CONFIG)
    report_config["output_dir"] = str(destination)
    write_discovery(report_config, records, discovery)

    checksums = {
        p.relative_to(destination).as_posix(): file_digest(p)
        for p in destination.rglob("*") if p.is_file()
    }
    atomic_write(
        destination / "checksums.json", json_text(checksums).encode("utf-8")
    )

    # Explicit finite file list: no output ZIP is recursively included.
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in paths:
            archive.write(
                path, Path(destination.name) / path.relative_to(destination)
            )
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity verification failed.")

    print(f"Package: {destination}")
    print(f"ZIP: {archive_path}")
    print(f"Self-tests passed: {len(validation['tests'])}")
    print(
        "Real image discovery: "
        + (
            f"{discovery['images_found']} images"
            if discovery["root_available"]
            else "root unavailable; configure image_root before annotation"
        )
    )


def default_config_path():
    parent = Path(__file__).resolve().parent
    if parent.name == "scripts":
        parent = parent.parent
    return parent / "annotation_config.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--archive-legacy", type=Path, help="Back up v1 state in this directory and stop without migration.")
    parser.add_argument(
        "--limit", type=int,
        help="GUI-only limit to the first N physical systems in saved order."
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    if args.archive_legacy:
        output=args.archive_legacy.expanduser().resolve()
        if not output.is_dir(): raise FileNotFoundError(output)
        lock=SessionLock(output/".annotation.lock")
        lock.acquire()
        try: guard_legacy(output)
        finally: lock.release()
        print("No legacy v1 state found.")
        return
    if args.package:
        build_package(args.package)
        return
    if args.self_test:
        result = run_self_test()
        if args.report:
            if args.report.exists():
                raise FileExistsError(args.report)
            atomic_write(args.report, json_text(result).encode("utf-8"))
        print(json_text(result))
        return
    if args.init:
        if args.config.exists():
            raise FileExistsError(args.config)
        atomic_write(
            args.config, json_text(DEFAULT_CONFIG).encode("utf-8")
        )
        print(f"Created {args.config}; review its paths and annotation targets.")
        return

    config = load_config(
        args.config, args.image_root, args.output_dir
    )
    records, report = discover(Path(config["image_root"]), config)

    if args.discover:
        complete = write_discovery(config, records, report)
        print(json_text({
            key: complete.get(key)
            for key in (
                "image_root", "root_available", "images_found", "physical_systems_found",
                    "included_bilayers", "excluded_bilayers", "excluded_systems_found",
                    "systems_by_bilayer_type", "systems_by_run", "images_per_system_distribution",
                "readable_images", "unreadable_images",
                "already_labelled_count", "unlabelled_current_units",
            )
        }))
        print(f"Reports: {config['output_dir']}")
        if not complete["root_available"]:
            raise SystemExit(2)
        return

    if not report["root_available"]:
        write_discovery(config, records, report)
        raise RuntimeError(
            "Image root is unavailable. No annotations were modified. "
            "Mount/copy the Photos tree or update image_root."
        )
    if not records:
        write_discovery(config, records, report)
        raise RuntimeError(
            "No physical system directories found under molecule/bilayer/run/. "
            "Review discovery_report.json."
        )

    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:
        raise RuntimeError(
            "An interactive Tk display is required. Run `python -m tkinter` "
            "to test it. Use a graphical/X11 session or run locally against "
            f"a copied/mounted Photos tree. Details: {exc}"
        ) from exc

    store = None
    try:
        store = Store(config)
        store.register(records)
        write_discovery(config, records, report, store)
        if store.counts()["labelled"] or store.counts()["uncertain"] or store.counts()["skipped"]:
            store.backup("startup")
        for notice in store.notices:
            print(notice)
        AnnotationGUI(root, store, args.limit)
        root.mainloop()
    finally:
        if store is not None:
            store.close()
        try:
            if root.winfo_exists():
                root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped; committed decisions remain saved.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)