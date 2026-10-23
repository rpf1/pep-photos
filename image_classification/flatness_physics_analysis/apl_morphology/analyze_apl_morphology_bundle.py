#!/usr/bin/env python3
"""
Comprehensive paired APL ↔ membrane-morphology analysis.

Designed for the PFPeptoids matched-pair table with columns such as:
    molecule, physical_run, m_delta_apl, b_delta_apl, ddAPL,
    m_label, b_label, m_curved, b_curved, m_deformed, b_deformed,
    m_delta_height_rms, b_delta_height_rms, dd_delta_height_rms, ...
and the analogous endpoint/baseline/excess-area/curvature/thickness/asymmetry columns.

INPUT
-----
Either:
  1) one CSV/XLSX already containing m_delta_apl and b_delta_apl, OR
  2) the matched-pair CSV/XLSX plus --dapl pointing to a 3-column file:
         name, m_delta_apl, b_delta_apl

OUTPUT
------
One timestamped analysis directory containing:
  analysis_master.csv
  analysis_master.xlsx
  tables/*.csv
  plots/*
  plot_index.csv
  SUMMARY.txt

Plot coverage:
  - m_delta_apl vs b_delta_apl coloured by phenotype classes
  - m_delta_apl vs b_delta_apl coloured continuously by every morphology metric
  - every APL summary metric vs every morphology metric
  - every pairwise combination of the three primary morphology families
    (height RMS, excess area, curvature)
  - optional --exhaustive-pair-plots: every numeric metric vs every other
    numeric metric, coloured by pair deformation state

Statistics:
  - all-vs-all Spearman rho, raw p, BH-FDR q
  - APL-vs-morphology Spearman subset
  - flat/deformed and curved/not-curved Mann–Whitney tests + rank-biserial effects
  - descriptive summaries by labels and pair states
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, spearmanr


PRIMARY_FAMILIES = ("height_rms", "excess_area", "curvature")
APL_BASE = ("m_delta_apl", "b_delta_apl", "ddAPL")
APL_DERIVED = ("mean_delta_apl", "response_magnitude", "abs_ddAPL")
CLASS_COLUMNS = (
    "m_label", "b_label",
    "m_curved", "b_curved",
    "m_deformed", "b_deformed",
    "pair_deformation_state", "pair_curvature_state",
    "deformation_transition", "curvature_transition",
)

# Change these if you want a different plotting appearance.
FIGSIZE = (7.0, 5.5)
DPI = 180
POINT_SIZE = 30
POINT_ALPHA = 0.78
EDGE_ALPHA = 0.25


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Comprehensive paired APL ↔ morphology analysis."
    )
    p.add_argument("input", help="Matched-pair CSV or XLSX.")
    p.add_argument(
        "--dapl",
        default=None,
        help="Optional CSV/XLSX containing name,m_delta_apl,b_delta_apl."
    )
    p.add_argument("--sheet", default=0, help="XLSX sheet name/index for input.")
    p.add_argument("--dapl-sheet", default=0, help="XLSX sheet name/index for --dapl.")
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: apl_morphology_analysis_YYYYMMDD_HHMMSS"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel plotting workers. Defaults to SLURM_CPUS_PER_TASK or min(8,cpu_count)."
    )
    p.add_argument(
        "--exhaustive-pair-plots",
        action="store_true",
        help="Also create every numeric-vs-numeric scatter plot. This can create >1000 PNGs."
    )
    p.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Skip consolidated XLSX output if you only want CSV/PNG files."
    )
    return p.parse_args()


def read_table(path, sheet=0):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported input format: {path}")


def norm_name(s):
    return str(s).strip()


def coerce_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s
    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    out = s.map(lambda x: mapping.get(str(x).strip().lower(), x) if pd.notna(x) else np.nan)
    try:
        return out.astype("boolean")
    except Exception:
        return out


def slug(s, max_len=120):
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    if len(x) <= max_len:
        return x
    return x[:max_len]


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    back = np.empty_like(adjusted)
    back[order] = adjusted
    q[valid] = back
    return q


def rank_biserial_from_u(u, n1, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    return 2.0 * u / (n1 * n2) - 1.0


def prepare_master(df, dapl=None):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "molecule" not in df.columns:
        raise ValueError("Input must contain a 'molecule' column.")
    df["molecule"] = df["molecule"].map(norm_name)

    if dapl is not None:
        d = dapl.copy()
        d.columns = [str(c).strip() for c in d.columns]
        if "name" in d.columns and "molecule" not in d.columns:
            d = d.rename(columns={"name": "molecule"})
        need = {"molecule", "m_delta_apl", "b_delta_apl"}
        missing = need - set(d.columns)
        if missing:
            raise ValueError(f"--dapl is missing columns: {sorted(missing)}")
        d["molecule"] = d["molecule"].map(norm_name)
        d = d[["molecule", "m_delta_apl", "b_delta_apl"]].drop_duplicates("molecule")
        for c in ["m_delta_apl", "b_delta_apl"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        df = df.merge(d, on="molecule", how="left", validate="many_to_one")

    missing_apl = {"m_delta_apl", "b_delta_apl"} - set(df.columns)
    if missing_apl:
        raise ValueError(
            "Input has no m_delta_apl/b_delta_apl. Supply --dapl with "
            "name,m_delta_apl,b_delta_apl."
        )

    for c in df.columns:
        if c.startswith(("m_", "b_", "dd_")) or c in {"ddAPL"}:
            if c not in CLASS_COLUMNS and c not in {"m_label", "b_label"}:
                converted = pd.to_numeric(df[c], errors="coerce")
                # Keep numeric conversion only when the column genuinely contains numeric data.
                if converted.notna().sum() >= max(3, int(0.5 * df[c].notna().sum())):
                    df[c] = converted

    for c in ["m_delta_apl", "b_delta_apl", "ddAPL"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "ddAPL" not in df.columns:
        df["ddAPL"] = df["b_delta_apl"] - df["m_delta_apl"]

    df["ddAPL_recomputed"] = df["b_delta_apl"] - df["m_delta_apl"]
    df["ddAPL_residual"] = df["ddAPL"] - df["ddAPL_recomputed"]
    df["mean_delta_apl"] = (df["m_delta_apl"] + df["b_delta_apl"]) / 2.0
    df["response_magnitude"] = np.sqrt(df["m_delta_apl"]**2 + df["b_delta_apl"]**2)
    df["abs_ddAPL"] = df["ddAPL"].abs()

    for c in ["m_curved", "b_curved", "m_deformed", "b_deformed"]:
        if c in df.columns:
            df[c] = coerce_bool(df[c])

    if {"m_deformed", "b_deformed"} <= set(df.columns):
        def pair_state(row):
            m = row["m_deformed"]
            b = row["b_deformed"]
            if pd.isna(m) or pd.isna(b):
                return "unknown"
            if bool(m) and bool(b):
                return "both_deformed"
            if bool(m):
                return "mammalian_only"
            if bool(b):
                return "bacterial_only"
            return "both_plain_flat"
        df["pair_deformation_state"] = df.apply(pair_state, axis=1)

    if {"m_curved", "b_curved"} <= set(df.columns):
        def curve_state(row):
            m = row["m_curved"]
            b = row["b_curved"]
            if pd.isna(m) or pd.isna(b):
                return "unknown"
            if bool(m) and bool(b):
                return "both_curved"
            if bool(m):
                return "mammalian_only"
            if bool(b):
                return "bacterial_only"
            return "neither_curved"
        df["pair_curvature_state"] = df.apply(curve_state, axis=1)

    if {"m_label", "b_label"} <= set(df.columns):
        df["label_pair"] = df["m_label"].astype(str) + " -> " + df["b_label"].astype(str)

    return df


def identify_numeric(df):
    cols = []
    for c in df.columns:
        if c in {"molecule"}:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 3:
            cols.append(c)
    return cols


def morphology_numeric_columns(df):
    exclude = set(APL_BASE) | set(APL_DERIVED) | {"ddAPL_recomputed", "ddAPL_residual"}
    out = []
    for c in identify_numeric(df):
        if c not in exclude and (
            c.startswith(("m_delta_", "b_delta_", "dd_delta_",
                          "m_endpoint_", "b_endpoint_", "dd_endpoint_",
                          "m_baseline_", "b_baseline_", "dd_baseline_"))
        ):
            out.append(c)
    return out


def primary_morphology_columns(df):
    morph = morphology_numeric_columns(df)
    return [c for c in morph if any(f in c for f in PRIMARY_FAMILIES)]


def spearman_long(df, columns):
    rows = []
    for a, b in itertools.combinations(columns, 2):
        x = pd.to_numeric(df[a], errors="coerce")
        y = pd.to_numeric(df[b], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < 3:
            continue
        rho, p = spearmanr(x[mask], y[mask])
        rows.append({"x": a, "y": b, "n": n, "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = bh_fdr(out["p"].values)
        out["abs_rho"] = out["rho"].abs()
        out = out.sort_values(["abs_rho", "q_bh"], ascending=[False, True])
    return out


def apl_vs_morph_spearman(df, apl_cols, morph_cols):
    rows = []
    for a in apl_cols:
        if a not in df.columns:
            continue
        for b in morph_cols:
            x = pd.to_numeric(df[a], errors="coerce")
            y = pd.to_numeric(df[b], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < 3:
                continue
            rho, p = spearmanr(x[mask], y[mask])
            rows.append({"apl_metric": a, "morphology_metric": b, "n": n, "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = bh_fdr(out["p"].values)
        out["abs_rho"] = out["rho"].abs()
        out = out.sort_values(["abs_rho", "q_bh"], ascending=[False, True])
    return out


def group_tests(df, numeric_cols):
    rows = []
    group_defs = [
        ("m_deformed", False, True),
        ("b_deformed", False, True),
        ("m_curved", False, True),
        ("b_curved", False, True),
    ]
    for group_col, g0, g1 in group_defs:
        if group_col not in df.columns:
            continue
        for metric in numeric_cols:
            vals = pd.to_numeric(df[metric], errors="coerce")
            a = vals[df[group_col] == g0].dropna()
            b = vals[df[group_col] == g1].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            u, p = mannwhitneyu(b, a, alternative="two-sided")
            rows.append({
                "group_column": group_col,
                "metric": metric,
                "group0": str(g0),
                "group1": str(g1),
                "n_group0": len(a),
                "n_group1": len(b),
                "median_group0": float(a.median()),
                "median_group1": float(b.median()),
                "median_difference_group1_minus_group0": float(b.median() - a.median()),
                "rank_biserial_group1_vs_group0": rank_biserial_from_u(u, len(b), len(a)),
                "p": p,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = bh_fdr(out["p"].values)
        out = out.sort_values(["q_bh", "group_column", "metric"])
    return out


def descriptives(df, numeric_cols):
    frames = []
    for g in ["m_label", "b_label", "pair_deformation_state", "pair_curvature_state",
              "deformation_transition", "curvature_transition"]:
        if g not in df.columns:
            continue
        tmp = df.groupby(g, dropna=False)[numeric_cols].agg(["count", "median", "mean", "std"])
        tmp.columns = [f"{a}__{b}" for a, b in tmp.columns]
        tmp = tmp.reset_index()
        tmp.insert(0, "grouping", g)
        tmp = tmp.rename(columns={g: "group"})
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def discrete_scatter(df, x, y, group, out, title=None, equality=False):
    cols = [x, y, group]
    d = df[cols].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if d.empty:
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    groups = list(pd.unique(d[group].astype(str)))
    cmap = plt.get_cmap("tab10", max(1, len(groups)))
    for i, value in enumerate(groups):
        g = d[d[group].astype(str) == value]
        ax.scatter(g[x], g[y], s=POINT_SIZE, alpha=POINT_ALPHA,
                   label=value, color=cmap(i), edgecolors="black", linewidths=EDGE_ALPHA)

    if equality:
        lo = np.nanmin(np.r_[d[x].values, d[y].values])
        hi = np.nanmax(np.r_[d[x].values, d[y].values])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.65)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x} | colour={group}")
    if len(groups) <= 12:
        ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return True


def continuous_scatter(df, x, y, c, out, title=None, equality=False):
    d = df[[x, y, c]].copy()
    for col in [x, y, c]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    if d.empty:
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(d[x], d[y], c=d[c], s=POINT_SIZE, alpha=POINT_ALPHA,
                    cmap="viridis", edgecolors="none")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(c)

    if equality:
        lo = np.nanmin(np.r_[d[x].values, d[y].values])
        hi = np.nanmax(np.r_[d[x].values, d[y].values])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.65)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x} | colour={c}")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return True


def heatmap_corr(df, cols, out):
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    n = len(cols)
    size = max(8, min(24, 0.33 * n + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=90, fontsize=max(5, 9 - n // 15))
    ax.set_yticklabels(cols, fontsize=max(5, 9 - n // 15))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    ax.set_title("All numeric metrics: Spearman correlation")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_worker(job):
    kind, csv_path, kwargs = job
    d = pd.read_csv(csv_path)
    if kind == "discrete":
        ok = discrete_scatter(d, **kwargs)
    elif kind == "continuous":
        ok = continuous_scatter(d, **kwargs)
    else:
        raise ValueError(kind)
    return kwargs["out"], ok


def make_jobs(df, csv_path, outdir, morph_cols, primary_cols, numeric_cols, exhaustive):
    jobs = []
    index_rows = []

    p0 = outdir / "plots" / "00_apl_plane_classes"
    p1 = outdir / "plots" / "01_apl_plane_continuous"
    p2 = outdir / "plots" / "02_apl_vs_morphology"
    p3 = outdir / "plots" / "03_primary_morphology_pairs"
    p4 = outdir / "plots" / "04_exhaustive_numeric_pairs"
    for p in [p0, p1, p2, p3] + ([p4] if exhaustive else []):
        p.mkdir(parents=True, exist_ok=True)

    # APL plane coloured by key classes.
    for group in ["pair_deformation_state", "pair_curvature_state", "m_label", "b_label"]:
        if group not in df.columns:
            continue
        out = p0 / f"m_vs_b_delta_apl__colour_{slug(group)}.png"
        kwargs = dict(
            x="m_delta_apl", y="b_delta_apl", group=group, out=str(out),
            title=f"Bacterial vs mammalian ΔAPL | colour={group}", equality=True
        )
        jobs.append(("discrete", str(csv_path), kwargs))
        index_rows.append({"path": str(out.relative_to(outdir)), "x": "m_delta_apl",
                           "y": "b_delta_apl", "colour": group, "colour_type": "categorical",
                           "section": "APL plane"})

    # APL plane coloured by every morphology metric.
    for c in morph_cols:
        out = p1 / f"m_vs_b_delta_apl__colour_{slug(c)}.png"
        kwargs = dict(
            x="m_delta_apl", y="b_delta_apl", c=c, out=str(out),
            title=f"Bacterial vs mammalian ΔAPL | colour={c}", equality=True
        )
        jobs.append(("continuous", str(csv_path), kwargs))
        index_rows.append({"path": str(out.relative_to(outdir)), "x": "m_delta_apl",
                           "y": "b_delta_apl", "colour": c, "colour_type": "continuous",
                           "section": "APL plane coloured by morphology"})

    # Every APL summary metric vs every morphology metric.
    apl_plot_cols = [c for c in (*APL_BASE, *APL_DERIVED) if c in df.columns]
    group = "pair_deformation_state" if "pair_deformation_state" in df.columns else None
    for a in apl_plot_cols:
        for m in morph_cols:
            if a == m:
                continue
            out = p2 / f"{slug(a)}__vs__{slug(m)}.png"
            if group:
                kwargs = dict(
                    x=a, y=m, group=group, out=str(out),
                    title=f"{m} vs {a} | colour={group}", equality=False
                )
                jobs.append(("discrete", str(csv_path), kwargs))
                colour, ctype = group, "categorical"
            else:
                # Fallback: colour by ddAPL.
                kwargs = dict(
                    x=a, y=m, c="ddAPL", out=str(out),
                    title=f"{m} vs {a} | colour=ddAPL", equality=False
                )
                jobs.append(("continuous", str(csv_path), kwargs))
                colour, ctype = "ddAPL", "continuous"
            index_rows.append({"path": str(out.relative_to(outdir)), "x": a, "y": m,
                               "colour": colour, "colour_type": ctype,
                               "section": "APL vs morphology"})

    # Every combination of the three primary morphology families.
    group = "pair_deformation_state" if "pair_deformation_state" in df.columns else None
    for x, y in itertools.combinations(primary_cols, 2):
        out = p3 / f"{slug(x)}__vs__{slug(y)}.png"
        if group:
            kwargs = dict(x=x, y=y, group=group, out=str(out),
                          title=f"{y} vs {x} | colour={group}", equality=False)
            jobs.append(("discrete", str(csv_path), kwargs))
            colour, ctype = group, "categorical"
        else:
            kwargs = dict(x=x, y=y, c="ddAPL", out=str(out),
                          title=f"{y} vs {x} | colour=ddAPL", equality=False)
            jobs.append(("continuous", str(csv_path), kwargs))
            colour, ctype = "ddAPL", "continuous"
        index_rows.append({"path": str(out.relative_to(outdir)), "x": x, "y": y,
                           "colour": colour, "colour_type": ctype,
                           "section": "Primary morphology pairwise"})

    # Optional: every numeric pair.
    if exhaustive:
        group = "pair_deformation_state" if "pair_deformation_state" in df.columns else None
        for x, y in itertools.combinations(numeric_cols, 2):
            out = p4 / f"{slug(x)}__vs__{slug(y)}.png"
            if group:
                kwargs = dict(x=x, y=y, group=group, out=str(out),
                              title=f"{y} vs {x} | colour={group}", equality=False)
                jobs.append(("discrete", str(csv_path), kwargs))
                colour, ctype = group, "categorical"
            else:
                kwargs = dict(x=x, y=y, c="ddAPL", out=str(out),
                              title=f"{y} vs {x} | colour=ddAPL", equality=False)
                jobs.append(("continuous", str(csv_path), kwargs))
                colour, ctype = "ddAPL", "continuous"
            index_rows.append({"path": str(out.relative_to(outdir)), "x": x, "y": y,
                               "colour": colour, "colour_type": ctype,
                               "section": "Exhaustive numeric pairwise"})

    return jobs, pd.DataFrame(index_rows)


def write_xlsx(out_path, master, tables):
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            master.to_excel(writer, sheet_name="master", index=False)
            for name, frame in tables.items():
                if frame is None or frame.empty:
                    continue
                safe = name[:31]
                frame.to_excel(writer, sheet_name=safe, index=False)
    except ImportError:
        print("WARNING: openpyxl unavailable; skipping XLSX output.", file=sys.stderr)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out or f"apl_morphology_analysis_{timestamp}").resolve()
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    workers = args.workers
    if workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
        if workers <= 0:
            workers = min(8, os.cpu_count() or 1)
    workers = max(1, workers)

    raw = read_table(args.input, args.sheet)
    dapl = read_table(args.dapl, args.dapl_sheet) if args.dapl else None
    master = prepare_master(raw, dapl)

    master_csv = outdir / "analysis_master.csv"
    master.to_csv(master_csv, index=False)

    numeric_cols = identify_numeric(master)
    morph_cols = morphology_numeric_columns(master)
    primary_cols = primary_morphology_columns(master)
    apl_cols = [c for c in (*APL_BASE, *APL_DERIVED) if c in master.columns]

    all_corr = spearman_long(master, numeric_cols)
    apl_morph_corr = apl_vs_morph_spearman(master, apl_cols, morph_cols)
    tests = group_tests(master, list(dict.fromkeys(apl_cols + morph_cols)))
    desc = descriptives(master, list(dict.fromkeys(apl_cols + morph_cols)))

    tables = {
        "all_numeric_spearman": all_corr,
        "apl_vs_morph_spearman": apl_morph_corr,
        "group_tests": tests,
        "descriptives": desc,
    }
    for name, frame in tables.items():
        frame.to_csv(outdir / "tables" / f"{name}.csv", index=False)

    # Useful strongest-relationship tables.
    if not apl_morph_corr.empty:
        apl_morph_corr.head(50).to_csv(
            outdir / "tables" / "top50_apl_morphology_relationships.csv", index=False
        )
    if not all_corr.empty:
        all_corr.head(100).to_csv(
            outdir / "tables" / "top100_all_relationships.csv", index=False
        )

    # Correlation heatmaps.
    heatmap_corr(master, numeric_cols, outdir / "plots" / "all_numeric_spearman_heatmap.png")
    core_heat_cols = list(dict.fromkeys(apl_cols + primary_cols))
    if len(core_heat_cols) >= 2:
        heatmap_corr(master, core_heat_cols, outdir / "plots" / "apl_primary_morphology_heatmap.png")

    jobs, plot_index = make_jobs(
        master, master_csv, outdir, morph_cols, primary_cols, numeric_cols,
        args.exhaustive_pair_plots
    )
    plot_index.to_csv(outdir / "plot_index.csv", index=False)

    print(f"Rows: {len(master)}")
    print(f"Numeric metrics: {len(numeric_cols)}")
    print(f"Morphology metrics: {len(morph_cols)}")
    print(f"Primary morphology metrics: {len(primary_cols)}")
    print(f"Plots queued: {len(jobs)}")
    print(f"Workers: {workers}")

    failures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(plot_worker, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                path, ok = fut.result()
                if not ok:
                    failures.append((path, "no complete rows"))
            except Exception as e:
                failures.append(("<worker>", repr(e)))
            if i % 100 == 0 or i == len(futures):
                print(f"Plots completed: {i}/{len(futures)}")

    pd.DataFrame(failures, columns=["path", "error"]).to_csv(
        outdir / "plot_failures.csv", index=False
    )

    # XLSX: master + compact statistical tables.
    if not args.no_xlsx:
        write_xlsx(outdir / "analysis_bundle.xlsx", master, tables)

    # Summary.
    residual = pd.to_numeric(master["ddAPL_residual"], errors="coerce")
    summary = [
        "APL ↔ morphology analysis bundle",
        "=" * 34,
        f"Input: {Path(args.input).resolve()}",
        f"dAPL input: {Path(args.dapl).resolve() if args.dapl else 'contained in main input'}",
        f"Rows: {len(master)}",
        f"Numeric metrics: {len(numeric_cols)}",
        f"Morphology metrics: {len(morph_cols)}",
        f"Primary morphology metrics: {len(primary_cols)}",
        f"Plots requested: {len(jobs)}",
        f"Plot failures: {len(failures)}",
        f"ddAPL check max abs residual (existing - [b-m]): "
        f"{float(residual.abs().max()) if residual.notna().any() else float('nan'):.10g}",
        "",
        "Key files:",
        "  analysis_master.csv",
        "  analysis_bundle.xlsx",
        "  tables/all_numeric_spearman.csv",
        "  tables/apl_vs_morph_spearman.csv",
        "  tables/group_tests.csv",
        "  plot_index.csv",
        "  plots/",
        "",
        "Definitions:",
        "  ddAPL = b_delta_apl - m_delta_apl",
        "  mean_delta_apl = (b_delta_apl + m_delta_apl) / 2",
        "  response_magnitude = sqrt(b_delta_apl^2 + m_delta_apl^2)",
        "  abs_ddAPL = abs(ddAPL)",
    ]
    (outdir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"\nDONE: {outdir}")
    print(f"Main spreadsheet: {outdir / 'analysis_bundle.xlsx'}")
    print(f"Plot index: {outdir / 'plot_index.csv'}")
    print(f"Summary: {outdir / 'SUMMARY.txt'}")


if __name__ == "__main__":
    main()
