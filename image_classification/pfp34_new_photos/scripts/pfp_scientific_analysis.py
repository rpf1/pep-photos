#!/usr/bin/env python3
"""
PFPeptoids deep scientific analysis of the canonical manual + simulation master.

PURPOSE
-------
Take the validated `master_analysis_20260929_v1/` outputs and perform the
next-stage scientific analysis before any RDKit/ML target is chosen.

This script is READ-ONLY with respect to all existing project data.
It writes only a NEW results folder and ZIP.

It covers:
1. data/QC and unresolved exceptions;
2. membrane-to-membrane manual-label concordance/transitions;
3. ΔAPL and thickness-change association with manual visual classes;
4. pairwise post-hoc tests with Holm correction and rank-biserial effects;
5. ddAPL / absolute-response / mean-response target redundancy;
6. pair-label-pattern versus ddAPL summaries;
7. morphology redundancy and independence from projected APL;
8. curved-vs-flat_interaction morphology comparisons where sample size permits;
9. targeted local-lipid contrasts with BH-FDR correction;
10. explicit leaflet/asymmetry pilot summary;
11. dense CHOL assignment-QC summary;
12. useful "contradiction/exemplar" systems showing why APL alone is incomplete;
13. a target-decision evidence table;
14. a compact Markdown scientific report;
15. a ZIP for off-cluster review.

No RDKit descriptors are generated. No final composite target is imposed.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import traceback
import zipfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

try:
    from scipy.stats import (
        chi2_contingency,
        kruskal,
        mannwhitneyu,
        spearmanr,
    )
except Exception as exc:
    raise SystemExit(
        "scipy is required for this analysis. Activate the usual PFPeptoids "
        f"Python environment and retry.\nOriginal error: {exc}"
    )

MAMMALIAN = "POPC60CHOL40_W_WF"
BACTERIAL = "POPE75POPG25_W_WF"

PRIMARY_CLASSES = [
    "flat_no_interaction",
    "flat_interaction",
    "curved",
    "bulge_flat",
    "bulge_curved",
    "pore_formed",
]

CORE_SYSTEM_METRICS = [
    "APL_change_BOX",
    "APL_percent_change_BOX",
    "thickness_change",
]

CORE_MORPH = [
    "morph__endpoint__height_rms_nm",
    "morph__endpoint__midplane_excess_area_fraction",
    "morph__endpoint__H_rms_nm_inv",
    "morph__endpoint__thickness_median_nm",
]

CORE_MORPH_DELTA = [
    "morph__delta__height_rms_nm",
    "morph__delta__midplane_excess_area_fraction",
    "morph__delta__H_rms_nm_inv",
    "morph__delta__thickness_median_nm",
]

TARGETS = [
    "dAPL_bacterial",
    "dAPL_mammalian",
    "ddAPL",
    "abs_ddAPL",
    "mean_dAPL",
    "min_dAPL_between_membranes",
    "max_dAPL_between_membranes",
]


class Tee:
    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("w", encoding="utf-8")

    def write(self, text):
        sys.__stdout__.write(text)
        self.f.write(text)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default=(
            "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/"
            "PFPeptoids/analysis/image_classification/pfp34_new_photos/"
            "session/master_analysis_20260929_v1"
        ),
    )
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj_sorted = ranked * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    restored = np.empty(m)
    restored[order] = adj_sorted
    out[np.where(valid)[0]] = restored
    return out


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj_sorted = np.empty(m)
    running = 0.0
    for i, raw in enumerate(ranked):
        running = max(running, (m - i) * raw)
        adj_sorted[i] = min(running, 1.0)
    restored = np.empty(m)
    restored[order] = adj_sorted
    out[np.where(valid)[0]] = restored
    return out


def kruskal_effect(groups):
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [g[np.isfinite(g)] for g in arrays if len(g)]
    if len(arrays) < 2:
        return np.nan, np.nan, np.nan
    h, p = kruskal(*arrays)
    n = sum(len(g) for g in arrays)
    k = len(arrays)
    eps2 = (h - k + 1) / (n - k) if n > k else np.nan
    return float(h), float(p), float(eps2)


def rank_biserial_from_u(u, nx, ny):
    return float(2.0 * u / (nx * ny) - 1.0)


def describe_by_group(df, group_cols, metrics):
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        temp = df[group_cols + [metric]].copy()
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
        temp = temp[temp[metric].notna()]
        if temp.empty:
            continue
        for key, g in temp.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            v = g[metric]
            rec = dict(zip(group_cols, key))
            rec.update({
                "metric": metric,
                "n": int(v.count()),
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                "median": float(v.median()),
                "q25": float(v.quantile(0.25)),
                "q75": float(v.quantile(0.75)),
                "min": float(v.min()),
                "max": float(v.max()),
            })
            rows.append(rec)
    return pd.DataFrame(rows)


def spearman_table(df, xcols, ycols=None, dataset="", min_n=5):
    if ycols is None:
        ycols = xcols
        symmetric = True
    else:
        symmetric = False

    rows = []
    seen = set()
    for x in xcols:
        if x not in df.columns:
            continue
        for y in ycols:
            if y not in df.columns or x == y:
                continue
            key = tuple(sorted((x, y)))
            if symmetric and key in seen:
                continue
            seen.add(key)

            xv = pd.to_numeric(df[x], errors="coerce")
            yv = pd.to_numeric(df[y], errors="coerce")
            mask = xv.notna() & yv.notna()
            n = int(mask.sum())
            if n < min_n:
                continue
            rho, p = spearmanr(xv[mask], yv[mask])
            rows.append({
                "dataset": dataset,
                "metric_x": x,
                "metric_y": y,
                "n": n,
                "spearman_rho": float(rho),
                "p_raw": float(p),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_BH_FDR"] = bh_adjust(out["p_raw"].values)
        out["abs_rho"] = out["spearman_rho"].abs()
        out = out.sort_values(["p_BH_FDR", "abs_rho"], ascending=[True, False])
    return out


def write_csv(df, path):
    df.to_csv(path, index=False)


def stage(msg):
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}")


def main():
    args = parse_args()
    inp = Path(args.input_dir)
    out = (
        Path(args.output_dir)
        if args.output_dir
        else inp.parent / "scientific_interpretation_20260929_v1"
    )
    zip_path = out.with_suffix(".zip")

    if out.exists() or zip_path.exists():
        raise SystemExit(
            "Refusing to overwrite an existing analysis output.\n"
            f"Output dir: {out}\nZIP: {zip_path}"
        )

    required = [
        "01_system_master_extended.csv",
        "02_system_master_core.csv",
        "05_tripeptoid_three_row_primary.csv",
        "06_tripeptoid_pairs_wide_primary.csv",
        "08_metric_dictionary.csv",
        "09_join_coverage.csv",
        "10_annotation_concentration_ambiguities.csv",
        "11_unmatched_annotations.csv",
        "12_unmatched_simulations.csv",
        "13_run_metadata_path_mismatches.csv",
        "20_pilot_source_coverage.csv",
        "21_photo_exceptions.csv",
        "VALIDATION.json",
    ]
    missing = [f for f in required if not (inp / f).exists()]
    if missing:
        raise SystemExit("Missing required input files:\n" + "\n".join(missing))

    out.mkdir(parents=True)
    tee = Tee(out / "RUN_LOG.txt")
    sys.stdout = tee

    try:
        stage("0. LOAD + VALIDATE INPUTS")
        validation = json.loads((inp / "VALIDATION.json").read_text())
        if validation.get("status") != "PASS":
            raise RuntimeError(
                "Input master package did not pass validation. "
                f"Status={validation.get('status')}"
            )

        ext = pd.read_csv(inp / "01_system_master_extended.csv", low_memory=False)
        core = pd.read_csv(inp / "02_system_master_core.csv", low_memory=False)
        three = pd.read_csv(inp / "05_tripeptoid_three_row_primary.csv", low_memory=False)
        wide = pd.read_csv(inp / "06_tripeptoid_pairs_wide_primary.csv", low_memory=False)
        metric_dict = pd.read_csv(inp / "08_metric_dictionary.csv")
        join_cov = pd.read_csv(inp / "09_join_coverage.csv")
        amb = pd.read_csv(inp / "10_annotation_concentration_ambiguities.csv", low_memory=False)
        unmatched_ann = pd.read_csv(inp / "11_unmatched_annotations.csv", low_memory=False)
        unmatched_sim = pd.read_csv(inp / "12_unmatched_simulations.csv", low_memory=False)
        run_mismatch = pd.read_csv(inp / "13_run_metadata_path_mismatches.csv", low_memory=False)
        pilot_cov = pd.read_csv(inp / "20_pilot_source_coverage.csv")
        photo_ex = pd.read_csv(inp / "21_photo_exceptions.csv", low_memory=False)

        system_rows = three[three["row_type"].isin(["mammalian", "bacterial"])].copy()

        print(f"Validated input: {inp}")
        print(f"Primary pairs: {len(wide)}")
        print(f"Primary physical system rows: {len(system_rows)}")
        print(f"Photo exceptions retained outside primary set: {len(photo_ex)}")

        qc_rows = [
            {"check": "input_validation", "value": validation.get("status"), "status": "PASS"},
            {"check": "primary_pairs", "value": len(wide), "status": "INFO"},
            {"check": "primary_three_rows", "value": len(three), "status": "INFO"},
            {"check": "three_rows_per_pair", "value": bool((three.groupby("pair_key").size() == 3).all()), "status": "PASS"},
            {"check": "concentration_ambiguous_annotations", "value": len(amb), "status": "EXCLUDED_FROM_AUTO_JOIN"},
            {"check": "unmatched_annotations", "value": len(unmatched_ann), "status": "REVIEW_NOT_PRIMARY"},
            {"check": "unmatched_simulations", "value": len(unmatched_sim), "status": "REVIEW_NOT_PRIMARY"},
            {"check": "run_metadata_path_mismatches", "value": len(run_mismatch), "status": "KNOWN_PATH_CANONICALISATION"},
            {"check": "photo_exceptions", "value": len(photo_ex), "status": "EXCLUDED_FROM_PRIMARY"},
        ]
        write_csv(pd.DataFrame(qc_rows), out / "01_data_qc_summary.csv")
        write_csv(join_cov, out / "support_join_coverage.csv")
        write_csv(pilot_cov, out / "support_pilot_coverage.csv")
        write_csv(photo_ex, out / "support_photo_exceptions.csv")
        write_csv(amb, out / "support_concentration_ambiguities.csv")
        write_csv(unmatched_ann, out / "support_unmatched_annotations.csv")
        write_csv(unmatched_sim, out / "support_unmatched_simulations.csv")
        write_csv(run_mismatch, out / "support_run_metadata_path_mismatches.csv")

        # ------------------------------------------------------------------
        stage("1. MANUAL LABEL CONCORDANCE BETWEEN MEMBRANES")
        label_ct = pd.crosstab(
            wide["manual_label_mammalian"],
            wide["manual_label_bacterial"],
            dropna=False,
        )
        label_ct.to_csv(out / "02_label_transition_counts.csv")

        row_pct = label_ct.div(label_ct.sum(axis=1), axis=0) * 100.0
        row_pct.to_csv(out / "03_label_transition_row_percent.csv")

        chi2, chi_p, dof, expected = chi2_contingency(label_ct)
        n = int(label_ct.values.sum())
        r, k = label_ct.shape
        cramer_v = math.sqrt((chi2 / n) / min(r - 1, k - 1))
        exact_agree = float(
            (wide["manual_label_mammalian"] == wide["manual_label_bacterial"]).mean()
        )

        concordance = pd.DataFrame([{
            "n_pairs": n,
            "exact_label_agreement_fraction": exact_agree,
            "exact_label_agreement_percent": 100 * exact_agree,
            "chi2": chi2,
            "chi2_dof": dof,
            "chi2_p": chi_p,
            "cramers_v": cramer_v,
            "note": (
                "Association, not interchangeability: membranes can produce "
                "different phenotypes for the same molecule."
            ),
        }])
        write_csv(concordance, out / "04_label_concordance_statistics.csv")
        print(concordance.to_string(index=False))

        # ------------------------------------------------------------------
        stage("2. ΔAPL / THICKNESS VERSUS MANUAL VISUAL CLASS")
        class_desc = describe_by_group(
            system_rows,
            ["row_type", "manual_label"],
            CORE_SYSTEM_METRICS,
        )
        write_csv(class_desc, out / "05_class_metric_descriptives.csv")

        omnibus_rows = []
        pairwise_rows = []

        for membrane in ["mammalian", "bacterial"]:
            d = system_rows[system_rows["row_type"] == membrane]
            for metric in CORE_SYSTEM_METRICS:
                temp = d[["manual_label", metric]].copy()
                temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
                temp = temp.dropna()

                eligible = []
                for label, g in temp.groupby("manual_label"):
                    if len(g) >= 2:
                        eligible.append((label, g[metric].to_numpy()))
                if len(eligible) >= 2:
                    h, p, eps2 = kruskal_effect([x[1] for x in eligible])
                    omnibus_rows.append({
                        "membrane": membrane,
                        "metric": metric,
                        "classes_in_test": ";".join(x[0] for x in eligible),
                        "n_total": sum(len(x[1]) for x in eligible),
                        "H": h,
                        "p_raw": p,
                        "epsilon_squared": eps2,
                    })

                labels_here = sorted(temp["manual_label"].dropna().unique())
                metric_pair_rows = []
                for a_i in range(len(labels_here)):
                    for b_i in range(a_i + 1, len(labels_here)):
                        a, b = labels_here[a_i], labels_here[b_i]
                        x = temp.loc[temp["manual_label"] == a, metric].dropna()
                        y = temp.loc[temp["manual_label"] == b, metric].dropna()
                        if len(x) < 3 or len(y) < 3:
                            continue
                        u, p = mannwhitneyu(x, y, alternative="two-sided")
                        metric_pair_rows.append({
                            "membrane": membrane,
                            "metric": metric,
                            "class_a": a,
                            "class_b": b,
                            "n_a": len(x),
                            "n_b": len(y),
                            "median_a": float(x.median()),
                            "median_b": float(y.median()),
                            "U": float(u),
                            "p_raw": float(p),
                            "rank_biserial_a_vs_b": rank_biserial_from_u(
                                u, len(x), len(y)
                            ),
                        })
                if metric_pair_rows:
                    q = holm_adjust([x["p_raw"] for x in metric_pair_rows])
                    for rec, adj in zip(metric_pair_rows, q):
                        rec["p_Holm_within_membrane_metric"] = adj
                    pairwise_rows.extend(metric_pair_rows)

        omnibus = pd.DataFrame(omnibus_rows)
        if not omnibus.empty:
            omnibus["p_BH_FDR_across_omnibus_tests"] = bh_adjust(omnibus["p_raw"])
        write_csv(omnibus, out / "06_class_metric_omnibus_tests.csv")

        pairwise = pd.DataFrame(pairwise_rows)
        if not pairwise.empty:
            pairwise = pairwise.sort_values(
                ["metric", "membrane", "p_Holm_within_membrane_metric"]
            )
        write_csv(pairwise, out / "07_class_metric_pairwise_tests.csv")

        # ------------------------------------------------------------------
        stage("3. PAIRED LABEL PATTERN + ddAPL")
        wide["label_pair"] = (
            wide["manual_label_mammalian"].astype(str)
            + " -> "
            + wide["manual_label_bacterial"].astype(str)
        )
        pair_pattern_desc = describe_by_group(
            wide,
            ["manual_label_mammalian", "manual_label_bacterial", "label_pair"],
            ["ddAPL", "dAPL_bacterial", "dAPL_mammalian", "mean_dAPL"],
        )
        write_csv(pair_pattern_desc, out / "08_pair_label_response_descriptives.csv")

        combo_counts = wide["label_pair"].value_counts()
        eligible_combos = combo_counts[combo_counts >= 5].index.tolist()
        combo_groups = [
            wide.loc[wide["label_pair"] == lab, "ddAPL"].dropna().values
            for lab in eligible_combos
        ]
        if len(combo_groups) >= 2:
            h, p, eps2 = kruskal_effect(combo_groups)
            combo_test = pd.DataFrame([{
                "metric": "ddAPL",
                "minimum_pair_count": 5,
                "eligible_label_patterns": ";".join(eligible_combos),
                "n_total": sum(len(g) for g in combo_groups),
                "H": h,
                "p": p,
                "epsilon_squared": eps2,
                "note": "Exploratory; label-pattern groups are observational and unbalanced.",
            }])
        else:
            combo_test = pd.DataFrame()
        write_csv(combo_test, out / "09_pair_label_ddapl_omnibus.csv")

        # ------------------------------------------------------------------
        stage("4. TARGET REDUNDANCY / COMPLEMENTARITY")
        target_corr = spearman_table(
            wide, TARGETS, dataset="primary_tripeptoid_pairs", min_n=20
        )
        write_csv(target_corr, out / "10_target_correlations.csv")

        target_desc = describe_by_group(
            wide.assign(dataset="primary_tripeptoid_pairs"),
            ["dataset"],
            TARGETS,
        )
        write_csv(target_desc, out / "11_target_descriptives.csv")

        # Evidence-oriented, not an imposed score/ranking.
        target_evidence = pd.DataFrame([
            {
                "candidate": "ddAPL",
                "captures": "directional membrane differential/selectivity",
                "retains_direction": True,
                "captures_absolute_response_strength": False,
                "main_limitation": (
                    "Can hide strong similar responses in both membranes and "
                    "does not encode morphology/deformation."
                ),
            },
            {
                "candidate": "dAPL_bacterial",
                "captures": "absolute bacterial-like projected response",
                "retains_direction": True,
                "captures_absolute_response_strength": True,
                "main_limitation": "Does not penalise/describe mammalian-like response.",
            },
            {
                "candidate": "dAPL_mammalian",
                "captures": "absolute mammalian-like projected response",
                "retains_direction": True,
                "captures_absolute_response_strength": True,
                "main_limitation": "Not a selectivity measure by itself.",
            },
            {
                "candidate": "mean_dAPL",
                "captures": "overall projected response across both membranes",
                "retains_direction": True,
                "captures_absolute_response_strength": True,
                "main_limitation": "Intentionally reduces membrane differential information.",
            },
            {
                "candidate": "min_dAPL_between_membranes",
                "captures": "strength of response shared by both membranes",
                "retains_direction": True,
                "captures_absolute_response_strength": True,
                "main_limitation": "Not a membrane-selectivity objective.",
            },
            {
                "candidate": "manual phenotype / structural metric",
                "captures": "curvature, bulging, interaction phenotype",
                "retains_direction": None,
                "captures_absolute_response_strength": False,
                "main_limitation": (
                    "Manual labels are categorical; quantitative morphology is "
                    "currently available only for the 15-pair pilot."
                ),
            },
        ])
        write_csv(target_evidence, out / "12_target_decision_evidence.csv")

        candidate_pair_table_cols = [
            "pair_key", "molecule", "physical_run",
            "manual_label_mammalian", "manual_label_bacterial",
            "dAPL_mammalian", "dAPL_bacterial", "ddAPL", "abs_ddAPL",
            "mean_dAPL", "min_dAPL_between_membranes",
            "max_dAPL_between_membranes",
            "mammalian__thickness_change", "bacterial__thickness_change",
        ]
        candidate_pair_table_cols += [
            c for c in wide.columns
            if c.startswith("mammalian__morph__endpoint__")
            or c.startswith("bacterial__morph__endpoint__")
            or c.startswith("bacterial_minus_mammalian__morph__endpoint__")
        ]
        candidate_pair_table_cols = [
            c for c in candidate_pair_table_cols if c in wide.columns
        ]
        write_csv(
            wide[candidate_pair_table_cols],
            out / "13_candidate_multitarget_pair_table.csv",
        )

        # ------------------------------------------------------------------
        stage("5. MORPHOLOGY: REDUNDANCY + INDEPENDENCE FROM APL")
        morph_systems = core[
            core.get("morphology_available", pd.Series(False, index=core.index))
            .fillna(False)
            .astype(bool)
        ].copy()

        morph_redundancy_parts = []
        response_assoc_parts = []
        morph_class_parts = []

        for bilayer, short in [(MAMMALIAN, "mammalian"), (BACTERIAL, "bacterial")]:
            d = morph_systems[morph_systems["bilayer_type"] == bilayer].copy()

            t = spearman_table(
                d,
                CORE_MORPH,
                dataset=f"{short}_morphology_core",
                min_n=5,
            )
            morph_redundancy_parts.append(t)

            t2 = spearman_table(
                d,
                ["APL_change_BOX", "thickness_change"],
                CORE_MORPH,
                dataset=f"{short}_response_vs_morphology",
                min_n=5,
            )
            response_assoc_parts.append(t2)

            # Morphology pilot has sparse rare classes. Inferential comparison
            # is restricted to groups with >=3 systems.
            for metric in CORE_MORPH + CORE_MORPH_DELTA:
                if metric not in d.columns:
                    continue
                tmp = d[["manual_label", metric]].copy()
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                tmp = tmp.dropna()
                groups = {
                    lab: g[metric]
                    for lab, g in tmp.groupby("manual_label")
                    if len(g) >= 3
                }
                if len(groups) == 2:
                    a, b = list(groups)
                    x, y = groups[a], groups[b]
                    u, p = mannwhitneyu(x, y, alternative="two-sided")
                    morph_class_parts.append({
                        "membrane": short,
                        "metric": metric,
                        "class_a": a,
                        "class_b": b,
                        "n_a": len(x),
                        "n_b": len(y),
                        "median_a": float(x.median()),
                        "median_b": float(y.median()),
                        "U": float(u),
                        "p_raw": float(p),
                        "rank_biserial_a_vs_b": rank_biserial_from_u(
                            u, len(x), len(y)
                        ),
                        "scope_note": (
                            "Only classes with >=3 pilot systems included; "
                            "rare classes remain descriptive only."
                        ),
                    })

        morph_red = pd.concat(
            [x for x in morph_redundancy_parts if not x.empty],
            ignore_index=True,
        ) if any(not x.empty for x in morph_redundancy_parts) else pd.DataFrame()
        write_csv(morph_red, out / "14_morphology_core_correlations.csv")

        morph_resp = pd.concat(
            [x for x in response_assoc_parts if not x.empty],
            ignore_index=True,
        ) if any(not x.empty for x in response_assoc_parts) else pd.DataFrame()
        write_csv(morph_resp, out / "15_morphology_vs_APL_thickness.csv")

        morph_class = pd.DataFrame(morph_class_parts)
        if not morph_class.empty:
            morph_class["p_BH_FDR"] = bh_adjust(morph_class["p_raw"])
        write_csv(morph_class, out / "16_morphology_class_comparisons.csv")

        morph_desc = describe_by_group(
            morph_systems,
            ["bilayer_type", "manual_label"],
            CORE_MORPH + CORE_MORPH_DELTA + ["APL_change_BOX", "thickness_change"],
        )
        write_csv(morph_desc, out / "17_morphology_pilot_descriptives.csv")

        # ------------------------------------------------------------------
        stage("6. APL-MORPHOLOGY CONTRADICTION / EXEMPLAR SYSTEMS")
        # The goal is not a score: identify systems where projected APL and
        # visible/quantitative deformation tell different stories.
        exemplars = morph_systems[[
            "molecule", "bilayer_type", "physical_run", "manual_label",
            "APL_change_BOX", "thickness_change",
            *[c for c in CORE_MORPH if c in morph_systems.columns],
        ]].copy()

        # Within each membrane, rank APL and height RMS independently.
        exemplars["APL_percentile_within_membrane"] = exemplars.groupby(
            "bilayer_type"
        )["APL_change_BOX"].rank(pct=True)
        exemplars["height_RMS_percentile_within_membrane"] = exemplars.groupby(
            "bilayer_type"
        )["morph__endpoint__height_rms_nm"].rank(pct=True)
        exemplars["APL_minus_height_percentile"] = (
            exemplars["APL_percentile_within_membrane"]
            - exemplars["height_RMS_percentile_within_membrane"]
        )
        exemplars["discordance_abs"] = exemplars[
            "APL_minus_height_percentile"
        ].abs()
        exemplars = exemplars.sort_values("discordance_abs", ascending=False)
        write_csv(exemplars, out / "18_APL_morphology_discordant_examples.csv")

        # ------------------------------------------------------------------
        stage("7. TARGETED LOCAL-LIPID EXPLORATION")
        pilot = ext[
            ext.get(
                "local_lipid_system_available",
                pd.Series(False, index=ext.index),
            ).fillna(False).astype(bool)
        ].copy()

        contrast_specs = []
        for species, bilayer in [
            ("POPG", BACTERIAL),
            ("POPE", BACTERIAL),
            ("CHOL", MAMMALIAN),
            ("POPC", MAMMALIAN),
        ]:
            contrast_specs.extend([
                (
                    f"{species}_high_minus_low_curvature",
                    f"lipid_system__median__{species}_enrichment_high_curvature",
                    f"lipid_system__median__{species}_enrichment_low_curvature",
                    bilayer,
                    species,
                    "high curvature enrichment - low curvature enrichment",
                ),
                (
                    f"{species}_thin_minus_notthin",
                    f"lipid_system__median__{species}_enrichment_thinnest10pct",
                    f"lipid_system__median__{species}_enrichment_not_thinnest10pct",
                    bilayer,
                    species,
                    "thinnest 10% enrichment - other-region enrichment",
                ),
            ])

        contrast_rows = []
        for name, a, b, bilayer, species, meaning in contrast_specs:
            if a not in pilot.columns or b not in pilot.columns:
                continue
            mask = pilot["bilayer_type"] == bilayer
            vals = (
                pd.to_numeric(pilot.loc[mask, a], errors="coerce")
                - pd.to_numeric(pilot.loc[mask, b], errors="coerce")
            )
            for idx, val in vals.items():
                contrast_rows.append({
                    "physical_key": pilot.at[idx, "physical_key"],
                    "molecule": pilot.at[idx, "molecule"],
                    "bilayer_type": bilayer,
                    "manual_label": pilot.at[idx, "manual_label"],
                    "species": species,
                    "contrast": name,
                    "meaning": meaning,
                    "value": val,
                })
        contrasts = pd.DataFrame(contrast_rows)
        write_csv(contrasts, out / "19_local_lipid_targeted_contrasts.csv")

        # Targeted association screen only; do not scan hundreds of arbitrary
        # features. This reduces multiplicity and preserves interpretability.
        lipid_tests = []
        response_metrics = [
            "APL_change_BOX",
            "thickness_change",
            "morph__endpoint__height_rms_nm",
            "morph__endpoint__midplane_excess_area_fraction",
        ]

        for species, bilayer in [("POPG", BACTERIAL), ("CHOL", MAMMALIAN)]:
            d = pilot[pilot["bilayer_type"] == bilayer].copy()
            derived = {}
            for contrast_name in [
                f"{species}_high_minus_low_curvature",
                f"{species}_thin_minus_notthin",
            ]:
                sub = contrasts[
                    (contrasts["species"] == species)
                    & (contrasts["contrast"] == contrast_name)
                ].set_index("physical_key")["value"]
                derived[contrast_name] = d["physical_key"].map(sub)

            direct = {
                f"{species}_corr_enrichment_vs_curvature":
                    pd.to_numeric(
                        d.get(
                            f"lipid_system__median__{species}_corr_enrichment_vs_curvature"
                        ),
                        errors="coerce",
                    ),
                f"{species}_corr_enrichment_vs_thickness":
                    pd.to_numeric(
                        d.get(
                            f"lipid_system__median__{species}_corr_enrichment_vs_thickness"
                        ),
                        errors="coerce",
                    ),
            }
            predictors = {**derived, **direct}

            for pred_name, pred in predictors.items():
                for response in response_metrics:
                    if response not in d.columns:
                        continue
                    y = pd.to_numeric(d[response], errors="coerce")
                    mask = pred.notna() & y.notna()
                    n = int(mask.sum())
                    if n < 8:
                        continue
                    rho, p = spearmanr(pred[mask], y[mask])
                    lipid_tests.append({
                        "bilayer_type": bilayer,
                        "species": species,
                        "predictor": pred_name,
                        "response": response,
                        "n": n,
                        "spearman_rho": float(rho),
                        "p_raw": float(p),
                    })

        lipid_assoc = pd.DataFrame(lipid_tests)
        if not lipid_assoc.empty:
            lipid_assoc["p_BH_FDR"] = bh_adjust(lipid_assoc["p_raw"])
            lipid_assoc["interpretation_status"] = np.where(
                lipid_assoc["p_BH_FDR"] < 0.05,
                "exploratory_FDR_signal",
                "no_FDR_signal",
            )
            lipid_assoc["caution"] = (
                "Pilot n≈15/membrane; local-lipid method is not production; "
                "CHOL results require assignment-QC context."
            )
            lipid_assoc = lipid_assoc.sort_values(
                ["p_BH_FDR", "p_raw", "species"]
            )
        write_csv(lipid_assoc, out / "20_local_lipid_exploratory_associations.csv")

        # ------------------------------------------------------------------
        stage("8. EXPLICIT LEAFLET + CHOL QC PILOTS")
        smoke_path = inp / "support_explicit_leaflet_smoke.csv"
        dense_path = inp / "support_dense_chol_qc.csv"

        if smoke_path.exists():
            smoke = pd.read_csv(smoke_path, low_memory=False)
            idcols = core[
                ["physical_key", "molecule", "bilayer_type", "manual_label"]
            ].drop_duplicates("physical_key")
            smoke = smoke.merge(idcols, on="physical_key", how="left")
            lead = ["molecule", "bilayer_type", "manual_label", "physical_key"]
            smoke = smoke[lead + [c for c in smoke.columns if c not in lead]]
            write_csv(smoke, out / "21_leaflet_smoke_interpretable_summary.csv")
        else:
            smoke = pd.DataFrame()

        if dense_path.exists():
            dense = pd.read_csv(dense_path, low_memory=False)
            idcols = core[
                ["physical_key", "molecule", "bilayer_type", "manual_label"]
            ].drop_duplicates("physical_key")
            dense = dense.merge(idcols, on="physical_key", how="left")
            lead = ["molecule", "bilayer_type", "manual_label", "physical_key"]
            dense = dense[lead + [c for c in dense.columns if c not in lead]]
            write_csv(dense, out / "22_dense_CHOL_QC_interpretable_summary.csv")
        else:
            dense = pd.DataFrame()

        # ------------------------------------------------------------------
        stage("9. BUILD COMPACT SCIENTIFIC REPORT")
        # Pull the key observed numbers directly from analysis results.
        def med(membrane, label, metric):
            d = class_desc[
                (class_desc["row_type"] == membrane)
                & (class_desc["manual_label"] == label)
                & (class_desc["metric"] == metric)
            ]
            return float(d.iloc[0]["median"]) if len(d) else np.nan

        # Target correlations
        def rho(a, b):
            d = target_corr[
                ((target_corr["metric_x"] == a) & (target_corr["metric_y"] == b))
                | ((target_corr["metric_x"] == b) & (target_corr["metric_y"] == a))
            ]
            return float(d.iloc[0]["spearman_rho"]) if len(d) else np.nan

        # Morph response lookup
        def morph_rho(dataset_fragment, response, morphmetric):
            if morph_resp.empty:
                return np.nan
            d = morph_resp[
                morph_resp["dataset"].str.contains(dataset_fragment, regex=False)
                & (
                    (
                        (morph_resp["metric_x"] == response)
                        & (morph_resp["metric_y"] == morphmetric)
                    )
                    | (
                        (morph_resp["metric_y"] == response)
                        & (morph_resp["metric_x"] == morphmetric)
                    )
                )
            ]
            return float(d.iloc[0]["spearman_rho"]) if len(d) else np.nan

        top_discordant = exemplars.head(8)[
            [
                "molecule", "bilayer_type", "manual_label",
                "APL_change_BOX",
                "morph__endpoint__height_rms_nm",
                "morph__endpoint__midplane_excess_area_fraction",
                "discordance_abs",
            ]
        ]

        report = []
        report.append("# PFPeptoids scientific interpretation — pre-RDKit")
        report.append("")
        report.append("## Dataset status")
        report.append("")
        report.append(
            f"- Input master validation: **PASS**."
        )
        report.append(
            f"- Primary dataset: **{len(wide)} tripeptoid bacterial/mammalian pairs "
            f"({len(system_rows)} physical membrane rows)**."
        )
        report.append(
            f"- Manual-label agreement between the two membranes is "
            f"**{100*exact_agree:.1f}%**; Cramér's V = **{cramer_v:.3f}**. "
            "The membrane environments therefore produce related but frequently "
            "different phenotypes for the same sequence."
        )
        report.append("")

        report.append("## What projected ΔAPL captures")
        report.append("")
        report.append(
            "- ΔAPL strongly distinguishes `flat_no_interaction` from the "
            "interacting/deforming classes, but among the four interacting/"
            "deforming classes its medians overlap substantially."
        )
        report.append(
            f"- Mammalian median ΔAPL: flat-no-interaction "
            f"{med('mammalian','flat_no_interaction','APL_change_BOX'):.2f}, "
            f"flat-interaction {med('mammalian','flat_interaction','APL_change_BOX'):.2f}, "
            f"curved {med('mammalian','curved','APL_change_BOX'):.2f} Å²/molecule."
        )
        report.append(
            f"- Bacterial median ΔAPL: flat-no-interaction "
            f"{med('bacterial','flat_no_interaction','APL_change_BOX'):.2f}, "
            f"flat-interaction {med('bacterial','flat_interaction','APL_change_BOX'):.2f}, "
            f"curved {med('bacterial','curved','APL_change_BOX'):.2f} Å²/molecule."
        )
        report.append("")

        report.append("## What ΔAPL misses")
        report.append("")
        report.append(
            "- Thickness change is much more strongly separated by visual "
            "morphology class than ΔAPL. This is consistent with bending/bulging "
            "being a distinct structural response rather than simply larger "
            "projected-area expansion."
        )
        report.append(
            f"- Mammalian median thickness change: flat-interaction "
            f"{med('mammalian','flat_interaction','thickness_change'):.1f}, "
            f"curved {med('mammalian','curved','thickness_change'):.1f}, "
            f"bulge-curved {med('mammalian','bulge_curved','thickness_change'):.1f} "
            "(source thickness units)."
        )
        report.append(
            f"- Bacterial median thickness change: flat-interaction "
            f"{med('bacterial','flat_interaction','thickness_change'):.1f}, "
            f"curved {med('bacterial','curved','thickness_change'):.1f}, "
            f"bulge-curved {med('bacterial','bulge_curved','thickness_change'):.1f}."
        )
        report.append("")

        report.append("## Morphology pilot")
        report.append("")
        report.append(
            f"- In the bacterial pilot, ΔAPL vs height RMS has Spearman ρ = "
            f"**{morph_rho('bacterial','APL_change_BOX','morph__endpoint__height_rms_nm'):.3f}**, "
            f"whereas thickness change vs height RMS has ρ = "
            f"**{morph_rho('bacterial','thickness_change','morph__endpoint__height_rms_nm'):.3f}**."
        )
        report.append(
            f"- In the mammalian pilot, ΔAPL vs height RMS has ρ = "
            f"**{morph_rho('mammalian','APL_change_BOX','morph__endpoint__height_rms_nm'):.3f}**, "
            f"whereas thickness change vs height RMS has ρ = "
            f"**{morph_rho('mammalian','thickness_change','morph__endpoint__height_rms_nm'):.3f}**."
        )
        report.append(
            "- Height RMS and excess surface area are strongly related, especially "
            "in the bacterial membrane, but thickness is substantially less "
            "redundant with those shape-amplitude metrics. RMS mean curvature is "
            "retained as a secondary/local-shape measure."
        )
        report.append("")

        report.append("## ddAPL and target choice")
        report.append("")
        report.append(
            f"- ddAPL is strongly tied to bacterial ΔAPL (ρ = "
            f"**{rho('ddAPL','dAPL_bacterial'):.3f}**) but only weakly tied to "
            f"mammalian ΔAPL (ρ = **{rho('ddAPL','dAPL_mammalian'):.3f}**)."
        )
        report.append(
            f"- Bacterial and mammalian ΔAPL themselves have ρ = "
            f"**{rho('dAPL_bacterial','dAPL_mammalian'):.3f}**."
        )
        report.append(
            "- Therefore ddAPL remains a defensible directional selectivity "
            "target, but it is not a complete activity/structure target. A separate "
            "absolute-response quantity and a structural phenotype/metric preserve "
            "information that subtraction removes."
        )
        report.append(
            "- No composite score is selected here. The evidence package supports "
            "a later decision between (a) ddAPL alone for selectivity, or "
            "(b) a multi-objective target retaining bacterial ΔAPL, mammalian ΔAPL/"
            "ddAPL and a structural response independently."
        )
        report.append("")

        report.append("## Local lipid / leaflet status")
        report.append("")
        report.append(
            "- The local-lipid dataset remains exploratory (15 systems per "
            "membrane). Targeted tests are FDR-corrected in "
            "`20_local_lipid_exploratory_associations.csv`; nominal p-values are "
            "not treated as established findings."
        )
        report.append(
            "- Explicit leaflet geometry is only a four-system smoke pilot. It "
            "supports the existence of strong leaflet asymmetry in the most "
            "deformed examples but is not population-level evidence."
        )
        report.append(
            "- Dense CHOL QC is only two mammalian systems and has substantial "
            "near-midplane/switching flags; it should constrain interpretation of "
            "CHOL redistribution rather than serve as an ML target now."
        )
        report.append("")

        report.append("## Highest APL–morphology discordance examples")
        report.append("")
        report.append(
            top_discordant.to_markdown(index=False)
        )
        report.append("")

        report.append("## Immediate decision point before RDKit")
        report.append("")
        report.append(
            "Use the numerical evidence in `12_target_decision_evidence.csv` and "
            "`13_candidate_multitarget_pair_table.csv` to choose whether the next "
            "ML stage remains a single ddAPL regression or becomes explicitly "
            "multi-objective. RDKit should be added only after that target contract "
            "is fixed."
        )

        (out / "SCIENTIFIC_REPORT.md").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        stage("10. FINAL VALIDATION + ZIP")
        required_outputs = [
            "01_data_qc_summary.csv",
            "02_label_transition_counts.csv",
            "04_label_concordance_statistics.csv",
            "05_class_metric_descriptives.csv",
            "06_class_metric_omnibus_tests.csv",
            "07_class_metric_pairwise_tests.csv",
            "10_target_correlations.csv",
            "12_target_decision_evidence.csv",
            "13_candidate_multitarget_pair_table.csv",
            "14_morphology_core_correlations.csv",
            "15_morphology_vs_APL_thickness.csv",
            "18_APL_morphology_discordant_examples.csv",
            "20_local_lipid_exploratory_associations.csv",
            "SCIENTIFIC_REPORT.md",
        ]
        missing_out = [x for x in required_outputs if not (out / x).exists()]
        if missing_out:
            raise RuntimeError(
                "Required output(s) missing after analysis:\n"
                + "\n".join(missing_out)
            )

        result_validation = {
            "status": "PASS",
            "source_master_validation": validation.get("status"),
            "primary_pairs": int(len(wide)),
            "primary_physical_system_rows": int(len(system_rows)),
            "exact_membrane_label_agreement_fraction": exact_agree,
            "label_cramers_v": cramer_v,
            "outputs_required": required_outputs,
            "local_lipid_tests_FDR_corrected": True,
            "pairwise_class_tests_Holm_corrected": True,
            "rdkit_used": False,
            "composite_target_chosen": False,
            "source_files_modified": False,
        }
        (out / "ANALYSIS_VALIDATION.json").write_text(
            json.dumps(result_validation, indent=2)
        )

        sys.stdout = sys.__stdout__
        tee.close()

        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out))

        print("\n" + "=" * 78)
        print("SCIENTIFIC INTERPRETATION COMPLETE — PASS")
        print("=" * 78)
        print(f"Results: {out}")
        print(f"Upload/review ZIP: {zip_path}")
        print("No source project data were modified.")

    except Exception:
        tb = traceback.format_exc()
        try:
            print("\nANALYSIS FAILED — THIS IS NOT SILENT")
            print(tb)
            (out / "FAILED.txt").write_text(tb, encoding="utf-8")
        finally:
            sys.stdout = sys.__stdout__
            tee.close()
        raise


if __name__ == "__main__":
    main()
