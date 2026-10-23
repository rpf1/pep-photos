#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def fmt_p(x):
    try:
        x=float(x)
        if not np.isfinite(x): return "NA"
        return f"{x:.2e}" if x < 0.001 else f"{x:.3f}"
    except Exception: return "NA"

def fmt(x, n=3):
    try:
        x=float(x)
        if not np.isfinite(x): return "NA"
        return f"{x:.{n}f}"
    except Exception: return "NA"

ap=argparse.ArgumentParser()
ap.add_argument("--out",required=True)
a=ap.parse_args()
out=Path(a.out)

lines=["# Flatness ↔ physical-response meeting summary",""]
qc_path=out/"task_0/qc_join_report.json"
if qc_path.exists():
    qc=json.loads(qc_path.read_text())
    lines += [
        "## Data/QC",
        f"- Morphology systems: **{qc['morphology_rows']}**.",
        f"- Strictly matched to nmol=80 master APL data: **{qc['strict_master_matches']}**; unmatched: **{qc['strict_master_unmatched']}**.",
        f"- Morphology QC: **{qc['morphology_PASS']} PASS**, **{qc['morphology_REVIEW_REQUIRED']} REVIEW_REQUIRED**.",
    ]
    ac=qc.get("annotation_crosscheck",{})
    if ac.get("provided"):
        lines.append(f"- Annotation cross-check: **{ac.get('label_exact_matches','?')}/{ac.get('morphology_rows','?')} exact manual-label matches**.")
    lines.append("")

cm_path=out/"task_1/class_morphology_tests.csv"
if cm_path.exists():
    cm=pd.read_csv(cm_path)
    wanted=["height_rms_delta_nm","excess_area_delta_fraction","curvature_delta_nm_inv","thickness_delta_nm"]
    s=cm[(cm["scope"]=="all_PASS")&(cm["grouping"]=="binary_inclusive")&cm["metric"].isin(wanted)].copy()
    lines += [
        "## Do visual curved/flat labels correspond to quantitative morphology?",
        "Inclusive sensitivity definition: curved = curved + bulge_curved; flat = flat_interaction + flat_no_interaction + bulge_flat; pore_formed excluded.",
    ]
    for _,r in s.iterrows():
        lines.append(
            f"- **{r['metric']}**: curved median {fmt(r['median_curved'])} vs flat {fmt(r['median_flat'])}; "
            f"rank-biserial = {fmt(r['rank_biserial_curved_vs_flat'])}; BH-adjusted p = {fmt_p(r['p_BH'])} "
            f"(n={int(r['n_curved'])} curved, {int(r['n_flat'])} flat)."
        )
    lines.append("")

apl_path=out/"task_2/class_APL_tests.csv"
if apl_path.exists():
    apd=pd.read_csv(apl_path)
    s=apd[(apd["grouping"]=="binary_inclusive") & apd["scope"].isin(["POPE75POPG25_W_WF","POPC60CHOL40_W_WF"])].copy()
    lines += ["## Does projected ΔAPL distinguish visually curved vs flat systems within each primary membrane?"]
    for _,r in s.iterrows():
        name="bacterial" if r["scope"].startswith("POPE") else "mammalian"
        lines.append(
            f"- **{name}**: curved median ΔAPL {fmt(r['median_curved'])} vs flat {fmt(r['median_flat'])}; "
            f"rank-biserial = {fmt(r['rank_biserial_curved_vs_flat'])}; BH-adjusted p = {fmt_p(r['p_BH'])}."
        )
    lines.append("")

pair_path=out/"task_3/paired_tests_and_correlations.csv"
if pair_path.exists():
    pr=pd.read_csv(pair_path)
    corr=pr[pr["analysis"]=="ddAPL_vs_differential_morphology"].copy()
    lines += ["## ddAPL versus bacterial–mammalian structural differences (primary matched tripeptoids)"]
    for _,r in corr.iterrows():
        lines.append(
            f"- **{r['metric']}**: Spearman ρ = {fmt(r['rho'])}, BH-adjusted p = {fmt_p(r['p_BH'])}, n={int(r['n'])} pairs."
        )
    lines.append("")

disc_path=out/"task_4/discordant_cases.csv"
if disc_path.exists():
    dd=pd.read_csv(disc_path)
    if len(dd):
        h=dd[(dd["discordance_type"]=="small_ddAPL_large_structural_difference")&
             (dd["metric"]=="height_rms_delta_nm")].head(5)
        lines += ["## Example systems: relatively small ddAPL but large bacterial–mammalian height-RMS difference"]
        for _,r in h.iterrows():
            lines.append(f"- `{r['molecule']}` / `{r['physical_run']}`: ddAPL={fmt(r['ddAPL'])}; ΔΔ height RMS={fmt(r['dd_metric'])} nm.")
        lines.append("")

red_path=out/"task_5/high_redundancy_abs_rho_ge_0p85.csv"
if red_path.exists():
    rd=pd.read_csv(red_path)
    rr=rd[rd["scope"]=="all_PASS"].sort_values("abs_rho",ascending=False).head(6)
    lines += ["## Strong metric relationships (all PASS systems; |ρ| ≥ 0.85)"]
    for _,r in rr.iterrows():
        lines.append(f"- {r['metric_a']} vs {r['metric_b']}: ρ={fmt(r['rho'])} (n={int(r['n'])}).")
    lines.append("")

temp_path=out/"task_6/temporal_persistence.csv"
if temp_path.exists():
    td=pd.read_csv(temp_path)
    lines += ["## Endpoint persistence check"]
    for _,r in td.iterrows():
        lines.append(f"- {r['metric']}: 100 ns endpoint vs 80–100 ns median ρ={fmt(r['endpoint_vs_80_100_median_rho'])} (n={int(r['n'])}).")
    lines.append("")

lines += [
    "## Interpretation to discuss",
    "- Quantitative morphology should be treated as complementary structural information, not as a replacement for ddAPL.",
    "- Strong separation of visual curved/flat classes by height RMS/excess area/curvature, alongside weak within-membrane ΔAPL separation, would directly support the argument that projected APL misses some visually obvious structural deformation.",
    "- Thickness is expected to track APL strongly and should not automatically be treated as an independent morphology axis.",
    "- Use the discordant systems as candidate supervisor-slide examples, then inspect their images before making a final qualitative claim.",
    "",
    "## Output locations",
    "- `task_0/`: strict merge/QC",
    "- `task_1/`: visual class vs morphology",
    "- `task_2/`: visual class vs ΔAPL",
    "- `task_3/`: matched-pair ddAPL vs structural differences",
    "- `task_4/`: discordant representative systems",
    "- `task_5/`: metric redundancy/correlations",
    "- `task_6/`: membrane-label association and temporal persistence",
]
(out/"MEETING_SUMMARY.md").write_text("\n".join(lines)+"\n")
print(out/"MEETING_SUMMARY.md")
