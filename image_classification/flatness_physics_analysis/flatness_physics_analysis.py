#!/usr/bin/env python3
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon, chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BILAYERS = {
    "POPE75POPG25_W_WF": "bacterial",
    "POPC60CHOL40_W_WF": "mammalian",
    "POPC80POPS20_W_WF": "POPC80POPS20",
}

CORE = {
    "height_rms_delta_nm": "morph__delta__height_rms_nm",
    "excess_area_delta_fraction": "morph__delta__midplane_excess_area_fraction",
    "curvature_delta_nm_inv": "morph__delta__H_rms_nm_inv",
    "thickness_delta_nm": "morph__delta__thickness_median_nm",
    "height_asym_delta_nm": "morph__delta__height_rms_abs_upper_minus_lower_nm",
    "excess_asym_delta": "morph__delta__excess_area_abs_upper_minus_lower",
    "curvature_asym_delta_nm_inv": "morph__delta__H_rms_abs_upper_minus_lower_nm_inv",
}

ENDPOINT = {
    "height_rms_endpoint_nm": "morph__endpoint__height_rms_nm",
    "excess_area_endpoint_fraction": "morph__endpoint__midplane_excess_area_fraction",
    "curvature_endpoint_nm_inv": "morph__endpoint__H_rms_nm_inv",
    "thickness_endpoint_nm": "morph__endpoint__thickness_median_nm",
}

LATE = {
    "height_rms_late_nm": "morph__relax_80_100__height_rms_nm__median",
    "excess_area_late_fraction": "morph__relax_80_100__midplane_excess_area_fraction__median",
    "curvature_late_nm_inv": "morph__relax_80_100__H_rms_nm_inv__median",
    "thickness_late_nm": "morph__relax_80_100__thickness_median_nm__median",
}

PLOT_LABELS = {
    "height_rms_delta_nm": "Δ height RMS (nm)",
    "excess_area_delta_fraction": "Δ excess area fraction",
    "curvature_delta_nm_inv": "Δ RMS mean curvature (nm⁻¹)",
    "thickness_delta_nm": "Δ local-normal thickness (nm)",
    "APL_change": "ΔAPL (Å² / membrane molecule)",
}

LABEL_ORDER = [
    "flat_no_interaction", "flat_interaction", "bulge_flat",
    "curved", "bulge_curved", "pore_formed"
]

def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    back = np.empty_like(q)
    back[order] = q
    out[np.where(ok)[0]] = back
    return out

def normalize_run(x):
    s = str(x)
    if s in {"0", "0.0"}:
        return "run_0"
    if s.startswith("run_"):
        return s
    return "run_" + s

def load_data(master_path, morph_path, ann_path=None):
    morph = pd.read_csv(morph_path)
    master = pd.read_csv(master_path)

    req_morph = {"molecule","bilayer","physical_run","manual_label","morph__qc_status"}
    missing = req_morph - set(morph.columns)
    if missing:
        raise ValueError(f"system_morphology.csv missing columns: {sorted(missing)}")

    # Production morphology population is nmol=80. Restrict master to exact concentration
    # before deriving join identities so concentration variants cannot collide.
    if "nmol" in master.columns:
        master = master.loc[pd.to_numeric(master["nmol"], errors="coerce") == 80].copy()
    master["molecule"] = master["molecule_name"].astype(str).str.replace(r"_\d+$", "", regex=True)
    master["bilayer"] = master["system_type"].astype(str)
    master["physical_run"] = master["run_number"].map(normalize_run)

    keys = ["molecule","bilayer","physical_run"]
    if master.duplicated(keys).any():
        d = master.loc[master.duplicated(keys, keep=False), keys]
        raise ValueError("Master has duplicate strict nmol=80 join keys:\n" + d.to_string(index=False))
    if morph.duplicated(keys).any():
        d = morph.loc[morph.duplicated(keys, keep=False), keys]
        raise ValueError("Morphology has duplicate physical-system keys:\n" + d.to_string(index=False))

    keep_master = [c for c in [
        *keys, "system_key","nmol","APL_change","APL_percent_change",
        "start_APL","finish_APL_100ns","APL_method",
        "start_thickness","finish_thickness_100ns","thickness_change",
        "analysis_complete","reached_100ns","system_dir"
    ] if c in master.columns]
    joined = morph.merge(master[keep_master], on=keys, how="left",
                         indicator="master_join", suffixes=("","__master"))

    joined["membrane_short"] = joined["bilayer"].map(BILAYERS).fillna(joined["bilayer"])
    joined["binary_conservative"] = np.select(
        [
            joined["manual_label"].eq("curved"),
            joined["manual_label"].isin(["flat_interaction","flat_no_interaction"])
        ],
        ["curved","flat"], default=None
    )
    joined["binary_inclusive"] = np.select(
        [
            joined["manual_label"].isin(["curved","bulge_curved"]),
            joined["manual_label"].isin(["flat_interaction","flat_no_interaction","bulge_flat"])
        ],
        ["curved","flat"], default=None
    )

    ann_report = {"provided": False}
    if ann_path and Path(ann_path).is_file():
        ann = pd.read_csv(ann_path)
        ann_report["provided"] = True
        run_col = "authoritative_run" if "authoritative_run" in ann.columns else "run"
        ann2 = ann.copy()
        ann2["bilayer"] = ann2["bilayer_type"]
        ann2["physical_run"] = ann2[run_col]
        if "canonical_include" in ann2.columns:
            ann2 = ann2.loc[ann2["canonical_include"].fillna(False).astype(bool)].copy()
        ann2 = ann2.drop_duplicates(keys, keep="last")
        chk = morph[keys+["manual_label"]].merge(
            ann2[keys+["label"]], on=keys, how="left", indicator=True
        )
        match = chk["manual_label"].fillna("<NA>").eq(chk["label"].fillna("<NA>"))
        ann_report.update({
            "morphology_rows": int(len(chk)),
            "annotation_joined": int((chk["_merge"]=="both").sum()),
            "label_exact_matches": int(match.sum()),
            "label_mismatches": int((~match & (chk["_merge"]=="both")).sum()),
        })
    return joined, ann_report

def valid_xy(df, col, group=None):
    d = df.copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.loc[np.isfinite(d[col])].copy()
    if group is not None:
        d = d.loc[d[group].notna() & ~d[group].astype(str).eq("nan")]
    return d

def mw_row(df, metric_name, col, grouping, scope):
    d = valid_xy(df, col, grouping)
    x = d.loc[d[grouping]=="curved", col].to_numpy()
    y = d.loc[d[grouping]=="flat", col].to_numpy()
    if len(x) < 2 or len(y) < 2:
        return None
    u,p = mannwhitneyu(x,y,alternative="two-sided")
    rb = 2*u/(len(x)*len(y))-1
    return {
        "scope": scope, "test":"Mann-Whitney U", "grouping": grouping,
        "metric": metric_name, "n_curved":len(x), "n_flat":len(y),
        "median_curved":float(np.median(x)), "median_flat":float(np.median(y)),
        "median_difference_curved_minus_flat":float(np.median(x)-np.median(y)),
        "rank_biserial_curved_vs_flat":float(rb), "statistic":float(u), "p":float(p)
    }

def omnibus_row(df, metric_name, col, scope):
    d = valid_xy(df, col, "manual_label")
    groups=[]; names=[]
    for lab in LABEL_ORDER:
        x=d.loc[d["manual_label"]==lab,col].to_numpy()
        if len(x)>=5:
            groups.append(x); names.append(lab)
    if len(groups)<2:
        return None
    stat,p=kruskal(*groups)
    return {
        "scope":scope,"test":"Kruskal-Wallis","grouping":"original_labels",
        "metric":metric_name,"classes":";".join(names),
        "n_total":int(sum(map(len,groups))),"statistic":float(stat),"p":float(p)
    }

def save_box(df, col, title, ylabel, path):
    labs=[x for x in LABEL_ORDER if (df["manual_label"]==x).sum()>0]
    vals=[pd.to_numeric(df.loc[df["manual_label"]==x,col],errors="coerce").dropna().to_numpy() for x in labs]
    if not vals or all(len(x)==0 for x in vals):
        return
    fig,ax=plt.subplots(figsize=(10,5))
    ax.boxplot(vals, tick_labels=labs, showfliers=False)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)

def add_q(rows):
    if not rows: return rows
    p=[r.get("p",np.nan) for r in rows]
    q=bh_adjust(p)
    for r,v in zip(rows,q): r["p_BH"]=v
    return rows

def task_qc(joined, ann_report, out):
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out/"merged_strict.csv",index=False)
    unmatched=joined.loc[joined["master_join"]!="both",
                         ["molecule","bilayer","physical_run","manual_label","morph__qc_status"]]
    unmatched.to_csv(out/"unmatched_master_rows.csv",index=False)
    labels=joined.groupby(["manual_label","morph__qc_status"],dropna=False).size().reset_index(name="n")
    labels.to_csv(out/"label_qc_counts.csv",index=False)
    report={
        "morphology_rows":int(len(joined)),
        "strict_master_matches":int((joined["master_join"]=="both").sum()),
        "strict_master_unmatched":int((joined["master_join"]!="both").sum()),
        "morphology_PASS":int((joined["morph__qc_status"]=="PASS").sum()),
        "morphology_REVIEW_REQUIRED":int((joined["morph__qc_status"]=="REVIEW_REQUIRED").sum()),
        "manual_label_counts":joined["manual_label"].value_counts(dropna=False).to_dict(),
        "annotation_crosscheck":ann_report,
    }
    (out/"qc_join_report.json").write_text(json.dumps(report,indent=2,default=str))

def task_class_morph(joined,out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; desc=[]
    base=joined.loc[joined["morph__qc_status"]=="PASS"].copy()
    metrics={**CORE,**ENDPOINT}
    for name,col in metrics.items():
        if col not in base: continue
        for scope,d in [("all_PASS",base)]+[(b,base.loc[base["bilayer"]==b]) for b in BILAYERS]:
            r=omnibus_row(d,name,col,scope)
            if r: rows.append(r)
            for g in ["binary_conservative","binary_inclusive"]:
                r=mw_row(d,name,col,g,scope)
                if r: rows.append(r)
        dd=valid_xy(base,col,"manual_label")
        for lab,x in dd.groupby("manual_label"):
            vals=x[col].to_numpy()
            desc.append({"metric":name,"manual_label":lab,"n":len(vals),
                         "median":float(np.median(vals)),
                         "q25":float(np.quantile(vals,.25)),
                         "q75":float(np.quantile(vals,.75))})
        if name in CORE:
            save_box(base,col,f"{name} by manual phenotype",PLOT_LABELS.get(name,name),
                     out/f"box_{name}.png")
    pd.DataFrame(add_q(rows)).to_csv(out/"class_morphology_tests.csv",index=False)
    pd.DataFrame(desc).to_csv(out/"class_morphology_descriptives.csv",index=False)

def task_class_apl(joined,out):
    out.mkdir(parents=True,exist_ok=True)
    base=joined.loc[(joined["morph__qc_status"]=="PASS") & (joined["master_join"]=="both")].copy()
    rows=[]; desc=[]
    for scope,d in [("all_PASS",base)]+[(b,base.loc[base["bilayer"]==b]) for b in BILAYERS]:
        r=omnibus_row(d,"APL_change","APL_change",scope)
        if r: rows.append(r)
        for g in ["binary_conservative","binary_inclusive"]:
            r=mw_row(d,"APL_change","APL_change",g,scope)
            if r: rows.append(r)
        dd=valid_xy(d,"APL_change","manual_label")
        for lab,x in dd.groupby("manual_label"):
            vals=x["APL_change"].to_numpy()
            desc.append({"scope":scope,"manual_label":lab,"n":len(vals),
                         "median_APL_change":float(np.median(vals)),
                         "q25":float(np.quantile(vals,.25)),
                         "q75":float(np.quantile(vals,.75))})
        if len(d):
            save_box(d,"APL_change",f"ΔAPL by manual phenotype — {scope}",
                     PLOT_LABELS["APL_change"],out/f"box_APL_change_{scope}.png")
    pd.DataFrame(add_q(rows)).to_csv(out/"class_APL_tests.csv",index=False)
    pd.DataFrame(desc).to_csv(out/"class_APL_descriptives.csv",index=False)

def make_pairs(joined, primary=True):
    d=joined.loc[
        (joined["morph__qc_status"]=="PASS") &
        (joined["master_join"]=="both") &
        (joined["bilayer"].isin(["POPE75POPG25_W_WF","POPC60CHOL40_W_WF"]))
    ].copy()
    if primary:
        d=d.loc[(d["molecule_type"]=="peptoid") & (pd.to_numeric(d["length"],errors="coerce")==3)]
    idx=["molecule","physical_run"]
    vals=["APL_change"]+[c for c in CORE.values() if c in d.columns]
    wide=d.pivot_table(index=idx,columns="membrane_short",values=vals,aggfunc="first")
    # Require both membranes for APL and all requested metrics independently later.
    flat=pd.DataFrame(index=wide.index).reset_index()
    for v in vals:
        for side in ["bacterial","mammalian"]:
            if (v,side) in wide.columns:
                flat[f"{v}__{side}"]=wide[(v,side)].values
    if "APL_change__bacterial" in flat and "APL_change__mammalian" in flat:
        flat["ddAPL"]=flat["APL_change__bacterial"]-flat["APL_change__mammalian"]
    for name,col in CORE.items():
        b=f"{col}__bacterial"; m=f"{col}__mammalian"
        if b in flat and m in flat:
            flat[f"dd__{name}"]=flat[b]-flat[m]
    return flat

def task_pairs(joined,out):
    out.mkdir(parents=True,exist_ok=True)
    pairs=make_pairs(joined,primary=True)
    pairs.to_csv(out/"primary_tripeptoid_pairs.csv",index=False)
    rows=[]
    for name,col in CORE.items():
        ddcol=f"dd__{name}"
        if ddcol not in pairs: continue
        d=pairs[["ddAPL",ddcol]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(d)>=5:
            rho,p=spearmanr(d["ddAPL"],d[ddcol])
            rows.append({"analysis":"ddAPL_vs_differential_morphology","metric":name,
                         "n":len(d),"rho":float(rho),"p":float(p)})
            fig,ax=plt.subplots(figsize=(6,5))
            ax.scatter(d["ddAPL"],d[ddcol],s=20)
            ax.axhline(0,linewidth=.8); ax.axvline(0,linewidth=.8)
            ax.set_xlabel("ddAPL (Å² / membrane molecule)")
            ax.set_ylabel("Bacterial − mammalian structural Δ")
            ax.set_title(f"ddAPL vs {name} (ρ={rho:.2f}, n={len(d)})")
            fig.tight_layout(); fig.savefig(out/f"scatter_ddAPL_vs_{name}.png",dpi=180); plt.close(fig)
        b=f"{col}__bacterial"; m=f"{col}__mammalian"
        if b in pairs and m in pairs:
            pm=pairs[[b,m]].apply(pd.to_numeric,errors="coerce").dropna()
            if len(pm)>=5:
                try:
                    w,pw=wilcoxon(pm[b],pm[m],zero_method="wilcox",alternative="two-sided")
                    rows.append({"analysis":"paired_bacterial_vs_mammalian",
                                 "metric":name,"n":len(pm),"statistic":float(w),"p":float(pw),
                                 "median_bacterial":float(pm[b].median()),
                                 "median_mammalian":float(pm[m].median()),
                                 "median_bacterial_minus_mammalian":float((pm[b]-pm[m]).median())})
                except ValueError:
                    pass
    pd.DataFrame(add_q(rows)).to_csv(out/"paired_tests_and_correlations.csv",index=False)

def task_discordance(joined,out):
    out.mkdir(parents=True,exist_ok=True)
    pairs=make_pairs(joined,primary=True)
    rows=[]
    if "ddAPL" not in pairs:
        pd.DataFrame().to_csv(out/"discordant_cases.csv",index=False); return
    ddabs=pairs["ddAPL"].abs()
    q25_dd=ddabs.quantile(.25); q75_dd=ddabs.quantile(.75)
    for name in CORE:
        c=f"dd__{name}"
        if c not in pairs: continue
        a=pairs[c].abs()
        q25_m=a.quantile(.25); q75_m=a.quantile(.75)
        mask1=(ddabs<=q25_dd)&(a>=q75_m)
        mask2=(ddabs>=q75_dd)&(a<=q25_m)
        for typ,mask in [
            ("small_ddAPL_large_structural_difference",mask1),
            ("large_ddAPL_small_structural_difference",mask2),
        ]:
            for _,r in pairs.loc[mask].iterrows():
                rows.append({"discordance_type":typ,"metric":name,
                             "molecule":r["molecule"],"physical_run":r["physical_run"],
                             "ddAPL":r["ddAPL"],"dd_metric":r[c],
                             "abs_ddAPL_q25":q25_dd,"abs_ddAPL_q75":q75_dd,
                             "abs_metric_q25":q25_m,"abs_metric_q75":q75_m})
    outdf=pd.DataFrame(rows)
    if len(outdf):
        outdf["abs_ddAPL"]=outdf["ddAPL"].abs()
        outdf["abs_dd_metric"]=outdf["dd_metric"].abs()
        outdf=outdf.sort_values(["discordance_type","metric","abs_dd_metric"],ascending=[True,True,False])
    outdf.to_csv(out/"discordant_cases.csv",index=False)
    # Also list strongest absolute structural differential examples metric-by-metric.
    top=[]
    for name in CORE:
        c=f"dd__{name}"
        if c not in pairs: continue
        d=pairs[["molecule","physical_run","ddAPL",c]].dropna().copy()
        d["abs_dd_metric"]=d[c].abs()
        for _,r in d.nlargest(10,"abs_dd_metric").iterrows():
            top.append({"metric":name,"molecule":r["molecule"],"physical_run":r["physical_run"],
                        "ddAPL":r["ddAPL"],"dd_metric":r[c],"abs_dd_metric":r["abs_dd_metric"]})
    pd.DataFrame(top).to_csv(out/"top_structural_differential_cases.csv",index=False)

def task_redundancy(joined,out):
    out.mkdir(parents=True,exist_ok=True)
    base=joined.loc[(joined["morph__qc_status"]=="PASS") & (joined["master_join"]=="both")].copy()
    cols={"APL_change":"APL_change",**CORE}
    rows=[]
    for scope,d in [("all_PASS",base)]+[(b,base.loc[base["bilayer"]==b]) for b in BILAYERS]:
        names=list(cols)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                x=pd.to_numeric(d[cols[a]],errors="coerce")
                y=pd.to_numeric(d[cols[b]],errors="coerce")
                ok=x.notna()&y.notna()
                if ok.sum()<5: continue
                rho,p=spearmanr(x[ok],y[ok])
                rows.append({"scope":scope,"metric_a":a,"metric_b":b,
                             "n":int(ok.sum()),"rho":float(rho),"p":float(p),
                             "abs_rho":float(abs(rho))})
    df=pd.DataFrame(add_q(rows))
    df.to_csv(out/"metric_spearman_long.csv",index=False)
    if len(df):
        df.loc[df["abs_rho"]>=.85].sort_values(["scope","abs_rho"],ascending=[True,False]).to_csv(
            out/"high_redundancy_abs_rho_ge_0p85.csv",index=False)

def cramers_v(tab):
    chi2,p,_,_=chi2_contingency(tab)
    n=tab.to_numpy().sum()
    r,k=tab.shape
    if n==0 or min(r-1,k-1)==0: return np.nan,p,chi2
    return math.sqrt((chi2/n)/min(r-1,k-1)),p,chi2

def task_labels(joined,out):
    out.mkdir(parents=True,exist_ok=True)
    d=joined.loc[joined["morph__qc_status"]=="PASS"].copy()
    tab=pd.crosstab(d["manual_label"],d["membrane_short"])
    tab.to_csv(out/"label_by_membrane_counts.csv")
    v,p,chi=cramers_v(tab)
    pd.DataFrame([{"analysis":"manual_label_vs_membrane","n":int(tab.to_numpy().sum()),
                   "chi2":chi,"p":p,"cramers_v":v}]).to_csv(out/"label_membrane_association.csv",index=False)

    # Matched B/M label transition table for primary tripeptoids.
    p=d.loc[
        d["bilayer"].isin(["POPE75POPG25_W_WF","POPC60CHOL40_W_WF"]) &
        (d["molecule_type"]=="peptoid") &
        (pd.to_numeric(d["length"],errors="coerce")==3)
    ].pivot_table(index=["molecule","physical_run"],columns="membrane_short",
                  values="manual_label",aggfunc="first")
    if {"bacterial","mammalian"}.issubset(p.columns):
        pd.crosstab(p["mammalian"],p["bacterial"]).to_csv(out/"paired_label_transition_mammalian_to_bacterial.csv")

def task_temporal(joined,out):
    out.mkdir(parents=True,exist_ok=True)
    d=joined.loc[joined["morph__qc_status"]=="PASS"].copy()
    rows=[]
    mapping=[
        ("height_rms","morph__endpoint__height_rms_nm",LATE["height_rms_late_nm"]),
        ("excess_area","morph__endpoint__midplane_excess_area_fraction",LATE["excess_area_late_fraction"]),
        ("curvature","morph__endpoint__H_rms_nm_inv",LATE["curvature_late_nm_inv"]),
        ("thickness","morph__endpoint__thickness_median_nm",LATE["thickness_late_nm"]),
    ]
    for name,end,late in mapping:
        if end not in d or late not in d: continue
        x=pd.to_numeric(d[end],errors="coerce"); y=pd.to_numeric(d[late],errors="coerce")
        ok=x.notna()&y.notna()
        if ok.sum()<5: continue
        rho,p=spearmanr(x[ok],y[ok])
        mad=(x[ok]-y[ok]).abs().median()
        rows.append({"metric":name,"n":int(ok.sum()),"endpoint_vs_80_100_median_rho":rho,
                     "p":p,"median_abs_endpoint_minus_late":mad})
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(y[ok],x[ok],s=18)
        ax.set_xlabel("80–100 ns median"); ax.set_ylabel("100 ns endpoint")
        ax.set_title(f"{name}: endpoint vs late-window median")
        fig.tight_layout(); fig.savefig(out/f"temporal_endpoint_vs_late_{name}.png",dpi=180); plt.close(fig)
    pd.DataFrame(add_q(rows)).to_csv(out/"temporal_persistence.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",type=int,required=True,choices=range(7))
    ap.add_argument("--master",required=True)
    ap.add_argument("--morphology",required=True)
    ap.add_argument("--annotations",default=None)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    out=Path(a.out)
    joined,ann_report=load_data(a.master,a.morphology,a.annotations)
    task_out=out/f"task_{a.task}"
    funcs=[task_qc,task_class_morph,task_class_apl,task_pairs,task_discordance,task_redundancy,task_labels]
    if a.task==0: task_qc(joined,ann_report,task_out)
    elif a.task==1: task_class_morph(joined,task_out)
    elif a.task==2: task_class_apl(joined,task_out)
    elif a.task==3: task_pairs(joined,task_out)
    elif a.task==4: task_discordance(joined,task_out)
    elif a.task==5: task_redundancy(joined,task_out)
    elif a.task==6:
        task_labels(joined,task_out)
        task_temporal(joined,task_out)
    print(json.dumps({"task":a.task,"status":"PASS","output":str(task_out)}))

if __name__=="__main__":
    main()
