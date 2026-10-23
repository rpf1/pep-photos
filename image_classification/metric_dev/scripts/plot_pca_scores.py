#!/usr/bin/env python3
"""
Export PC1/PC2/PC3 scores from the frozen PCA parameters and plot sample-level PCA space.

Important:
- Does NOT refit PCA.
- Reconstructs scores from the saved scale, centre, and oriented PCA components.
- Verifies reconstructed PC1/PC2 against the original exported scores before accepting PC3.
- For paired B/M PCA models, shows the same coordinates twice:
    left = coloured by bacterial phenotype label
    right = coloured by mammalian phenotype label
  This avoids inventing an arbitrary combined B/M class.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = [
    "flat_no_interaction",
    "flat_interaction",
    "bulge_flat",
    "curved",
    "bulge_curved",
    "pore_formed",
]

# Canonical class colours from metric_dev/scripts/plot_metric_analysis.py
CLASS_COLOURS = dict(
    zip(
        LABELS,
        ["#657685", "#4796BA", "#DD9A35", "#427943", "#963966", "#111111"],
    )
)


def parse_view(model: str) -> str:
    if model.startswith("endpoint_vs_ref__"):
        return "endpoint_vs_ref"
    if model.startswith("late_vs_ref__"):
        return "late_vs_ref"
    raise ValueError(f"Cannot determine PCA view from model: {model}")


def parse_side(model: str) -> str:
    if "__B__" in model:
        return "B"
    if "__M__" in model:
        return "M"
    if "__paired__" in model:
        return "paired"
    raise ValueError(f"Cannot determine PCA side from model: {model}")


def feature_to_unified_column(feature: str, model: str, view: str) -> str:
    if feature.startswith("B_"):
        side, metric = "B", feature[2:]
    elif feature.startswith("M_"):
        side, metric = "M", feature[2:]
    else:
        side = parse_side(model)
        if side == "paired":
            raise ValueError(
                f"Paired model feature lacks B_/M_ prefix: {model} / {feature}"
            )
        metric = feature

    return f"{side}__{metric}__{view}"


def component_variance(summary: pd.DataFrame, model: str, component: int) -> float:
    hit = summary[
        (summary["model"] == model) & (summary["component"] == component)
    ]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0]["explained_variance"]) * 100.0


def axis_label(summary: pd.DataFrame, model: str, pc: str) -> str:
    component = int(pc.replace("PC", ""))
    pct = component_variance(summary, model, component)
    return f"{pc} ({pct:.1f}%)" if np.isfinite(pct) else pc


def scatter_2d(ax, df, x, y, label_col, title):
    for label in LABELS:
        sub = df[df[label_col] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub[x],
            sub[y],
            s=35,
            alpha=0.85,
            label=label,
            c=CLASS_COLOURS[label],
            edgecolors="none",
        )

    # Any unexpected labels remain visible rather than silently disappearing.
    known = set(LABELS)
    for label in sorted(set(df[label_col].dropna()) - known):
        sub = df[df[label_col] == label]
        ax.scatter(
            sub[x],
            sub[y],
            s=35,
            alpha=0.85,
            label=str(label),
            c="#808080",
            edgecolors="none",
        )

    ax.axhline(0, linewidth=0.8, color="0.6")
    ax.axvline(0, linewidth=0.8, color="0.6")
    ax.set_title(title)


def scatter_3d(ax, df, label_col, title):
    for label in LABELS:
        sub = df[df[label_col] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            sub["PC3"],
            s=28,
            alpha=0.85,
            label=label,
            c=CLASS_COLOURS[label],
            depthshade=True,
        )

    known = set(LABELS)
    for label in sorted(set(df[label_col].dropna()) - known):
        sub = df[df[label_col] == label]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            sub["PC3"],
            s=28,
            alpha=0.85,
            label=str(label),
            c="#808080",
            depthshade=True,
        )

    ax.set_title(title)


def save_both(fig, png_dir: Path, svg_dir: Path, stem: str):
    fig.savefig(png_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(svg_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def make_pairplot(df, model, x, y, summary, png_dir, svg_dir):
    side = parse_side(model)

    xlabel = axis_label(summary, model, x)
    ylabel = axis_label(summary, model, y)

    if side == "paired":
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained"
        )
        scatter_2d(
            axes[0], df, x, y, "label_B", "Same paired PCA coordinates — B labels"
        )
        scatter_2d(
            axes[1], df, x, y, "label_M", "Same paired PCA coordinates — M labels"
        )
        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(title="Curvature label", fontsize=8)
        fig.suptitle(f"{model}: {x} vs {y}")
    else:
        label_col = "label_B" if side == "B" else "label_M"
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        scatter_2d(ax, df, x, y, label_col, f"{model}: {x} vs {y}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title="Curvature label", fontsize=8)

    save_both(fig, png_dir, svg_dir, f"{model}__{x}_vs_{y}")


def make_3d(df, model, summary, png_dir, svg_dir):
    side = parse_side(model)

    xlab = axis_label(summary, model, "PC1")
    ylab = axis_label(summary, model, "PC2")
    zlab = axis_label(summary, model, "PC3")

    if side == "paired":
        fig = plt.figure(figsize=(14, 6), layout="constrained")
        ax_b = fig.add_subplot(1, 2, 1, projection="3d")
        ax_m = fig.add_subplot(1, 2, 2, projection="3d")

        scatter_3d(ax_b, df, "label_B", "Same paired PCA coordinates — B labels")
        scatter_3d(ax_m, df, "label_M", "Same paired PCA coordinates — M labels")

        for ax in (ax_b, ax_m):
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_zlabel(zlab)
            ax.legend(title="Curvature label", fontsize=7)

        fig.suptitle(f"{model}: PC1 vs PC2 vs PC3")
    else:
        label_col = "label_B" if side == "B" else "label_M"
        fig = plt.figure(figsize=(8, 7), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        scatter_3d(ax, df, label_col, f"{model}: PC1 vs PC2 vs PC3")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_zlabel(zlab)
        ax.legend(title="Curvature label", fontsize=8)

    save_both(fig, png_dir, svg_dir, f"{model}__PC1_PC2_PC3_3D")


def reconstruct_scores(results: Path):
    old_scores = pd.read_csv(results / "data" / "pca_scores.csv")
    unified = pd.read_csv(results / "data" / "unified_metrics.csv")
    summary = pd.read_csv(results / "tables" / "pca_summary.csv")

    with open(results / "data" / "pca_parameters.json", encoding="utf-8") as fh:
        params = json.load(fh)

    output = []
    max_error_pc1 = 0.0
    max_error_pc2 = 0.0

    for model, state in params.items():
        original = old_scores[old_scores["model"] == model].copy()
        if original.empty:
            raise ValueError(f"No original PCA scores found for model: {model}")

        # Preserve exactly the population/order used in the original exported PCA scores.
        merged = original[
            ["model", "pair_id", "molecule", "label_B", "label_M", "PC1", "PC2"]
        ].merge(
            unified,
            on="pair_id",
            how="left",
            validate="one_to_one",
            sort=False,
            suffixes=("", "__unified"),
        )

        view = parse_view(model)
        cols = [
            feature_to_unified_column(feature, model, view)
            for feature in state["features"]
        ]

        missing = [c for c in cols if c not in merged.columns]
        if missing:
            raise KeyError(f"{model}: missing unified metric columns: {missing}")

        X = merged[cols].to_numpy(float)
        if not np.isfinite(X).all():
            raise ValueError(f"{model}: non-finite PCA input values")

        scale = np.asarray(state["scale"], dtype=float)
        centre = np.asarray(state["centre"], dtype=float)
        components = np.asarray(state["components"], dtype=float)

        if components.shape[0] < 3:
            raise ValueError(f"{model}: fewer than three PCA components saved")

        Z = X / scale
        scores = (Z - centre) @ components[:3].T

        err1 = float(np.max(np.abs(scores[:, 0] - merged["PC1"].to_numpy(float))))
        err2 = float(np.max(np.abs(scores[:, 1] - merged["PC2"].to_numpy(float))))
        max_error_pc1 = max(max_error_pc1, err1)
        max_error_pc2 = max(max_error_pc2, err2)

        for j, (_, row) in enumerate(merged.iterrows()):
            output.append(
                {
                    "model": model,
                    "pair_id": row["pair_id"],
                    "molecule": row["molecule"],
                    "label_B": row["label_B"],
                    "label_M": row["label_M"],
                    "PC1": scores[j, 0],
                    "PC2": scores[j, 1],
                    "PC3": scores[j, 2],
                }
            )

    reconstructed = pd.DataFrame(output)

    # The reconstruction should reproduce the existing PC1/PC2 essentially exactly.
    tolerance = 1e-8
    if max_error_pc1 > tolerance or max_error_pc2 > tolerance:
        raise ValueError(
            "Frozen PCA reconstruction did not reproduce existing PC1/PC2 scores. "
            f"max PC1 error={max_error_pc1:.3g}; "
            f"max PC2 error={max_error_pc2:.3g}"
        )

    return reconstructed, summary, max_error_pc1, max_error_pc2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        required=True,
        help="Path to results/archie_8193723_complete",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Optional output directory; defaults to RESULTS/pca_review",
    )
    args = ap.parse_args()

    results = Path(args.results).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else results / "pca_review"
    )

    png_dir = output / "figures" / "png"
    svg_dir = output / "figures" / "svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    scores, summary, err1, err2 = reconstruct_scores(results)

    score_csv = output / "pca_scores_PC1_PC2_PC3.csv"
    scores.to_csv(score_csv, index=False)

    # Also save a compact first-three-PC variance table for agent analysis.
    first3 = (
        summary[summary["component"].isin([1, 2, 3])]
        .pivot(
            index=["model", "view", "side", "feature_set"],
            columns="component",
            values="explained_variance",
        )
        .reset_index()
        .rename(columns={1: "PC1_variance", 2: "PC2_variance", 3: "PC3_variance"})
    )
    for col in ["PC1_variance", "PC2_variance", "PC3_variance"]:
        first3[col] *= 100.0
    first3["PC1_PC2_PC3_cumulative"] = (
        first3["PC1_variance"]
        + first3["PC2_variance"]
        + first3["PC3_variance"]
    )
    first3.to_csv(output / "pca_first3_variance.csv", index=False)

    for model, df in scores.groupby("model", sort=False):
        make_pairplot(df, model, "PC1", "PC2", summary, png_dir, svg_dir)
        make_pairplot(df, model, "PC1", "PC3", summary, png_dir, svg_dir)
        make_pairplot(df, model, "PC2", "PC3", summary, png_dir, svg_dir)
        make_3d(df, model, summary, png_dir, svg_dir)

    print("Frozen PCA reconstruction successful.")
    print(f"Maximum PC1 reconstruction error: {err1:.3g}")
    print(f"Maximum PC2 reconstruction error: {err2:.3g}")
    print(f"Scores with PC3: {score_csv}")
    print(f"PNG figures: {png_dir}")
    print(f"SVG figures: {svg_dir}")


if __name__ == "__main__":
    main()

