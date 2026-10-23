
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(
    "/users/pfb19164/Desktop/post_medical_leave/PFPeptoids/"
    "PFPeptoids/analysis/image_classification/metric_dev/"
    "results/archie_8193723_complete/surface_papl_analysis/"
    "pristine_extraction_8202573"
)

OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = {
    "B": {
        "label": "Pristine bacterial (POPE75:POPG25)",
        "color": "tab:blue",
    },
    "M": {
        "label": "Pristine mammalian (POPC60:CHOL40)",
        "color": "tab:orange",
    },
}

data = {}

for side in SYSTEMS:
    df = pd.read_csv(BASE / f"pristine_frames_{side}.csv")
    df = df.sort_values("time_ns")
    data[side] = df

def format_axes(ax):
    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel("Projected APL (Å²/lipid)")
    ax.grid(alpha=0.25)

def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)

# Individual membrane plots
for side, config in SYSTEMS.items():
    df = data[side]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        df["time_ns"],
        df["projected_APL_A2"],
        color=config["color"],
        linewidth=1.8,
    )

    ax.set_title(config["label"])
    format_axes(ax)
    save(fig, f"pristine_pAPL_{side}")

# Combined plot
fig, ax = plt.subplots(figsize=(10, 5.5))

for side, config in SYSTEMS.items():
    df = data[side]

    ax.plot(
        df["time_ns"],
        df["projected_APL_A2"],
        label=config["label"],
        color=config["color"],
        linewidth=1.8,
    )

ax.set_title("Pristine membrane projected APL over time")
format_axes(ax)
ax.legend(frameon=False)
save(fig, "pristine_pAPL_combined")

print(f"Figures saved to: {OUT}")

