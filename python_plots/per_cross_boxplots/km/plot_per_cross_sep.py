import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

FILES = {
    "2": SCRIPT_DIR / "case_2.csv",
    "5": SCRIPT_DIR / "case_5.csv",
    "10": SCRIPT_DIR / "case_10.csv",
}

CASE_LABELS = list(FILES.keys())
TARGET_RPS = [20, 40, 60, 80, 100, 150, 300, 450]

# ============================================================
# Load + clean
# ============================================================

def load_all():
    all_data = []

    for case, path in FILES.items():
        df = pd.read_csv(path)

        # remove benchmark
        df = df[~df["base_name"].str.contains("HourlyBenchmark", na=False)]

        # extract method and type
                
        parts = df["base_name"].str.split("_")

        df["method"] = parts.str[0]
        df["type"]   = parts.str[1]

        # add scenario label
        df["scenario"] = case

        df["loss_of_load_e_demand"] = pd.to_numeric(
            df["loss_of_load_e_demand"], errors="coerce"
        )

        # normalize by number of scenarios
        df["loss_of_load_e_demand"] = (
            df["loss_of_load_e_demand"] / df["scenario"].astype(int)
        )


        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

# ============================================================
# Plotting
# ============================================================

def plot_grid(df, method, savepath):

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for i, rp in enumerate(TARGET_RPS):
        ax = axes[i]

        sub = df[
            (df["method"] == method) &
            (df["rp"] == rp)
        ]

        if sub.empty:
            continue

        # -------- Build boxplot data --------
        positions = []
        data = []
        colors = []

        base_x = np.arange(len(CASE_LABELS))
        width = 0.25

        for j, case in enumerate(CASE_LABELS):

            for k, typ in enumerate(["per", "cross"]):
                vals = sub[
                    (sub["scenario"] == case) &
                    (sub["type"] == typ)
                ]["loss_of_load_e_demand"].dropna()

                if len(vals) == 0:
                    continue

                pos = j + (k - 0.5) * width
                positions.append(pos)
                data.append(vals)

                colors.append("#1f77b4" if typ == "per" else "#ff7f0e")

        # -------- Plot boxplots --------
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.8,
            patch_artist=True,
            medianprops={
                    "color": "black",
                    "linewidth": 1.0                }

        )

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # -------- Axes formatting --------
        ax.set_title(f"RP = {rp}", fontsize=16)
        ax.set_xticks(base_x)
        ax.set_xticklabels(CASE_LABELS)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis='both', labelsize=13)

    fig.supxlabel("Number of scenarios", fontsize=18)
    fig.supylabel("Loss of load per scenario (MWh)", fontsize=18)

    plt.subplots_adjust(
        left=0.08, right=0.97,
        top=0.92, bottom=0.08,
        wspace=0.25, hspace=0.35
    )
    
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Per-scenario"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="Cross-scenario"),
    ]
    
        
    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        title="Selection methods",
            fontsize=14,
            title_fontsize=15

    )




    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    df = load_all()

    plot_grid(
        df,
        method="kmeans",
        savepath=PLOTS_DIR / "lol_kmeans_boxplot.png"
    )

    plot_grid(
        df,
        method="kmedoids",
        savepath=PLOTS_DIR / "lol_kmedoids_boxplot.png"
    )

if __name__ == "__main__":
    main()