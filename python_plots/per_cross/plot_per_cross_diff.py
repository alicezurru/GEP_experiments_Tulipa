import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Inputs
# ============================================================

FILES = {
    "2": SCRIPT_DIR / "case_2.csv",
    "4": SCRIPT_DIR / "case_5.csv",
    "10": SCRIPT_DIR / "case_10.csv",
}

CASE_LABELS = list(FILES.keys())

TARGET_RPS = [20, 40, 60, 80, 100, 150, 300, 450]

# ============================================================
# Helpers
# ============================================================

def map_rp(rp):
    """Map RP to closest target (±7 tolerance implicitly handled)."""
    return min(TARGET_RPS, key=lambda x: abs(x - rp))


def compute_improvement(per, cross):
    """(per - cross) / per"""
    if per == 0:
        return np.nan
    return (per - cross) / per


# ============================================================
# Core computation
# ============================================================

def compute_all_results():
    results = {
        rp: {"kmeans": [], "kmedoids": []}
        for rp in TARGET_RPS
    }

    for case in CASE_LABELS:
        df = pd.read_csv(FILES[case])

        # remove benchmark
        df = df[df["base_name"] != "0_HourlyBenchmark"].copy()

        # map RP
        df["rp_mapped"] = df["rp"].apply(map_rp)

        for rp in TARGET_RPS:
            row = {}

            def get_value(name):
                sub = df[
                    (df["base_name"] == name) &
                    (df["rp_mapped"] == rp)
                ]
                if sub.empty:
                    return np.nan
                return sub["num_loss_of_load_e_demand_mean"].values[0]

            # --- kmeans ---
            cross = get_value("kmeans_cross_dirac")
            per = get_value("kmeans_per_dirac")
            results[rp]["kmeans"].append(
                compute_improvement(per, cross)
            )

            # --- kmedoids ---
            cross = get_value("kmedoids_cross_dirac")
            per = get_value("kmedoids_per_dirac")
            results[rp]["kmedoids"].append(
                compute_improvement(per, cross)
            )

    return results


# ============================================================
# Plot
# ============================================================

def plot_grid(results, savepath):
    fig, axes = plt.subplots(2, 4, figsize=(15, 12))
    axes = axes.flatten()

    x = np.arange(len(CASE_LABELS))
    width = 0.35

    for i, rp in enumerate(TARGET_RPS):
        ax = axes[i]

        kmeans_vals = results[rp]["kmeans"]
        kmedoids_vals = results[rp]["kmedoids"]

        # bars
        ax.bar(
            x - width / 2,
            kmeans_vals,
            width,
            label="k-means",
            color="orange",
            edgecolor="black"
        )

        ax.bar(
            x + width / 2,
            kmedoids_vals,
            width,
            label="k-medoids",
            color="purple",
            edgecolor="black"
        )

        # zero line
        ax.axhline(0, color="black", linewidth=1)

        # formatting
        ax.set_title(f"RP = {rp}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(CASE_LABELS)

        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)

    # global labels
    fig.supxlabel("Number of scenarios", fontsize=14)
    fig.supylabel("Improvement (per - cross) / per", fontsize=14)

    # legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="orange", ec="black", label="k-means"),
        plt.Rectangle((0, 0), 1, 1, color="purple", ec="black", label="k-medoids"),
    ]

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        title="Method"
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.9,
        top=0.93,
        bottom=0.08,
        wspace=0.25,
        hspace=0.3,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    results = compute_all_results()

    plot_grid(
        results,
        PLOTS_DIR / "improvement_grid.png",
    )


if __name__ == "__main__":
    main()