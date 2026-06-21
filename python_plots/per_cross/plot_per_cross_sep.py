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
    "5": SCRIPT_DIR / "case_5.csv",
    "10": SCRIPT_DIR / "case_10.csv",
}

CASE_LABELS = list(FILES.keys())

TARGET_RPS = [20, 40, 60, 80, 100, 150, 300, 450]

# ============================================================
# Helpers
# ============================================================

def map_rp(rp):
    return min(TARGET_RPS, key=lambda x: abs(x - rp))


def compute_improvement(per, cross):
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

        # map return periods
        df["rp_mapped"] = df["rp"].apply(map_rp)

        def get_value(name, rp):
            sub = df[
                (df["base_name"] == name) &
                (df["rp_mapped"] == rp)
            ]
            if sub.empty:
                return np.nan
            return sub["num_loss_of_load_e_demand_mean"].values[0]

        for rp in TARGET_RPS:
            # k-means
            cross = get_value("kmeans_cross_dirac", rp)
            per = get_value("kmeans_per_dirac", rp)
            results[rp]["kmeans"].append(
                compute_improvement(per, cross)
            )

            # k-medoids
            cross = get_value("kmedoids_cross_dirac", rp)
            per = get_value("kmedoids_per_dirac", rp)
            results[rp]["kmedoids"].append(
                compute_improvement(per, cross)
            )

    return results


# ============================================================
# Plot
# ============================================================

def plot_grid(results, savepath, method):
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    x = np.arange(len(CASE_LABELS))
    x_numeric = np.array([int(c) for c in CASE_LABELS])  # for trend

    for i, rp in enumerate(TARGET_RPS):
        ax = axes[i]

        values = results[rp][method]

        # ---------- COLORS ----------
        colors = [
            "#2E86DE" if v >= 0 else "#E74C3C"
            for v in values
        ]

        # ---------- GOOD REGION ----------
        ax.axhspan(0, 1, color="green", alpha=0.05, zorder=0)

        # ---------- BARS ----------
        ax.bar(
            x,
            values,
            width=0.55,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7
        )

        # ---------- TREND LINE (KEY ADDITION) ----------
        ax.plot(
            x,
            values,
            color="black",
            marker="o",
            linewidth=1.8,
            zorder=3
        )


        # zero line
        ax.axhline(0, color="black", linewidth=1)

        # titles and ticks
        ax.set_title(f"RP = {rp}", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(CASE_LABELS, fontsize=13)

        # fixed y-axis
        ax.set_ylim(-0.5, 1)
        ax.set_yticks(np.linspace(-0.5, 1, 4))
        ax.tick_params(axis='y', labelsize=13)

        # grid
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    # global labels
    fig.supxlabel("Number of scenarios", fontsize=16)
    fig.supylabel(
        "Relative improvement of cross-scenario compared to per-scenario",
        fontsize=16
    )

    # layout
    plt.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.92,
        bottom=0.08,
        wspace=0.25,
        hspace=0.35,
    )

    fig.patch.set_facecolor("white")

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    results = compute_all_results()

    plot_grid(
        results,
        PLOTS_DIR / "improvement_kmeans.png",
        method="kmeans"
    )

    plot_grid(
        results,
        PLOTS_DIR / "improvement_kmedoids.png",
        method="kmedoids"
    )


if __name__ == "__main__":
    main()
