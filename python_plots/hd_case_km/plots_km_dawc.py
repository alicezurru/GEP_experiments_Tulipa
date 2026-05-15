import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Constants (UNCHANGED STYLE)
# ============================================================

COLS = ["DISTANT", "HALFMIXED", "CLOSE", "MIXED"]

ROWS = [
    ("DAWC CONVEX WEIGHTS", "_dconvex"),
]

# ✅ UPDATED: full names instead of abbreviations
FILE_PREFIX = {
    "DISTANT": "stats_distant",
    "HALFMIXED": "stats_halfmixed",
    "CLOSE": "stats_close",
    "MIXED": "stats_mixed",
}

MARKER_MAP = {
    "per_scenario": "^",
    "cross_scenario": "o",
}

COLOR_MAP = {
    ("k_means", "per_scenario"): "gold",
    ("k_means", "cross_scenario"): "orange",
    ("k_medoids", "per_scenario"): "blue",
    ("k_medoids", "cross_scenario"): "purple",
}

VALUE_MAP = {
    "rel_regret": "Relative regret",
    "num_loss_of_load_e_demand": "Number of timesteps with loss of load",
    "time_to_cluster": "Time to cluster (s)",
    "time_to_create": "Time to create (s)",
    "time_to_solve": "Time to solve (s)",
    "total_time": "Total time (s)",
}

# ============================================================
# Panel plotting (IDENTICAL SEMANTICS)
# ============================================================

def plot_panel(ax, stats_df, case_df, value, rp_to_pos):
    df = stats_df.merge(case_df, on="base_name")
    df = df[df.base_name != "0_HourlyBenchmark"]

    for (method, stochastic), g in df.groupby(["method", "stochastic_method"]):
        g = g.sort_values("rp")

        stochastic_clean = (
            "per_scenario"
            if stochastic.endswith("per_scenario")
            else "cross_scenario"
        )

        xpos = [rp_to_pos[rp] for rp in g.rp]

        ax.plot(
            xpos,
            g[f"{value}_mean"],
            color=COLOR_MAP[(method, stochastic_clean)],
            marker=MARKER_MAP[stochastic_clean],
            lw=2,
            ms=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

        ax.fill_between(
            xpos,
            g[f"{value}_q25"],
            g[f"{value}_q75"],
            color=COLOR_MAP[(method, stochastic_clean)],
            alpha=0.25,
            linewidth=0,
        )

# ============================================================
# Grid plotting (3 × 4, SAME STYLE)
# ============================================================

def plot_values_quantiles_grid(stats_dict, case_df, value, savepath):
    
    row_suffix = ROWS[0][1]

    rp_vals = sorted(
        v for v in stats_dict[COLS[0]][row_suffix]["rp"].unique() if v != 1
    )

    rp_pos = list(range(len(rp_vals)))
    rp_to_pos = dict(zip(rp_vals, rp_pos))

        
    fig, axes = plt.subplots(
        1, 4,
        figsize=(17, 4.5),
        sharex=True,
        sharey=True,
    )

    # normalize axes indexing
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)


    for row_idx, (row_title, suffix) in enumerate(ROWS):
        for col_idx, col in enumerate(COLS):
            ax = axes[row_idx, col_idx]
            stats_df = stats_dict[col][suffix]

            plot_panel(ax, stats_df, case_df, value, rp_to_pos)

            if row_idx == 0:
                ax.set_title(col, fontsize=16)

            # if col_idx == 0:
            #     ax.set_ylabel(row_title, fontsize=15, rotation=90, labelpad=22)
            # else:
            #     ax.set_ylabel("")

            if row_idx == 0:
                ax.set_xticks(rp_pos)
                ax.set_xticklabels(
                    [str(rp) for rp in rp_vals],
                    rotation=45,
                    ha="right",
                )
            else:
                ax.set_xlabel("")

            ax.set_axisbelow(True)
            ax.grid(axis="both", alpha=0.2)
            ax.tick_params(axis="both", labelsize=11)
            
            if value == "rel_regret":
                ax.set_ylim(0.0, 0.15)
    
        
    for row_idx, (row_title, _) in enumerate(ROWS):
        # Get vertical center of the row from the first column
        bbox = axes[row_idx, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)

        fig.text(
            0.035, y_center,
            row_title,
            rotation=90,
            fontsize=15,
            va="center",
            ha="center",
        )



    fig.supxlabel("Number of representative periods", fontsize=15)
    fig.supylabel(VALUE_MAP[value], fontsize=15, x=0.045)

    legend_handles = [
        Line2D([0], [0], color="gold", marker="^", lw=2,
               markeredgecolor="black", label="K-means: per-scenario"),
        Line2D([0], [0], color="orange", marker="o", lw=2,
               markeredgecolor="black", label="K-means: cross-scenario"),
        Line2D([0], [0], color="blue", marker="^", lw=2,
               markeredgecolor="black", label="K-medoids: per-scenario"),
        Line2D([0], [0], color="purple", marker="o", lw=2,
               markeredgecolor="black", label="K-medoids: cross-scenario"),
    ]

    fig.legend(
        handles=legend_handles,
        title="Clustering methods",
        title_fontsize=17,
        fontsize=16,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.8,
    )

    plt.subplots_adjust(
        left=0.09,
        right=0.86,
        top=0.92,
        bottom=0.20,
        wspace=0.05,
        hspace=0.08,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

    stats = {}
    for col in COLS:
        base = FILE_PREFIX[col]
        
        stats[col] = {
            "_dconvex": pd.read_csv(SCRIPT_DIR / f"{base}_dconvex.csv"),
        }


    for value in VALUE_MAP:
        plot_values_quantiles_grid(
            stats,
            case_df,
            value,
            PLOTS_DIR / f"{value}.png",
        )

if __name__ == "__main__":
    main()
