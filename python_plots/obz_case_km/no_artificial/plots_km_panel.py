import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# ============================================================
# Constants (faithful translation of Julia constants)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

MARKER_MAP = {
    "per_scenario": "^",
    "cross_scenario": "o",
}

COLOR_MAP_METHOD_WEIGHT_STMETHOD = {
    "k_means_convex_per_scenario": "gold",
    "k_means_convex_cross_scenario": "orange",
    "k_means_dirac_per_scenario": "gold",
    "k_means_dirac_cross_scenario": "orange",
    "k_medoids_convex_per_scenario": "blue",
    "k_medoids_convex_cross_scenario": "purple",
    "k_medoids_dirac_per_scenario": "blue",
    "k_medoids_dirac_cross_scenario": "purple",
}

LEGEND_MAP = {
    "k_means_convex_per_scenario": "K-means: per-scenario",
    "k_means_convex_cross_scenario": "K-means: cross-scenario",
    "k_means_dirac_per_scenario": "K-means: dirac weights, per-scenario",
    "k_means_dirac_cross_scenario": "K-means: dirac weights, cross-scenario",
    "k_medoids_convex_per_scenario": "K-medoids: per-scenario",
    "k_medoids_convex_cross_scenario": "K-medoids: cross-scenario",
    "k_medoids_dirac_per_scenario": "K-medoids: dirac weights, per-scenario",
    "k_medoids_dirac_cross_scenario": "K-medoids: dirac weights, cross-scenario",
}

VALUE_MAP = {
    "rel_regret": "Relative regret",
    "num_loss_of_load_e_demand": "Number of timesteps with electricity loss of load",
    "num_loss_of_load_h2_demand": "Number of timesteps with hydrogen loss of load",
    "total_steps_loss_of_load" : "Total number of timesteps with loss of load",
    "time_to_cluster": "Time to cluster (s)",
    "time_to_create": "Time to create (s)",
    "time_to_solve": "Time to solve (s)",
    "total_time": "Total time (s)",
}



# ============================================================
# Panel plotting (equivalent to plot_values_quantiles_panel)
# ============================================================

def plot_panel(ax, stats_df, case_df, value, method, rp_to_pos):
    df = stats_df.merge(case_df, on="base_name")
    df = df[df.base_name != "0_HourlyBenchmark"]
    df = df[df.method == method]

    for base_name, g in df.groupby("base_name"):
        g = g.sort_values("rp")

            
        stochastic = g.stochastic_method.iloc[0]
        weight = g.weight_type.iloc[0]
        key = f"{method}_{weight}_{stochastic}"

        color = COLOR_MAP_METHOD_WEIGHT_STMETHOD[key]

        if stochastic.endswith("per_scenario"):
            stochastic_clean = "per_scenario"
        else:
            stochastic_clean = "cross_scenario"

        marker = MARKER_MAP[stochastic_clean]


        mean = g[f"{value}_mean"]
        q25 = g[f"{value}_q25"]
        q75 = g[f"{value}_q75"]

        xpos = [rp_to_pos[rp] for rp in g.rp]

        ax.plot(
            xpos,
            mean,
            color=color,
            lw=2,
            marker=marker,
            ms=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=LEGEND_MAP[key],
        )

        ax.fill_between(
            xpos,
            q25,
            q75,
            color=color,
            alpha=0.25,
            linewidth=0,
        )


# ============================================================
# Grid plotting (equivalent to plot_values_quantiles_grid)
# ============================================================

def plot_values_quantiles_grid(stats_df, case_df, value, savepath):
    rp_vals = sorted(v for v in stats_df["rp"].unique() if v != 1)
    rp_pos = list(range(len(rp_vals)))               # 0,1,2,...
    rp_to_pos = dict(zip(rp_vals, rp_pos))           # rp → position

    fig, ax = plt.subplots(
        1,1,
        figsize=(14, 5),
        sharex=True,
        sharey=False,
    )

    plot_panel(ax, stats_df, case_df, value, "k_medoids", rp_to_pos)

    ax.set_xticks(rp_pos)
    #ax.set_xticklabels([str(rp) for rp in rp_vals])
    ax.set_xticklabels([str(rp) for rp in rp_vals], rotation=45, ha="right")

    # Subtle grid, attached panels
    ax.set_axisbelow(True)
    ax.grid(axis="both", alpha=0.2)
    ax.tick_params(axis="both", labelsize=11)


    # Global labels
    fig.supxlabel("Number of representative periods", fontsize=15)
    fig.supylabel(VALUE_MAP[value], fontsize=15)

    # # Single legend (outside)
    # handles, labels = axes[0, -1].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="center left",
    #     bbox_to_anchor=(1.01, 0.5),
    #     frameon=True,
    # )
        
    legend_handles = [
        # Line2D(
        #     [0], [0],
        #     color="gold",
        #     marker="^",
        #     lw=2,
        #     markersize=6,
        #     markeredgecolor="black",
        #     markeredgewidth=0.5,
        #     label="K-means: per-scenario",
        # ),
        # Line2D(
        #     [0], [0],
        #     color="orange",
        #     marker="o",
        #     lw=2,
        #     markersize=6,
        #     markeredgecolor="black",
        #     markeredgewidth=0.5,
        #     label="K-means: cross-scenario",
        # ),
        # Line2D(
        #     [0], [0],
        #     color="blue",
        #     marker="^",
        #     lw=2,
        #     markersize=6,
        #     markeredgecolor="black",
        #     markeredgewidth=0.5,
        #     label="K-medoids: per-scenario",
        # ),
        Line2D(
            [0], [0],
            color="purple",
            marker="o",
            lw=2,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="K-medoids: cross-scenario",
        ),
    ]


    fig.legend(
    handles=legend_handles,
    title="Clustering methods",
    title_fontsize=17,      # title size
    fontsize=16,            # entry text size
    loc="center left",
    bbox_to_anchor=(0.86, 0.5), 
    frameon=True,
    borderpad=1.2,
    labelspacing=0.8,
    handlelength=2.5,
    handletextpad=0.8,
)



    plt.subplots_adjust(
        left=0.06,
        right=0.82,
        top=0.90,
        bottom=0.10,
        wspace=0.05,
        hspace=0.05,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_combined_relregret_lol(stats_df, case_df, savepath):
    rp_vals = sorted(v for v in stats_df["rp"].unique() if v != 1)
    rp_pos = list(range(len(rp_vals)))
    rp_to_pos = dict(zip(rp_vals, rp_pos))

    # --- CREATE 1x2 LAYOUT ---
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharex=True,
        sharey=False,
    )

    ax1, ax2 = axes

    # --- LEFT: RELATIVE REGRET ---
    plot_panel(ax1, stats_df, case_df, "rel_regret", "k_medoids", rp_to_pos)
    ax1.set_title("Relative regret", fontsize=16)

    # --- RIGHT: TOTAL LOSS OF LOAD ---
    plot_panel(ax2, stats_df, case_df, "total_lol", "k_medoids", rp_to_pos)
    ax2.set_title("Total loss of load (MW)", fontsize=16)

    # --- SHARED X AXIS ---
    for ax in axes:
        ax.set_xticks(rp_pos)
        ax.set_xticklabels([str(rp) for rp in rp_vals], rotation=45, ha="right")
        ax.set_axisbelow(True)
        ax.grid(axis="both", alpha=0.2)
        ax.tick_params(axis="both", labelsize=14)

    # --- GLOBAL LABELS ---
    fig.supxlabel("Number of representative periods", fontsize=18, y=-0.01)


    # --- LEGEND (single, right side) ---
    legend_handles = [
        Line2D(
            [0], [0],
            color="purple",
            marker="o",
            lw=2,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="K-medoids",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        title="Clustering methods",
        title_fontsize=17,
        fontsize=16,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        frameon=True,
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.85,
        top=0.90,
        bottom=0.15,
        wspace=0.25,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main (mirrors your Julia driver)
# ============================================================

def main():
    plots_dir = Path(SCRIPT_DIR / "plots")
    plots_dir.mkdir(exist_ok=True)

    stats = pd.read_csv(SCRIPT_DIR / "stats.csv")


    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

    for value in [
        "rel_regret",
        "num_loss_of_load_e_demand",
        "time_to_cluster",
        "time_to_create",
        "time_to_solve",
        "total_time",
        "total_steps_loss_of_load",
        "num_loss_of_load_h2_demand"
    ]:
        plot_values_quantiles_grid(
            stats,
            case_df,
            value,
            plots_dir / f"{value}.png",
        )
    plot_combined_relregret_lol(
    stats,
    case_df,
    plots_dir / "combined_relregret_lol2.png",
    )



if __name__ == "__main__":
    main()