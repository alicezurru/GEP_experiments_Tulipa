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


# Columns are beta values
BETAS = ["b0", "b0.05", "b0.1", "b0.2"]

MARKER_MAP = {
    "per_scenario": "^",
    "cross_scenario": "o",
}

COLOR_MAP = {
    ("k_means", "per_scenario"): "gold",
    ("k_means", "cross_scenario"): "orange",
    ("k_medoids", "per_scenario"): "blue",
    ("k_medoids", "cross_scenario"): "#6a0dad",
}

VALUE_MAP = {
    "rel_regret": "Relative regret",
    "num_loss_of_load_e_demand": "Number of timesteps with electricity loss of load",
    "num_loss_of_load_h2_demand": "Number of timesteps with hydrogen loss of load",
    "total_steps_loss_of_load" : "Total number of timesteps with loss of load",
    "total_lol" : "Total loss of load (GWh)",
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
        
    rp_vals = sorted(
        v for v in stats_dict["b0"]["rp"].unique() if v != 1
    )

    rp_pos = list(range(len(rp_vals)))
    rp_to_pos = dict(zip(rp_vals, rp_pos))

        
    fig, axes = plt.subplots(
        1, 4,
        figsize=(17, 4),
        sharex=True,
        sharey=True,
    )

                
    for col_idx, beta in enumerate(BETAS):
        ax = axes[col_idx]
        stats_df = stats_dict[beta]


        plot_panel(ax, stats_df, case_df, value, rp_to_pos)

                
        ax.set_title(r"$\beta$ = " + beta[1:], fontsize=20)

        ax.set_xticks(rp_pos)
        ax.set_xticklabels(
            [str(rp) for rp in rp_vals],
            rotation=45,
            ha="right",
        )


        ax.set_axisbelow(True)
        ax.grid(axis="both", alpha=0.2)
        ax.tick_params(axis="both", labelsize=14)
        
        if value == "rel_regret":
            ax.set_ylim(0.0, 0.4)
        if value == "total_steps_loss_of_load":
            ax.set_ylim(-100, 700)
        if value == "total_lol":
            ax.set_ylim(-100, 2000)




    fig.supxlabel("Number of representative periods", fontsize=17, y=-0.1)
    fig.supylabel(VALUE_MAP[value], fontsize=17, x=0.03)

    legend_handles = [
        Line2D([0], [0], color="#6a0dad", marker="o", lw=2,
               markeredgecolor="black", label="K-medoids"),
    ]

    fig.legend(
        handles=legend_handles,
        title="Clustering methods",
        title_fontsize=18,
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
        left=0.10,
        right=0.86,
        top=0.92,
        bottom=0.10,
        wspace=0.05,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

        
    stats = {}

    for beta in BETAS:
        stats[beta] = pd.read_csv(
            SCRIPT_DIR / f"stats_{beta}.csv"
        )



    for value in VALUE_MAP:
        plot_values_quantiles_grid(
            stats,
            case_df,
            value,
            PLOTS_DIR / f"{value}.png",
        )

if __name__ == "__main__":
    main()
