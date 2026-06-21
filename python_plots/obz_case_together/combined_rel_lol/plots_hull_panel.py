import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

COLS = ["NO ARTIFICIAL", "HCWC", "DAWC"]

METHODS = ["convex_hull", "convex_hull_with_null"]

MARKER_MAP = {
    "per_scenario": "^",
    "cross_scenario": "o",
}

COLOR_MAP = {
    "convex_hull_per_scenario": "#4d4d4d",
    "convex_hull_cross_scenario": "#9e9e9e",
    "convex_hull_with_null_per_scenario": "#2ca02c",
    "convex_hull_with_null_cross_scenario": "#98df8a",
}

LEGEND_MAP = {
    "convex_hull_per_scenario": "Convex hull: per-scenario",
    "convex_hull_cross_scenario": "Convex hull: cross-scenario",
    "convex_hull_with_null_per_scenario": "Bounded conical hull: per-scenario",
    "convex_hull_with_null_cross_scenario": "Bounded conical hull: cross-scenario",
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

YLIM_MAP = {
    "rel_regret": (-0.1, 8.0),
    #  "total_steps_loss_of_load": (-100, 2000),
    "total_lol": (-500, 200000),
}

# ============================================================
# Data preparation (UNCHANGED)
# ============================================================

def prepare_results(results_df, case_df):
    results_df = results_df.merge(case_df, on="base_name", how="left")
    hourly = results_df[results_df.base_name == "0_HourlyBenchmark"].iloc[0]

    results_df["rel_regret"] = [
        0.0 if r.base_name == "0_HourlyBenchmark"
        else (r.objective_value_resolve_benchmark - hourly.objective_value)
             / hourly.objective_value
        for r in results_df.itertuples()
    ]

    results_df["total_time"] = (
        results_df.time_to_cluster
        + results_df.time_to_read
        + results_df.time_to_create
        + results_df.time_to_solve
        + results_df.time_to_save
    )

    results_df["num_loss_of_load_e_demand"] = (
        results_df.num_loss_of_load_e_demand
        - hourly.num_loss_of_load_e_demand
    )

    results_df["num_loss_of_load_h2_demand"] = (
        results_df.num_loss_of_load_h2_demand
        - hourly.num_loss_of_load_h2_demand
    )

    results_df["total_steps_loss_of_load"] = (
        results_df.num_loss_of_load_h2_demand
        + results_df.num_loss_of_load_e_demand
    )
    results_df["loss_of_load_e_demand"] = (
        results_df.loss_of_load_e_demand
        - hourly.loss_of_load_e_demand
    )

    results_df["loss_of_load_h2_demand"] = (
        results_df.loss_of_load_h2_demand
        - hourly.loss_of_load_h2_demand
    )

    results_df["total_lol"] = (
        results_df.loss_of_load_h2_demand
        + results_df.loss_of_load_e_demand
    )


    return results_df

# ============================================================
# Panel plotting (both methods inside)
# ============================================================

def plot_panel(ax, df, stats_df, case_df, value, rp_to_pos):

    df = df[df.base_name != "0_HourlyBenchmark"]

    for method in METHODS:
        df_m = df[df.method == method]

        for stoch, g in df_m.groupby("stochastic_method"):
            g = g.sort_values("rp")
            key = f"{method}_{stoch}"
            xpos = [rp_to_pos[rp] for rp in g.rp]

            ax.plot(
                xpos,
                g[value],
                color=COLOR_MAP[key],
                lw=2,
                marker=MARKER_MAP[stoch],
                ms=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        
    df_stats = stats_df.merge(case_df, on="base_name")
    df_stats = df_stats[df_stats.base_name != "0_HourlyBenchmark"]
    df_stats = df_stats[df_stats.method == "k_medoids"]

    for base_name, g in df_stats.groupby("base_name"):
        g = g.sort_values("rp")

        stochastic = g.stochastic_method.iloc[0]

        if stochastic.endswith("per_scenario"):
            stoch_clean = "per_scenario"
        else:
            stoch_clean = "cross_scenario"

        marker = MARKER_MAP[stoch_clean]

        xpos = [rp_to_pos[rp] for rp in g.rp]

        mean = g[f"{value}_mean"]
        q25 = g[f"{value}_q25"]
        q75 = g[f"{value}_q75"]

        # choose ONE color for k-medoids (simple + consistent)
        color = "#6a0dad"   # purple

        ax.plot(
            xpos,
            mean,
            color=color,
            lw=2,
            marker=marker,
            ms=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
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
# Grid plotting (3 × 4)
# ============================================================

# def plot_values_grid(results_sets, stats_sets, case_df, value, savepath):

#     rp_vals = sorted(v for v in results_sets[COLS[0]].rp.unique() if v != 1)
#     rp_pos = list(range(len(rp_vals)))
#     rp_to_pos = dict(zip(rp_vals, rp_pos))

#     fig, ax = plt.subplots(
#         1, 1,
#         figsize=(6, 5),
#         sharex=True,
#         sharey=True,
#     )

        
#     col = "NO ARTIFICIAL"
#     df = results_sets[col]
#     stats_df = stats_sets[col]

#     plot_panel(ax, df, stats_df, case_df, value, rp_to_pos)

#     # No title here

#     ax.set_xticks(rp_pos)
#     ax.set_xticklabels(
#         [str(rp) for rp in rp_vals],
#         rotation=45,
#         ha="right",
#     )

#     ax.set_axisbelow(True)
#     ax.grid(alpha=0.2)
#     ax.tick_params(labelsize=14)


#     fig.supxlabel("Number of representative periods", fontsize=17, y=-0.05)
#     fig.supylabel(VALUE_MAP[value], fontsize=17, x=0.01)

#     if value in YLIM_MAP:
#         ymin, ymax = YLIM_MAP[value]
#         ax.set_ylim(ymin, ymax)

#     # handles = [
#     #     Line2D([0], [0], color="#4d4d4d", marker="^", lw=2, label="Convex hull: per-scenario"),
#     #     Line2D([0], [0], color="#9e9e9e", marker="o", lw=2, label="Convex hull: cross-scenario"),
#     #     Line2D([0], [0], color="#2ca02c", marker="^", lw=2, label="Bounded conical hull: per-scenario"),
#     #     Line2D([0], [0], color="#98df8a", marker="o", lw=2, label="Bounded conical hull: cross-scenario"),
#     # ]
#     handles = [
#         Line2D([0], [0], color="#9e9e9e", marker="o", lw=2, label="Convex hull"),
#         Line2D([0], [0], color="#98df8a", marker="o", lw=2, label="Bounded conical hull"),
#         Line2D([0], [0], color="#6a0dad", marker="o", lw=2, label="K-medoids"),
#     ]

#     fig.legend(
#         handles=handles,
#         title="Clustering methods",
#         title_fontsize=18,
#         fontsize=16,
#         loc="center left",
#         bbox_to_anchor=(0.88, 0.5),
#         frameon=True,
#     )

        
#     plt.subplots_adjust(
#         left=0.12,
#         right=0.80,
#         top=0.92,
#         bottom=0.15,
#     )


#     fig.savefig(savepath, dpi=300, bbox_inches="tight")
#     plt.close(fig)

def plot_two_values(results_sets, stats_sets, case_df, savepath):

    values = ["rel_regret", "total_lol"]

    rp_vals = sorted(v for v in results_sets["NO ARTIFICIAL"].rp.unique() if v != 1)
    rp_pos = list(range(len(rp_vals)))
    rp_to_pos = dict(zip(rp_vals, rp_pos))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharex=True,
    )

    col = "NO ARTIFICIAL"

    for i, value in enumerate(values):
        ax = axes[i]

        df = results_sets[col]
        stats_df = stats_sets[col]

        plot_panel(ax, df, stats_df, case_df, value, rp_to_pos)

        # axis formatting
        ax.set_xticks(rp_pos)
        ax.set_xticklabels(
            [str(rp) for rp in rp_vals],
            rotation=45,
            ha="right",
        )

        ax.set_axisbelow(True)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=14)

        # individual y-labels
        ax.set_title(VALUE_MAP[value], fontsize=17)

        if value in YLIM_MAP:
            ymin, ymax = YLIM_MAP[value]
            ax.set_ylim(ymin, ymax)

    fig.supxlabel("Number of representative periods", fontsize=16)

    # legend (same as before)
    handles = [
        Line2D([0], [0], color="#9e9e9e", marker="o", lw=2, label="Convex hull"),
        Line2D([0], [0], color="#98df8a", marker="o", lw=2, label="Bounded conical hull"),
        Line2D([0], [0], color="#6a0dad", marker="o", lw=2, label="K-medoids"),
    ]

    fig.legend(
        handles=handles,
        title="Clustering methods",
        title_fontsize=18,
        fontsize=16,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        frameon=True,
    )

    plt.subplots_adjust(
        left=0.10,
        right=0.85,
        bottom=0.15,
        wspace=0.5,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    
    stats_sets = {
        "NO ARTIFICIAL": pd.read_csv(SCRIPT_DIR / "stats.csv"),
    }

    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

    results = {
        "NO ARTIFICIAL": pd.read_csv(SCRIPT_DIR / "results.csv"),
    }

    for k in results:
        results[k] = prepare_results(results[k], case_df)

    
    # for value in VALUE_MAP:
    #     plot_values_grid(
    #         results,
    #         stats_sets, 
    #         case_df,
    #         value,
    #         plots_dir / f"{value}.png",
    #     )
    
    plot_two_values(
        results,
        stats_sets,
        case_df,
        plots_dir / "regret_vs_lol.png",
    )


if __name__ == "__main__":
    main()