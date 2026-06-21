import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

COLS = ["REFERENCE METHODS", "APGS"]

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
    "loss_of_load_e_demand": "Total electricity loss of load (GWh)",
    "loss_of_load_h2_demand": "Total hydrogen loss of load (GWh)",
    "total_steps_loss_of_load" : "Total number of timesteps with loss of load",
    "total_lol" : "Total loss of load (GWh)",
    "time_to_cluster": "Time to cluster (s)",
    "time_to_create": "Time to create (s)",
    "time_to_solve": "Time to solve (s)",
    "total_time": "Total time (s)",
}

YLIM_MAP = {
     "rel_regret": (-0.1, 1.5),
    "total_lol": (-1000, 40000),
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

    results_df["speedup"] = 671.7 / results_df["total_time"] 


    return results_df

# ============================================================
# Panel plotting (both methods inside)
# ============================================================

def plot_panel(ax, df, stats_dict, case_df, value, rp_to_pos, col_name):

    df = df[df.base_name != "0_HourlyBenchmark"]

    for method in METHODS:
        df_m = df[df.method == method]

        for stoch, g in df_m.groupby("stochastic_method"):
            g = g.sort_values("rp")
            key = f"{method}_{stoch}"
            xpos = g["speedup"]

            ax.plot(
                xpos,
                g[value],
                color=COLOR_MAP[key],
                linestyle="None",
                marker=MARKER_MAP[stoch],
                ms=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        
    # ---- STANDARD K-MEDOIDS ----
    df_stats = stats_dict["kmedoids"].merge(case_df, on="base_name")
    df_stats = df_stats[df_stats.base_name != "0_HourlyBenchmark"]
    df_stats = df_stats[df_stats.method == "k_medoids"]
    df_stats["speedup_mean"] = 671.7 / df_stats["total_time_mean"]

    for base_name, g in df_stats.groupby("base_name"):
        g = g.sort_values("rp")

        stochastic = g.stochastic_method.iloc[0]
        stoch_clean = "per_scenario" if stochastic.endswith("per_scenario") else "cross_scenario"
        marker = MARKER_MAP[stoch_clean]

        xpos = g["speedup_mean"]

        mean = g[f"{value}_mean"]
        q25 = g[f"{value}_q25"]
        q75 = g[f"{value}_q75"]

        color = "#6a0dad"  # purple

        ax.plot(
            xpos, mean,
            color=color,
            linestyle="None",
            marker=marker,
            ms=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
    
    # ---- DIRAC K-MEDOIDS (ONLY FIRST COLUMN) ----
    if col_name == "REFERENCE METHODS" and "dirac" in stats_dict:

        df_dir = stats_dict["dirac"].merge(case_df, on="base_name")
        df_dir = df_dir[df_dir.base_name != "0_HourlyBenchmark"]
        df_dir = df_dir[df_dir.method == "k_medoids"]

        for base_name, g in df_dir.groupby("base_name"):
            g = g.sort_values("rp")

            stochastic = g.stochastic_method.iloc[0]
            stoch_clean = "per_scenario" if stochastic.endswith("per_scenario") else "cross_scenario"
            marker = MARKER_MAP[stoch_clean]

            xpos = [rp_to_pos[rp] for rp in g.rp]

            mean = g[f"{value}_mean"]
            q25 = g[f"{value}_q25"]
            q75 = g[f"{value}_q75"]

            color = "#6a0dad"

            # DOTTED LINE
            ax.plot(
                xpos,
                mean,
                color=color,
                linestyle="None",
                marker=marker,
                ms=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

        # ---- BENCHMARK LINE (only for time_to_solve) ----
    if value == "time_to_solve":
        bm_y = 526.3

        ax.axhline(
            y=bm_y,
            color="grey",
            linestyle=":",
            linewidth=2,
            alpha=0.8,
        )

        # label "BM" slightly above the line, on the right side
        xmin, xmax = ax.get_xlim()
        ax.text(
            xmax,
            bm_y,
            " BM",
            color="grey",
            fontsize=12,
            va="bottom",
            ha="right",
        )




# ============================================================
# Grid plotting (3 × 4)
# ============================================================

def plot_values_grid(results_sets, stats_sets, case_df, value, savepath):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharex=True,
        sharey=True,
    )

    for col_idx, col in enumerate(COLS):
        ax = axes[col_idx]
        df = results_sets[col]
        stats_dict = stats_sets[col]

        plot_panel(ax, df, stats_dict, case_df, value, None,col)

        ax.set_title(col, fontsize=20)

        ax.set_axisbelow(True)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=14)

    fig.supxlabel("Speedup (total time BM / total time reduced model)", fontsize=17, y=-0.05)
    fig.supylabel(VALUE_MAP[value], fontsize=17, x=0.01)

    if value in YLIM_MAP:
        ymin, ymax = YLIM_MAP[value]
        for ax in axes:
            ax.set_ylim(ymin, ymax)
    if value in YLIM_MAP:
        xmin, xmax = (0.0,50)
        for ax in axes:
            ax.set_xlim(xmin, xmax)
    
    handles = [
        Line2D([0], [0], color="#9e9e9e", marker="o", linestyle="None", markersize=8, label="Convex hull"),
        Line2D([0], [0], color="#98df8a", marker="o", linestyle="None", markersize=8, label="Bounded conical hull"),
        Line2D([0], [0], color="#6a0dad", marker="o", linestyle="None", markersize=8, label="K-medoids (Dirac weights)"),
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
        left=0.08,
        right=0.86,
        top=0.92,
        bottom=0.10,
        wspace=0.05,
        hspace=0.08,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    
    stats_sets = {
        "REFERENCE METHODS": {
            "kmedoids": pd.read_csv(SCRIPT_DIR / "stats_dir.csv"),
        },
        "APGS": {
            "kmedoids": pd.read_csv(SCRIPT_DIR / "stats_kapgs.csv"),
        },
    }

    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

        
    results_sets = {
        "REFERENCE METHODS": pd.read_csv(SCRIPT_DIR / "results_hull.csv"),
        "APGS": pd.read_csv(SCRIPT_DIR / "results_hapgs.csv"),
    }

    for k in results_sets:
        results_sets[k] = prepare_results(results_sets[k], case_df)

        
    for value in VALUE_MAP:
        plot_values_grid(
            results_sets,
            stats_sets, 
            case_df,
            value,
            plots_dir / f"{value}.png",
        )

if __name__ == "__main__":
    main()