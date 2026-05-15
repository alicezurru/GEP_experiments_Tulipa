import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# ============================================================
# Constants (close to k-means / k-medoids version)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

COLS = ["DISTANT", "HALFMIXED", "CLOSE", "MIXED"]
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
    "num_loss_of_load_e_demand": "Number of timesteps with loss of load",
    "time_to_cluster": "Time to cluster (s)",
    "time_to_create": "Time to create (s)",
    "time_to_solve": "Time to solve (s)",
    "total_time": "Total time (s)",
}

YLIM_MAP = {
    #"rel_regret": (0.0, 0.5),
    "num_loss_of_load_e_demand": (-18.0, 150),
}


# ============================================================
# Data preparation (Julia-equivalent preprocessing)
# ============================================================

def prepare_results(results_df, case_df):
    # merge with case studies
    results_df = results_df.merge(case_df, on="base_name", how="left")

    # hourly benchmark
    hourly = results_df[results_df.base_name == "0_HourlyBenchmark"].iloc[0]

    # relative regret
    results_df["rel_regret"] = [
        0.0 if r.base_name == "0_HourlyBenchmark"
        else (r.objective_value_resolve_benchmark - hourly.objective_value)
             / hourly.objective_value
        for r in results_df.itertuples()
    ]

    # total time
    results_df["total_time"] = (
        results_df.time_to_cluster
        + results_df.time_to_read
        + results_df.time_to_create
        + results_df.time_to_solve
        + results_df.time_to_save
    )

    # normalize loss of load
    results_df["num_loss_of_load_e_demand"] = (
        results_df.num_loss_of_load_e_demand
        - hourly.num_loss_of_load_e_demand
    )

    return results_df

# ============================================================
# Panel plotting (no ribbons, deterministic)
# ============================================================

def plot_panel(ax, df, value, method, rp_to_pos):

    df = df[df.method == method]
    df = df[df.base_name != "0_HourlyBenchmark"]

    for stoch, g in df.groupby("stochastic_method"):
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
            label=LEGEND_MAP[key],
        )

# ============================================================
# Grid plotting (very close to k-means version)
# ============================================================

def plot_values_grid(results_dict, value, savepath):

    rp_vals = sorted(v for v in results_dict[COLS[0]].rp.unique() if v != 1)
    rp_pos = list(range(len(rp_vals)))
    rp_to_pos = dict(zip(rp_vals, rp_pos))

    fig, axes = plt.subplots(
        2, 4,
        figsize=(17, 9),
        sharex=True,
        sharey=True,
    )

    for col_idx, col in enumerate(COLS):
        df = results_dict[col]

        for row_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]

            plot_panel(ax, df, value, method, rp_to_pos)

            if row_idx == 0:
                ax.set_title(col, fontsize=16)

            if col_idx != 0:
                ax.set_ylabel("")

            if row_idx == 1:
                ax.set_xticks(rp_pos)
                ax.set_xticklabels(
                    [str(rp) for rp in rp_vals],
                    rotation=45,
                    ha="right",
                )

            ax.set_axisbelow(True)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=11)

    fig.supxlabel("Number of representative periods", fontsize=15)
    fig.supylabel(VALUE_MAP[value], fontsize=15)
    
    if value in YLIM_MAP:
        ymin, ymax = YLIM_MAP[value]
        for ax in axes.flat:
            ax.set_ylim(ymin, ymax)


    # Legend (manual, like before)
    handles = [
        Line2D([0], [0], color="#4d4d4d", marker="^", lw=2, label="Convex hull: per-scenario"),
        Line2D([0], [0], color="#9e9e9e", marker="o", lw=2, label="Convex hull: cross-scenario"),
        Line2D([0], [0], color="#2ca02c", marker="^", lw=2, label="Bounded conical hull: per-scenario"),
        Line2D([0], [0], color="#98df8a", marker="o", lw=2, label="Bounded conical hull: cross-scenario"),
    ]

    fig.legend(
        handles=handles,
        title="Clustering methods",
        title_fontsize=17,
        fontsize=15,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
    )

    plt.subplots_adjust(
        left=0.06,
        right=0.86,
        top=0.90,
        bottom=0.10,
        wspace=0.05,
        hspace=0.05,
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():

    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    results = {
        "DISTANT": pd.read_csv(SCRIPT_DIR / "results_dist.csv"),
        "HALFMIXED": pd.read_csv(SCRIPT_DIR / "results_hmix.csv"),
        "CLOSE": pd.read_csv(SCRIPT_DIR / "results_cl.csv"),
        "MIXED": pd.read_csv(SCRIPT_DIR / "results_mix.csv"),
    }

    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

    for k in results:
        results[k] = prepare_results(results[k], case_df)

    for value in VALUE_MAP:
        plot_values_grid(
            results,
            value,
            plots_dir / f"{value}_deterministic.png",
        )

if __name__ == "__main__":
    main()
