import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

# Columns (beta values)
BETAS = [0, 0.2, 0.5, 1]
BETA_LABELS = [r"$\beta=0$", r"$\beta=0.2$", r"$\beta=0.5$", r"$\beta=1$"]

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
    "convex_hull_cross_scenario": "Convex hull",
    "convex_hull_with_null_cross_scenario": "Bounded conical hull",
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
    "rel_regret": (0.0, 0.4),
    "total_steps_loss_of_load": (-100.0, 700),
    "total_lol": (-100.0, 2000),
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

def plot_panel(ax, df, value, rp_to_pos):

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

# ============================================================
# Grid plotting (3 × 4)
# ============================================================

def plot_values_grid(results_sets, value, savepath):
    rp_vals = sorted(
        v for v in results_sets[0][BETAS[0]].rp.unique() if v != 1
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
        df = results_sets[0][beta]


        plot_panel(ax, df, value, rp_to_pos)

                
        ax.set_title(BETA_LABELS[col_idx], fontsize=20)

        ax.set_xticks(rp_pos)
        ax.set_xticklabels(
            [str(rp) for rp in rp_vals],
            rotation=45,
            ha="right",
        )


        ax.set_axisbelow(True)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=14)

    fig.supxlabel("Number of representative periods", fontsize=17, y = -0.1)
    fig.supylabel(VALUE_MAP[value], fontsize=17, x=0.03)

    if value in YLIM_MAP:
        ymin, ymax = YLIM_MAP[value]
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    handles = [
        Line2D([0], [0], color="#9e9e9e", marker="o", lw=2, label="Convex hull"),
        Line2D([0], [0], color="#98df8a", marker="o", lw=2, label="Bounded conical hull"),
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
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():

    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

    results_sets = []

    beta_dict = {}

    for beta in BETAS:
        beta_str = str(beta)
        df = pd.read_csv(
            SCRIPT_DIR / f"results_b{beta_str}.csv"
        )
        beta_dict[beta] = prepare_results(df, case_df)

    results_sets.append(beta_dict)

    for value in VALUE_MAP:
        plot_values_grid(
            results_sets,
            value,
            plots_dir / f"{value}_three_rows.png",
        )

if __name__ == "__main__":
    main()