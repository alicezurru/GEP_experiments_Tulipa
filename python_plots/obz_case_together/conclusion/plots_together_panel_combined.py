import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

METHODS = ["convex_hull"]

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
     "rel_regret": (-0.04, 1.0),
     "total_steps_loss_of_load": (-200, 2000),
     "total_lol": (-200, 5000),
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

# def plot_panel(ax, df, stats_dict, case_df, value, rp_to_pos, col_name):

#     df = df[df.base_name != "0_HourlyBenchmark"]

#     for method in METHODS:
#         df_m = df[df.method == method]

#         for stoch, g in df_m.groupby("stochastic_method"):
#             g = g.sort_values("rp")
#             key = f"{method}_{stoch}"
#             xpos = [rp_to_pos[rp] for rp in g.rp]

  
#             ax.plot(
#                 xpos,
#                 g[value],
#                 color=COLOR_MAP[key],
#                 lw=2,
#                 linestyle=linestyle,
#                 marker=MARKER_MAP[stoch],
#                 ms=6,
#                 markerfacecolor=COLOR_MAP[key] if is_ref else "white",
#                 markeredgecolor=COLOR_MAP[key],
#                 markeredgewidth=1.2 if not is_ref else 0.5,
#             )

                    
#     # ---- STANDARD K-MEDOIDS ----
#     df_stats = stats_dict["kmedoids"].merge(case_df, on="base_name")
#     df_stats = df_stats[df_stats.base_name != "0_HourlyBenchmark"]
#     df_stats = df_stats[df_stats.method == "k_medoids"]

#     for base_name, g in df_stats.groupby("base_name"):
#         g = g.sort_values("rp")

#         stochastic = g.stochastic_method.iloc[0]
#         stoch_clean = "per_scenario" if stochastic.endswith("per_scenario") else "cross_scenario"
#         marker = MARKER_MAP[stoch_clean]

#         xpos = [rp_to_pos[rp] for rp in g.rp]

#         mean = g[f"{value}_mean"]
#         q25 = g[f"{value}_q25"]
#         q75 = g[f"{value}_q75"]

#         color = "#6a0dad"  # purple


#         ax.plot(
#             xpos, mean,
#             color=color,
#             lw=2,
#             linestyle=linestyle,
#             marker=marker,
#             ms=6,
#             markerfacecolor=color if is_ref else "white",
#             markeredgecolor=color,
#             markeredgewidth=1.2 if not is_ref else 0.5,
#         )


#         ax.fill_between(xpos, q25, q75, color=color, alpha=0.25, linewidth=0)


    
#     # ---- DIRAC K-MEDOIDS (ONLY FIRST COLUMN) ----
#     if col_name == "REFERENCE METHODS" and "dirac" in stats_dict:

#         df_dir = stats_dict["dirac"].merge(case_df, on="base_name")
#         df_dir = df_dir[df_dir.base_name != "0_HourlyBenchmark"]
#         df_dir = df_dir[df_dir.method == "k_medoids"]

#         for base_name, g in df_dir.groupby("base_name"):
#             g = g.sort_values("rp")

#             stochastic = g.stochastic_method.iloc[0]
#             stoch_clean = "per_scenario" if stochastic.endswith("per_scenario") else "cross_scenario"
#             marker = MARKER_MAP[stoch_clean]

#             xpos = [rp_to_pos[rp] for rp in g.rp]

#             mean = g[f"{value}_mean"]
#             q25 = g[f"{value}_q25"]
#             q75 = g[f"{value}_q75"]

#             color = "#6a0dad"

#             # DOTTED LINE
           
#             ax.plot(
#                 xpos, mean,
#                 color=color,
#                 lw=2,
#                 linestyle=":",
#                 marker=marker,
#                 ms=6,
#                 markerfacecolor=color if is_ref else "white",
#                 markeredgecolor=color,
#                 markeredgewidth=1.2 if not is_ref else 0.5,
#             )


#             # ✅ RIBBON (same style as normal k-medoids)
#             ax.fill_between(
#                 xpos,
#                 q25,
#                 q75,
#                 color=color,
#                 alpha=0.15,   # slightly lighter so it doesn’t overpower
#                 linewidth=0,
#             )
#         # ---- BENCHMARK LINE (only for time_to_solve) ----
#     if value == "time_to_solve":
#         bm_y = 526.3

#         ax.axhline(
#             y=bm_y,
#             color="grey",
#             linestyle=":",
#             linewidth=2,
#             alpha=0.8,
#         )

#         # label "BM" slightly above the line, on the right side
#         xmin, xmax = ax.get_xlim()
#         ax.text(
#             xmax,
#             bm_y,
#             " BM",
#             color="grey",
#             fontsize=12,
#             va="bottom",
#             ha="right",
#         )
def plot_panel(ax, results_sets, stats_sets, case_df, value, rp_to_pos):

    # ---- DEFINE ALL LINES ----
    lines = [
        # k-medoids
        ("kmedoids_dirac", "stats"),
        ("kmedoids_hcwc", "stats"),
        ("kmedoids_apgs0", "stats"),
        ("kmedoids_apgs001", "stats"),

        # convex hull
        ("hull", "results"),
        ("hull_hcwc", "results"),
        ("hull_apgs0", "results"),
        ("hull_apgs001", "results"),
    ]

    for name, dtype in lines:

        color, linestyle, mfc, mew = get_style(name)
                
        is_apgs = "apgs" in name

        lw = 3.2 if is_apgs else 1.6
        alpha = 1.0 if is_apgs else 0.45
        marker = "o"


        # ---- LOAD DATA ----
        if dtype == "results":
            df = results_sets[name]
            df = df[df.base_name != "0_HourlyBenchmark"]

            for stoch, g in df.groupby("stochastic_method"):
                g = g.sort_values("rp")

                xpos = [rp_to_pos[rp] for rp in g.rp]

                ax.plot(
                    xpos,
                    g[value],
                    color=color,
                    lw=lw,
                    linestyle=linestyle,
                    alpha=alpha,
                    marker=marker,
                    ms=6,
                    markerfacecolor=mfc,
                    markeredgecolor=color,
                    markeredgewidth=mew,
                    zorder=3 if is_apgs else 1,
                )

        else:
            df = stats_sets[name].merge(case_df, on="base_name")
            df = df[df.base_name != "0_HourlyBenchmark"]

            for base_name, g in df.groupby("base_name"):
                g = g.sort_values("rp")

                xpos = [rp_to_pos[rp] for rp in g.rp]

                ax.plot(
                    xpos,
                    g[f"{value}_mean"],
                    color=color,
                    lw=lw,
                    linestyle=linestyle,
                    alpha=alpha,
                    marker=marker,
                    ms=6,
                    markerfacecolor=mfc,
                    markeredgecolor=color,
                    markeredgewidth=mew,
                    zorder=3 if is_apgs else 1,
                )

                ax.fill_between(
                    xpos,
                    g[f"{value}_q25"],
                    g[f"{value}_q75"],
                    color=color,
                    alpha=0.12,
    
                )
    if value == "total_time":
        bm_y = 671.7

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
def plot_values_grid(results_sets, stats_sets, case_df, savepath):

    values = ["rel_regret", "total_lol", "total_time"]

    sample_df = next(iter(results_sets.values()))
    rp_vals = sorted(v for v in sample_df.rp.unique() if v != 1)

    rp_pos = list(range(len(rp_vals)))
    rp_to_pos = dict(zip(rp_vals, rp_pos))

    # 🔴 CHANGE: 3 subplots (vertical)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4),
        sharex=True
    )

    for ax, value in zip(axes, values):

        plot_panel(ax, results_sets, stats_sets, case_df, value, rp_to_pos)

        ax.set_axisbelow(True)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=14)

        ax.set_title(VALUE_MAP[value], fontsize=17, pad=10)

        if value in YLIM_MAP:
            ymin, ymax = YLIM_MAP[value]
            ax.set_ylim(ymin, ymax)
        if value == "total_time":
            bm_y = 671.7

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

   
    for ax in axes:
        ax.set_xticks(rp_pos)
        ax.set_xticklabels(
            [str(rp) for rp in rp_vals],
            rotation=45,
            ha="right",
        )


    fig.supxlabel("Number of representative periods", fontsize=17, y=-0.1)

    # legend (UNCHANGED)
    method_handles = [
        Line2D([0], [0], color="#6a0dad", lw=3, label="K-medoids"),
        Line2D([0], [0], color="#9e9e9e", lw=3, label="Convex hull"),
    ]

    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o",
               markerfacecolor="black", label="No Artificial"),

        Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o",
               markerfacecolor="white", label="HCWC"),

        Line2D([0], [0], color="black", lw=2, linestyle=(0, (4, 2)),
               marker="o", markerfacecolor="black", label="APGS α=0"),

        Line2D([0], [0], color="black", lw=2, linestyle=(0, (2, 1)),
               marker="o", markerfacecolor="black", label="APGS α=0.01"),
    ]
    
    handles = [
        Line2D([], [], linestyle="none", label="Clustering methods"),
        *method_handles,

        Line2D([], [], linestyle="none", label=""),


        Line2D([], [], linestyle="none", label="Configurations"),
        *style_handles,
    ]


    fig.legend(
        handles=handles,
        title_fontsize=18,
        fontsize=16,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        frameon=True,
    )

    plt.subplots_adjust(
        left=0.12,
        right=0.82,
        top=0.95,
        bottom=0.08,
        hspace=0.25,
        wspace=0.25
    )

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def get_style(name):
    # defaults
    color = None
    linestyle = "-"
    mfc = None
    mew = 0.5

    # --- K-MEDOIDS ---
    if "kmedoids" in name:
        color = "#6a0dad"

        if "dirac" in name:
            mfc = color
            linestyle = "-"

        elif "hcwc" in name:
            mfc = "white"
            mew = 1.4

        elif "apgs001" in name:
            mfc = color
            linestyle = (0, (2, 1))

        elif "apgs0" in name:
            mfc = color
            linestyle = (0, (8, 3))

    # --- CONVEX HULL ---
    elif "hull" in name:
        color = "#9e9e9e"

        if "hcwc" in name:
            mfc = "white"
            mew = 1.4

        elif "apgs001" in name:
            mfc = color
            linestyle = (0, (2, 1))

        elif "apgs0" in name:
            mfc = color
            linestyle = (0, (8, 3))

        else:
            mfc = color

    return color, linestyle, mfc, mew

# ============================================================
# Main
# ============================================================

def main():
    
    
    stats_sets = {
        "kmedoids_dirac": pd.read_csv(SCRIPT_DIR / "stats_dir.csv"),
        "kmedoids_hcwc": pd.read_csv(SCRIPT_DIR / "stats_conv.csv"),
        "kmedoids_apgs0": pd.read_csv(SCRIPT_DIR / "stats_apgs0.csv"),
        "kmedoids_apgs001": pd.read_csv(SCRIPT_DIR / "stats_apgs0.01.csv"),
    }

    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    case_df = pd.read_csv(SCRIPT_DIR / "case-studies-info.csv")

    
    results_sets = {
        "hull": pd.read_csv(SCRIPT_DIR / "results_hull.csv"),
        "hull_hcwc": pd.read_csv(SCRIPT_DIR / "results_hull_h.csv"),
        "hull_apgs0": pd.read_csv(SCRIPT_DIR / "results_apgs0.csv"),
        "hull_apgs001": pd.read_csv(SCRIPT_DIR / "results_apgs0.01.csv"),
    }


    for k in results_sets:
        results_sets[k] = prepare_results(results_sets[k], case_df)

        

    plot_values_grid(
        results_sets,
        stats_sets, 
        case_df,
        plots_dir / "combined_plot.png",
    )

if __name__ == "__main__":
    main()