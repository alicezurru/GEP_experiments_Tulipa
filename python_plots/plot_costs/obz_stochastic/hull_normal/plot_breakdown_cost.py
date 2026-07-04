import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def plot_cost_breakdown_pretty(df, save_path):

    # Sort by rp for a clean x-axis
    df = df.sort_values("rp")

    # X-axis labels: rp only
    labels = df["rp"].astype(str).tolist()
    labels[0] = "BM"

    # Recreate cumulative stacking exactly like your Julia logic
    investment_storage = df["investment_cost_storage"]

    investment_renewable = (
        df["investment_cost_renewable"] + investment_storage
    )

    investment_non_renewable = (
        df["investment_cost_non_renewable"] + investment_renewable
    )

    investment_electrolyzer = (
        df["investment_cost_electrolyzer"] + investment_non_renewable
    )

    operational = (
        df["operational_cost"]
        + investment_electrolyzer
        - df["penalty_tot_loss_of_load"]
    )

    penalty = (
        df["operational_cost"] + investment_electrolyzer
    )

    # Stack components (bottom → top)
    data = [
        investment_storage,
        df["investment_cost_renewable"],
        df["investment_cost_non_renewable"],
        df["investment_cost_electrolyzer"],
        df["operational_cost"],
        df["penalty_tot_loss_of_load"],
    ]

    labels_stack = [
        "Storage Investment",
        "Renewable Investment",
        "Non-Renewable Investment",
        "Electrolyzer Investment",
        "Operational Cost",
        "Penalty Loss",
    ]

    # More meaningful colors
    colors = [
        "#00E5FF",  # green → storage
        "#81C784",  # light green → renewable
        "#9E9E9E",  # gray → non-renewable
        "#1E88E5",  # blue → electrolyzer
        "#FB8C00",  # orange → operational
        "#E53935",  # red → penalty
    ]
    plt.rcParams.update({
    "font.size": 12,        # base font size
    "axes.titlesize": 16,   # title
    "axes.labelsize": 14,   # axis labels
    "xtick.labelsize": 12,  # x-axis ticks
    "ytick.labelsize": 12,  # y-axis ticks
    "legend.fontsize": 12   # legend
    })


    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = None
    for i in range(len(data)):
        if bottom is None:
            ax.bar(labels, data[i], label=labels_stack[i], color=colors[i])
            bottom = data[i]
        else:
            ax.bar(labels, data[i], bottom=bottom, label=labels_stack[i], color=colors[i])
            bottom = bottom + data[i]

    # Formatting
    ax.set_xlabel("Number of Representative Periods")
    ax.set_ylabel("Cost (kEUR)")
    ax.set_title("Cost Breakdown")
    ax.set_ylim(0, 5.5e8)

    # Rotate x labels slightly
    plt.xticks(rotation=30)

    # Move legend outside so it doesn't block data
    # ax.legend(
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1),
    #     borderaxespad=0
    # )
    ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)



    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():

    plots_dir = SCRIPT_DIR / "plots_per"

    results =  pd.read_csv(SCRIPT_DIR / "results_per.csv")

    plot_cost_breakdown_pretty(results,plots_dir)

if __name__ == "__main__":
    main()