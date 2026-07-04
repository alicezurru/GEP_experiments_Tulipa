import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def plot_cost_breakdown_on_ax(ax, df, title):
    df = df.sort_values("rp")

    labels = df["rp"].astype(str).tolist()
    labels[0] = "BM"

    # Stack data (same as before)
    data = [
        df["investment_cost_storage"],
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

    colors = [
        "#00E5FF",
        "#81C784",
        "#9E9E9E",
        "#1E88E5",
        "#FB8C00",
        "#E53935",
    ]

    bottom = None
    for i in range(len(data)):
        if bottom is None:
            bars = ax.bar(labels, data[i], color=colors[i])
            bottom = data[i]
        else:
            bars = ax.bar(labels, data[i], bottom=bottom, color=colors[i])
            bottom = bottom + data[i]

    ax.set_title(title)
    ax.set_xlabel("Number of Representative Periods")
    ax.tick_params(axis="x", rotation=30)

    return labels_stack, colors


def plot_two_cost_breakdowns(df1, df2, save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    labels_stack, colors = plot_cost_breakdown_on_ax(ax1, df1, "DAWC")
    plot_cost_breakdown_on_ax(ax2, df2, "DAWC APGS")
    ax1.set_ylim(0, 7.5e8)

    # Common Y label and global title
    ax1.set_ylabel("Cost (kEUR)")
    fig.suptitle("Cost Breakdown", fontsize=16)

    # Single legend (shared)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in colors
    ]
    fig.legend(
        handles,
        labels_stack,
        loc="center left",
        bbox_to_anchor=(0.78, 0.5)
    )

    plt.tight_layout(rect=[0, 0, 0.78, 0.95])  # leave space for legend & title
    plt.savefig(save_path, dpi=300)
    plt.close()



def main():
    plots_dir = SCRIPT_DIR / "plots"

    # ✅ create folder if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_d = pd.read_csv(SCRIPT_DIR / "results_d.csv")
    df_dapgs = pd.read_csv(SCRIPT_DIR / "results_dapgs.csv")

    plot_two_cost_breakdowns(
        df_d,
        df_dapgs,
        plots_dir / "cost_breakdown_comparison.png"
    )


if __name__ == "__main__":
    main()