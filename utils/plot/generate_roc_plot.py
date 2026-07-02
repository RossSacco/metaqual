import os
import argparse
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from metaqual.utils.plot.scorer_config import (
    get_enabled_scorers,
    get_label,
    get_color,
    get_linestyle,
    prepare_scores_for_roc,
)


def plot_roc(
    input_file,
    output_path,
    x_min=0.80,
    x_max=1.00,
    y_min=0.80,
    y_max=1.00,
):
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    labels = data["labels"]

    models_to_plot = get_enabled_scorers()

    print("Scorers presenti nel file:")
    for scorer in data["scorers"].keys():
        print(f"- {scorer}")

    print("\nScorers abilitati nel config e da plottare:")
    for scorer in models_to_plot:
        print(f"- {scorer}")

    fig, ax = plt.subplots(figsize=(8, 8))
    axins = ax.inset_axes([0.05, 0.70, 0.25, 0.25])

    plotted = 0

    for name in models_to_plot:
        if name not in data["scorers"]:
            print(f"[WARNING] Scorer enabled=True ma non trovato nel pickle: {name}")
            continue

        raw_scores = data["scorers"][name]
        scores = prepare_scores_for_roc(name, raw_scores)

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        label_text = f"{get_label(name)} ({roc_auc:.2f})"

        ax.plot(
            fpr,
            tpr,
            label=label_text,
            color=get_color(name),
            linestyle=get_linestyle(name),
            linewidth=2.5,
        )

        axins.plot(
            fpr,
            tpr,
            color=get_color(name),
            linestyle=get_linestyle(name),
            linewidth=2.0,
        )

        plotted += 1

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
        label="Random (0.50)",
    )

    axins.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
    )

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    ax.set_xlabel("False Positive Rate\n(higher -> more non-relevant passages kept)")
    ax.set_ylabel("True Positive Rate\n(higher -> fewer relevant passages pruned)")
    ax.set_title("ROC Curves for Passage Quality Estimators")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    axins.set_xlim([0.0, 1.0])
    axins.set_ylim([0.0, 1.0])
    axins.set_xticks([0, 1])
    axins.set_yticks([0, 1])
    axins.set_title("detail", fontsize=10, loc="left", pad=-12)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    if plotted == 0:
        raise ValueError(
            "Nessuno scorer è stato plottato. "
            "Controlla enabled=True in scorer_config.py e il contenuto di roc_data.pkl."
        )

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")

    print(f"\nGrafico salvato in: {output_path}")
    print(f"Grafico salvato in: {output_path.replace('.png', '.pdf')}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        default="roc_data.pkl",
    )

    parser.add_argument(
        "--output_path",
        default="metaqual/utils/plot/roc_modular.png",
    )

    parser.add_argument("--x_min", type=float, default=0.80)
    parser.add_argument("--x_max", type=float, default=1.00)
    parser.add_argument("--y_min", type=float, default=0.80)
    parser.add_argument("--y_max", type=float, default=1.00)

    args = parser.parse_args()

    plot_roc(
        input_file=args.input_file,
        output_path=args.output_path,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()