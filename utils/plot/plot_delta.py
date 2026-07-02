import os
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from metaqual.utils.plot.scorer_config import (
    BASE_MODEL,
    get_enabled_scorers,
    get_label,
    get_color,
    get_linestyle,
    prepare_scores_for_roc,
)


def interpolate_tpr(fpr, tpr, grid):
    """
    Interpola la TPR su una griglia comune di FPR.
    Gestisce FPR duplicati prendendo la TPR massima per ogni FPR.
    """

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    unique_fpr = np.unique(fpr)

    unique_tpr = np.array([
        tpr[fpr == value].max()
        for value in unique_fpr
    ])

    return np.interp(grid, unique_fpr, unique_tpr)


def plot_delta_tpr(
    input_file,
    output_path,
    base_model=BASE_MODEL,
    fpr_min=0.80,
    fpr_max=1.00,
):
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    labels = data["labels"]

    if base_model not in data["scorers"]:
        raise ValueError(f"Base model non trovato nel pickle: {base_model}")

    enabled_scorers = get_enabled_scorers()

    compare_models = [
        name
        for name in enabled_scorers
        if name != base_model
    ]

    print("Base model:")
    print(f"- {base_model}")

    print("\nScorers abilitati nel config:")
    for scorer in enabled_scorers:
        print(f"- {scorer}")

    print("\nModelli da confrontare:")
    for scorer in compare_models:
        print(f"- {scorer}")

    base_raw_scores = data["scorers"][base_model]
    base_scores = prepare_scores_for_roc(base_model, base_raw_scores)

    base_fpr, base_tpr, _ = roc_curve(labels, base_scores)
    base_auc = auc(base_fpr, base_tpr)

    fpr_grid = np.linspace(fpr_min, fpr_max, 1000)
    base_tpr_interp = interpolate_tpr(base_fpr, base_tpr, fpr_grid)

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    all_deltas = []

    for name in compare_models:
        if name not in data["scorers"]:
            print(f"[WARNING] Scorer enabled=True ma non trovato nel pickle: {name}")
            continue

        raw_scores = data["scorers"][name]
        scores = prepare_scores_for_roc(name, raw_scores)

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        tpr_interp = interpolate_tpr(fpr, tpr, fpr_grid)
        delta_tpr = tpr_interp - base_tpr_interp

        all_deltas.append(delta_tpr)

        label_text = f"{get_label(name)} ({roc_auc:.2f})"

        ax.plot(
            fpr_grid,
            delta_tpr,
            label=label_text,
            color=get_color(name),
            linestyle=get_linestyle(name),
            linewidth=2.8,
        )

        plotted += 1

    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=1.2,
        alpha=0.8,
        label=f"{get_label(base_model)} baseline ({base_auc:.2f})",
    )

    ax.set_xlim([fpr_min, fpr_max])

    if all_deltas:
        stacked = np.concatenate(all_deltas)
        max_abs_delta = max(abs(stacked.min()), abs(stacked.max()), 0.005)
    else:
        max_abs_delta = 0.005

    ax.set_ylim([-max_abs_delta * 1.15, max_abs_delta * 1.15])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel(f"Δ TPR vs {get_label(base_model)}")
    ax.set_title(f"TPR Difference with respect to {get_label(base_model)}")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)

    if plotted == 0:
        raise ValueError(
            "Nessuno scorer di confronto è stato plottato. "
            "Devi avere almeno due scorer enabled=True: il base model e un modello di confronto."
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
        default="metaqual/utils/plot/delta_tpr_modular.png",
    )

    parser.add_argument(
        "--base_model",
        default=BASE_MODEL,
        help="Scorer baseline rispetto a cui calcolare il delta.",
    )

    parser.add_argument("--fpr_min", type=float, default=0.80)
    parser.add_argument("--fpr_max", type=float, default=1.00)

    args = parser.parse_args()

    plot_delta_tpr(
        input_file=args.input_file,
        output_path=args.output_path,
        base_model=args.base_model,
        fpr_min=args.fpr_min,
        fpr_max=args.fpr_max,
    )


if __name__ == "__main__":
    main()