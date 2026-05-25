import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def interpolate_tpr(fpr, tpr, grid):
    """
    Interpola la TPR su una griglia comune di FPR.
    Serve perché le ROC possono avere punti diversi.
    """
    order = np.argsort(fpr)
    fpr = np.asarray(fpr)[order]
    tpr = np.asarray(tpr)[order]

    return np.interp(grid, fpr, tpr)


def plot_delta_tpr_vs_qualt5(input_file="roc_data.pkl"):
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    labels = data["labels"]

    base_model = "finetuned_qualt5"

    compare_models = [
        "metadata_qualt5_concat",
        "metadata_qualt5_MP",
        "metadata_qualt5_msmarco_passage_NUOVO",
    ]

    colors = {
        "metadata_qualt5_MP": "#ff7f0e",
        "metadata_qualt5_concat": "#000000",
        "metadata_qualt5_msmarco_passage_NUOVO": "#9467bd",
    }

    linestyles = {
        "metadata_qualt5_concat": "--",
        "metadata_qualt5_MP": "-.",
        "metadata_qualt5_msmarco_passage_NUOVO": ":",
    }

    legend_names = {
        "metadata_qualt5_msmarco_passage_NUOVO": "Metadata-QualT5-pooled",
        "metadata_qualt5_concat": "Metadata-QualT5-CONCAT",
        "metadata_qualt5_MP": "Metadata-QualT5-MP",
    }

    print("Scorers presenti nel file:")
    print(list(data["scorers"].keys()))

    if base_model not in data["scorers"]:
        raise ValueError(f"Base model non trovato nel file: {base_model}")

    # Calcolo ROC del modello base.
    base_scores = data["scorers"][base_model]
    base_fpr, base_tpr, _ = roc_curve(labels, base_scores)
    base_auc = auc(base_fpr, base_tpr)

    # Griglia comune nella zona di interesse.
    fpr_grid = np.linspace(0.80, 1.00, 1000)
    base_tpr_interp = interpolate_tpr(base_fpr, base_tpr, fpr_grid)

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in compare_models:
        if name not in data["scorers"]:
            print(f"[WARNING] Scorer non trovato nel file: {name}")
            continue

        scores = data["scorers"][name]

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        tpr_interp = interpolate_tpr(fpr, tpr, fpr_grid)
        delta_tpr = tpr_interp - base_tpr_interp

        display_name = legend_names.get(name, name)
        label_text = f"{display_name} ({roc_auc:.2f})"

        ax.plot(
            fpr_grid,
            delta_tpr,
            label=label_text,
            color=colors.get(name, "black"),
            linestyle=linestyles.get(name, "-"),
            linewidth=2.8,
        )

    # Linea zero: stesso comportamento del QualT5-Finetuned.
    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=1.2,
        alpha=0.8,
        label=f"QualT5-Finetuned baseline ({base_auc:.2f})",
    )

    ax.set_xlim([0.80, 1.00])

    # Se le differenze sono più grandi o più piccole, modifica questi valori.
    ax.set_ylim([-0.015, 0.015])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Δ TPR vs QualT5-Finetuned")
    ax.set_title("TPR Difference with respect to QualT5-Finetuned")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)

    output_path = "metaqual/utils/plot/figure3_delta_tpr_vs_qualt5.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")

    print(f"Grafico salvato in: {output_path}")
    print(f"Grafico salvato in: {output_path.replace('.png', '.pdf')}")


if __name__ == "__main__":
    plot_delta_tpr_vs_qualt5()
    