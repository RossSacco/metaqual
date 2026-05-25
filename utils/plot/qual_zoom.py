import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_qualt5_detail(input_file="roc_data.pkl"):
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    labels = data["labels"]

    qualt5_models = [
        "finetuned_qualt5",
        "metadata_qualt5_concat",
        "metadata_qualt5_MP",
        "metadata_qualt5_msmarco_passage_NUOVO",
    ]

    colors = {
        "metadata_qualt5_MP": "#ff7f0e",
        "metadata_qualt5_concat": "#000000",
        "metadata_qualt5_msmarco_passage_NUOVO": "#9467bd",
        "finetuned_qualt5": "#1f77b4",
    }

    linestyles = {
        "finetuned_qualt5": "-",
        "metadata_qualt5_concat": "--",
        "metadata_qualt5_MP": "-.",
        "metadata_qualt5_msmarco_passage_NUOVO": ":",
    }

    markers = {
        "finetuned_qualt5": "o",
        "metadata_qualt5_concat": "s",
        "metadata_qualt5_MP": "^",
        "metadata_qualt5_msmarco_passage_NUOVO": "D",
    }

    legend_names = {
        "finetuned_qualt5": "QualT5-Finetuned",
        "metadata_qualt5_msmarco_passage_NUOVO": "Metadata-QualT5-pooled",
        "metadata_qualt5_concat": "Metadata-QualT5-CONCAT",
        "metadata_qualt5_MP": "Metadata-QualT5-MP",
    }

    print("Scorers presenti nel file:")
    print(list(data["scorers"].keys()))

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in qualt5_models:
        if name not in data["scorers"]:
            print(f"[WARNING] Scorer non trovato nel file: {name}")
            continue

        scores = data["scorers"][name]

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        display_name = legend_names.get(name, name)
        label_text = f"{display_name} ({roc_auc:.2f})"

        ax.plot(
            fpr,
            tpr,
            label=label_text,
            color=colors.get(name, "black"),
            linestyle=linestyles.get(name, "-"),
            linewidth=2.8,
            marker=markers.get(name, None),
            markevery=80,
            markersize=5,
            alpha=0.95,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label="Random (0.50)",
    )

    # Zoom sulla zona in cui le curve sono molto vicine.
    ax.set_xlim([0.80, 1.00])
    ax.set_ylim([0.935, 1.00])

    ax.set_xlabel("False Positive Rate\n(higher -> more non-relevant passages kept)")
    ax.set_ylabel("True Positive Rate\n(higher -> fewer relevant passages pruned)")
    ax.set_title("ROC Detail for QualT5-based Passage Quality Estimators")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    output_path = "metaqual/utils/plot/figure2_qualt5_roc_detail.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")

    print(f"Grafico salvato in: {output_path}")
    print(f"Grafico salvato in: {output_path.replace('.png', '.pdf')}")


if __name__ == "__main__":
    plot_roc_qualt5_detail()