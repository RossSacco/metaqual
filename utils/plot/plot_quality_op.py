import os
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from metaqual.utils.plot.scorer_config import (
    get_enabled_scorers,
    get_label,
    get_color,
    get_linestyle,
    prepare_scores_for_roc,
)


def ensure_output_dir(output_path):
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def save_figure(output_path):
    ensure_output_dir(output_path)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if output_path.endswith(".png"):
        pdf_path = output_path.replace(".png", ".pdf")
    else:
        pdf_path = output_path + ".pdf"

    plt.savefig(pdf_path, bbox_inches="tight")

    print(f"Grafico salvato in: {output_path}")
    print(f"Grafico salvato in: {pdf_path}")


def load_roc_data(input_file):
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    labels = np.asarray(data["labels"]).astype(int)

    if "scorers" not in data:
        raise ValueError("Il pickle deve contenere la chiave 'scorers'.")

    return labels, data["scorers"]


def get_models_to_plot(scorers_dict):
    models_to_plot = get_enabled_scorers()

    print("Scorers presenti nel file:")
    for scorer in scorers_dict.keys():
        print(f"- {scorer}")

    print("\nScorers abilitati nel config e da plottare:")
    for scorer in models_to_plot:
        print(f"- {scorer}")

    valid_models = []

    for name in models_to_plot:
        if name not in scorers_dict:
            print(f"[WARNING] Scorer enabled=True ma non trovato nel pickle: {name}")
            continue

        valid_models.append(name)

    if not valid_models:
        raise ValueError(
            "Nessuno scorer valido da plottare. "
            "Controlla enabled=True in scorer_config.py e il contenuto di roc_data.pkl."
        )

    return valid_models


def compute_roc_curves(labels, scorers_dict, models_to_plot):
    curves = {}

    for name in models_to_plot:
        raw_scores = scorers_dict[name]
        scores = prepare_scores_for_roc(name, raw_scores)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        curves[name] = {
            "scores": np.asarray(scores),
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
        }

        print(f"{name}: AUC={roc_auc:.6f}")

    return curves


def plot_roc_full(
    input_file,
    output_path,
):
    labels, scorers_dict = load_roc_data(input_file)
    models_to_plot = get_models_to_plot(scorers_dict)
    curves = compute_roc_curves(labels, scorers_dict, models_to_plot)

    fig, ax = plt.subplots(figsize=(8, 8))

    for name in models_to_plot:
        c = curves[name]

        label_text = f"{get_label(name)} AUC={c['auc']:.4f}"

        ax.plot(
            c["fpr"],
            c["tpr"],
            label=label_text,
            color=get_color(name),
            linestyle=get_linestyle(name),
            linewidth=2.5,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
        linewidth=2.0,
        label="Random AUC=0.5000",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel("False Positive Rate\n(higher -> more non-relevant passages kept)")
    ax.set_ylabel("True Positive Rate\n(higher -> fewer relevant passages pruned)")
    ax.set_title("ROC Curves for Passage Quality Estimators")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def plot_roc_operational(
    input_file,
    output_path,
    x_min=0.30,
    x_max=0.90,
    y_min=0.70,
    y_max=1.00,
    add_inset=True,
):
    labels, scorers_dict = load_roc_data(input_file)
    models_to_plot = get_models_to_plot(scorers_dict)
    curves = compute_roc_curves(labels, scorers_dict, models_to_plot)

    fig, ax = plt.subplots(figsize=(8, 8))

    if add_inset:
        axins = ax.inset_axes([0.05, 0.70, 0.25, 0.25])
    else:
        axins = None

    for name in models_to_plot:
        c = curves[name]

        label_text = f"{get_label(name)} AUC={c['auc']:.4f}"

        ax.plot(
            c["fpr"],
            c["tpr"],
            label=label_text,
            color=get_color(name),
            linestyle=get_linestyle(name),
            linewidth=2.5,
        )

        if axins is not None:
            axins.plot(
                c["fpr"],
                c["tpr"],
                color=get_color(name),
                linestyle=get_linestyle(name),
                linewidth=2.0,
            )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle=":",
        linewidth=2.0,
        label="Random AUC=0.5000",
    )

    if axins is not None:
        axins.plot(
            [0, 1],
            [0, 1],
            color="black",
            linestyle=":",
            linewidth=1.5,
        )

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    ax.set_xlabel("False Positive Rate\n(higher -> more non-relevant passages kept)")
    ax.set_ylabel("True Positive Rate\n(higher -> fewer relevant passages pruned)")
    ax.set_title("Operational ROC Region for Static Pruning")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    if axins is not None:
        axins.set_xlim([0.0, 1.0])
        axins.set_ylim([0.0, 1.0])
        axins.set_xticks([0, 1])
        axins.set_yticks([0, 1])
        axins.set_title("full", fontsize=10, loc="left", pad=-12)
        ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def plot_delta_tpr(
    input_file,
    output_path,
    baseline_scorer="finetuned_qualt5",
    x_min=0.30,
    x_max=0.90,
):
    labels, scorers_dict = load_roc_data(input_file)
    models_to_plot = get_models_to_plot(scorers_dict)
    curves = compute_roc_curves(labels, scorers_dict, models_to_plot)

    if baseline_scorer not in curves:
        raise ValueError(
            f"baseline_scorer={baseline_scorer} non presente tra le curve. "
            f"Disponibili: {list(curves.keys())}"
        )

    baseline_curve = curves[baseline_scorer]

    grid = np.linspace(0.0, 1.0, 2000)
    baseline_tpr_interp = np.interp(
        grid,
        baseline_curve["fpr"],
        baseline_curve["tpr"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0

    for name in models_to_plot:
        if name == baseline_scorer:
            continue

        c = curves[name]

        model_tpr_interp = np.interp(
            grid,
            c["fpr"],
            c["tpr"],
        )

        delta = model_tpr_interp - baseline_tpr_interp

        ax.plot(
            grid,
            delta,
            label=f"{get_label(name)} - {get_label(baseline_scorer)}",
            color=get_color(name),
            linestyle=get_linestyle(name),
            linewidth=2.5,
        )

        plotted += 1

    if plotted == 0:
        raise ValueError(
            "Nessun modello diverso dalla baseline da plottare nel delta TPR."
        )

    ax.axhline(
        0.0,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"{get_label(baseline_scorer)} baseline",
    )

    ax.set_xlim([x_min, x_max])

    ax.set_xlabel("False Positive Rate\n(higher -> more non-relevant passages kept)")
    ax.set_ylabel(f"Δ TPR vs {get_label(baseline_scorer)}")
    ax.set_title(f"TPR Difference with respect to {get_label(baseline_scorer)}")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def compute_pruning_stats(labels, scores, prune_fraction):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    total = len(labels)
    n_pruned = int(round(total * prune_fraction))

    order = np.argsort(scores)

    pruned_mask = np.zeros(total, dtype=bool)

    if n_pruned > 0:
        pruned_mask[order[:n_pruned]] = True

    kept_mask = ~pruned_mask

    positive_mask = labels == 1
    negative_mask = labels == 0

    total_positive = int(positive_mask.sum())
    total_negative = int(negative_mask.sum())

    positives_kept = int((kept_mask & positive_mask).sum())
    positives_pruned = int((pruned_mask & positive_mask).sum())

    negatives_kept = int((kept_mask & negative_mask).sum())
    negatives_pruned = int((pruned_mask & negative_mask).sum())

    if n_pruned > 0:
        threshold = float(scores[order[n_pruned - 1]])
    else:
        threshold = float("-inf")

    return {
        "threshold": threshold,
        "total": total,
        "n_pruned": n_pruned,
        "n_kept": int(kept_mask.sum()),
        "total_positive": total_positive,
        "total_negative": total_negative,
        "positives_kept": positives_kept,
        "positives_pruned": positives_pruned,
        "negatives_kept": negatives_kept,
        "negatives_pruned": negatives_pruned,
        "kept_positive_rate": positives_kept / total_positive if total_positive else np.nan,
        "removed_negative_rate": negatives_pruned / total_negative if total_negative else np.nan,
        "positive_loss": positives_pruned / total_positive if total_positive else np.nan,
        "kept_negative_rate": negatives_kept / total_negative if total_negative else np.nan,
    }


def plot_pruning_operational(
    input_file,
    output_path,
    csv_output_path=None,
    prune_fractions="0.15,0.25,0.30,0.45",
):
    labels, scorers_dict = load_roc_data(input_file)
    models_to_plot = get_models_to_plot(scorers_dict)

    fractions = [
        float(x.strip())
        for x in prune_fractions.split(",")
        if x.strip()
    ]

    if not fractions:
        raise ValueError("Nessuna prune fraction valida.")

    rows = []

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in models_to_plot:
        raw_scores = scorers_dict[name]
        scores = prepare_scores_for_roc(name, raw_scores)

        y = []

        for fraction in fractions:
            stats = compute_pruning_stats(labels, scores, fraction)

            row = {
                "prune_fraction": fraction,
                "model": name,
                "label": get_label(name),
                **stats,
            }

            rows.append(row)
            y.append(stats["kept_positive_rate"])

        ax.plot(
            fractions,
            y,
            label=get_label(name),
            color=get_color(name),
            linestyle=get_linestyle(name),
            marker="o",
            linewidth=2.5,
        )

    ax.set_xlabel("Pruned fraction")
    ax.set_ylabel("Relevant passages kept rate")
    ax.set_title("Operational Static Pruning Performance")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    save_figure(output_path)
    plt.close()

    pruning_df = pd.DataFrame(rows)

    if csv_output_path is None:
        if output_path.endswith(".png"):
            csv_output_path = output_path.replace(".png", ".csv")
        else:
            csv_output_path = output_path + ".csv"

    ensure_output_dir(csv_output_path)

    pruning_df.to_csv(csv_output_path, index=False)

    print(f"Metriche pruning salvate in: {csv_output_path}")

    print("\nPruning metrics:")
    cols_to_print = [
        "prune_fraction",
        "model",
        "positives_kept",
        "positives_pruned",
        "negatives_pruned",
        "kept_positive_rate",
        "removed_negative_rate",
        "positive_loss",
    ]

    print(pruning_df[cols_to_print].to_string(index=False))


def plot_all(
    input_file,
    output_dir,
    baseline_scorer,
    operational_x_min,
    operational_x_max,
    operational_y_min,
    operational_y_max,
    prune_fractions,
):
    os.makedirs(output_dir, exist_ok=True)

    plot_roc_full(
        input_file=input_file,
        output_path=os.path.join(output_dir, "roc_full_modular.png"),
    )

    plot_roc_operational(
        input_file=input_file,
        output_path=os.path.join(output_dir, "roc_operational_modular.png"),
        x_min=operational_x_min,
        x_max=operational_x_max,
        y_min=operational_y_min,
        y_max=operational_y_max,
        add_inset=True,
    )

    plot_delta_tpr(
        input_file=input_file,
        output_path=os.path.join(output_dir, "delta_tpr_operational_modular.png"),
        baseline_scorer=baseline_scorer,
        x_min=operational_x_min,
        x_max=operational_x_max,
    )

    plot_pruning_operational(
        input_file=input_file,
        output_path=os.path.join(output_dir, "pruning_operational_modular.png"),
        csv_output_path=os.path.join(output_dir, "pruning_operational_modular.csv"),
        prune_fractions=prune_fractions,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        default="roc_data.pkl",
    )

    parser.add_argument(
        "--output_dir",
        default="metaqual/utils/plot/plots_operational",
    )

    parser.add_argument(
        "--plot_type",
        choices=[
            "all",
            "roc_full",
            "roc_operational",
            "delta_tpr",
            "pruning",
        ],
        default="all",
    )

    parser.add_argument(
        "--baseline_scorer",
        default="finetuned_qualt5",
    )

    parser.add_argument("--x_min", type=float, default=0.30)
    parser.add_argument("--x_max", type=float, default=0.90)
    parser.add_argument("--y_min", type=float, default=0.70)
    parser.add_argument("--y_max", type=float, default=1.00)

    parser.add_argument(
        "--prune_fractions",
        type=str,
        default="0.15,0.25,0.30,0.45",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_type == "all":
        plot_all(
            input_file=args.input_file,
            output_dir=args.output_dir,
            baseline_scorer=args.baseline_scorer,
            operational_x_min=args.x_min,
            operational_x_max=args.x_max,
            operational_y_min=args.y_min,
            operational_y_max=args.y_max,
            prune_fractions=args.prune_fractions,
        )

    elif args.plot_type == "roc_full":
        plot_roc_full(
            input_file=args.input_file,
            output_path=os.path.join(args.output_dir, "roc_full_modular.png"),
        )

    elif args.plot_type == "roc_operational":
        plot_roc_operational(
            input_file=args.input_file,
            output_path=os.path.join(args.output_dir, "roc_operational_modular.png"),
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            add_inset=True,
        )

    elif args.plot_type == "delta_tpr":
        plot_delta_tpr(
            input_file=args.input_file,
            output_path=os.path.join(args.output_dir, "delta_tpr_operational_modular.png"),
            baseline_scorer=args.baseline_scorer,
            x_min=args.x_min,
            x_max=args.x_max,
        )

    elif args.plot_type == "pruning":
        plot_pruning_operational(
            input_file=args.input_file,
            output_path=os.path.join(args.output_dir, "pruning_operational_modular.png"),
            csv_output_path=os.path.join(args.output_dir, "pruning_operational_modular.csv"),
            prune_fractions=args.prune_fractions,
        )


if __name__ == "__main__":
    main()