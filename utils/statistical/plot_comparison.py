import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from metaqual.utils.plot.scorer_config import (
    SCORERS,
    parse_csv_arg,
    resolve_scorer_name,
    get_active_scorers,
    get_label,
    get_scorer_style,
)


BASE1 = "results2"
RES = "metaqual/utils/statistical/results_stsRRv1"

# Se vuoi plottare sempre RR@10, lascia True.
# Se invece vuoi mantenere la logica:
# - dev.small -> RR@10
# - test-2019/test-2020 -> nDCG@10
# metti False.
FORCE_RR10 = False


BASELINE_SCORER = "finetuned_qualt5"
BASELINE_LABEL = "QualT5-Finetuned"

# Nome esatto dello scorer ricavato dai file:
#   results2/compare_all_finetuned_qualt5_<qrels>_<threshold>.csv

def normalize_scorer_token(value):
    """Normalizza un nome scorer senza cambiarne il significato."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def resolve_compare_scorer(raw_scorer):
    """
    Riconosce esplicitamente la baseline dal nome esatto del file.
    Non converte `qualt5` o altri alias in `finetuned_qualt5`.
    """
    if normalize_scorer_token(raw_scorer) == BASELINE_SCORER:
        return BASELINE_SCORER
    return resolve_scorer_name(raw_scorer)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help=(
            "Scorer da plottare, separati da virgola. "
            "Accetta nome canonico, alias o substring. "
            "Esempio: --include finetuned_qualt5,attfus"
        ),
    )

    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help=(
            "Scorer da escludere, separati da virgola. "
            "Accetta nome canonico, alias o substring. "
            "Esempio: --exclude pooled,mp"
        ),
    )

    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Filtra per gruppo del config. Esempio: --group metadata_qualt5",
    )

    parser.add_argument(
        "--list-scorers",
        action="store_true",
        help="Mostra gli scorer configurati e termina.",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE1,
        help=f"Directory con i compare_all_*.csv. Default: {BASE1}",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default=RES,
        help=(
            f"Directory di supporto e, se --plot-dir non è specificato, anche output grafici. "
            f"Default: {RES}"
        ),
    )

    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help=(
            "Directory in cui salvare i grafici e i CSV plot-ready. "
            "Se non specificata, usa --res-dir."
        ),
    )

    parser.add_argument(
        "--force-rr10",
        action="store_true",
        help="Forza RR@10 per tutti i qrels.",
    )

    parser.add_argument(
        "--safe-retained",
        type=float,
        default=0.99,
        help=("Soglia minima di efficacia mantenuta rispetto a "
              "QualT5-Finetuned allo stesso livello di pruning. Default: 0.99"),
    )

    parser.add_argument(
        "--safe-delta",
        type=float,
        default=None,
        help=("Alternativa a --safe-retained: soglia minima del delta rispetto a "
              "QualT5-Finetuned, es. -0.005."),
    )

    return parser.parse_args()


def safe_name(x):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(x))


def parse_compare_filename(path):
    """
    Parsing dei file nel formato:
        compare_all_<scorer_name>_<qrels_variant>_<threshold>.csv

    Esempi:
        compare_all_finetuned_qualt5_test-2019_test-2020_0.6.csv
        compare_all_metadata_qualt5_ATTFUS_ck5_test-2019_test-2020_0.6.csv

    Output:
        scorer_name, qrels_variant, threshold
    """
    name = os.path.basename(path)

    if not name.startswith("compare_all_") or not name.endswith(".csv"):
        raise ValueError(f"Filename non riconosciuto: {path}")

    stem = name[len("compare_all_"):-len(".csv")]

    try:
        prefix, threshold_str = stem.rsplit("_", 1)
        threshold = float(threshold_str)
    except ValueError:
        raise ValueError(f"Impossibile estrarre threshold dal filename: {path}")

    match = re.search(
        r"_(test-\d{4}(?:_test-\d{4})*|dev\.small|dev_small)$",
        prefix,
    )

    if match:
        qrels_variant = match.group(1).replace("dev_small", "dev.small")
        scorer_name = prefix[:match.start()]
    else:
        qrels_variant = None
        scorer_name = prefix

    return scorer_name, qrels_variant, threshold


def load_performance_dataframe(summary_files, active_scorers):
    rows = []

    for f in summary_files:
        name = os.path.basename(f)

        try:
            raw_scorer, qrels_variant, threshold = parse_compare_filename(f)
        except ValueError:
            print(f"[WARNING] Nome file non compatibile, salto: {name}")
            continue

        scorer = resolve_compare_scorer(raw_scorer)

        if scorer is None:
            print(f"[WARNING] Scorer non presente nel config, salto: {raw_scorer} | file={name}")
            continue

        if scorer not in active_scorers:
            continue

        df = pd.read_csv(f)
        metric_cols = df.columns.tolist()

        if "name" not in df.columns:
            print(f"[WARNING] Colonna 'name' mancante in {name}, salto.")
            continue

        for _, r in df.iterrows():
            run_name = str(r["name"])

            if "Pruned" not in run_name:
                continue

            pipeline = run_name.replace(" Pruned", "").strip()

            rows.append({
                "scorer": scorer,
                "qrels_variant": qrels_variant,
                "threshold": threshold,
                "pruning_percent": threshold * 100.0,
                "pipeline": pipeline,
                "RR@10": float(r["RR@10"]) if "RR@10" in metric_cols else np.nan,
                "nDCG@10": float(r["nDCG@10"]) if "nDCG@10" in metric_cols else np.nan,
                "R@100": float(r["R@100"]) if "R@100" in metric_cols else np.nan,
            })

    return pd.DataFrame(rows)


def make_long_plot_dataframe(perf_df):
    """Converte i risultati in formato long, senza usare riferimenti al modello non potato."""
    long_rows = []

    for _, r in perf_df.iterrows():
        for metric in ["RR@10", "nDCG@10", "R@100"]:
            long_rows.append({
                "scorer": r["scorer"],
                "qrels_variant": r["qrels_variant"],
                "threshold": r["threshold"],
                "pruning_percent": r["pruning_percent"],
                "pipeline": r["pipeline"],
                "metric": metric,
                "value": r[metric],
            })

    plot_df = pd.DataFrame(long_rows)

    if plot_df.empty:
        return plot_df

    # Retrocompatibilità per vecchie run senza qrels_variant nel nome.
    plot_df["qrels_variant"] = plot_df["qrels_variant"].fillna("dev.small")
    return plot_df


def choose_metric(qrels_val, force_rr10=False):
    if force_rr10:
        return "RR@10"

    qrels_low = str(qrels_val).lower()

    if "dev.small" in qrels_low or "dev_small" in qrels_low:
        return "RR@10"

    return "nDCG@10"


def plot_all(plot_df, scorer_order, scorer_labels, scorer_styles, plot_dir, force_rr10=False):
    """
    Disegna le curve potate dei modelli selezionati.

    QualT5-Finetuned è mostrato come curva di baseline. Non viene aggiunto alcun
    punto artificiale al pruning 0% e non viene tracciata alcuna linea del modello
    non potato.
    """
    pipelines = ["BM25", "SPLADE", "TAS-B"]

    for qrels_val in sorted(plot_df["qrels_variant"].dropna().unique()):
        metric_to_plot = choose_metric(qrels_val, force_rr10=force_rr10)

        print(
            f"\n[INFO] Generazione grafici standard per QRELS: {qrels_val} "
            f"- Metrica: {metric_to_plot}"
        )

        qrels_sub = plot_df[
            (plot_df["qrels_variant"] == qrels_val)
            & (plot_df["metric"] == metric_to_plot)
        ].copy()

        if qrels_sub.empty:
            print(f"[WARNING] Nessun dato trovato per {metric_to_plot} su {qrels_val}. Salto...")
            continue

        for pipeline in pipelines:
            sub = qrels_sub[qrels_sub["pipeline"] == pipeline].copy()

            if sub.empty:
                continue

            plt.figure(figsize=(9.6, 5.8))
            ax = plt.gca()
            present_scorers = []

            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if s.empty:
                    continue

                present_scorers.append(scorer)
                style = scorer_styles[scorer]

                line, = ax.plot(
                    s["pruning_percent"],
                    s["value"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    marker=style.get("marker", "o"),
                    markersize=5,
                )

                if style.get("dashes") is not None:
                    line.set_dashes(style["dashes"])

            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel(metric_to_plot)
            ax.set_title(
                f"{pipeline} – {metric_to_plot} ({qrels_val})\n"
                f"Baseline: {BASELINE_LABEL}"
            )

            xticks = sorted(sub["pruning_percent"].dropna().unique().tolist())
            ax.set_xticks(xticks)
            ax.set_xticklabels([
                f"{int(x)}%" if x == int(x) else f"{x:g}%"
                for x in xticks
            ])

            legend_handles = []
            for scorer in present_scorers:
                style = scorer_styles[scorer]
                label = scorer_labels[scorer]
                if scorer == BASELINE_SCORER:
                    label = f"{label} (baseline)"

                h = Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=2.0,
                    linestyle=style["linestyle"],
                    marker=style.get("marker", "o"),
                    markersize=5,
                    label=label,
                )

                if style.get("dashes") is not None:
                    h.set_dashes(style["dashes"])

                legend_handles.append(h)

            ax.legend(
                handles=legend_handles,
                loc="best",
                frameon=True,
                handlelength=3.2,
                handletextpad=0.8,
            )

            ax.grid(True, alpha=0.25)
            plt.tight_layout()

            metric_clean = metric_to_plot.lower().replace("@", "")
            pipeline_clean = pipeline.lower().replace("-", "").replace(" ", "_")
            qrels_clean = safe_name(qrels_val)

            out = os.path.join(
                plot_dir,
                f"{metric_clean}_pruning_plot_{pipeline_clean}_{qrels_clean}_selected_scorers.png",
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")


def make_selected_metric_df(plot_df, force_rr10=False):
    """
    Seleziona la metrica da visualizzare e associa a ogni risultato il valore
    prodotto da QualT5-Finetuned nella stessa identica configurazione:

      - stesso qrels_variant;
      - stessa retrieval pipeline;
      - stessa metrica;
      - stesso threshold/livello di pruning.

    Aggiunge:
        baseline_value = valore di QualT5-Finetuned
        delta_value    = value - baseline_value
        retained       = value / baseline_value
    """
    parts = []

    for qrels_val in sorted(plot_df["qrels_variant"].dropna().unique()):
        metric_to_plot = choose_metric(qrels_val, force_rr10=force_rr10)

        sub = plot_df[
            (plot_df["qrels_variant"] == qrels_val)
            & (plot_df["metric"] == metric_to_plot)
        ].copy()

        if sub.empty:
            continue

        sub["selected_metric"] = metric_to_plot
        parts.append(sub)

    if not parts:
        raise ValueError("Nessun dato disponibile per le metriche selezionate.")

    df = pd.concat(parts, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["pruning_percent"] = pd.to_numeric(df["pruning_percent"], errors="coerce")

    # Una chiave arrotondata evita mancate corrispondenze dovute alla
    # rappresentazione floating point di soglie come 0.3 o 0.6.
    df["pruning_key"] = df["pruning_percent"].round(8)

    baseline_keys = [
        "qrels_variant",
        "pipeline",
        "metric",
        "pruning_key",
    ]

    baseline_df = (
        df[df["scorer"] == BASELINE_SCORER]
        .groupby(baseline_keys, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "baseline_value"})
    )

    if baseline_df.empty:
        raise ValueError(
            f"Non sono stati trovati risultati per la baseline {BASELINE_LABEL} "
            f"({BASELINE_SCORER}). Controlla che esistano i file "
            "compare_all_finetuned_qualt5_*.csv."
        )

    df = df.merge(baseline_df, on=baseline_keys, how="left")

    missing = df["baseline_value"].isna()
    if missing.any():
        missing_cfg = (
            df.loc[missing, ["qrels_variant", "pipeline", "metric", "threshold"]]
            .drop_duplicates()
            .sort_values(["qrels_variant", "pipeline", "metric", "threshold"])
        )
        print(
            "[WARNING] Mancano risultati QualT5-Finetuned per alcune configurazioni; "
            "queste righe non saranno usate nei confronti relativi:"
        )
        print(missing_cfg.to_string(index=False))

    df = df.dropna(
        subset=["value", "baseline_value", "pruning_percent"]
    ).copy()

    df["baseline_scorer"] = BASELINE_SCORER
    df["baseline_label"] = BASELINE_LABEL
    df["delta_value"] = df["value"] - df["baseline_value"]

    nonzero_baseline = ~np.isclose(df["baseline_value"], 0.0)
    df["retained"] = np.where(
        nonzero_baseline,
        df["value"] / df["baseline_value"],
        np.nan,
    )

    return df


def plot_extra_line(
    metric_df,
    scorer_order,
    scorer_labels,
    scorer_styles,
    plot_dir,
    y_col,
    y_label,
    baseline_value,
    title_suffix,
    file_prefix,
):
    pipelines = ["BM25", "SPLADE", "TAS-B"]

    for qrels_val in sorted(metric_df["qrels_variant"].dropna().unique()):
        qsub = metric_df[metric_df["qrels_variant"] == qrels_val].copy()

        if qsub.empty:
            continue

        metric_name = qsub["selected_metric"].iloc[0]

        for pipeline in pipelines:
            sub = qsub[qsub["pipeline"] == pipeline].copy()

            if sub.empty:
                continue

            plt.figure(figsize=(9.8, 5.8))
            ax = plt.gca()

            ax.axhline(
                baseline_value,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"{BASELINE_LABEL} reference",
            )

            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if s.empty:
                    continue

                style = scorer_styles[scorer]
                label = scorer_labels[scorer]
                if scorer == BASELINE_SCORER:
                    label = f"{label} (baseline)"

                line, = ax.plot(
                    s["pruning_percent"],
                    s[y_col],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    marker=style.get("marker", "o"),
                    markersize=5,
                    label=label,
                )

                if style.get("dashes") is not None:
                    line.set_dashes(style["dashes"])

            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel(y_label)
            ax.set_title(f"{pipeline} – {title_suffix} ({qrels_val}, {metric_name})")

            xticks = sorted(sub["pruning_percent"].dropna().unique().tolist())
            ax.set_xticks(xticks)
            ax.set_xticklabels([
                f"{int(x)}%" if x == int(x) else f"{x:g}%"
                for x in xticks
            ])

            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", frameon=True)
            plt.tight_layout()

            metric_clean = metric_name.lower().replace("@", "")
            pipeline_clean = pipeline.lower().replace("-", "").replace(" ", "_")
            qrels_clean = safe_name(qrels_val)

            out = os.path.join(
                plot_dir,
                f"{file_prefix}_{metric_clean}_{pipeline_clean}_{qrels_clean}.png",
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")


def plot_extra_heatmap(
    metric_df,
    scorer_order,
    scorer_labels,
    plot_dir,
    value_col,
    title_suffix,
    file_prefix,
    center_zero=False,
    fmt="{:.4f}",
):
    pipelines = ["BM25", "SPLADE", "TAS-B"]

    for qrels_val in sorted(metric_df["qrels_variant"].dropna().unique()):
        qsub = metric_df[metric_df["qrels_variant"] == qrels_val].copy()

        if qsub.empty:
            continue

        metric_name = qsub["selected_metric"].iloc[0]

        for pipeline in pipelines:
            sub = qsub[qsub["pipeline"] == pipeline].copy()

            # La baseline serve solo per calcolare delta e retained: non deve
            # comparire come riga nelle heatmap.
            sub = sub[sub["scorer"] != BASELINE_SCORER].copy()

            if sub.empty:
                print(
                    f"[WARNING] Nessun modello diverso da {BASELINE_LABEL} "
                    f"per la heatmap {pipeline} / {qrels_val}. Salto..."
                )
                continue

            sub["scorer_label"] = sub["scorer"].map(
                lambda s: scorer_labels.get(s, s)
            )

            pivot = sub.pivot_table(
                index="scorer_label",
                columns="pruning_percent",
                values=value_col,
                aggfunc="mean",
            )

            ordered_labels = [
                scorer_labels[s]
                for s in scorer_order
                if s != BASELINE_SCORER
                and scorer_labels.get(s, s) in pivot.index
            ]

            pivot = pivot.reindex(ordered_labels)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)

            values = pivot.values.astype(float)

            fig_w = max(8.5, 1.2 * len(pivot.columns) + 3)
            fig_h = max(4.5, 0.6 * len(pivot.index) + 2)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            if center_zero:
                vmax = np.nanmax(np.abs(values))
                vmin = -vmax
                cmap = "coolwarm"
            else:
                vmin = np.nanmin(values)
                vmax = np.nanmax(values)
                cmap = "viridis"

            im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_title(f"{pipeline} – {title_suffix} ({qrels_val}, {metric_name})")
            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel("Scorer")

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([
                f"{int(c)}%" if c == int(c) else f"{c:g}%"
                for c in pivot.columns
            ])

            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)

            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if not np.isnan(values[i, j]):
                        ax.text(
                            j,
                            i,
                            fmt.format(values[i, j]),
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

            cbar = fig.colorbar(im, ax=ax)
            if value_col == "delta_value":
                cbar.set_label(f"Δ metric vs {BASELINE_LABEL}")
            elif value_col == "retained":
                cbar.set_label(f"Retained effectiveness vs {BASELINE_LABEL}")
            else:
                cbar.set_label(value_col)

            fig.tight_layout()

            metric_clean = metric_name.lower().replace("@", "")
            pipeline_clean = pipeline.lower().replace("-", "").replace(" ", "_")
            qrels_clean = safe_name(qrels_val)

            out = os.path.join(
                plot_dir,
                f"{file_prefix}_{metric_clean}_{pipeline_clean}_{qrels_clean}.png",
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")


def plot_maximum_safe_pruning(
    metric_df,
    scorer_labels,
    plot_dir,
    retained_threshold=0.99,
    delta_threshold=None,
):
    rows = []

    for (qrels_val, pipeline, scorer), g in metric_df.groupby(
        ["qrels_variant", "pipeline", "scorer"]
    ):
        g = g.sort_values("pruning_percent").copy()

        if delta_threshold is not None:
            good = g[g["delta_value"] >= delta_threshold]
            criterion = f"Δ vs {BASELINE_LABEL} >= {delta_threshold}"
        else:
            good = g[g["retained"] >= retained_threshold]
            criterion = f"retained vs {BASELINE_LABEL} >= {retained_threshold}"

        safe_pruning = good["pruning_percent"].max() if not good.empty else 0.0

        rows.append({
            "qrels_variant": qrels_val,
            "pipeline": pipeline,
            "scorer": scorer,
            "scorer_label": scorer_labels.get(scorer, scorer),
            "safe_pruning": safe_pruning,
            "criterion": criterion,
        })

    safe_df = pd.DataFrame(rows)

    out_csv = os.path.join(plot_dir, "extra_maximum_safe_pruning.csv")
    safe_df.to_csv(out_csv, index=False)
    print(f"[INFO] Salvato: {out_csv}")

    for qrels_val in sorted(safe_df["qrels_variant"].dropna().unique()):
        sub = safe_df[safe_df["qrels_variant"] == qrels_val].copy()

        if sub.empty:
            continue

        sub["label"] = sub["pipeline"] + " | " + sub["scorer_label"]
        sub = sub.sort_values("safe_pruning", ascending=True)

        fig_h = max(5.0, 0.35 * len(sub))
        fig, ax = plt.subplots(figsize=(11, fig_h))

        ax.barh(sub["label"], sub["safe_pruning"])
        ax.set_xlabel("Maximum safe pruning percentage")
        ax.set_title(f"Maximum safe pruning ({qrels_val}) – {sub['criterion'].iloc[0]}")
        ax.grid(True, axis="x", alpha=0.3)

        for i, v in enumerate(sub["safe_pruning"]):
            ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)

        fig.tight_layout()

        out = os.path.join(
            plot_dir,
            f"extra_maximum_safe_pruning_{safe_name(qrels_val)}.png",
        )

        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Salvato: {out}")


def compute_aupc(metric_df, scorer_labels):
    """Area media sotto la curva della retained effectiveness vs QualT5-Finetuned."""
    rows = []

    for (qrels_val, pipeline, scorer), g in metric_df.groupby(
        ["qrels_variant", "pipeline", "scorer"]
    ):
        g = g.sort_values("pruning_percent").dropna(subset=["retained"]).copy()

        x = (g["pruning_percent"] / 100.0).to_numpy(dtype=float)
        y = g["retained"].to_numpy(dtype=float)

        if len(x) == 0:
            aupc = np.nan
            mean_retained = np.nan
            min_retained = np.nan
        elif len(x) < 2 or np.max(x) == np.min(x):
            aupc = float(np.nanmean(y))
            mean_retained = float(np.nanmean(y))
            min_retained = float(np.nanmin(y))
        else:
            aupc = float(np.trapz(y, x) / (np.max(x) - np.min(x)))
            mean_retained = float(np.nanmean(y))
            min_retained = float(np.nanmin(y))

        rows.append({
            "qrels_variant": qrels_val,
            "pipeline": pipeline,
            "scorer": scorer,
            "scorer_label": scorer_labels.get(scorer, scorer),
            "baseline": BASELINE_LABEL,
            "aupc_retained": aupc,
            "mean_retained": mean_retained,
            "min_retained": min_retained,
        })

    return pd.DataFrame(rows)


def plot_aupc(metric_df, scorer_labels, plot_dir):
    aupc_df = compute_aupc(metric_df, scorer_labels)

    out_csv = os.path.join(plot_dir, "extra_aupc_retained.csv")
    aupc_df.to_csv(out_csv, index=False)
    print(f"[INFO] Salvato: {out_csv}")

    for qrels_val in sorted(aupc_df["qrels_variant"].dropna().unique()):
        sub = aupc_df[aupc_df["qrels_variant"] == qrels_val].copy()

        if sub.empty:
            continue

        sub["label"] = sub["pipeline"] + " | " + sub["scorer_label"]
        sub = sub.sort_values("aupc_retained", ascending=True)

        fig_h = max(5.0, 0.35 * len(sub))
        fig, ax = plt.subplots(figsize=(11, fig_h))

        ax.barh(sub["label"], sub["aupc_retained"])
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5)

        ax.set_xlabel("AUPC retained effectiveness")
        ax.set_title(f"Area under pruning curve vs {BASELINE_LABEL} ({qrels_val})")
        ax.grid(True, axis="x", alpha=0.3)

        for i, v in enumerate(sub["aupc_retained"]):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

        fig.tight_layout()

        out = os.path.join(
            plot_dir,
            f"extra_aupc_retained_{safe_name(qrels_val)}.png",
        )

        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Salvato: {out}")


def plot_rank_heatmap(metric_df, scorer_order, scorer_labels, plot_dir):
    rank_df = metric_df.copy()

    rank_df["rank"] = rank_df.groupby(
        ["qrels_variant", "pipeline", "pruning_percent"]
    )["value"].rank(ascending=False, method="min")

    rank_df["scorer_label"] = rank_df["scorer"].map(
        lambda s: scorer_labels.get(s, s)
    )

    pipelines = ["BM25", "SPLADE", "TAS-B"]

    for qrels_val in sorted(rank_df["qrels_variant"].dropna().unique()):
        qsub = rank_df[rank_df["qrels_variant"] == qrels_val].copy()

        if qsub.empty:
            continue

        metric_name = qsub["selected_metric"].iloc[0]

        for pipeline in pipelines:
            sub = qsub[qsub["pipeline"] == pipeline].copy()

            # Escludi QualT5-Finetuned dalla visualizzazione: rimane la
            # baseline usata per i confronti, ma non occupa una riga.
            sub = sub[sub["scorer"] != BASELINE_SCORER].copy()

            if sub.empty:
                continue

            # Ricalcola il rank solo tra i modelli visualizzati.
            sub["rank"] = sub.groupby("pruning_percent")["value"].rank(
                ascending=False, method="min"
            )

            pivot = sub.pivot_table(
                index="scorer_label",
                columns="pruning_percent",
                values="rank",
                aggfunc="mean",
            )

            ordered_labels = [
                scorer_labels[s]
                for s in scorer_order
                if s != BASELINE_SCORER
                and scorer_labels.get(s, s) in pivot.index
            ]

            pivot = pivot.reindex(ordered_labels)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)

            values = pivot.values.astype(float)

            fig_w = max(8.5, 1.2 * len(pivot.columns) + 3)
            fig_h = max(4.5, 0.6 * len(pivot.index) + 2)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            im = ax.imshow(values, aspect="auto", cmap="viridis_r")

            ax.set_title(f"{pipeline} – model rank ({qrels_val}, {metric_name})")
            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel("Scorer")

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([
                f"{int(c)}%" if c == int(c) else f"{c:g}%"
                for c in pivot.columns
            ])

            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)

            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if not np.isnan(values[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{int(values[i, j])}",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Rank, 1 = best")

            fig.tight_layout()

            metric_clean = metric_name.lower().replace("@", "")
            pipeline_clean = pipeline.lower().replace("-", "").replace(" ", "_")
            qrels_clean = safe_name(qrels_val)

            out = os.path.join(
                plot_dir,
                f"extra_rank_heatmap_{metric_clean}_{pipeline_clean}_{qrels_clean}.png",
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")


def pareto_mask(points):
    """
    Pareto front massimizzando:
        x = pruning_percent
        y = value
    """
    n = len(points)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        x_i, y_i = points[i]

        for j in range(n):
            if i == j:
                continue

            x_j, y_j = points[j]

            dominates = (
                x_j >= x_i
                and y_j >= y_i
                and (x_j > x_i or y_j > y_i)
            )

            if dominates:
                keep[i] = False
                break

    return keep


def plot_pareto(metric_df, scorer_order, scorer_labels, scorer_styles, plot_dir):
    pipelines = ["BM25", "SPLADE", "TAS-B"]

    for qrels_val in sorted(metric_df["qrels_variant"].dropna().unique()):
        qsub = metric_df[metric_df["qrels_variant"] == qrels_val].copy()

        if qsub.empty:
            continue

        metric_name = qsub["selected_metric"].iloc[0]

        for pipeline in pipelines:
            sub = qsub[qsub["pipeline"] == pipeline].copy()

            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if s.empty:
                    continue

                style = scorer_styles[scorer]
                label = scorer_labels[scorer]
                if scorer == BASELINE_SCORER:
                    label = f"{label} (baseline)"

                line, = ax.plot(
                    s["pruning_percent"],
                    s["value"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    marker=style.get("marker", "o"),
                    markersize=5,
                    label=label,
                    alpha=0.9,
                )

                if style.get("dashes") is not None:
                    line.set_dashes(style["dashes"])

            points = sub[["pruning_percent", "value"]].to_numpy()
            mask = pareto_mask(points)
            pareto = sub.iloc[np.where(mask)[0]].copy()

            ax.scatter(
                pareto["pruning_percent"],
                pareto["value"],
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                label="Pareto front",
                zorder=8,
            )

            for _, r in pareto.iterrows():
                ax.annotate(
                    f"{scorer_labels.get(r['scorer'], r['scorer'])} {r['pruning_percent']:.0f}%",
                    (r["pruning_percent"], r["value"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )

            ax.set_title(
                f"{pipeline} – Pareto plot ({qrels_val}, {metric_name})\n"
                f"Reference model: {BASELINE_LABEL}"
            )
            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", frameon=True)

            fig.tight_layout()

            metric_clean = metric_name.lower().replace("@", "")
            pipeline_clean = pipeline.lower().replace("-", "").replace(" ", "_")
            qrels_clean = safe_name(qrels_val)

            out = os.path.join(
                plot_dir,
                f"extra_pareto_{metric_clean}_{pipeline_clean}_{qrels_clean}.png",
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")


def plot_extra_comparisons(
    plot_df,
    scorer_order,
    scorer_labels,
    scorer_styles,
    plot_dir,
    force_rr10=False,
    retained_threshold=0.99,
    delta_threshold=None,
):
    """Crea tutti i confronti usando QualT5-Finetuned come unica baseline."""
    metric_df = make_selected_metric_df(
        plot_df,
        force_rr10=force_rr10,
    )

    out_ready = os.path.join(
        plot_dir,
        "extra_plot_ready_selected_metric_vs_qualt5_finetuned.csv",
    )
    metric_df.to_csv(out_ready, index=False)
    print(f"[INFO] Salvato dataset extra plot-ready: {out_ready}")

    print(f"[INFO] Creo extra plot: Δ metrica vs {BASELINE_LABEL}...")
    plot_extra_line(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        plot_dir=plot_dir,
        y_col="delta_value",
        y_label=f"Δ metric vs {BASELINE_LABEL}",
        baseline_value=0.0,
        title_suffix=f"Δ metric vs {BASELINE_LABEL}",
        file_prefix="extra_delta_vs_qualt5_finetuned",
    )

    print(f"[INFO] Creo extra plot: retained effectiveness vs {BASELINE_LABEL}...")
    plot_extra_line(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        plot_dir=plot_dir,
        y_col="retained",
        y_label=f"Retained effectiveness vs {BASELINE_LABEL}",
        baseline_value=1.0,
        title_suffix=f"Retained effectiveness vs {BASELINE_LABEL}",
        file_prefix="extra_retained_vs_qualt5_finetuned",
    )

    print(f"[INFO] Creo extra heatmap: Δ metrica vs {BASELINE_LABEL}...")
    plot_extra_heatmap(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
        value_col="delta_value",
        title_suffix=f"Δ metric vs {BASELINE_LABEL}",
        file_prefix="extra_heatmap_delta_vs_qualt5_finetuned",
        center_zero=True,
        fmt="{:.4f}",
    )

    print(f"[INFO] Creo extra heatmap: retained effectiveness vs {BASELINE_LABEL}...")
    plot_extra_heatmap(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
        value_col="retained",
        title_suffix=f"Retained effectiveness vs {BASELINE_LABEL}",
        file_prefix="extra_heatmap_retained_vs_qualt5_finetuned",
        center_zero=False,
        fmt="{:.3f}",
    )

    print(f"[INFO] Creo extra plot: maximum safe pruning vs {BASELINE_LABEL}...")
    plot_maximum_safe_pruning(
        metric_df=metric_df,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
        retained_threshold=retained_threshold,
        delta_threshold=delta_threshold,
    )

    print(f"[INFO] Creo extra plot: AUPC retained vs {BASELINE_LABEL}...")
    plot_aupc(
        metric_df=metric_df,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
    )

    print("[INFO] Creo extra plot: rank heatmap...")
    plot_rank_heatmap(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
    )

    print("[INFO] Creo extra plot: Pareto...")
    plot_pareto(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        plot_dir=plot_dir,
    )


def main():
    args = parse_args()

    include_scorers = parse_csv_arg(args.include)
    exclude_scorers = parse_csv_arg(args.exclude)

    if args.list_scorers:
        print("\n[INFO] Scorer nel config:")
        for scorer_name, cfg in SCORERS.items():
            enabled = cfg.get("enabled", False)
            group = cfg.get("group", None)
            label = cfg.get("label", scorer_name)
            aliases = ", ".join(cfg.get("compare_aliases", []))
            print(
                f"  - {scorer_name:32s} enabled={str(enabled):5s} "
                f"group={str(group):16s} label={label} aliases=[{aliases}]"
            )
        raise SystemExit(0)

    os.makedirs(args.res_dir, exist_ok=True)

    plot_dir = args.plot_dir if args.plot_dir is not None else args.res_dir
    os.makedirs(plot_dir, exist_ok=True)

    selected_scorers = get_active_scorers(
        include=include_scorers,
        exclude=exclude_scorers,
        group=args.group,
        only_enabled=True,
    )

    if not selected_scorers:
        raise ValueError(
            "Nessuno scorer attivo/selezionato.\n"
            f"include={include_scorers}\n"
            f"exclude={exclude_scorers}\n"
            f"group={args.group}"
        )

    if BASELINE_SCORER not in SCORERS:
        raise ValueError(
            f"La baseline richiesta '{BASELINE_SCORER}' non è presente in SCORERS."
        )

    # La baseline viene sempre caricata, anche se --include o --group selezionano
    # soltanto modelli metadata. Serve per effettuare il confronto corretto.
    load_scorers = list(dict.fromkeys([BASELINE_SCORER] + selected_scorers))

    print("\n[INFO] Scorer richiesti:")
    for scorer in selected_scorers:
        print(f"  - {scorer} -> {get_label(scorer)}")

    print(f"\n[INFO] Baseline obbligatoria: {BASELINE_SCORER} -> {get_label(BASELINE_SCORER)}")
    print(f"[INFO] Directory risultati compare_all: {args.base_dir}")
    print(f"[INFO] Directory output grafici: {plot_dir}")

    summary_files = sorted(glob.glob(os.path.join(args.base_dir, "compare_all_*.csv")))
    summary_files = [
        f for f in summary_files
        if "_perquery_" not in os.path.basename(f)
        and "_timings" not in os.path.basename(f)
    ]

    baseline_file_pattern = os.path.join(
        args.base_dir,
        "compare_all_finetuned_qualt5_*.csv",
    )
    baseline_files = sorted(glob.glob(baseline_file_pattern))

    if not baseline_files:
        raise FileNotFoundError(
            "Baseline QualT5-Finetuned non trovata.\n"
            f"Pattern cercato: {baseline_file_pattern}\n"
            "Il nome atteso è "
            "compare_all_finetuned_qualt5_<qrels>_<threshold>.csv."
        )

    print(
        f"[INFO] File baseline compare_all_finetuned_qualt5 trovati: "
        f"{len(baseline_files)}"
    )
    for baseline_file in baseline_files:
        print(f"  - {os.path.basename(baseline_file)}")

    perf_df = load_performance_dataframe(summary_files, load_scorers)

    if perf_df.empty:
        raise ValueError(
            "Nessun risultato trovato per gli scorer selezionati.\n"
            f"Scorer caricati: {load_scorers}\n"
            f"Directory risultati: {args.base_dir}\n"
            "Controlla i nomi dei file compare_all_*.csv e gli alias in scorer_config.py."
        )

    baseline_perf = perf_df[perf_df["scorer"] == BASELINE_SCORER].copy()
    if baseline_perf.empty:
        raise ValueError(
            f"Mancano i risultati della baseline {BASELINE_LABEL}.\n"
            "Devono essere presenti file con formato "
            "compare_all_finetuned_qualt5_<qrels>_<threshold>.csv "
            "nella directory indicata da --base-dir."
        )

    print(
        f"[INFO] Righe baseline {BASELINE_LABEL} caricate: "
        f"{len(baseline_perf)}"
    )
    print(
        "[INFO] Livelli di pruning baseline disponibili: "
        + ", ".join(
            f"{x:g}%" for x in sorted(
                baseline_perf["pruning_percent"].dropna().unique().tolist()
            )
        )
    )

    plot_df = make_long_plot_dataframe(perf_df)
    available_scorers = plot_df["scorer"].dropna().unique().tolist()

    scorer_order = [
        scorer
        for scorer in load_scorers
        if scorer in available_scorers
    ]

    if not scorer_order:
        raise ValueError(
            "Nessuno scorer selezionato ha dati disponibili nei CSV.\n"
            f"Scorer richiesti: {load_scorers}\n"
            f"Scorer disponibili nei risultati: {available_scorers}"
        )

    scorer_labels = {scorer: get_label(scorer) for scorer in scorer_order}
    scorer_styles = {scorer: get_scorer_style(scorer) for scorer in scorer_order}

    plot_ready_out = os.path.join(
        plot_dir,
        "plot_ready_pruning_selected_scorers.csv",
    )
    plot_df.to_csv(plot_ready_out, index=False)

    print("\n[INFO] Scorer che verranno effettivamente plottati:")
    for scorer in scorer_order:
        suffix = " [BASELINE]" if scorer == BASELINE_SCORER else ""
        print(f"  - {scorer} -> {scorer_labels[scorer]}{suffix}")

    force_rr10 = FORCE_RR10 or args.force_rr10

    plot_all(
        plot_df=plot_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        plot_dir=plot_dir,
        force_rr10=force_rr10,
    )

    plot_extra_comparisons(
        plot_df=plot_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        plot_dir=plot_dir,
        force_rr10=force_rr10,
        retained_threshold=args.safe_retained,
        delta_threshold=args.safe_delta,
    )

    print("\n[DONE] Creati i grafici e i dataset plot-ready.")
    print(f"[DONE] Baseline usata in tutti i confronti: {BASELINE_LABEL}")
    print(f"[DONE] Dataset principale: {plot_ready_out}")
    print(f"[DONE] Output grafici: {plot_dir}")


if __name__ == "__main__":
    main()
    
    
"""
python -m metaqual.utils.statistical.plot_comparison \
  --base-dir results2 \
  --res-dir metaqual/utils/statistical/results_sts4 \
  --plot-dir metaqual/utils/statistical/results_sts4/plots_extra \
  --include metadata_qualt5_concat_v2,metadata_qualt5_allmetapj,metadata_qualt5_attfus \
  --safe-delta -0.005
"""