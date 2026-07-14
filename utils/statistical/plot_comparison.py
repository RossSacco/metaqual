import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
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
RES = "metaqual/utils/statistical/results_sts4"

# Se vuoi plottare sempre RR@10, lascia True.
# Se invece vuoi mantenere la logica:
# - dev.small -> RR@10
# - test-2019/test-2020 -> nDCG@10
# metti False.
FORCE_RR10 = False


LEGACY_VARIANTS = {
    "CONCAT-V2-FA-ck1": "metadata_qualt5_concat_v2",
    "CONCAT-V2-FA_ck1": "metadata_qualt5_concat_v2",
    "ATTFUS-ck5": "metadata_qualt5_attfus",
    "ATTFUS_ck5": "metadata_qualt5_attfus",
    "ALLMETAPJ-ck1": "metadata_qualt5_allmetapj",
    "ALLMETAPJ_ck1": "metadata_qualt5_allmetapj",
    "MP": "metadata_qualt5_mp",
    "CONCAT": "metadata_qualt5_concat_v1",
    "POOLEDCONCAT": "metadata_qualt5_pooled",
    "POOLED_CONCAT": "metadata_qualt5_pooled",
}


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
            f"Directory con TOST e, se --plot-dir non è specificato, anche output grafici. "
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
        help="Soglia retained effectiveness per maximum safe pruning. Default: 0.99",
    )

    parser.add_argument(
        "--safe-delta",
        type=float,
        default=None,
        help="Alternativa a --safe-retained: soglia delta metrica, es. -0.005.",
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


def fix_legacy_metadata_names(df):
    """
    Retrocompatibilità per vecchi CSV TOST.

    Caso legacy:
        scorer = metadata_qualt5
        qrels_variant = ATTFUS-ck5_test-2019_test-2020

    Diventa:
        scorer = metadata_qualt5_attfus
        qrels_variant = test-2019_test-2020
    """
    if df.empty:
        return df

    df = df.copy()

    if "qrels_variant" not in df.columns:
        df["qrels_variant"] = None

    for idx, row in df.iterrows():
        scorer = str(row.get("scorer", ""))
        qrels_variant = row.get("qrels_variant", None)

        if scorer != "metadata_qualt5":
            continue

        if pd.isna(qrels_variant):
            continue

        qrels_str = str(qrels_variant)

        for variant, canonical_scorer in LEGACY_VARIANTS.items():
            prefix = variant + "_"

            if qrels_str.startswith(prefix):
                df.at[idx, "scorer"] = canonical_scorer
                df.at[idx, "qrels_variant"] = qrels_str[len(prefix):]
                break

    return df


def load_performance_dataframe(summary_files, active_scorers):
    rows = []

    for f in summary_files:
        name = os.path.basename(f)

        try:
            raw_scorer, qrels_variant, threshold = parse_compare_filename(f)
        except ValueError:
            print(f"[WARNING] Nome file non compatibile, salto: {name}")
            continue

        scorer = resolve_scorer_name(raw_scorer)

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


def load_tost_dataframe(tost_path, active_scorers):
    if not os.path.exists(tost_path):
        raise FileNotFoundError(
            f"File TOST non trovato: {tost_path}\n"
            "Esegui prima lo script TOST aggiornato."
        )

    tost_df = pd.read_csv(tost_path)
    tost_df = fix_legacy_metadata_names(tost_df)

    tost_df["scorer"] = tost_df["scorer"].apply(resolve_scorer_name)
    tost_df = tost_df[tost_df["scorer"].notna()].copy()

    tost_keep = tost_df.rename(
        columns={"passes_pruning_tost_p_lt_0.05": "equivalent"}
    ).copy()

    required_tost_cols = [
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
        "metric",
        "mean_full",
        "equivalent",
    ]

    missing_cols = [c for c in required_tost_cols if c not in tost_keep.columns]
    if missing_cols:
        raise ValueError(
            f"Nel file TOST mancano queste colonne: {missing_cols}\n"
            "Rigenera il file TOST con lo script aggiornato."
        )

    tost_keep = tost_keep[required_tost_cols].copy()
    tost_keep = tost_keep[tost_keep["scorer"].isin(active_scorers)].copy()

    return tost_keep


def make_long_plot_dataframe(perf_df, tost_keep):
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

    # Retrocompatibilità per vecchie run senza qrels_variant nel nome.
    plot_df["qrels_variant"] = plot_df["qrels_variant"].fillna("dev.small")
    tost_keep["qrels_variant"] = tost_keep["qrels_variant"].fillna("dev.small")

    plot_df = plot_df.merge(
        tost_keep,
        on=["scorer", "qrels_variant", "threshold", "pipeline", "metric"],
        how="left",
    )

    plot_df["equivalent"] = plot_df["equivalent"].fillna(False)

    return plot_df


def choose_metric(qrels_val, force_rr10=False):
    if force_rr10:
        return "RR@10"

    qrels_low = str(qrels_val).lower()

    if "dev.small" in qrels_low or "dev_small" in qrels_low:
        return "RR@10"

    return "nDCG@10"


def plot_all(plot_df, scorer_order, scorer_labels, scorer_styles, plot_dir, force_rr10=False):
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

            first_nonempty = None

            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if not s.empty and "mean_full" in s.columns and pd.notna(s["mean_full"].iloc[0]):
                    first_nonempty = s
                    break

            if first_nonempty is not None:
                full_val = float(first_nonempty["mean_full"].iloc[0])
                x_max = max([0.0] + sub["pruning_percent"].dropna().tolist())

                full_line, = ax.plot(
                    [0.0, x_max],
                    [full_val, full_val],
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                )
                full_line.set_dashes((6, 3))

            present_scorers = []

            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if s.empty:
                    continue

                present_scorers.append(scorer)

                if "mean_full" in s.columns and pd.notna(s["mean_full"].iloc[0]):
                    full_val = float(s["mean_full"].iloc[0])
                elif first_nonempty is not None:
                    full_val = float(first_nonempty["mean_full"].iloc[0])
                else:
                    full_val = float(s["value"].max())

                x_vals = [0.0] + s["pruning_percent"].tolist()
                y_vals = [full_val] + s["value"].tolist()

                style = scorer_styles[scorer]

                line, = ax.plot(
                    x_vals,
                    y_vals,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    marker=style.get("marker", "o"),
                    markersize=5,
                )

                if style.get("dashes") is not None:
                    line.set_dashes(style["dashes"])

                eq = s[s["equivalent"] == True]

                if not eq.empty:
                    ax.scatter(
                        eq["pruning_percent"],
                        eq["value"],
                        s=180,
                        facecolors=style["color"],
                        alpha=0.5,
                        edgecolors=style["color"],
                        linewidths=1.5,
                        zorder=5,
                    )

            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel(metric_to_plot)
            ax.set_title(f"{pipeline} – {metric_to_plot} ({qrels_val})")

            xticks = [0.0] + sorted(sub["pruning_percent"].dropna().unique().tolist())
            ax.set_xticks(xticks)
            ax.set_xticklabels([
                f"{int(x)}%" if x == int(x) else f"{x:g}%"
                for x in xticks
            ])

            legend_handles = []

            h_full = Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.8,
                linestyle="--",
                marker=None,
                label="Full",
            )
            h_full.set_dashes((6, 3))
            legend_handles.append(h_full)

            for scorer in present_scorers:
                style = scorer_styles[scorer]

                h = Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=2.0,
                    linestyle=style["linestyle"],
                    marker=style.get("marker", "o"),
                    markersize=5,
                    label=scorer_labels[scorer],
                )

                if style.get("dashes") is not None:
                    h.set_dashes(style["dashes"])

                legend_handles.append(h)

            h_equiv = Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=12,
                alpha=0.5,
                label="TOST equivalent",
            )
            legend_handles.append(h_equiv)

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
    Usa lo stesso plot_df creato dallo script.

    Output:
        una riga per scorer/pipeline/qrels/pruning solo per la metrica scelta:
        - dev.small -> RR@10
        - test-* -> nDCG@10
        - oppure RR@10 se force_rr10=True

    Aggiunge:
        delta_value = value - mean_full
        retained = value / mean_full
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
    df["mean_full"] = pd.to_numeric(df["mean_full"], errors="coerce")
    df["pruning_percent"] = pd.to_numeric(df["pruning_percent"], errors="coerce")

    df = df.dropna(subset=["value", "mean_full", "pruning_percent"]).copy()

    df["delta_value"] = df["value"] - df["mean_full"]
    df["retained"] = df["value"] / df["mean_full"]

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
                label="Full baseline",
            )

            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if s.empty:
                    continue

                x_vals = [0.0] + s["pruning_percent"].tolist()

                if y_col == "delta_value":
                    y_vals = [0.0] + s[y_col].tolist()
                elif y_col == "retained":
                    y_vals = [1.0] + s[y_col].tolist()
                else:
                    y_vals = [float(s["mean_full"].iloc[0])] + s[y_col].tolist()

                style = scorer_styles[scorer]

                line, = ax.plot(
                    x_vals,
                    y_vals,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    marker=style.get("marker", "o"),
                    markersize=5,
                    label=scorer_labels[scorer],
                )

                if style.get("dashes") is not None:
                    line.set_dashes(style["dashes"])

            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel(y_label)
            ax.set_title(f"{pipeline} – {title_suffix} ({qrels_val}, {metric_name})")

            xticks = [0.0] + sorted(sub["pruning_percent"].dropna().unique().tolist())
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

            if sub.empty:
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
                if scorer_labels.get(s, s) in pivot.index
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
            criterion = f"Δ >= {delta_threshold}"
        else:
            good = g[g["retained"] >= retained_threshold]
            criterion = f"retained >= {retained_threshold}"

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
    rows = []

    for (qrels_val, pipeline, scorer), g in metric_df.groupby(
        ["qrels_variant", "pipeline", "scorer"]
    ):
        g = g.sort_values("pruning_percent").copy()

        x = np.array([0.0] + (g["pruning_percent"] / 100.0).tolist())
        y = np.array([1.0] + g["retained"].tolist())

        if len(x) < 2 or np.max(x) == np.min(x):
            aupc = np.nanmean(y)
        else:
            aupc = np.trapz(y, x) / (np.max(x) - np.min(x))

        rows.append({
            "qrels_variant": qrels_val,
            "pipeline": pipeline,
            "scorer": scorer,
            "scorer_label": scorer_labels.get(scorer, scorer),
            "aupc_retained": aupc,
            "mean_retained": np.nanmean(y),
            "min_retained": np.nanmin(y),
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
        ax.set_title(f"Area under pruning curve ({qrels_val})")
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

            if sub.empty:
                continue

            pivot = sub.pivot_table(
                index="scorer_label",
                columns="pruning_percent",
                values="rank",
                aggfunc="mean",
            )

            ordered_labels = [
                scorer_labels[s]
                for s in scorer_order
                if scorer_labels.get(s, s) in pivot.index
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

                line, = ax.plot(
                    s["pruning_percent"],
                    s["value"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    marker=style.get("marker", "o"),
                    markersize=5,
                    label=scorer_labels[scorer],
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

            full_val = float(sub["mean_full"].dropna().iloc[0])
            ax.axhline(
                full_val,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Full",
            )

            ax.set_title(f"{pipeline} – Pareto plot ({qrels_val}, {metric_name})")
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
    """
    Crea grafici extra usando lo stesso plot_df dello script principale.
    """
    metric_df = make_selected_metric_df(
        plot_df,
        force_rr10=force_rr10,
    )

    out_ready = os.path.join(
        plot_dir,
        "extra_plot_ready_selected_metric_with_delta.csv",
    )
    metric_df.to_csv(out_ready, index=False)
    print(f"[INFO] Salvato dataset extra plot-ready: {out_ready}")

    print("[INFO] Creo extra plot: Δ metrica vs Full...")
    plot_extra_line(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        plot_dir=plot_dir,
        y_col="delta_value",
        y_label="Δ metric vs Full",
        baseline_value=0.0,
        title_suffix="Δ metric vs Full",
        file_prefix="extra_delta",
    )

    print("[INFO] Creo extra plot: retained effectiveness...")
    plot_extra_line(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        plot_dir=plot_dir,
        y_col="retained",
        y_label="Retained effectiveness",
        baseline_value=1.0,
        title_suffix="Retained effectiveness",
        file_prefix="extra_retained",
    )

    print("[INFO] Creo extra heatmap: Δ metric...")
    plot_extra_heatmap(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
        value_col="delta_value",
        title_suffix="Δ metric vs Full",
        file_prefix="extra_heatmap_delta",
        center_zero=True,
        fmt="{:.4f}",
    )

    print("[INFO] Creo extra heatmap: retained effectiveness...")
    plot_extra_heatmap(
        metric_df=metric_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
        value_col="retained",
        title_suffix="Retained effectiveness",
        file_prefix="extra_heatmap_retained",
        center_zero=False,
        fmt="{:.3f}",
    )

    print("[INFO] Creo extra plot: maximum safe pruning...")
    plot_maximum_safe_pruning(
        metric_df=metric_df,
        scorer_labels=scorer_labels,
        plot_dir=plot_dir,
        retained_threshold=retained_threshold,
        delta_threshold=delta_threshold,
    )

    print("[INFO] Creo extra plot: AUPC retained...")
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

    active_scorers = get_active_scorers(
        include=include_scorers,
        exclude=exclude_scorers,
        group=args.group,
        only_enabled=True,
    )

    if not active_scorers:
        raise ValueError(
            "Nessuno scorer attivo/selezionato.\n"
            f"include={include_scorers}\n"
            f"exclude={exclude_scorers}\n"
            f"group={args.group}"
        )

    print("\n[INFO] Scorer attivi/selezionati dal config:")
    for scorer in active_scorers:
        print(f"  - {scorer} -> {get_label(scorer)}")

    print(f"\n[INFO] Directory risultati compare_all: {args.base_dir}")
    print(f"[INFO] Directory TOST: {args.res_dir}")
    print(f"[INFO] Directory output grafici: {plot_dir}")

    summary_files = sorted(glob.glob(os.path.join(args.base_dir, "compare_all_*.csv")))

    summary_files = [
        f for f in summary_files
        if "_perquery_" not in os.path.basename(f)
        and "_timings" not in os.path.basename(f)
    ]

    perf_df = load_performance_dataframe(summary_files, active_scorers)

    if perf_df.empty:
        raise ValueError(
            "Nessun risultato trovato per gli scorer selezionati.\n"
            f"Scorer selezionati: {active_scorers}\n"
            f"Directory risultati: {args.base_dir}\n"
            "Controlla i nomi dei file compare_all_*.csv e gli alias in scorer_config.py."
        )

    tost_path = os.path.join(
        args.res_dir,
        "tost_pruning_noninferiority_5pct_full_vs_pruned.csv",
    )
    tost_keep = load_tost_dataframe(tost_path, active_scorers)

    plot_df = make_long_plot_dataframe(perf_df, tost_keep)

    available_scorers = plot_df["scorer"].dropna().unique().tolist()

    scorer_order = [
        scorer
        for scorer in active_scorers
        if scorer in available_scorers
    ]

    if not scorer_order:
        raise ValueError(
            "Nessuno scorer selezionato ha dati disponibili nei CSV.\n"
            f"Scorer attivi da config/CLI: {active_scorers}\n"
            f"Scorer disponibili nei risultati: {available_scorers}"
        )

    scorer_labels = {
        scorer: get_label(scorer)
        for scorer in scorer_order
    }

    scorer_styles = {
        scorer: get_scorer_style(scorer)
        for scorer in scorer_order
    }

    plot_ready_out = os.path.join(
        plot_dir,
        "plot_ready_pruning_selected_scorers.csv",
    )
    plot_df.to_csv(plot_ready_out, index=False)

    print("\n[INFO] Scorer che verranno effettivamente plottati:")
    for scorer in scorer_order:
        print(f"  - {scorer} -> {scorer_labels[scorer]}")

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
    print(f"[DONE] Dataset principale: {plot_ready_out}")
    print(f"[DONE] Output grafici: {plot_dir}")


if __name__ == "__main__":
    main()
    
    
"""
python -m metaqual.utils.statistical.plot_comparison \
  --base-dir results2 \
  --res-dir metaqual/utils/statistical/results_sts4 \
  --plot-dir metaqual/utils/statistical/results_sts4/plots_extra \
  --include finetuned_qualt5,metadata_qualt5_concat_v2,metadata_qualt5_allmetapj,metadata_qualt5_attfus \
  --safe-delta -0.005
"""