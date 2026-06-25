import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE1 = "results2"
RES = "metaqual/utils/statistical/results_sts2"

# Se vuoi plottare sempre RR@10, lascia True.
# Se invece vuoi mantenere la logica originale:
# - dev.small -> RR@10
# - test-2019/test-2020 -> nDCG@10
# metti False.
FORCE_RR10 = False


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)

    summary_files = sorted(glob.glob(os.path.join(BASE1, "compare_all_*.csv")))
    summary_files = [
        f for f in summary_files
        if "_perquery_" not in os.path.basename(f)
        and "_timings" not in os.path.basename(f)
    ]

    tost_path = os.path.join(
        RES,
        "tost_pruning_noninferiority_5pct_full_vs_pruned.csv"
    )

    if not os.path.exists(tost_path):
        raise FileNotFoundError(
            f"File TOST non trovato: {tost_path}\n"
            "Esegui prima TOST.py aggiornato anche per metadata_qualt5."
        )

    tost_df = pd.read_csv(tost_path)

    rows = []

    # Manteniamo SOLO questi due scorer.
    # Nota: nel tuo codice lo scorer si chiama finetuned_qualt5,
    # anche se a parole lo chiami qualt5_finetuned.
    KNOWN_SCORERS = [
        "metadata_qualt5",
        "finetuned_qualt5",
    ]

    for f in summary_files:
        name = os.path.basename(f)

        if not name.startswith("compare_all_") or not name.endswith(".csv"):
            continue

        stem = name[len("compare_all_"):-len(".csv")]

        try:
            prefix, threshold_str = stem.rsplit("_", 1)
            threshold = float(threshold_str)
        except ValueError:
            print(f"[WARNING] Nome file non compatibile, salto: {name}")
            continue

        scorer = None
        qrels_variant = None

        for candidate in sorted(KNOWN_SCORERS, key=len, reverse=True):
            if prefix == candidate:
                scorer = candidate
                qrels_variant = None
                break
            elif prefix.startswith(candidate + "_"):
                scorer = candidate
                qrels_variant = prefix[len(candidate) + 1:]
                break

        if scorer is None:
            # Qui vengono scartati itn, tasb, cdd, perplexity, ecc.
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

    perf_df = pd.DataFrame(rows)

    if perf_df.empty:
        raise ValueError(
            "Nessun risultato trovato per finetuned_qualt5 o metadata_qualt5.\n"
            "Controlla che in results2 esistano file tipo:\n"
            "  compare_all_finetuned_qualt5_...csv\n"
            "  compare_all_metadata_qualt5_...csv"
        )

    # TOST: serve per disegnare i pallini delle configurazioni equivalenti.
    tost_keep = tost_df.rename(
        columns={"passes_pruning_tost_p_lt_0.05": "equivalent"}
    )[[
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
        "metric",
        "mean_full",
        "equivalent",
    ]].copy()

    # Teniamo solo i due scorer anche dal file TOST.
    tost_keep = tost_keep[tost_keep["scorer"].isin(KNOWN_SCORERS)].copy()

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
        how="left"
    )

    plot_df["equivalent"] = plot_df["equivalent"].fillna(False)

    plot_ready_out = os.path.join(
        RES,
        "plot_ready_pruning_finetuned_vs_metadata_qualt5.csv"
    )
    plot_df.to_csv(plot_ready_out, index=False)

    scorer_order = [
        "finetuned_qualt5",
        "metadata_qualt5",
    ]

    pipelines = ["BM25", "SPLADE", "TAS-B"]

    scorer_labels = {
        "finetuned_qualt5": "Finetuned QualT5",
        "metadata_qualt5": "Metadata QualT5",
    }

    scorer_styles = {
        "finetuned_qualt5": {
            "color": "tab:blue",
            "linestyle": "-",
            "dashes": None,
        },
        "metadata_qualt5": {
            "color": "tab:orange",
            "linestyle": "-",
            "dashes": None,
        },
    }

    for qrels_val in sorted(plot_df["qrels_variant"].dropna().unique()):

        if FORCE_RR10:
            METRIC_TO_PLOT = "RR@10"
        else:
            if "dev.small" in str(qrels_val).lower() or "dev_small" in str(qrels_val).lower():
                METRIC_TO_PLOT = "RR@10"
            else:
                METRIC_TO_PLOT = "nDCG@10"

        print(
            f"\n[INFO] Generazione grafici per QRELS: {qrels_val} "
            f"- Metrica: {METRIC_TO_PLOT}"
        )

        qrels_sub = plot_df[
            (plot_df["qrels_variant"] == qrels_val)
            & (plot_df["metric"] == METRIC_TO_PLOT)
        ].copy()

        if qrels_sub.empty:
            print(f"[WARNING] Nessun dato trovato per {METRIC_TO_PLOT} su {qrels_val}. Salto...")
            continue

        for pipeline in pipelines:
            sub = qrels_sub[qrels_sub["pipeline"] == pipeline].copy()

            if sub.empty:
                continue

            plt.figure(figsize=(8.6, 5.4))
            ax = plt.gca()

            # Linea baseline Full.
            # Prendiamo mean_full dal primo scorer disponibile.
            first_nonempty = None
            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")
                if not s.empty and pd.notna(s["mean_full"].iloc[0]):
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

            # Linee pruned.
            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if s.empty:
                    continue

                if pd.notna(s["mean_full"].iloc[0]):
                    full_val = float(s["mean_full"].iloc[0])
                else:
                    full_val = s["value"].max()

                x_vals = [0.0] + s["pruning_percent"].tolist()
                y_vals = [full_val] + s["value"].tolist()

                style = scorer_styles[scorer]

                line, = ax.plot(
                    x_vals,
                    y_vals,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                )

                if style["dashes"] is not None:
                    line.set_dashes(style["dashes"])

                # Pallini grandi sulle configurazioni equivalenti secondo TOST.
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
            ax.set_ylabel(METRIC_TO_PLOT)
            ax.set_title(f"{pipeline} – {METRIC_TO_PLOT} ({qrels_val})")

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

            for scorer in scorer_order:
                style = scorer_styles[scorer]
                h = Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=2.0,
                    linestyle=style["linestyle"],
                    marker="o",
                    markersize=5,
                    label=scorer_labels[scorer],
                )

                if style["dashes"] is not None:
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

            plt.tight_layout()

            metric_clean = METRIC_TO_PLOT.lower().replace("@", "")
            pipeline_clean = pipeline.lower().replace("-", "").replace(" ", "_")
            qrels_clean = str(qrels_val).replace("/", "_").replace(" ", "_")

            out = os.path.join(
                RES,
                f"{metric_clean}_pruning_plot_{pipeline_clean}_{qrels_clean}_finetuned_vs_metadata.png"
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")

    print("\nCreati i grafici e il dataset plot-ready:")
    print(plot_ready_out)