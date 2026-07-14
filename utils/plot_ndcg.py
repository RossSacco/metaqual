import os
import re
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE1 = "results2"
RES = "metaqual/utils/statistical/results_sts4"

# Se vuoi plottare sempre RR@10, lascia True.
# Se invece vuoi mantenere la logica:
# - dev.small -> RR@10
# - test-2019/test-2020 -> nDCG@10
# metti False.
FORCE_RR10 = False


def parse_compare_filename(path):
    """
    Parsing dei file nel formato:

        compare_all_<scorer_name>_<qrels_variant>_<threshold>.csv

    Esempi:
        compare_all_finetuned_qualt5_test-2019_test-2020_0.6.csv
        compare_all_metadata_qualt5_mp_test-2019_test-2020_0.6.csv
        compare_all_metadata_qualt5_concat_test-2019_test-2020_0.6.csv
        compare_all_metadata_qualt5_pooledconcat_test-2019_test-2020_0.6.csv

    Output:
        scorer_name
        qrels_variant
        threshold
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

    # qrels riconosciuti:
    # test-2019
    # test-2020
    # test-2019_test-2020
    # dev.small / dev_small
    match = re.search(
        r"_(test-\d{4}(?:_test-\d{4})*|dev\.small|dev_small)$",
        prefix
    )

    if match:
        qrels_variant = match.group(1).replace("dev_small", "dev.small")
        scorer_name = prefix[:match.start()]
    else:
        qrels_variant = None
        scorer_name = prefix

    return scorer_name, qrels_variant, threshold


def is_wanted_scorer(scorer):
    """
    Teniamo:
    - finetuned_qualt5
    - tutte le varianti metadata_qualt5*
    """
    scorer = str(scorer)
    return scorer == "finetuned_qualt5" or scorer.startswith("metadata_qualt5")


def fix_legacy_metadata_names(df):
    """
    Serve per retrocompatibilità.

    Se in vecchi CSV TOST hai righe del tipo:
        scorer = metadata_qualt5
        qrels_variant = mp_test-2019_test-2020

    le converte in:
        scorer = metadata_qualt5_mp
        qrels_variant = test-2019_test-2020

    Stessa cosa per concat e pooledconcat.
    """

    if df.empty:
        return df

    df = df.copy()

    if "qrels_variant" not in df.columns:
        df["qrels_variant"] = None

    known_variants = [
        #"pooledconcat",
        #"pooled_concat",
        #"concat",
        #"mp",
        #"POOLEDCONCAT",
        #"POOLED_CONCAT",
        #"CONCAT",
        #"MP",
        "CONCAT-V2-FA-ck1",
        "ATTFUS-ck5",
        "ALLMETAPJ-ck1",
    ]

    for idx, row in df.iterrows():
        scorer = str(row.get("scorer", ""))
        qrels_variant = row.get("qrels_variant", None)

        if scorer != "metadata_qualt5":
            continue

        if pd.isna(qrels_variant):
            continue

        qrels_str = str(qrels_variant)

        for variant in known_variants:
            prefix = variant + "_"

            if qrels_str.startswith(prefix):
                clean_variant = variant.lower()
                clean_variant = clean_variant.replace("pooled_concat", "pooledconcat")

                df.at[idx, "scorer"] = f"metadata_qualt5_{clean_variant}"
                df.at[idx, "qrels_variant"] = qrels_str[len(prefix):]
                break

    return df


def scorer_label(scorer):
    labels = {
        "finetuned_qualt5": "Finetuned QualT5",
        "metadata_qualt5": "Metadata QualT5",
        "metadata_qualt5_mp": "Metadata QualT5 MP",
        "metadata_qualt5_concat": "Metadata QualT5 CONCAT",
        "metadata_qualt5_pooledconcat": "Metadata QualT5 POOLEDCONCAT",
    }

    if scorer in labels:
        return labels[scorer]

    if scorer.startswith("metadata_qualt5_"):
        suffix = scorer.replace("metadata_qualt5_", "")
        return "Metadata QualT5 " + suffix.upper()

    return scorer


def build_scorer_styles(scorer_order):
    """
    Stili per distinguere finetuned e le diverse varianti metadata.
    """

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "--",
        "--",
        "-.",
        ":",
    ]

    styles = {}

    for i, scorer in enumerate(scorer_order):
        styles[scorer] = {
            "color": colors[i % len(colors)],
            "linestyle": linestyles[i % len(linestyles)],
            "dashes": None,
        }

    return styles


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)

    summary_files = sorted(glob.glob(os.path.join(BASE1, "compare_all_*.csv")))

    # Teniamo solo i summary csv, non i per-query e non i timings.
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
            "Esegui prima lo script TOST aggiornato, quello che riconosce "
            "metadata_qualt5_mp / metadata_qualt5_concat / metadata_qualt5_pooledconcat."
        )

    tost_df = pd.read_csv(tost_path)
    tost_df = fix_legacy_metadata_names(tost_df)

    rows = []

    for f in summary_files:
        name = os.path.basename(f)

        try:
            scorer, qrels_variant, threshold = parse_compare_filename(f)
        except ValueError:
            print(f"[WARNING] Nome file non compatibile, salto: {name}")
            continue

        # Teniamo finetuned_qualt5 + tutte le varianti metadata_qualt5*
        if not is_wanted_scorer(scorer):
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
            "Nessun risultato trovato per finetuned_qualt5 o metadata_qualt5*.\n"
            "Controlla che in results2 esistano file tipo:\n"
            "  compare_all_finetuned_qualt5_test-2019_test-2020_0.6.csv\n"
            "  compare_all_metadata_qualt5_mp_test-2019_test-2020_0.6.csv\n"
            "  compare_all_metadata_qualt5_concat_test-2019_test-2020_0.6.csv\n"
            "  compare_all_metadata_qualt5_pooledconcat_test-2019_test-2020_0.6.csv"
        )

    # TOST: serve per disegnare i pallini delle configurazioni equivalenti.
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

    # Teniamo solo finetuned_qualt5 + metadata_qualt5*
    tost_keep = tost_keep[tost_keep["scorer"].apply(is_wanted_scorer)].copy()

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
        "plot_ready_pruning_finetuned_vs_all_metadata_qualt5.csv"
    )

    plot_df.to_csv(plot_ready_out, index=False)

    # Ordine: prima baseline finetuned, poi tutte le varianti metadata trovate nei dati.
    available_scorers = sorted(plot_df["scorer"].dropna().unique().tolist())

    metadata_scorers = sorted([
        s for s in available_scorers
        if str(s).startswith("metadata_qualt5")
    ])

    scorer_order = []

    if "finetuned_qualt5" in available_scorers:
        scorer_order.append("finetuned_qualt5")

    scorer_order.extend(metadata_scorers)

    pipelines = ["BM25", "SPLADE", "TAS-B"]

    scorer_labels = {
        scorer: scorer_label(scorer)
        for scorer in scorer_order
    }

    scorer_styles = build_scorer_styles(scorer_order)

    print("\n[INFO] Scorer che verranno plottati:")
    for scorer in scorer_order:
        print(f"  - {scorer} -> {scorer_labels[scorer]}")

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

            plt.figure(figsize=(9.6, 5.8))
            ax = plt.gca()

            # Linea baseline Full.
            # Prendiamo mean_full dal primo scorer disponibile.
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

            # Linee pruned.
            present_scorers = []

            for scorer in scorer_order:
                s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")

                if s.empty:
                    continue

                present_scorers.append(scorer)

                # Punto iniziale della linea = valore Full.
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

            for scorer in present_scorers:
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

            ax.grid(True, alpha=0.25)

            plt.tight_layout()

            metric_clean = METRIC_TO_PLOT.lower().replace("@", "")
            pipeline_clean = pipeline.lower().replace("-", "").replace(" ", "_")
            qrels_clean = str(qrels_val).replace("/", "_").replace(" ", "_")

            out = os.path.join(
                RES,
                f"{metric_clean}_pruning_plot_{pipeline_clean}_{qrels_clean}_finetuned_vs_all_metadata.png"
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")

    print("\nCreati i grafici e il dataset plot-ready:")
    print(plot_ready_out)