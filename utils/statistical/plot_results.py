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
        help=f"Directory con TOST e output grafici. Default: {RES}",
    )

    parser.add_argument(
        "--force-rr10",
        action="store_true",
        help="Forza RR@10 per tutti i qrels.",
    )

    return parser.parse_args()


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


def plot_all(plot_df, scorer_order, scorer_labels, scorer_styles, res_dir, force_rr10=False):
    pipelines = ["BM25", "SPLADE", "TAS-B"]

    for qrels_val in sorted(plot_df["qrels_variant"].dropna().unique()):
        metric_to_plot = choose_metric(qrels_val, force_rr10=force_rr10)

        print(
            f"\n[INFO] Generazione grafici per QRELS: {qrels_val} "
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

            # Linea baseline Full: prendiamo mean_full dal primo scorer disponibile.
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
            qrels_clean = str(qrels_val).replace("/", "_").replace(" ", "_")

            out = os.path.join(
                res_dir,
                f"{metric_clean}_pruning_plot_{pipeline_clean}_{qrels_clean}_selected_scorers.png",
            )

            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Salvato: {out}")


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
        args.res_dir,
        "plot_ready_pruning_selected_scorers.csv",
    )
    plot_df.to_csv(plot_ready_out, index=False)

    print("\n[INFO] Scorer che verranno effettivamente plottati:")
    for scorer in scorer_order:
        print(f"  - {scorer} -> {scorer_labels[scorer]}")

    plot_all(
        plot_df=plot_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        res_dir=args.res_dir,
        force_rr10=(FORCE_RR10 or args.force_rr10),
    )

    print("\nCreati i grafici e il dataset plot-ready:")
    print(plot_ready_out)


if __name__ == "__main__":
    main()
