import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE1 = "results2"
RES = "metaqual/utils/statistical/results_sts"

if __name__ == "__main__":
    summary_files = sorted(glob.glob(os.path.join(BASE1, "compare_all_*.csv")))
    summary_files = [f for f in summary_files if "_perquery_" not in os.path.basename(f)]

    tost_path = os.path.join(RES, "tost_pruning_noninferiority_5pct_full_vs_pruned.csv")
    tost_df = pd.read_csv(tost_path)

    rows = []
    for f in summary_files:
        name = os.path.basename(f)
        m = re.match(r"compare_all_(.+?)_(\d+(?:\.\d+)?)\.csv$", name)
        if not m:
            continue

        scorer = m.group(1)
        threshold = float(m.group(2))

        df = pd.read_csv(f)
        metric_cols = df.columns.tolist()

        for _, r in df.iterrows():
            run_name = str(r["name"])
            if "Pruned" not in run_name:
                continue

            pipeline = run_name.replace(" Pruned", "").strip()

            rows.append({
                "scorer": scorer,
                "threshold": threshold,
                "pruning_percent": threshold * 100.0,
                "pipeline": pipeline,
                "RR@10": float(r["RR@10"]) if "RR@10" in metric_cols else np.nan,
                "nDCG@10": float(r["nDCG@10"]) if "nDCG@10" in metric_cols else np.nan,
                "R@100": float(r["R@100"]) if "R@100" in metric_cols else np.nan,
            })

    perf_df = pd.DataFrame(rows)

    tost_keep = tost_df.rename(
        columns={"passes_pruning_tost_p_lt_0.05": "equivalent"}
    )[[
        "scorer", "threshold", "pipeline", "metric", "mean_full", "equivalent"
    ]].copy()

    long_rows = []
    for _, r in perf_df.iterrows():
        for metric in ["RR@10", "nDCG@10", "R@100"]:
            long_rows.append({
                "scorer": r["scorer"],
                "threshold": r["threshold"],
                "pruning_percent": r["pruning_percent"],
                "pipeline": r["pipeline"],
                "metric": metric,
                "value": r[metric],
            })

    plot_df = pd.DataFrame(long_rows)
    plot_df = plot_df.merge(
        tost_keep,
        on=["scorer", "threshold", "pipeline", "metric"],
        how="left"
    )
    plot_df["equivalent"] = plot_df["equivalent"].fillna(False)

    plot_ready_out = os.path.join(RES, "plot_ready_pruning_with_equivalence_fixed.csv")
    plot_df.to_csv(plot_ready_out, index=False)

    rr_df = plot_df[plot_df["metric"] == "RR@10"].copy()

    scorer_order = ["itn", "qualt5", "tasb", "cdd", "perplexity"]
    pipelines = ["BM25", "SPLADE", "TAS-B"]

    scorer_labels = {
        "itn": "ITN",
        "qualt5": "QualT5-Small",
        "tasb": "TASB-Mag",
        "cdd": "CDD",
        "perplexity": "T5-Ppl",
    }

    scorer_styles = {
        "itn": {
            "color": "red",
            "linestyle": "-",
            "dashes": None,
        },
        "qualt5": {
            "color": "tab:blue",
            "linestyle": "-",
            "dashes": None,
        },
        "tasb": {
            "color": "tab:green",
            "linestyle": "-",
            "dashes": None,
        },
        "cdd": {
            "color": "red",
            "linestyle": "--",
            "dashes": (6, 3),
        },
        "perplexity": {
            "color": "#e377c2",
            "linestyle": "--",
            "dashes": (6, 3),
        },
    }

    for pipeline in pipelines:
        sub = rr_df[rr_df["pipeline"] == pipeline].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(8.6, 5.4))
        ax = plt.gca()

        first_nonempty = None
        for scorer in scorer_order:
            s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")
            if not s.empty:
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

        for scorer in scorer_order:
            s = sub[sub["scorer"] == scorer].sort_values("pruning_percent")
            if s.empty:
                continue

            full_val = float(s["mean_full"].iloc[0])

            x_vals = [0.0] + s["pruning_percent"].tolist()
            y_vals = [full_val] + s["value"].tolist()

            style = scorer_styles[scorer]

            line, = ax.plot(
                x_vals,
                y_vals,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )

            if style["dashes"] is not None:
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
        ax.set_ylabel("RR@10")
        ax.set_title(f"{pipeline} – RR@10")

        xticks = [0.0] + sorted(sub["pruning_percent"].dropna().unique().tolist())
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x)}%" if x == int(x) else f"{x:g}%" for x in xticks])

        legend_handles = []

        h_full = Line2D(
            [0], [0],
            color="black",
            linewidth=1.8,
            linestyle="--",
            marker=None,
            label="Full"
        )
        h_full.set_dashes((6, 3))
        legend_handles.append(h_full)

        for scorer in scorer_order:
            style = scorer_styles[scorer]
            h = Line2D(
                [0], [0],
                color=style["color"],
                linewidth=2.0,
                linestyle=style["linestyle"],
                marker=None,
                label=scorer_labels[scorer],
            )
            if style["dashes"] is not None:
                h.set_dashes(style["dashes"])
            legend_handles.append(h)

        ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=True,
            handlelength=3.2,
            handletextpad=0.8
        )

        plt.tight_layout()

        out = os.path.join(
            RES,
            f"rr10_pruning_plot_{pipeline.lower().replace('-', '').replace(' ', '_')}.png"
        )
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

    print("Creati i grafici e il dataset plot-ready:")
    print(plot_ready_out)