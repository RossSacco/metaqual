import os
import re
import glob
from matplotlib.pyplot import stem
from nltk.data import path
from nltk.data import path
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttost_paired
from scipy import stats

INPUT_DIR = "results2"
BASE_DIR = "metaqual/utils/statistical/results_sts2"
ALPHA = 0.05
REL_LOWER_BOUND = 0.05
VERY_LARGE_UPPER = 1e6

def paired_ci(diffs, alpha=0.05):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_diff = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha/2, df=n - 1)
    return mean_diff - tcrit * se, mean_diff + tcrit * se

def run_pruning_tost(path, alpha=0.05, rel_lower_bound=0.05, upper_bound=1e6):
    df = pd.read_csv(path)
    KNOWN_SCORERS = [
        "finetuned_qualt5",
        "perplexity",
        "qualt5",
        "tasb",
        "itn",
        "cdd",
    ]
    
    name = os.path.basename(path)

    if not name.startswith("compare_all_") or not name.endswith("_perquery_wide.csv"):
        raise ValueError(f"Filename non riconosciuto: {path}")

    stem = name[len("compare_all_"):-len("_perquery_wide.csv")]

# threshold = ultima parte dopo l'ultimo underscore
    prefix, threshold_str = stem.rsplit("_", 1)
    threshold = float(threshold_str)
    
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
        raise ValueError(f"Scorer non riconosciuto nel filename: {path}")
    
    systems = ["BM25", "SPLADE", "TAS-B"]
    metrics = ["RR@10", "nDCG@10", "R@100"]

    rows = []
    for system in systems:
        for metric in metrics:
            col_full = f"{system} Full__{metric}"
            col_pruned = f"{system} Pruned__{metric}"

            if col_full not in df.columns or col_pruned not in df.columns:
                continue

            full = pd.to_numeric(df[col_full], errors="coerce")
            pruned = pd.to_numeric(df[col_pruned], errors="coerce")
            mask = full.notna() & pruned.notna()
            full = full[mask].to_numpy(dtype=float)
            pruned = pruned[mask].to_numpy(dtype=float)

            if len(full) < 2:
                continue

            diffs = pruned - full
            mean_full = float(np.mean(full))
            mean_pruned = float(np.mean(pruned))
            mean_diff = float(np.mean(diffs))

            lower_bound_abs = -rel_lower_bound * mean_full

            pvalue, lower_test, upper_test = ttost_paired(
                pruned, full, lower_bound_abs, upper_bound
            )
            t_low, p_low, df_low = lower_test

            ci_low, ci_high = paired_ci(diffs, alpha=alpha)

            rows.append({
                "scorer": scorer,
                "qrels_variant": qrels_variant,
                "threshold": threshold,
                "pruning_percent": threshold * 100.0,
                "pipeline": system,
                "metric": metric,
                "n_queries": len(diffs),
                "mean_full": mean_full,
                "mean_pruned": mean_pruned,
                "mean_diff_pruned_minus_full": mean_diff,
                "lower_bound_relative": -rel_lower_bound,
                "lower_bound_absolute": lower_bound_abs,
                "upper_bound_absolute": upper_bound,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "p_noninferiority": float(p_low),
                "t_noninferiority": float(t_low),
                "passes_pruning_tost_p_lt_0.05": bool(p_low < alpha),
                "margin_respected_by_mean": bool(mean_diff > lower_bound_abs),
                "margin_respected_by_ci": bool(ci_low > lower_bound_abs),
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    wide_files = sorted(glob.glob(os.path.join(INPUT_DIR, "compare_all_*_perquery_wide.csv")))
    if not wide_files:
        raise FileNotFoundError("Nessun file compare_all_*_perquery_wide.csv trovato nella cartella corrente.")

    results = pd.concat(
        [run_pruning_tost(f, alpha=ALPHA, rel_lower_bound=REL_LOWER_BOUND, upper_bound=VERY_LARGE_UPPER)
         for f in wide_files],
        ignore_index=True
    )

    csv_out = os.path.join(BASE_DIR, "tost_pruning_noninferiority_5pct_full_vs_pruned.csv")

    results.to_csv(csv_out, index=False)

    print(results)
    print("\nSalvati:")
    print(csv_out)
    
