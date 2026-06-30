import os
import re
import glob

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ttost_paired


INPUT_DIR = "results2"
BASE_DIR = "metaqual/utils/statistical/results_sts3"

ALPHA = 0.05
REL_LOWER_BOUND = 0.05
VERY_LARGE_UPPER = 1e6


def parse_compare_filename(path):
    """
    Parsing robusto dei file nel formato:

        compare_all_<experiment_name>_<qrels_variant>_<threshold>_perquery_wide.csv

    Esempio:
        compare_all_metadata_qualt5_concat_test-2019_test-2020_0.6_perquery_wide.csv

    Output:
        experiment_name = metadata_qualt5_concat
        qrels_variant = test-2019_test-2020
        threshold = 0.6
    """

    name = os.path.basename(path)

    if not name.startswith("compare_all_") or not name.endswith("_perquery_wide.csv"):
        raise ValueError(f"Filename non riconosciuto: {path}")

    stem = name[len("compare_all_"):-len("_perquery_wide.csv")]

    # Ultima parte dopo underscore = threshold
    try:
        prefix, threshold_str = stem.rsplit("_", 1)
        threshold = float(threshold_str)
    except ValueError:
        raise ValueError(f"Impossibile estrarre threshold dal filename: {path}")

    # Cerca qrels del tipo:
    # test-2019
    # test-2019_test-2020
    match = re.search(r"_(test-\d{4}(?:_test-\d{4})*)$", prefix)

    if match:
        qrels_variant = match.group(1)
        experiment_name = prefix[:match.start()]
    else:
        qrels_variant = None
        experiment_name = prefix

    return experiment_name, qrels_variant, threshold


def paired_ci(diffs, alpha=0.05):
    """
    Intervallo di confidenza al 95% sulla differenza paired:
        diff = pruned - full
    """

    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)

    if n < 2:
        return np.nan, np.nan

    mean_diff = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / np.sqrt(n)

    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    return mean_diff - tcrit * se, mean_diff + tcrit * se


def run_pruning_tost(path, alpha=0.05, rel_lower_bound=0.05, upper_bound=1e6):
    """
    Esegue il test di non-inferiorità tra Full e Pruned.

    Ipotesi pratica:
        voglio verificare che Pruned non sia peggiore di Full
        oltre il margine del 5%.

    Differenza:
        diff = pruned - full

    Margine assoluto:
        lower_bound_abs = -0.05 * mean_full

    Se il test passa:
        Pruned è considerato non-inferiore a Full.
    """

    df = pd.read_csv(path)

    scorer, qrels_variant, threshold = parse_compare_filename(path)

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

            # TOST paired:
            # x1 = pruned
            # x2 = full
            # quindi la differenza testata è pruned - full
            pvalue, lower_test, upper_test = ttost_paired(
                pruned,
                full,
                lower_bound_abs,
                upper_bound
            )

            t_low, p_low, df_low = lower_test
            t_up, p_up, df_up = upper_test

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

                "p_tost": float(pvalue),
                "p_noninferiority": float(p_low),
                "t_noninferiority": float(t_low),

                "p_upper_test": float(p_up),
                "t_upper_test": float(t_up),

                "passes_pruning_tost_p_lt_0.05": bool(p_low < alpha),
                "margin_respected_by_mean": bool(mean_diff > lower_bound_abs),
                "margin_respected_by_ci": bool(ci_low > lower_bound_abs),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    wide_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "compare_all_*_perquery_wide.csv"))
    )

    if not wide_files:
        raise FileNotFoundError(
            f"Nessun file compare_all_*_perquery_wide.csv trovato in: {INPUT_DIR}"
        )

    all_results = []

    for f in wide_files:
        print(f"[INFO] Elaboro: {f}")
        res = run_pruning_tost(
            f,
            alpha=ALPHA,
            rel_lower_bound=REL_LOWER_BOUND,
            upper_bound=VERY_LARGE_UPPER
        )

        if not res.empty:
            all_results.append(res)

    if not all_results:
        raise RuntimeError("Nessun risultato valido prodotto dai file trovati.")

    results = pd.concat(all_results, ignore_index=True)

    csv_out = os.path.join(
        BASE_DIR,
        "tost_pruning_noninferiority_5pct_full_vs_pruned.csv"
    )

    results.to_csv(csv_out, index=False)

    print("\nRisultati:")
    print(results)

    print("\nSalvato:")
    print(csv_out)