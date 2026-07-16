import glob
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ttost_paired


INPUT_DIR = "results2"
BASE_DIR = "metaqual/utils/statistical/results_stsRR"

ALPHA = 0.05
REL_LOWER_BOUND = 0.05
VERY_LARGE_UPPER = 1e6

SYSTEMS = ("BM25", "SPLADE", "TAS-B")
METRICS = ("RR@10", "nDCG@10", "R@100")


def normalize_qrels_variant(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value.replace("dev_small", "dev.small")


def normalize_threshold(value: float) -> float:
    return round(float(value), 10)


def parse_compare_filename(path: str) -> Tuple[str, str, float]:
    """
    Parsing dei file nel formato:

        compare_all_<experiment_name>_<qrels_variant>_<threshold>_perquery_wide.csv

    Varianti qrels supportate:
        - dev.small
        - dev_small
        - test-2019
        - test-2020
        - test-2019_test-2020
        - altre concatenazioni test-YYYY_test-YYYY

    Esempi:
        compare_all_finetuned_qualt5_dev.small_0.15_perquery_wide.csv
        compare_all_metadata_qualt5_concat_test-2019_test-2020_0.6_perquery_wide.csv

    Restituisce:
        experiment_name, qrels_variant, threshold
    """
    name = os.path.basename(path)

    expected_suffix = "_perquery_wide.csv"

    if not name.startswith("compare_all_") or not name.endswith(expected_suffix):
        raise ValueError(f"Filename non riconosciuto: {path}")

    stem = name[len("compare_all_"):-len(expected_suffix)]

    try:
        prefix, threshold_str = stem.rsplit("_", 1)
        threshold = normalize_threshold(float(threshold_str))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Impossibile estrarre threshold dal filename: {path}"
        ) from exc

    qrels_match = re.search(
        r"_(dev\.small|dev_small|test-\d{4}(?:_test-\d{4})*)$",
        prefix,
    )

    if qrels_match is None:
        raise ValueError(
            "Impossibile estrarre qrels_variant dal filename: "
            f"{path}. Varianti supportate: dev.small, dev_small, test-YYYY."
        )

    qrels_variant = normalize_qrels_variant(qrels_match.group(1))
    experiment_name = prefix[:qrels_match.start()].strip("_")

    if not experiment_name:
        raise ValueError(
            f"Nome esperimento vuoto dopo il parsing del filename: {path}"
        )

    return experiment_name, qrels_variant, threshold


def paired_ci(
    diffs: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Intervallo di confidenza bilaterale sulla differenza paired:

        diff = pruned - full
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)

    if n < 2:
        return np.nan, np.nan

    mean_diff = float(np.mean(diffs))
    standard_deviation = float(np.std(diffs, ddof=1))
    standard_error = standard_deviation / np.sqrt(n)

    t_critical = stats.t.ppf(
        1.0 - alpha / 2.0,
        df=n - 1,
    )

    return (
        mean_diff - t_critical * standard_error,
        mean_diff + t_critical * standard_error,
    )


def run_pruning_tost(
    path: str,
    alpha: float = 0.05,
    rel_lower_bound: float = 0.05,
    upper_bound: float = 1e6,
) -> pd.DataFrame:
    """
    Test paired di non-inferiorità tra Full e Pruned.

    Differenza:
        diff = pruned - full

    Margine assoluto:
        lower_bound_abs = -rel_lower_bound * mean_full

    Il pruning è considerato non-inferiore quando il test inferiore
    rifiuta l'ipotesi:

        diff <= lower_bound_abs

    a favore di:

        diff > lower_bound_abs
    """
    dataframe = pd.read_csv(path)

    scorer, qrels_variant, threshold = parse_compare_filename(path)

    rows = []

    for system in SYSTEMS:
        for metric in METRICS:
            full_column = f"{system} Full__{metric}"
            pruned_column = f"{system} Pruned__{metric}"

            if (
                full_column not in dataframe.columns
                or pruned_column not in dataframe.columns
            ):
                print(
                    "[WARNING] Colonne mancanti, salto: "
                    f"file={os.path.basename(path)}, "
                    f"pipeline={system}, metric={metric}"
                )
                continue

            full_series = pd.to_numeric(
                dataframe[full_column],
                errors="coerce",
            )
            pruned_series = pd.to_numeric(
                dataframe[pruned_column],
                errors="coerce",
            )

            valid_mask = full_series.notna() & pruned_series.notna()

            full = full_series[valid_mask].to_numpy(dtype=float)
            pruned = pruned_series[valid_mask].to_numpy(dtype=float)

            if len(full) < 2:
                print(
                    "[WARNING] Meno di due coppie valide, salto: "
                    f"file={os.path.basename(path)}, "
                    f"pipeline={system}, metric={metric}"
                )
                continue

            differences = pruned - full

            mean_full = float(np.mean(full))
            mean_pruned = float(np.mean(pruned))
            mean_difference = float(np.mean(differences))

            lower_bound_absolute = -rel_lower_bound * mean_full

            # ttost_paired restituisce:
            # pvalue complessivo, test inferiore, test superiore.
            #
            # Per la non-inferiorità usiamo il test inferiore:
            # H0: pruned - full <= lower_bound_absolute
            # H1: pruned - full >  lower_bound_absolute
            _, lower_test, upper_test = ttost_paired(
                pruned,
                full,
                lower_bound_absolute,
                upper_bound,
            )

            t_lower, p_lower, df_lower = lower_test
            t_upper, p_upper, df_upper = upper_test

            ci_low, ci_high = paired_ci(
                differences,
                alpha=alpha,
            )

            rows.append({
                "scorer": scorer,
                "qrels_variant": qrels_variant,
                "threshold": threshold,
                "pruning_percent": threshold * 100.0,
                "pipeline": system,
                "metric": metric,
                "n_queries": len(differences),
                "mean_full": mean_full,
                "mean_pruned": mean_pruned,
                "mean_diff_pruned_minus_full": mean_difference,
                "lower_bound_relative": -rel_lower_bound,
                "lower_bound_absolute": lower_bound_absolute,
                "upper_bound_absolute": upper_bound,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "p_noninferiority": float(p_lower),
                "t_noninferiority": float(t_lower),
                "df_noninferiority": float(df_lower),
                "p_upper_test": float(p_upper),
                "t_upper_test": float(t_upper),
                "df_upper_test": float(df_upper),
                "passes_pruning_tost_p_lt_0.05": bool(p_lower < alpha),
                "margin_respected_by_mean": bool(
                    mean_difference > lower_bound_absolute
                ),
                "margin_respected_by_ci": bool(
                    ci_low > lower_bound_absolute
                ),
            })

    return pd.DataFrame(rows)


def validate_output(results: pd.DataFrame) -> None:
    required_columns = {
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
        "metric",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Colonne mancanti nell'output TOST: {sorted(missing_columns)}"
        )

    duplicate_keys = [
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
        "metric",
    ]

    duplicates = results[
        results.duplicated(
            subset=duplicate_keys,
            keep=False,
        )
    ]

    if not duplicates.empty:
        print("\n[ERROR] Righe TOST duplicate:")
        print(
            duplicates[
                duplicate_keys
            ].sort_values(duplicate_keys).to_string(index=False)
        )
        raise ValueError(
            "Sono presenti risultati TOST duplicati per le stesse chiavi."
        )


def main() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)

    wide_files = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "compare_all_*_perquery_wide.csv",
            )
        )
    )

    if not wide_files:
        raise FileNotFoundError(
            "Nessun file compare_all_*_perquery_wide.csv trovato in: "
            f"{INPUT_DIR}"
        )

    print(f"[INFO] File per-query wide trovati: {len(wide_files)}")

    all_results = []
    skipped_files = []

    for file_path in wide_files:
        print(f"[INFO] Elaboro: {file_path}")

        try:
            result = run_pruning_tost(
                file_path,
                alpha=ALPHA,
                rel_lower_bound=REL_LOWER_BOUND,
                upper_bound=VERY_LARGE_UPPER,
            )
        except ValueError as exc:
            print(f"[WARNING] {exc}")
            skipped_files.append(file_path)
            continue

        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise RuntimeError(
            "Nessun risultato valido prodotto dai file trovati."
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    validate_output(results)

    results = results.sort_values(
        [
            "qrels_variant",
            "scorer",
            "threshold",
            "pipeline",
            "metric",
        ]
    ).reset_index(drop=True)

    output_path = os.path.join(
        BASE_DIR,
        "tost_pruning_noninferiority_5pct_full_vs_pruned.csv",
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\n[INFO] Riepilogo righe TOST per qrels:")
    print(
        results.groupby(
            ["qrels_variant", "metric"],
            dropna=False,
        ).size()
    )

    dev_small_rows = results[
        results["qrels_variant"] == "dev.small"
    ]

    print(
        f"\n[INFO] Righe TOST dev.small prodotte: "
        f"{len(dev_small_rows)}"
    )

    if skipped_files:
        print("\n[WARNING] File saltati:")
        for file_path in skipped_files:
            print(f"  - {file_path}")

    print("\n[INFO] Salvato:")
    print(output_path)


if __name__ == "__main__":
    main()