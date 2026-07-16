import os

import pandas as pd


BASE_DIR = "metaqual/utils/statistical/results_stsRR"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "tost_pruning_noninferiority_5pct_full_vs_pruned.csv",
)


def parse_bool_series(series: pd.Series) -> pd.Series:
    """
    Converte correttamente bool, 0/1 e stringhe True/False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {"true", "1", "yes", "y", "si", "sì", "t"}
    )


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"File TOST non trovato: {INPUT_CSV}"
        )

    dataframe = pd.read_csv(INPUT_CSV)

    required_columns = [
        "scorer",
        "qrels_variant",
        "threshold",
        "pruning_percent",
        "pipeline",
        "metric",
        "mean_full",
        "mean_pruned",
        "mean_diff_pruned_minus_full",
        "p_noninferiority",
        "lower_bound_absolute",
        "passes_pruning_tost_p_lt_0.05",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Nel CSV TOST mancano le colonne: {missing_columns}"
        )

    dataframe["passes_pruning_tost_p_lt_0.05"] = parse_bool_series(
        dataframe["passes_pruning_tost_p_lt_0.05"]
    )

    equivalent = dataframe[
        dataframe["passes_pruning_tost_p_lt_0.05"]
    ].copy()

    equivalent["retained_for_plot"] = True

    plot_ready = equivalent[
        [
            "scorer",
            "qrels_variant",
            "threshold",
            "pruning_percent",
            "pipeline",
            "metric",
            "mean_full",
            "mean_pruned",
            "mean_diff_pruned_minus_full",
            "p_noninferiority",
            "lower_bound_absolute",
            "retained_for_plot",
        ]
    ].copy()

    plot_ready["config_id"] = (
        plot_ready["scorer"].astype(str)
        + " | "
        + plot_ready["qrels_variant"].astype(str)
        + " | "
        + plot_ready["pipeline"].astype(str)
        + " | "
        + plot_ready["metric"].astype(str)
        + " | thr="
        + plot_ready["threshold"].astype(str)
    )

    equivalent_output = os.path.join(
        BASE_DIR,
        "equivalent_configs_only_5pct.csv",
    )

    plot_output = os.path.join(
        BASE_DIR,
        "equivalent_configs_plot_ready_5pct.csv",
    )

    equivalent.to_csv(
        equivalent_output,
        index=False,
    )

    plot_ready.to_csv(
        plot_output,
        index=False,
    )

    print("[INFO] Salvati:")
    print(equivalent_output)
    print(plot_output)

    print("\n[INFO] Configurazioni equivalenti per qrels e metrica:")

    if plot_ready.empty:
        print("Nessuna configurazione equivalente trovata.")
    else:
        print(
            plot_ready.groupby(
                ["qrels_variant", "metric", "pipeline"],
                dropna=False,
            ).size()
        )

    dev_small_equivalent = plot_ready[
        (plot_ready["qrels_variant"] == "dev.small")
        & (plot_ready["metric"] == "RR@10")
    ]

    print(
        "\n[INFO] Configurazioni equivalenti "
        f"dev.small / RR@10: {len(dev_small_equivalent)}"
    )


if __name__ == "__main__":
    main()