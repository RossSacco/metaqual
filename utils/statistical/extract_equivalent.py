import os
import pandas as pd

BASE_DIR = "metaqual/utils/statistical/results_sts"
INPUT_CSV = os.path.join(BASE_DIR, "tost_pruning_noninferiority_5pct_full_vs_pruned.csv")

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    eq = df[df["passes_pruning_tost_p_lt_0.05"] == True].copy()
    eq["retained_for_plot"] = True

    plot_ready = eq[[
        "scorer",
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
    ]].copy()

    plot_ready["config_id"] = (
        plot_ready["scorer"].astype(str) + " | " +
        plot_ready["pipeline"].astype(str) + " | " +
        plot_ready["metric"].astype(str) + " | thr=" +
        plot_ready["threshold"].astype(str)
    )

    eq_csv = os.path.join(BASE_DIR, "equivalent_configs_only_5pct.csv")
    plot_csv = os.path.join(BASE_DIR, "equivalent_configs_plot_ready_5pct.csv")

    eq.to_csv(eq_csv, index=False)
    plot_ready.to_csv(plot_csv, index=False)

    print("Salvati:")
    print(eq_csv)
    print(plot_csv)
