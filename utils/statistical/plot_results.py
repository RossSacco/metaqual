import argparse
import glob
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from metaqual.utils.plot.scorer_config import (
    SCORERS,
    get_active_scorers,
    get_label,
    get_scorer_style,
    parse_csv_arg,
    resolve_scorer_name,
)


BASE1 = "results2"
RES = "metaqual/utils/statistical/2s-vs-or--ALLMETAPJ"

# False:
#   - dev.small -> RR@10
#   - test-2019/test-2020 -> nDCG@10
#
# True:
#   - RR@10 per tutti i dataset.
FORCE_RR10 = False

METRICS = ("RR@10", "nDCG@10", "R@100")
PIPELINE_ORDER = ("BM25", "SPLADE", "TAS-B")


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


# =========================================================
# ARGOMENTI
# =========================================================

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


# =========================================================
# NORMALIZZAZIONE
# =========================================================

def normalize_qrels_variant(value: Any) -> Optional[str]:
    """
    Normalizza il nome del set di qrels.

    IMPORTANTE:
    un valore mancante NON viene interpretato come dev.small.
    """
    if value is None or pd.isna(value):
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    normalized = normalized.replace("dev_small", "dev.small")
    return normalized


def normalize_threshold(value: Any) -> float:
    """
    Normalizza la soglia per rendere stabile il merge tra CSV risultati e TOST.
    """
    return round(float(value), 10)


def normalize_pipeline(value: Any) -> str:
    """
    Uniforma le principali varianti dei nomi delle pipeline.
    """
    pipeline = str(value).strip()
    compact = pipeline.lower().replace("_", "").replace("-", "").replace(" ", "")

    mapping = {
        "bm25": "BM25",
        "splade": "SPLADE",
        "tasb": "TAS-B",
    }

    return mapping.get(compact, pipeline)


def normalize_metric(value: Any) -> str:
    """
    Normalizza le varianti più comuni dei nomi delle metriche.
    """
    metric = str(value).strip()
    compact = metric.lower().replace(" ", "")

    aliases = {
        "rr@10": "RR@10",
        "rr(rel=1)@10": "RR@10",
        "recip_rank@10": "RR@10",
        "reciprank@10": "RR@10",
        "ndcg@10": "nDCG@10",
        "ndcg_cut_10": "nDCG@10",
        "r@100": "R@100",
        "recall@100": "R@100",
        "recall_100": "R@100",
    }

    return aliases.get(compact, metric)


def to_optional_float(value: Any) -> float:
    return pd.to_numeric(value, errors="coerce")


def parse_bool_value(value: Any) -> bool:
    """
    Converte correttamente bool, numeri e stringhe.

    Evita il bug di astype(bool), per cui la stringa "False"
    verrebbe interpretata come True perché non vuota.
    """
    if value is None or pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)

    text = str(value).strip().lower()

    true_values = {"true", "1", "yes", "y", "si", "sì", "t"}
    false_values = {"false", "0", "no", "n", "", "nan", "none", "f"}

    if text in true_values:
        return True
    if text in false_values:
        return False

    raise ValueError(f"Valore booleano non riconosciuto: {value!r}")


# =========================================================
# PARSING FILE
# =========================================================

def parse_compare_filename(path: str) -> Tuple[str, Optional[str], float]:
    """
    Parsing dei file nel formato:

        compare_all_<scorer_name>_<qrels_variant>_<threshold>.csv

    Esempi:

        compare_all_finetuned_qualt5_dev.small_0.6.csv
        compare_all_finetuned_qualt5_test-2019_test-2020_0.6.csv
        compare_all_metadata_qualt5_ATTFUS_ck5_dev.small_0.6.csv

    Restituisce:
        scorer_name, qrels_variant, threshold
    """
    name = os.path.basename(path)

    if not name.startswith("compare_all_") or not name.endswith(".csv"):
        raise ValueError(f"Filename non riconosciuto: {path}")

    stem = name[len("compare_all_"):-len(".csv")]

    try:
        prefix, threshold_str = stem.rsplit("_", 1)
        threshold = normalize_threshold(threshold_str)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Impossibile estrarre threshold dal filename: {path}"
        ) from exc

    match = re.search(
        r"_(test-\d{4}(?:_test-\d{4})*|dev\.small|dev_small)$",
        prefix,
    )

    if match:
        qrels_variant = normalize_qrels_variant(match.group(1))
        scorer_name = prefix[:match.start()]
    else:
        qrels_variant = None
        scorer_name = prefix

    return scorer_name, qrels_variant, threshold


def fix_legacy_metadata_names(df: pd.DataFrame) -> pd.DataFrame:
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
        return df.copy()

    fixed = df.copy()

    if "qrels_variant" not in fixed.columns:
        fixed["qrels_variant"] = None

    for idx, row in fixed.iterrows():
        scorer = str(row.get("scorer", "")).strip()
        qrels_variant = normalize_qrels_variant(
            row.get("qrels_variant", None)
        )

        if scorer != "metadata_qualt5" or qrels_variant is None:
            continue

        for variant, canonical_scorer in LEGACY_VARIANTS.items():
            prefix = variant + "_"

            if qrels_variant.startswith(prefix):
                fixed.at[idx, "scorer"] = canonical_scorer
                fixed.at[idx, "qrels_variant"] = qrels_variant[len(prefix):]
                break

    return fixed


# =========================================================
# CARICAMENTO RISULTATI
# =========================================================

def extract_pipeline_and_kind(run_name: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Restituisce:
        pipeline, kind

    kind è "full" oppure "pruned".
    """
    text = str(run_name).strip()

    if text.endswith(" Full"):
        return normalize_pipeline(text[:-len(" Full")]), "full"

    if text.endswith(" Pruned"):
        return normalize_pipeline(text[:-len(" Pruned")]), "pruned"

    return None, None


def load_performance_dataframe(
    summary_files: Iterable[str],
    active_scorers: List[str],
) -> pd.DataFrame:
    """
    Carica sia le righe Full sia le righe Pruned.

    La baseline Full viene letta dallo STESSO CSV della run Pruned.
    Non viene più recuperata dal file TOST.
    """
    rows: List[Dict[str, Any]] = []

    for file_path in summary_files:
        filename = os.path.basename(file_path)

        try:
            raw_scorer, qrels_variant, threshold = parse_compare_filename(
                file_path
            )
        except ValueError as exc:
            print(f"[WARNING] {exc}. Salto: {filename}")
            continue

        scorer = resolve_scorer_name(raw_scorer)

        if scorer is None:
            print(
                "[WARNING] Scorer non presente nel config, salto: "
                f"{raw_scorer} | file={filename}"
            )
            continue

        if scorer not in active_scorers:
            continue

        if qrels_variant is None:
            print(
                "[WARNING] Qrels non riconoscibili nel filename. "
                f"Il file NON verrà attribuito automaticamente a dev.small: "
                f"{filename}"
            )
            continue

        df = pd.read_csv(file_path)

        if "name" not in df.columns:
            print(
                f"[WARNING] Colonna 'name' mancante in {filename}, salto."
            )
            continue

        missing_metric_columns = [
            metric for metric in METRICS if metric not in df.columns
        ]
        if missing_metric_columns:
            print(
                f"[WARNING] In {filename} mancano le metriche "
                f"{missing_metric_columns}. I relativi valori saranno NaN."
            )

        full_by_pipeline: Dict[str, Dict[str, float]] = {}
        pruned_rows: List[Tuple[str, pd.Series]] = []

        for _, row in df.iterrows():
            pipeline, kind = extract_pipeline_and_kind(row["name"])

            if pipeline is None:
                continue

            if kind == "full":
                if pipeline in full_by_pipeline:
                    raise ValueError(
                        f"Più righe Full per {pipeline} in {filename}."
                    )

                full_by_pipeline[pipeline] = {
                    metric: to_optional_float(row.get(metric, np.nan))
                    for metric in METRICS
                }

            elif kind == "pruned":
                pruned_rows.append((pipeline, row))

        if not full_by_pipeline:
            print(
                f"[WARNING] Nessuna baseline Full trovata in {filename}. "
                "Salto il file per evitare una baseline inventata."
            )
            continue

        for pipeline, row in pruned_rows:
            if pipeline not in full_by_pipeline:
                print(
                    "[WARNING] Baseline Full mancante per "
                    f"{pipeline} nel file {filename}. Salto la riga Pruned."
                )
                continue

            full_metrics = full_by_pipeline[pipeline]

            output_row: Dict[str, Any] = {
                "source_file": filename,
                "scorer": scorer,
                "qrels_variant": qrels_variant,
                "threshold": threshold,
                "pruning_percent": threshold * 100.0,
                "pipeline": pipeline,
            }

            for metric in METRICS:
                output_row[metric] = to_optional_float(
                    row.get(metric, np.nan)
                )
                output_row[f"full_{metric}"] = full_metrics[metric]

            rows.append(output_row)

    return pd.DataFrame(rows)


def load_tost_dataframe(
    tost_path: str,
    active_scorers: List[str],
) -> pd.DataFrame:
    """
    Carica il TOST.

    Le righe senza qrels_variant NON vengono eliminate qui:
    verranno associate successivamente ai risultati solo quando il dataset
    può essere inferito in modo univoco.

    mean_full viene mantenuto come segnale aggiuntivo per distinguere,
    quando necessario, dev.small dalle collezioni TREC.
    """
    if not os.path.exists(tost_path):
        raise FileNotFoundError(
            f"File TOST non trovato: {tost_path}\n"
            "Esegui prima lo script TOST aggiornato."
        )

    tost_df = pd.read_csv(tost_path)
    tost_df = fix_legacy_metadata_names(tost_df)

    if "scorer" not in tost_df.columns:
        raise ValueError("Nel file TOST manca la colonna 'scorer'.")

    equivalence_source = "passes_pruning_tost_p_lt_0.05"

    if "equivalent" not in tost_df.columns:
        if equivalence_source not in tost_df.columns:
            raise ValueError(
                "Nel file TOST manca sia la colonna "
                f"'{equivalence_source}' sia la colonna 'equivalent'."
            )

        tost_df = tost_df.rename(
            columns={equivalence_source: "equivalent"}
        )

    required_cols = [
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
        "metric",
        "equivalent",
    ]

    missing_cols = [
        column for column in required_cols if column not in tost_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Nel file TOST mancano queste colonne: {missing_cols}\n"
            "Rigenera il file TOST con lo script aggiornato."
        )

    # mean_full è opzionale ma molto utile per inferire il dataset legacy.
    keep_cols = required_cols.copy()
    if "mean_full" in tost_df.columns:
        keep_cols.append("mean_full")

    tost_keep = tost_df[keep_cols].copy()

    tost_keep["scorer"] = tost_keep["scorer"].apply(resolve_scorer_name)
    tost_keep = tost_keep[tost_keep["scorer"].notna()].copy()
    tost_keep = tost_keep[
        tost_keep["scorer"].isin(active_scorers)
    ].copy()

    tost_keep["qrels_variant"] = tost_keep[
        "qrels_variant"
    ].apply(normalize_qrels_variant)

    # Nel CSV TOST legacy, le righe senza qrels_variant appartengono
    # a dev.small. Questa retrocompatibilità viene applicata SOLO al TOST:
    # i CSV delle performance devono invece avere il dataset nel filename.
    missing_qrels_mask = tost_keep["qrels_variant"].isna()
    missing_qrels_count = int(missing_qrels_mask.sum())

    if missing_qrels_count > 0:
        print(
            f"[INFO] Assegno dev.small a {missing_qrels_count} "
            "righe TOST legacy senza qrels_variant."
        )
        tost_keep.loc[
            missing_qrels_mask,
            "qrels_variant",
        ] = "dev.small"

    tost_keep["threshold"] = tost_keep[
        "threshold"
    ].apply(normalize_threshold)

    tost_keep["pipeline"] = tost_keep[
        "pipeline"
    ].apply(normalize_pipeline)

    tost_keep["metric"] = tost_keep[
        "metric"
    ].apply(normalize_metric)

    tost_keep["equivalent"] = tost_keep[
        "equivalent"
    ].apply(parse_bool_value)

    if "mean_full" in tost_keep.columns:
        tost_keep["mean_full"] = pd.to_numeric(
            tost_keep["mean_full"],
            errors="coerce",
        )

    missing_count = int(tost_keep["qrels_variant"].isna().sum())
    print(
        f"[INFO] Righe TOST caricate: {len(tost_keep)} "
        f"(senza qrels_variant: {missing_count})"
    )

    return tost_keep


# =========================================================
# VALIDAZIONE E DATAFRAME LONG
# =========================================================

def validate_performance_duplicates(perf_df: pd.DataFrame) -> None:
    """
    Impedisce che due CSV diversi producano due punti per la stessa
    combinazione scorer/qrels/threshold/pipeline.
    """
    duplicate_keys = [
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
    ]

    duplicates = perf_df[
        perf_df.duplicated(subset=duplicate_keys, keep=False)
    ].sort_values(duplicate_keys)

    if duplicates.empty:
        return

    print("\n[ERROR] Risultati duplicati trovati:")
    print(
        duplicates[
            duplicate_keys + ["source_file"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Sono presenti più CSV per la stessa combinazione "
        "scorer/qrels/threshold/pipeline. "
        "Elimina o sposta i CSV duplicati prima di creare i grafici."
    )


def infer_missing_tost_qrels(
    tost_keep: pd.DataFrame,
    plot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Associa le righe TOST legacy senza qrels_variant a un dataset soltanto
    quando il match con i risultati è univoco.

    Criteri:
      1. scorer
      2. threshold
      3. pipeline
      4. metric
      5. mean_full, quando disponibile

    Questo recupera correttamente i TOST di dev.small senza attribuire
    indiscriminatamente ogni valore mancante a dev.small.
    """
    resolved = tost_keep.copy()

    missing_indices = resolved.index[
        resolved["qrels_variant"].isna()
    ].tolist()

    if not missing_indices:
        return resolved

    inferred = 0
    unresolved_rows = []

    for row_index in missing_indices:
        row = resolved.loc[row_index]

        candidates = plot_df[
            (plot_df["scorer"] == row["scorer"])
            & (
                np.isclose(
                    plot_df["threshold"].astype(float),
                    float(row["threshold"]),
                    rtol=0.0,
                    atol=1e-10,
                )
            )
            & (plot_df["pipeline"] == row["pipeline"])
            & (plot_df["metric"] == row["metric"])
        ].copy()

        # Quando il TOST contiene mean_full, usalo per distinguere i dataset.
        if (
            "mean_full" in resolved.columns
            and pd.notna(row.get("mean_full", np.nan))
            and not candidates.empty
        ):
            candidate_full = pd.to_numeric(
                candidates["mean_full"],
                errors="coerce",
            )

            candidates = candidates[
                np.isclose(
                    candidate_full,
                    float(row["mean_full"]),
                    rtol=1e-7,
                    atol=1e-10,
                    equal_nan=False,
                )
            ].copy()

        candidate_qrels = sorted(
            candidates["qrels_variant"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(candidate_qrels) == 1:
            resolved.at[row_index, "qrels_variant"] = candidate_qrels[0]
            inferred += 1
        else:
            unresolved_rows.append({
                "scorer": row["scorer"],
                "threshold": row["threshold"],
                "pipeline": row["pipeline"],
                "metric": row["metric"],
                "mean_full": row.get("mean_full", np.nan),
                "candidate_qrels": "|".join(candidate_qrels),
            })

    print(
        f"[INFO] Qrels TOST legacy inferite con match univoco: "
        f"{inferred}/{len(missing_indices)}"
    )

    if unresolved_rows:
        unresolved_df = pd.DataFrame(unresolved_rows)

        print(
            "[WARNING] Alcune righe TOST senza qrels_variant non sono "
            "associabili in modo univoco e verranno escluse:"
        )
        print(unresolved_df.head(20).to_string(index=False))

    return resolved[
        resolved["qrels_variant"].notna()
    ].copy()


def prepare_tost_for_merge(tost_keep: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove duplicati identici dal TOST e blocca duplicati conflittuali.
    """
    merge_keys = [
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
        "metric",
    ]

    duplicated = tost_keep[
        tost_keep.duplicated(subset=merge_keys, keep=False)
    ].copy()

    if not duplicated.empty:
        conflict_counts = (
            duplicated.groupby(merge_keys, dropna=False)["equivalent"]
            .nunique()
        )

        conflicts = conflict_counts[conflict_counts > 1]

        if not conflicts.empty:
            raise ValueError(
                "Nel TOST esistono righe duplicate con valori di equivalenza "
                "conflittuali per le stesse chiavi:\n"
                f"{conflicts.to_string()}"
            )

    return tost_keep.drop_duplicates(
        subset=merge_keys,
        keep="first",
    ).copy()


def make_long_plot_dataframe(
    perf_df: pd.DataFrame,
    tost_keep: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converte i risultati in formato lungo e aggiunge l'informazione TOST.

    mean_full arriva dai CSV compare_all_*.csv, non dal TOST.
    """
    long_rows: List[Dict[str, Any]] = []

    for _, row in perf_df.iterrows():
        for metric in METRICS:
            long_rows.append({
                "source_file": row["source_file"],
                "scorer": row["scorer"],
                "qrels_variant": normalize_qrels_variant(
                    row["qrels_variant"]
                ),
                "threshold": normalize_threshold(row["threshold"]),
                "pruning_percent": float(row["pruning_percent"]),
                "pipeline": normalize_pipeline(row["pipeline"]),
                "metric": normalize_metric(metric),
                "value": to_optional_float(row[metric]),
                "mean_full": to_optional_float(row[f"full_{metric}"]),
            })

    plot_df = pd.DataFrame(long_rows)

    if plot_df.empty:
        return plot_df

    missing_qrels = plot_df["qrels_variant"].isna()
    if missing_qrels.any():
        print(
            "[WARNING] Elimino "
            f"{int(missing_qrels.sum())} righe performance senza "
            "qrels_variant. Non vengono interpretate come dev.small."
        )
        plot_df = plot_df[~missing_qrels].copy()

    # Recupera in modo sicuro le qrels mancanti dei TOST legacy.
    tost_resolved = infer_missing_tost_qrels(
        tost_keep=tost_keep,
        plot_df=plot_df,
    )

    tost_for_merge = prepare_tost_for_merge(tost_resolved)

    merge_keys = [
        "scorer",
        "qrels_variant",
        "threshold",
        "pipeline",
        "metric",
    ]

    plot_df = plot_df.merge(
        tost_for_merge[merge_keys + ["equivalent"]],
        on=merge_keys,
        how="left",
        validate="many_to_one",
        indicator="_tost_merge",
    )

    matched_count = int((plot_df["_tost_merge"] == "both").sum())
    unmatched_count = int((plot_df["_tost_merge"] == "left_only").sum())

    print(
        f"[INFO] Match TOST completati: {matched_count}; "
        f"righe performance senza match TOST: {unmatched_count}"
    )

    # Diagnostica specifica per dev.small / RR@10.
    dev_mask = (
        plot_df["qrels_variant"].astype(str).str.lower().eq("dev.small")
        & plot_df["metric"].eq("RR@10")
    )

    dev_total = int(dev_mask.sum())
    dev_matched = int(
        (dev_mask & plot_df["_tost_merge"].eq("both")).sum()
    )

    print(
        f"[INFO] Match TOST dev.small / RR@10: "
        f"{dev_matched}/{dev_total}"
    )

    if dev_total > 0 and dev_matched < dev_total:
        unmatched_dev = plot_df[
            dev_mask & plot_df["_tost_merge"].eq("left_only")
        ][
            [
                "scorer",
                "threshold",
                "pipeline",
                "metric",
                "qrels_variant",
            ]
        ].drop_duplicates()

        print(
            "[WARNING] Chiavi dev.small / RR@10 senza match nel TOST:"
        )
        print(unmatched_dev.head(30).to_string(index=False))

    # parse_bool_value gestisce già NaN/None come False.
    # Evita il FutureWarning causato da fillna(False) su dtype object.
    plot_df["equivalent"] = (
        plot_df["equivalent"]
        .apply(parse_bool_value)
        .astype(bool)
    )

    # Riepilogo dei punti equivalenti effettivamente disponibili.
    equivalent_summary = (
        plot_df[plot_df["equivalent"]]
        .groupby(
            ["qrels_variant", "metric", "pipeline"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_equivalent")
    )

    if equivalent_summary.empty:
        print("[WARNING] Nessun punto TOST equivalent associato ai plot.")
    else:
        print("\n[INFO] Punti TOST equivalent associati:")
        print(equivalent_summary.to_string(index=False))

    plot_df = plot_df.drop(columns=["_tost_merge"])

    return plot_df


# =========================================================
# PLOT
# =========================================================

def choose_metric(qrels_val: str, force_rr10: bool = False) -> str:
    if force_rr10:
        return "RR@10"

    qrels_low = str(qrels_val).lower()

    if "dev.small" in qrels_low or "dev_small" in qrels_low:
        return "RR@10"

    return "nDCG@10"


def get_unique_full_value(
    sub: pd.DataFrame,
    pipeline: str,
    qrels_variant: str,
    metric: str,
) -> float:
    """
    La baseline Full deve essere unica per dataset, pipeline e metrica.

    Non dipende dallo scorer usato per il pruning.
    """
    full_values = (
        pd.to_numeric(sub["mean_full"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(full_values) == 0:
        raise ValueError(
            "Baseline Full mancante per "
            f"pipeline={pipeline}, qrels={qrels_variant}, metric={metric}."
        )

    reference = float(full_values[0])

    if not np.allclose(
        full_values,
        reference,
        rtol=1e-8,
        atol=1e-10,
    ):
        unique_values = sorted(set(full_values.tolist()))

        raise ValueError(
            "Baseline Full non coerenti per "
            f"pipeline={pipeline}, qrels={qrels_variant}, metric={metric}: "
            f"{unique_values}. "
            "Probabile riuso di run appartenenti a dataset differenti."
        )

    return reference


def plot_all(
    plot_df: pd.DataFrame,
    scorer_order: List[str],
    scorer_labels: Dict[str, str],
    scorer_styles: Dict[str, Dict[str, Any]],
    res_dir: str,
    force_rr10: bool = False,
) -> None:
    for qrels_val in sorted(
        plot_df["qrels_variant"].dropna().unique()
    ):
        metric_to_plot = choose_metric(
            qrels_val,
            force_rr10=force_rr10,
        )

        print(
            f"\n[INFO] Generazione grafici per QRELS: {qrels_val} "
            f"- Metrica: {metric_to_plot}"
        )

        qrels_sub = plot_df[
            (plot_df["qrels_variant"] == qrels_val)
            & (plot_df["metric"] == metric_to_plot)
        ].copy()

        qrels_sub["value"] = pd.to_numeric(
            qrels_sub["value"],
            errors="coerce",
        )
        qrels_sub["mean_full"] = pd.to_numeric(
            qrels_sub["mean_full"],
            errors="coerce",
        )

        qrels_sub = qrels_sub[
            qrels_sub["value"].notna()
        ].copy()

        if qrels_sub.empty:
            print(
                f"[WARNING] Nessun dato valido trovato per "
                f"{metric_to_plot} su {qrels_val}. Salto."
            )
            continue

        for pipeline in PIPELINE_ORDER:
            sub = qrels_sub[
                qrels_sub["pipeline"] == pipeline
            ].copy()

            if sub.empty:
                continue

            full_val = get_unique_full_value(
                sub=sub,
                pipeline=pipeline,
                qrels_variant=qrels_val,
                metric=metric_to_plot,
            )

            plt.figure(figsize=(9.6, 5.8))
            ax = plt.gca()

            x_max = max(
                [0.0]
                + sub["pruning_percent"].dropna().astype(float).tolist()
            )

            full_line, = ax.plot(
                [0.0, x_max],
                [full_val, full_val],
                color="black",
                linestyle="--",
                linewidth=1.4,
            )
            full_line.set_dashes((6, 3))

            present_scorers: List[str] = []

            for scorer in scorer_order:
                scorer_sub = sub[
                    sub["scorer"] == scorer
                ].copy()

                if scorer_sub.empty:
                    continue

                scorer_sub = scorer_sub.sort_values(
                    ["pruning_percent", "threshold"]
                )

                # Non deve esistere più di un punto per la stessa percentuale.
                duplicated_x = scorer_sub[
                    scorer_sub.duplicated(
                        subset=["pruning_percent"],
                        keep=False,
                    )
                ]

                if not duplicated_x.empty:
                    raise ValueError(
                        "Più punti per lo stesso scorer e la stessa "
                        "percentuale di pruning:\n"
                        f"{duplicated_x[['source_file', 'scorer', 'qrels_variant', 'pipeline', 'pruning_percent']].to_string(index=False)}"
                    )

                present_scorers.append(scorer)

                x_vals = [0.0] + scorer_sub[
                    "pruning_percent"
                ].astype(float).tolist()

                y_vals = [full_val] + scorer_sub[
                    "value"
                ].astype(float).tolist()

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

                equivalent_sub = scorer_sub[
                    scorer_sub["equivalent"]
                ]

                if not equivalent_sub.empty:
                    ax.scatter(
                        equivalent_sub["pruning_percent"],
                        equivalent_sub["value"],
                        s=180,
                        facecolors=style["color"],
                        alpha=0.5,
                        edgecolors=style["color"],
                        linewidths=1.5,
                        zorder=5,
                    )

            ax.set_xlabel("Pruning percentage")
            ax.set_ylabel(metric_to_plot)
            ax.set_title(
                f"{pipeline} – {metric_to_plot} ({qrels_val})"
            )

            xticks = [0.0] + sorted(
                sub["pruning_percent"]
                .dropna()
                .astype(float)
                .unique()
                .tolist()
            )

            ax.set_xticks(xticks)
            ax.set_xticklabels([
                f"{int(x)}%" if float(x).is_integer() else f"{x:g}%"
                for x in xticks
            ])

            legend_handles: List[Line2D] = []

            full_handle = Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.8,
                linestyle="--",
                marker=None,
                label="Full",
            )
            full_handle.set_dashes((6, 3))
            legend_handles.append(full_handle)

            for scorer in present_scorers:
                style = scorer_styles[scorer]

                scorer_handle = Line2D(
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
                    scorer_handle.set_dashes(style["dashes"])

                legend_handles.append(scorer_handle)

            equivalence_handle = Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=12,
                alpha=0.5,
                label="TOST equivalent",
            )
            legend_handles.append(equivalence_handle)

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
            pipeline_clean = (
                pipeline.lower()
                .replace("-", "")
                .replace(" ", "_")
            )
            qrels_clean = (
                str(qrels_val)
                .replace("/", "_")
                .replace(" ", "_")
                .replace(".", "_")
            )

            output_path = os.path.join(
                res_dir,
                (
                    f"{metric_clean}_pruning_plot_"
                    f"{pipeline_clean}_{qrels_clean}_"
                    "selected_scorers.png"
                ),
            )

            plt.savefig(
                output_path,
                dpi=220,
                bbox_inches="tight",
            )
            plt.close()

            print(f"[INFO] Salvato: {output_path}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
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
                f"  - {scorer_name:32s} "
                f"enabled={str(enabled):5s} "
                f"group={str(group):16s} "
                f"label={label} "
                f"aliases=[{aliases}]"
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

    summary_files = sorted(
        glob.glob(
            os.path.join(
                args.base_dir,
                "compare_all_*.csv",
            )
        )
    )

    summary_files = [
        file_path
        for file_path in summary_files
        if "_perquery_" not in os.path.basename(file_path)
        and "_timings" not in os.path.basename(file_path)
    ]

    print(
        f"\n[INFO] File summary candidati trovati: "
        f"{len(summary_files)}"
    )

    perf_df = load_performance_dataframe(
        summary_files=summary_files,
        active_scorers=active_scorers,
    )

    if perf_df.empty:
        raise ValueError(
            "Nessun risultato trovato per gli scorer selezionati.\n"
            f"Scorer selezionati: {active_scorers}\n"
            f"Directory risultati: {args.base_dir}\n"
            "Controlla i nomi dei file compare_all_*.csv, "
            "la presenza delle righe Full/Pruned e gli alias "
            "in scorer_config.py."
        )

    validate_performance_duplicates(perf_df)

    tost_path = os.path.join(
        args.res_dir,
        "tost_pruning_noninferiority_5pct_full_vs_pruned.csv",
    )

    tost_keep = load_tost_dataframe(
        tost_path=tost_path,
        active_scorers=active_scorers,
    )

    plot_df = make_long_plot_dataframe(
        perf_df=perf_df,
        tost_keep=tost_keep,
    )

    if plot_df.empty:
        raise ValueError(
            "Il DataFrame plot-ready è vuoto dopo normalizzazione e merge."
        )

    available_scorers = (
        plot_df["scorer"]
        .dropna()
        .unique()
        .tolist()
    )

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

    print("\n[INFO] Qrels disponibili:")
    for qrels_variant in sorted(
        plot_df["qrels_variant"].dropna().unique()
    ):
        selected_metric = choose_metric(
            qrels_variant,
            force_rr10=(
                FORCE_RR10 or args.force_rr10
            ),
        )
        print(
            f"  - {qrels_variant}: "
            f"metrica plottata = {selected_metric}"
        )

    plot_all(
        plot_df=plot_df,
        scorer_order=scorer_order,
        scorer_labels=scorer_labels,
        scorer_styles=scorer_styles,
        res_dir=args.res_dir,
        force_rr10=(
            FORCE_RR10 or args.force_rr10
        ),
    )

    print("\n[INFO] Creati i grafici e il dataset plot-ready:")
    print(plot_ready_out)


if __name__ == "__main__":
    main()