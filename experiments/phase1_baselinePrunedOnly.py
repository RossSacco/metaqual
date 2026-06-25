import os
import sys
import re
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import yaml
import torch
import pyterrier as pt
import pyterrier_dr
import pyterrier_pisa
import pyt_splade
import ir_measures
import pandas as pd

from metaqual.data.loaders.msmarco.dataset_loader import DatasetLoader
from metaqual.retrieval.pyterrier_pipe import RetrievalPipelines


DEFAULT_SCORERS = [
    "tasb",
    "perplexity",
    "itn",
    "cdd",
    "finetuned_qualt5",
    "metadata_qualt5",
]
SUPPORTED_SCORERS = DEFAULT_SCORERS


# =========================================================
# UTILS
# =========================================================

def get_qrels_name(config: Dict[str, Any]) -> str:
    """
    Se qrels_variant è una lista, per esempio ["test-2019", "test-2020"],
    la converte in "test-2019_test-2020".
    """
    variant = config["dataset"]["qrels_variant"]
    if isinstance(variant, list):
        return "_".join(str(v) for v in variant)
    return str(variant)


def get_index_scorer_name(config: Dict[str, Any], scorer_name: str) -> str:
    """
    Nome fisico dello scorer usato per cercare la cartella degli indici pruned.

    Esempio:
        scorer_name = "metadata_qualt5"
        experiment.index_scorer_name = "metadata_qualt5_MP"

    Allora lo script userà:
        /data/data-sacco/indexes/metadata_qualt5_MP_pruned_0.15

    ma nei risultati continuerà a salvare lo scorer logico:
        metadata_qualt5
    """
    exp_cfg = config.get("experiment", {})

    # Caso semplice: un solo scorer attivo.
    if "index_scorer_name" in exp_cfg:
        return str(exp_cfg["index_scorer_name"])

    # Caso avanzato: scorer="all" con mapping.
    # Esempio:
    # index_scorer_names:
    #   metadata_qualt5: metadata_qualt5_MP
    #   finetuned_qualt5: finetuned_qualt5
    mapping = exp_cfg.get("index_scorer_names", {})
    if isinstance(mapping, dict) and scorer_name in mapping:
        return str(mapping[scorer_name])

    return scorer_name


def ensure_pyterrier_started() -> None:
    print("[DEBUG] Inizializzazione di PyTerrier/Java...")
    if not pt.java.started():
        pt.java.init()
    print("[DEBUG] PyTerrier inizializzato correttamente.")


def ensure_path_exists(path: str, description: str = "Path") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} non trovato: {path}")


def clean_query_for_terrier(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"[:(){}\[\]^\"~*?\\/+!\-]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def move_model_to_cuda(model: Any, model_name: str = "model") -> Any:
    print(f"\n--- [DEBUG] Procedura GPU per: {model_name} ---")

    if not torch.cuda.is_available():
        print("[WARNING] CUDA non disponibile: verrà usata la CPU.")
        return model

    moved = False
    candidate_attrs = [None, "model", "_model", "encoder", "_encoder"]

    for attr in candidate_attrs:
        target = model if attr is None else getattr(model, attr, None)
        if target is not None and hasattr(target, "to"):
            try:
                target.to("cuda")
                moved = True
            except Exception:
                pass

    if hasattr(model, "device"):
        try:
            model.device = torch.device("cuda")
            moved = True
            print(f"[DEBUG] Attributo 'device' di {model_name} forzato a CUDA.")
        except Exception:
            pass

    verified = False
    for attr in candidate_attrs:
        target = model if attr is None else getattr(model, attr, None)
        if target is not None and hasattr(target, "parameters"):
            try:
                first_param = next(target.parameters())
                print(f"[DEBUG] Device reale dei parametri di {model_name}: {first_param.device}")
                verified = True
                break
            except Exception:
                pass

    if moved:
        print(f"[INFO] 🟢 {model_name} configurato per GPU.")
    else:
        print(f"[WARNING] 🔴 Non sono riuscito a configurare {model_name} su GPU.")

    if not verified:
        print(f"[WARNING] Impossibile verificare il device reale dei parametri di {model_name}.")

    return model


def load_config(config_path: str) -> Dict[str, Any]:
    ensure_path_exists(config_path, "File di configurazione")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict[str, Any]) -> None:
    scorer_name = config["experiment"]["scorer"]
    active_retriever = config["experiment"].get("retriever", "all").lower()

    valid_retrievers = {"bm25", "splade", "tasb", "all"}
    if active_retriever not in valid_retrievers:
        raise ValueError(
            f"Retriever non valido: {active_retriever}. "
            f"Valori ammessi: {sorted(valid_retrievers)}"
        )

    if scorer_name != "all" and scorer_name not in SUPPORTED_SCORERS:
        raise ValueError(
            f"Scorer non valido: {scorer_name}. "
            f"Valori ammessi: {SUPPORTED_SCORERS + ['all']}"
        )


def get_eval_metrics() -> List[Any]:
    return [
        ir_measures.RR@10,
        ir_measures.nDCG@10,
        ir_measures.Recall@100,
    ]


def get_base_runs_dir(config: Dict[str, Any]) -> str:
    return config["paths"].get("runs_dir", config["paths"]["results_dir"])


def get_full_runs_dir(config: Dict[str, Any]) -> str:
    """
    Directory stabile per i full.
    Se nel config è presente paths.full_runs_dir, usa quella.
    Altrimenti crea una cartella standard.
    """
    dataset_name = config["dataset"]["name"]
    active_retriever = config["experiment"].get("retriever", "all").lower()
    base_runs_dir = get_base_runs_dir(config)

    full_runs_dir = config["paths"].get(
        "full_runs_dir",
        os.path.join(base_runs_dir, f"runs_full_{dataset_name}_{active_retriever}")
    )

    os.makedirs(full_runs_dir, exist_ok=True)
    return full_runs_dir


def get_pruned_runs_dir(config: Dict[str, Any], scorer_name: str) -> str:
    """
    Directory per salvare/riusare le run pruned.

    Usa lo stesso nome fisico usato per l'indice, cioè index_scorer_name.
    Esempio:
        scorer_name = metadata_qualt5
        index_scorer_name = metadata_qualt5_MP

    Allora salva in:
        runs_all_metadata_qualt5_MP_test-2019_test-2020_0.6
    """
    active_retriever = config["experiment"].get("retriever", "all").lower()
    threshold = config["experiment"]["threshold"]
    qrels_name = get_qrels_name(config)
    base_runs_dir = get_base_runs_dir(config)

    index_scorer_name = get_index_scorer_name(config, scorer_name)

    pruned_runs_dir = os.path.join(
        base_runs_dir,
        f"runs_{active_retriever}_{index_scorer_name}_{qrels_name}_{threshold}"
    )

    os.makedirs(pruned_runs_dir, exist_ok=True)
    return pruned_runs_dir


# =========================================================
# DATASET
# =========================================================

def prepare_topics_and_qrels(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_name = config["dataset"]["name"]
    topics_variants = config["dataset"]["topics_variant"]
    qrels_variants = config["dataset"]["qrels_variant"]

    if isinstance(topics_variants, str):
        topics_variants = [topics_variants]
    if isinstance(qrels_variants, str):
        qrels_variants = [qrels_variants]

    print(f"\n[DEBUG] Caricamento dataset ({dataset_name}) per varianti topics: {topics_variants}")
    print(f"[DEBUG] Caricamento qrels per varianti: {qrels_variants}")

    loader = DatasetLoader(dataset_name)

    topics_list = []
    qrels_list = []

    for tv in topics_variants:
        topics_list.append(loader.get_topics(tv).copy())

    for qv in qrels_variants:
        qrels_list.append(loader.get_qrels(qv).copy())

    topics = pd.concat(topics_list, ignore_index=True)
    qrels = pd.concat(qrels_list, ignore_index=True)

    topics = topics.drop_duplicates(subset=["qid"])

    if "docno" in qrels.columns:
        qrels = qrels.drop_duplicates(subset=["qid", "docno"])

    topics = topics[topics["qid"].isin(qrels["qid"])].copy()
    topics["query"] = topics["query"].apply(clean_query_for_terrier)
    topics = topics[topics["query"].astype(str).str.len() > 0].copy()

    print(f"[DEBUG] Totale query unificate: {len(topics)}")
    print(f"[DEBUG] Totale qrels unificate: {len(qrels)}")

    return topics, qrels


# =========================================================
# PIPELINES - FULL
# =========================================================

def build_full_systems(config: Dict[str, Any]) -> Tuple[List[Any], List[str]]:
    active_retriever = config["experiment"].get("retriever", "all").lower()
    dataset_name = config["dataset"]["name"]
    indexes_dir = config["paths"]["indexes_dir"]

    systems = []
    names = []

    # BM25 FULL
    if active_retriever in ["bm25", "all"]:
        print("\n[DEBUG] Preparazione pipeline BM25 FULL...")
        pipe_bm25_full = pt.terrier.Retriever.from_dataset(
            dataset_name,
            "terrier_stemmed",
            wmodel="BM25",
            verbose=True
        )
        systems.append(pipe_bm25_full)
        names.append("BM25 Full")

    # SPLADE FULL
    if active_retriever in ["splade", "all"]:
        print("\n[INFO] Caricamento modello SPLADE FULL...")
        splade_model = pyt_splade.Splade("naver/efficient-splade-VI-BT-large-doc")
        splade_model = move_model_to_cuda(splade_model, "SPLADE Full")
        encoder_gpu = splade_model.query_encoder(batch_size=32, verbose=True)

        full_splade_index_path = os.path.join(indexes_dir, "splade_full")
        ensure_path_exists(full_splade_index_path, "Indice SPLADE full")

        pipe_splade_full = (
            encoder_gpu
            >> pyterrier_pisa.PisaIndex(full_splade_index_path, stemmer="none").quantized()
        ) % 100

        systems.append(pipe_splade_full)
        names.append("SPLADE Full")

    # TAS-B FULL
    if active_retriever in ["tasb", "all"]:
        print("\n[INFO] Caricamento modello TAS-B FULL...")
        tasb_model = pyterrier_dr.TasB.dot()
        tasb_model = move_model_to_cuda(tasb_model, "TAS-B Full")

        print("[DEBUG] Caricamento FlexIndex per TAS-B FULL da HuggingFace...")
        full_tasb_index = pyterrier_dr.FlexIndex.from_hf(
            "macavaney/msmarco-passage.tasb.flex"
        )

        tasb_encoder_gpu = tasb_model.query_encoder(batch_size=64, verbose=True)
        pipe_tasb_full = tasb_encoder_gpu >> full_tasb_index

        systems.append(pipe_tasb_full)
        names.append("TAS-B Full")

    if not systems:
        raise ValueError(f"Nessun sistema FULL caricato per retriever={active_retriever}.")

    return systems, names


# =========================================================
# PIPELINES - PRUNED
# =========================================================

def build_pruned_systems(
    config: Dict[str, Any],
    scorer_name: str,
) -> Tuple[List[Any], List[str], str]:
    threshold = config["experiment"]["threshold"]
    active_retriever = config["experiment"].get("retriever", "all").lower()
    indexes_dir = config["paths"]["indexes_dir"]

    index_scorer_name = get_index_scorer_name(config, scorer_name)

    pruned_root = os.path.join(
        indexes_dir,
        f"{index_scorer_name}_pruned_{threshold}"
    )

    ensure_path_exists(
        pruned_root,
        (
            f"Cartella root degli indici pruned per scorer logico={scorer_name}, "
            f"index_scorer_name={index_scorer_name}"
        )
    )

    print("\n[DEBUG] Indice pruned selezionato:")
    print(f"  - scorer logico: {scorer_name}")
    print(f"  - index_scorer_name fisico: {index_scorer_name}")
    print(f"  - pruned_root: {pruned_root}")

    systems = []
    names = []

    # BM25 PRUNED
    if active_retriever in ["bm25", "all"]:
        print("\n[DEBUG] Preparazione pipeline BM25 PRUNED...")
        bm25_pruned_path = os.path.join(pruned_root, "pisa_bm25")
        ensure_path_exists(bm25_pruned_path, "Indice BM25 pruned")
        pipe_bm25_p = RetrievalPipelines(bm25_pruned_path).get_bm25()

        systems.append(pipe_bm25_p)
        names.append("BM25 Pruned")

    # SPLADE PRUNED
    if active_retriever in ["splade", "all"]:
        print("\n[INFO] Caricamento modello SPLADE PRUNED...")
        splade_model = pyt_splade.Splade("naver/efficient-splade-VI-BT-large-doc")
        splade_model = move_model_to_cuda(splade_model, "SPLADE Pruned")

        encoder_gpu = splade_model.query_encoder(batch_size=32, verbose=True)

        splade_pruned_path = os.path.join(pruned_root, "pisa_splade")
        ensure_path_exists(splade_pruned_path, "Indice SPLADE pruned")

        pipe_splade_p = (
            encoder_gpu
            >> pyterrier_pisa.PisaIndex(splade_pruned_path, stemmer="none").quantized()
        ) % 100

        systems.append(pipe_splade_p)
        names.append("SPLADE Pruned")

    # TAS-B PRUNED
    if active_retriever in ["tasb", "all"]:
        print("\n[INFO] Caricamento modello TAS-B PRUNED...")
        tasb_model = pyterrier_dr.TasB.dot()
        tasb_model = move_model_to_cuda(tasb_model, "TAS-B Pruned")

        tasb_pruned_path = os.path.join(pruned_root, "tasb.flex")
        ensure_path_exists(tasb_pruned_path, "Indice TAS-B pruned")

        pipe_tasb_p = RetrievalPipelines(
            tasb_pruned_path,
            query_encoder=tasb_model
        ).get_tasb()

        systems.append(pipe_tasb_p)
        names.append("TAS-B Pruned")

    if not systems:
        raise ValueError(
            f"Nessun sistema PRUNED caricato. "
            f"Verifica retriever={active_retriever}, scorer={scorer_name}, pruned_root={pruned_root}."
        )

    return systems, names, pruned_root


# =========================================================
# EXPERIMENT + TIMINGS
# =========================================================

def _add_timing_columns_to_avg(
    avg_df: pd.DataFrame,
    system_name: str,
    elapsed_seconds: float,
    num_queries: int,
) -> pd.DataFrame:
    """
    Aggiunge le metriche temporali al DataFrame medio prodotto da pt.Experiment.
    """
    df = avg_df.copy()

    if "name" not in df.columns:
        df["name"] = system_name

    df["runtime_seconds"] = elapsed_seconds

    if num_queries > 0:
        df["ms_per_query"] = (elapsed_seconds * 1000.0) / num_queries
        df["queries_per_second"] = num_queries / elapsed_seconds if elapsed_seconds > 0 else float("inf")
    else:
        df["ms_per_query"] = None
        df["queries_per_second"] = None

    return df


def _make_timing_row(
    system_name: str,
    save_dir: str,
    elapsed_seconds: float,
    num_queries: int,
) -> Dict[str, Any]:
    """
    Riga compatta per il CSV separato dei tempi.
    """
    if num_queries > 0:
        ms_per_query = (elapsed_seconds * 1000.0) / num_queries
        qps = num_queries / elapsed_seconds if elapsed_seconds > 0 else float("inf")
    else:
        ms_per_query = None
        qps = None

    return {
        "name": system_name,
        "save_dir": save_dir,
        "num_queries": num_queries,
        "runtime_seconds": elapsed_seconds,
        "ms_per_query": ms_per_query,
        "queries_per_second": qps,
    }


def run_cached_experiment(
    systems: List[Any],
    names: List[str],
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
    save_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Esegue pt.Experiment sistema per sistema per poter misurare i tempi separatamente.

    Restituisce:
    - res_avg: metriche medie + colonne di tempo
    - res_perq: metriche per-query
    - timings_df: solo tempi, comodo per analisi separate

    Nota importante:
    save_mode="reuse" rimane attivo. Quindi il tempo misurato può essere:
    - tempo reale di retrieval/evaluation, se la run non esisteva;
    - tempo di riuso/lettura/evaluation, se la run era già salvata.
    """
    print("\n========================================================")
    print(f"[INFO] Avvio / riuso metriche per i sistemi: {', '.join(names)}")
    print(f"[INFO] save_dir: {save_dir}")
    print("========================================================\n")

    os.makedirs(save_dir, exist_ok=True)

    all_avg = []
    all_perq = []
    timing_rows = []

    num_queries = len(topics)

    for system, name in zip(systems, names):
        print("\n" + "-" * 80)
        print(f"[INFO] Avvio sistema: {name}")
        print("-" * 80)

        start = time.perf_counter()

        avg_df, perq_df = pt.Experiment(
            [system],
            topics,
            qrels,
            eval_metrics=get_eval_metrics(),
            names=[name],
            verbose=True,
            save_dir=save_dir,
            save_mode="reuse",
            perquery="both",
        )

        elapsed = time.perf_counter() - start

        print(f"[TIME] {name}: {elapsed:.3f} secondi")
        if num_queries > 0:
            print(f"[TIME] {name}: {(elapsed * 1000.0) / num_queries:.3f} ms/query")
            print(f"[TIME] {name}: {num_queries / elapsed:.3f} query/s")

        avg_df = _add_timing_columns_to_avg(
            avg_df=avg_df,
            system_name=name,
            elapsed_seconds=elapsed,
            num_queries=num_queries,
        )

        timing_rows.append(
            _make_timing_row(
                system_name=name,
                save_dir=save_dir,
                elapsed_seconds=elapsed,
                num_queries=num_queries,
            )
        )

        all_avg.append(avg_df)
        all_perq.append(perq_df)

    res_avg = pd.concat(all_avg, ignore_index=True) if all_avg else pd.DataFrame()
    res_perq = pd.concat(all_perq, ignore_index=True) if all_perq else pd.DataFrame()
    timings_df = pd.DataFrame(timing_rows)

    return res_avg, res_perq, timings_df


def get_or_compute_full_results(
    config: Dict[str, Any],
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_runs_dir = get_full_runs_dir(config)
    full_systems, full_names = build_full_systems(config)

    return run_cached_experiment(
        systems=full_systems,
        names=full_names,
        topics=topics,
        qrels=qrels,
        save_dir=full_runs_dir,
    )


def compute_pruned_results(
    config: Dict[str, Any],
    scorer_name: str,
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pruned_runs_dir = get_pruned_runs_dir(config, scorer_name)
    pruned_systems, pruned_names, _ = build_pruned_systems(config, scorer_name)

    return run_cached_experiment(
        systems=pruned_systems,
        names=pruned_names,
        topics=topics,
        qrels=qrels,
        save_dir=pruned_runs_dir,
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_pairwise_perquery_outputs(
    res_avg: pd.DataFrame,
    res_perq: pd.DataFrame,
    timings_df: pd.DataFrame,
    scorer_name: str,
    threshold: Any,
    active_retriever: str,
    qrels_variant: str,
    results_dir: str,
) -> None:
    """
    Salva:
    1. CSV medio complessivo, con metriche IR + tempi
    2. CSV per-query lungo
    3. CSV per-query pivotato
    4. CSV differenze Full - Pruned per query
    5. CSV solo tempi
    """
    os.makedirs(results_dir, exist_ok=True)

    prefix = f"{active_retriever}_{scorer_name}_{qrels_variant}_{threshold}"

    avg_path = os.path.join(results_dir, f"compare_{prefix}.csv")
    perq_long_path = os.path.join(results_dir, f"compare_{prefix}_perquery_long.csv")
    perq_wide_path = os.path.join(results_dir, f"compare_{prefix}_perquery_wide.csv")
    perq_diff_path = os.path.join(results_dir, f"compare_{prefix}_perquery_diffs.csv")
    timings_path = os.path.join(results_dir, f"compare_{prefix}_timings.csv")

    # 1. risultati medi + tempi
    res_avg.to_csv(avg_path, index=False)

    # 2. timing separati
    timings_df.to_csv(timings_path, index=False)

    # 3. risultati per-query lunghi
    res_perq.to_csv(perq_long_path, index=False)

    df = res_perq.copy()

    expected_cols = {"qid", "measure", "value", "name"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Le colonne per-query attese non sono presenti. Mancano: {missing}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    # 4. pivotato: una riga per qid, colonne tipo "BM25 Full__RR@10"
    df_wide = df.pivot_table(
        index="qid",
        columns=["name", "measure"],
        values="value"
    )

    df_wide.columns = [
        f"{sys_name}__{metric}"
        for sys_name, metric in df_wide.columns
    ]

    df_wide = df_wide.reset_index()
    df_wide.to_csv(perq_wide_path, index=False)

    # 5. differenze Full - Pruned per query
    diff_rows = []
    systems = sorted(df["name"].unique())

    families: Dict[str, Dict[str, str]] = {}

    for sys_name in systems:
        if sys_name.endswith(" Full"):
            fam = sys_name[:-5]
            families.setdefault(fam, {})["full"] = sys_name
        elif sys_name.endswith(" Pruned"):
            fam = sys_name[:-7]
            families.setdefault(fam, {})["pruned"] = sys_name

    for fam, pair in families.items():
        if "full" not in pair or "pruned" not in pair:
            continue

        full_name = pair["full"]
        pruned_name = pair["pruned"]

        full_df = df[df["name"] == full_name].copy()
        pruned_df = df[df["name"] == pruned_name].copy()

        merged = full_df.merge(
            pruned_df,
            on=["qid", "measure"],
            suffixes=("_full", "_pruned")
        )

        merged["system_family"] = fam
        merged["full_name"] = full_name
        merged["pruned_name"] = pruned_name
        merged["diff_full_minus_pruned"] = merged["value_full"] - merged["value_pruned"]
        merged["diff_pruned_minus_full"] = merged["value_pruned"] - merged["value_full"]

        diff_rows.append(
            merged[
                [
                    "qid",
                    "measure",
                    "system_family",
                    "full_name",
                    "pruned_name",
                    "value_full",
                    "value_pruned",
                    "diff_full_minus_pruned",
                    "diff_pruned_minus_full",
                ]
            ]
        )

    if diff_rows:
        df_diffs = pd.concat(diff_rows, ignore_index=True)
        df_diffs.to_csv(perq_diff_path, index=False)
    else:
        pd.DataFrame().to_csv(perq_diff_path, index=False)

    print(f"[INFO] ✅ Salvato: {avg_path}")
    print(f"[INFO] ✅ Salvato: {perq_long_path}")
    print(f"[INFO] ✅ Salvato: {perq_wide_path}")
    print(f"[INFO] ✅ Salvato: {perq_diff_path}")
    print(f"[INFO] ✅ Salvato: {timings_path}")


# =========================================================
# RUN
# =========================================================

def run_single_scorer_evaluation(
    config: Dict[str, Any],
    scorer_name: str,
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
) -> None:
    threshold = config["experiment"]["threshold"]
    active_retriever = config["experiment"].get("retriever", "all").lower()
    qrels_name = get_qrels_name(config)

    indexes_dir = config["paths"]["indexes_dir"]
    results_dir = config["paths"]["results_dir"]

    index_scorer_name = get_index_scorer_name(config, scorer_name)

    ensure_path_exists(indexes_dir, "Cartella indexes_dir")
    os.makedirs(results_dir, exist_ok=True)

    print("\n[DEBUG] Configurazione:")
    print(f"  - Scorer logico: {scorer_name}")
    print(f"  - Index scorer fisico: {index_scorer_name}")
    print(f"  - Threshold: {threshold}")
    print(f"  - Retriever attivo: {active_retriever.upper()}")
    print(f"  - Full runs dir: {get_full_runs_dir(config)}")
    print(f"  - Pruned runs dir: {get_pruned_runs_dir(config, scorer_name)}")
    print(
        "  - Pruned index root attesa: "
        f"{os.path.join(indexes_dir, f'{index_scorer_name}_pruned_{threshold}')}"
    )

    # FULL: calcolo o riuso da cartella stabile
    full_avg, full_perq, full_timings = get_or_compute_full_results(
        config,
        topics,
        qrels
    )

    # PRUNED: calcolo o riuso per scorer + threshold corrente
    pruned_avg, pruned_perq, pruned_timings = compute_pruned_results(
        config,
        scorer_name,
        topics,
        qrels
    )

    # Merge risultati
    res_avg = pd.concat([full_avg, pruned_avg], ignore_index=True)
    res_perq = pd.concat([full_perq, pruned_perq], ignore_index=True)
    timings_df = pd.concat([full_timings, pruned_timings], ignore_index=True)

    # Ordinamento opzionale per leggibilità
    if "name" in res_avg.columns:
        res_avg = res_avg.sort_values(by=["name"]).reset_index(drop=True)

    if {"name", "qid", "measure"}.issubset(res_perq.columns):
        res_perq = res_perq.sort_values(
            by=["name", "qid", "measure"]
        ).reset_index(drop=True)

    if "name" in timings_df.columns:
        timings_df = timings_df.sort_values(by=["name"]).reset_index(drop=True)

    save_pairwise_perquery_outputs(
        res_avg=res_avg,
        res_perq=res_perq,
        timings_df=timings_df,
        scorer_name=index_scorer_name,
        threshold=threshold,
        active_retriever=active_retriever,
        qrels_variant=qrels_name,
        results_dir=results_dir,
    )

    print("\n--- Risultati medi + tempi ---")
    print(res_avg.head())

    print("\n--- Timings ---")
    print(timings_df)

    print("\n--- Risultati per-query ---")
    print(res_perq.head())


def run_evaluation(config: Dict[str, Any]) -> None:
    ensure_pyterrier_started()
    validate_config(config)

    topics, qrels = prepare_topics_and_qrels(config)
    scorer_name = config["experiment"]["scorer"]

    if scorer_name == "all":
        for scorer in DEFAULT_SCORERS:
            print("\n" + "=" * 80)
            print(f"[INFO] Avvio esperimento per scorer: {scorer}")
            print("=" * 80 + "\n")
            run_single_scorer_evaluation(config, scorer, topics, qrels)
    else:
        run_single_scorer_evaluation(config, scorer_name, topics, qrels)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valutazione full vs pruned con cache separata dei full e timing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path del file YAML di configurazione"
    )

    args = parser.parse_args()

    print(f"[DEBUG] Avvio script con config: {args.config}")

    try:
        config = load_config(args.config)
        run_evaluation(config)
    except Exception as e:
        print(f"\n[ERROR] Lo script è terminato con errore: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
    
    
