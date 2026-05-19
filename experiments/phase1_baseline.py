import os
import sys
import re
import argparse
from typing import Any, Dict, List, Tuple

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


DEFAULT_SCORERS = ["qualt5", "tasb", "perplexity", "itn", "cdd"]
SUPPORTED_SCORERS = DEFAULT_SCORERS + ["finetuned_qualt5", "metadata_qualt5", "metadata_enriched_qualt5"]


# =========================================================
# UTILS
# =========================================================

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


# =========================================================
# DATASET
# =========================================================

def prepare_topics_and_qrels(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_name = config["dataset"]["name"]
    topics_variant = config["dataset"]["topics_variant"]
    qrels_variant = config["dataset"]["qrels_variant"]

    print("\n[DEBUG] Caricamento dataset e pulizia query...")
    loader = DatasetLoader(dataset_name)
    topics = loader.get_topics(topics_variant).copy()
    qrels = loader.get_qrels(qrels_variant)

    topics = topics[topics["qid"].isin(qrels["qid"])].copy()
    topics["query"] = topics["query"].apply(clean_query_for_terrier)
    topics = topics[topics["query"].astype(str).str.len() > 0].copy()

    return topics, qrels


# =========================================================
# PIPELINES
# =========================================================

def build_systems_for_scorer(
    config: Dict[str, Any],
    scorer_name: str,
) -> Tuple[List[Any], List[str], str]:
    threshold = config["experiment"]["threshold"]
    active_retriever = config["experiment"].get("retriever", "all").lower()
    dataset_name = config["dataset"]["name"]
    indexes_dir = config["paths"]["indexes_dir"]

    pruned_root = os.path.join(indexes_dir, f"{scorer_name}_pruned_{threshold}")
    ensure_path_exists(pruned_root, f"Cartella root degli indici pruned per scorer={scorer_name}")

    systems = []
    names = []

    # BM25
    if active_retriever in ["bm25", "all"]:
        print("\n[DEBUG] Preparazione pipeline BM25...")
        pipe_bm25_full = pt.terrier.Retriever.from_dataset(
            dataset_name,
            "terrier_stemmed",
            wmodel="BM25",
            verbose=True
        )

        bm25_pruned_path = os.path.join(pruned_root, "pisa_bm25")
        ensure_path_exists(bm25_pruned_path, "Indice BM25 pruned")
        pipe_bm25_p = RetrievalPipelines(bm25_pruned_path).get_bm25()

        systems.extend([pipe_bm25_full, pipe_bm25_p])
        names.extend(["BM25 Full", "BM25 Pruned"])

    # SPLADE
    if active_retriever in ["splade", "all"]:
        print("\n[INFO] Caricamento modello SPLADE...")
        splade_model = pyt_splade.Splade("naver/efficient-splade-VI-BT-large-doc")
        splade_model = move_model_to_cuda(splade_model, "SPLADE")

        encoder_gpu = splade_model.query_encoder(batch_size=32, verbose=True)

        full_splade_index_path = os.path.join(indexes_dir, "splade_full")
        ensure_path_exists(full_splade_index_path, "Indice SPLADE full")

        pipe_splade_full = (
            encoder_gpu
            >> pyterrier_pisa.PisaIndex(full_splade_index_path, stemmer="none").quantized()
        ) % 100

        splade_pruned_path = os.path.join(pruned_root, "pisa_splade")
        ensure_path_exists(splade_pruned_path, "Indice SPLADE pruned")

        pipe_splade_p = (
            encoder_gpu
            >> pyterrier_pisa.PisaIndex(splade_pruned_path, stemmer="none").quantized()
        ) % 100

        systems.extend([pipe_splade_full, pipe_splade_p])
        names.extend(["SPLADE Full", "SPLADE Pruned"])

    # TAS-B
    if active_retriever in ["tasb", "all"]:
        print("\n[INFO] Caricamento modello TAS-B Full...")
        tasb_model = pyterrier_dr.TasB.dot()
        tasb_model = move_model_to_cuda(tasb_model, "TAS-B Full")

        print("[DEBUG] Caricamento FlexIndex per TAS-B da HuggingFace...")
        full_tasb_index = pyterrier_dr.FlexIndex.from_hf("macavaney/msmarco-passage.tasb.flex")

        tasb_encoder_gpu = tasb_model.query_encoder(batch_size=64, verbose=True)
        pipe_tasb_full = tasb_encoder_gpu >> full_tasb_index

        tasb_pruned_path = os.path.join(pruned_root, "tasb.flex")
        ensure_path_exists(tasb_pruned_path, "Indice TAS-B pruned")

        pipe_tasb_p = RetrievalPipelines(tasb_pruned_path, query_encoder=tasb_model).get_tasb()

        systems.extend([pipe_tasb_full, pipe_tasb_p])
        names.extend(["TAS-B Full", "TAS-B Pruned"])

    if not systems:
        raise ValueError(
            f"Nessun sistema caricato. Verifica retriever={active_retriever} e scorer={scorer_name}."
        )

    return systems, names, pruned_root


# =========================================================
# SAVE PER-QUERY
# =========================================================

def save_pairwise_perquery_outputs(
    res_avg: pd.DataFrame,
    res_perq: pd.DataFrame,
    scorer_name: str,
    threshold: Any,
    active_retriever: str,
    qrels_variant: str,
    results_dir: str,
) -> None:
    """
    Salva:
    1. CSV medio complessivo
    2. CSV per-query lungo
    3. CSV per-query pivotato: una riga per qid, colonne separate per sistema e metrica
    4. CSV differenze Full - Pruned per query, utile per test statistici
    """
    prefix = f"{active_retriever}_{scorer_name}_{qrels_variant}_{threshold}"

    avg_path = os.path.join(results_dir, f"compare_{prefix}.csv")
    perq_long_path = os.path.join(results_dir, f"compare_{prefix}_perquery_long.csv")
    perq_wide_path = os.path.join(results_dir, f"compare_{prefix}_perquery_wide.csv")
    perq_diff_path = os.path.join(results_dir, f"compare_{prefix}_perquery_diffs.csv")

    # 1. risultati medi
    res_avg.to_csv(avg_path, index=False)

    # 2. risultati per-query in formato lungo
    res_perq.to_csv(perq_long_path, index=False)

    # Normalizzazione nomi colonne
    df = res_perq.copy()

    # PyTerrier di solito restituisce colonne tipo:
    # qid, measure, value, name
    # oppure name, qid, measure, value
    expected_cols = {"qid", "measure", "value", "name"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Le colonne per-query attese non sono presenti. Mancano: {missing}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    # 3. pivotato: una riga per qid, colonne tipo "BM25 Full__RR@10"
    df_wide = df.pivot_table(
        index="qid",
        columns=["name", "measure"],
        values="value"
    )
    df_wide.columns = [f"{sys_name}__{metric}" for sys_name, metric in df_wide.columns]
    df_wide = df_wide.reset_index()
    df_wide.to_csv(perq_wide_path, index=False)

    # 4. differenze Full - Pruned per query
    #    utile per TOST / t-test / Wilcoxon
    diff_rows = []
    systems = sorted(df["name"].unique())

    # raggruppa per prefisso sistema: BM25, SPLADE, TAS-B
    families = {}
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
    qrels_variant = config["dataset"]["qrels_variant"]

    indexes_dir = config["paths"]["indexes_dir"]
    results_dir = config["paths"]["results_dir"]
    base_runs_dir = config["paths"].get("runs_dir", results_dir)

    ensure_path_exists(indexes_dir, "Cartella indexes_dir")
    os.makedirs(results_dir, exist_ok=True)

    runs_dir = os.path.join(
        base_runs_dir,
        f"runs_{active_retriever}_{scorer_name}_{qrels_variant}_{threshold}"
    )
    os.makedirs(runs_dir, exist_ok=True)

    print(f"[DEBUG] Configurazione:")
    print(f"  - Scorer: {scorer_name} (Threshold: {threshold})")
    print(f"  - Retriever Attivo: {active_retriever.upper()}")

    systems, names, _ = build_systems_for_scorer(config, scorer_name)

    print("\n========================================================")
    print(f"[INFO] Avvio calcolo metriche per i sistemi: {', '.join(names)}")
    print("========================================================\n")

    eval_metrics = [
        ir_measures.RR@10,
        ir_measures.nDCG@10,
        ir_measures.Recall@100,
    ]

    # perquery="both" restituisce (media, per-query)
    res_avg, res_perq = pt.Experiment(
        systems,
        topics,
        qrels,
        eval_metrics=eval_metrics,
        names=names,
        verbose=True,
        save_dir=runs_dir,
        save_mode="reuse",
        perquery="both",
    )

    save_pairwise_perquery_outputs(
        res_avg=res_avg,
        res_perq=res_perq,
        scorer_name=scorer_name,
        threshold=threshold,
        active_retriever=active_retriever,
        qrels_variant=qrels_variant,
        results_dir=results_dir,
    )

    print("\n--- Risultati medi ---")
    print(res_avg.head())

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
        description="Valutazione full vs pruned con salvataggio risultati medi e per-query."
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
