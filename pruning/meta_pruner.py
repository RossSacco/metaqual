import os
import argparse
import yaml
import torch
import pyterrier as pt
import pyterrier_pisa
import pyterrier_dr
import pyt_splade

from pyterrier_quality import QualCache, Filter
from metaqual.data.loaders.msmarco.dataset_loader import DatasetLoader


def _move_model_to_cuda(model, model_name="model"):
    """
    Prova a spostare un modello su GPU.

    Alcuni wrapper espongono direttamente .to(...),
    altri tengono il modello PyTorch interno in attributi come:
    .model, ._model, .encoder, ._encoder.
    """
    moved = False

    if hasattr(model, "to"):
        try:
            model = model.to("cuda")
            moved = True
        except Exception as e:
            print(f"[WARNING] {model_name}.to('cuda') fallito: {e}")

    for attr in ["model", "_model", "encoder", "_encoder"]:
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if hasattr(inner, "to"):
                try:
                    inner.to("cuda")
                    moved = True
                    print(f"[INFO] {model_name}.{attr} spostato su CUDA.")
                    break
                except Exception as e:
                    print(f"[WARNING] Tentativo {model_name}.{attr}.to('cuda') fallito: {e}")

    if not moved:
        print(f"[WARNING] Non sono riuscita a forzare {model_name} su CUDA.")
        print("[WARNING] Verifica l'API della tua versione del wrapper.")
    else:
        print(f"[INFO] {model_name} spostato su GPU.")

    return model


def _is_index_complete(path):
    """
    Controllo semplice per capire se un indice esiste già.

    Per FlexIndex TAS-B, pt_meta.json è un buon indicatore.
    Per PISA possono esserci file diversi, quindi qui lo usiamo
    soprattutto per controllare TAS-B come indice finale della pipeline.
    """
    return os.path.exists(os.path.join(path, "pt_meta.json"))


def _resolve_scorers_to_run(scorer_config):
    """
    Gestisce:
    - scorer: "all"
    - scorer: "metadata_qualt5"
    - scorer: ["qualt5", "metadata_qualt5", ...]
    """
    if scorer_config == "all":
        return [
            "qualt5",
            "tasb",
            "perplexity",
            "itn",
            "cdd",
            "finetuned_qualt5",
            "metadata_qualt5",
        ]

    if isinstance(scorer_config, list):
        return scorer_config

    return [scorer_config]


def run_full_pruning_and_indexing(config):
    # --------------------------------------------------
    # Check GPU e init PyTerrier
    # --------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile: questo script è configurato per forzare l'uso della GPU."
        )

    print(f"[INFO] CUDA disponibile: {torch.cuda.is_available()}")
    print(f"[INFO] GPU in uso: {torch.cuda.get_device_name(0)}")

    if not pt.started():
        pt.init()

    # --------------------------------------------------
    # Parametri dal config
    # --------------------------------------------------
    scorer_config = config["experiment"]["scorer"]
    dataset_name = config["dataset"]["name"]
    threshold_value = config["experiment"]["threshold"]

    cache_dir = config["paths"]["cache_dir"]
    indexes_dir = config["paths"]["indexes_dir"]

    # Nome opzionale per distinguere esperimenti diversi dello stesso scorer.
    # Esempio:
    # scorer = metadata_qualt5
    # scorer_output_name = metadata_qualt5_MP
    scorer_output_name = config["experiment"].get("scorer_output_name", None)

    # Nome opzionale della cache specifica.
    # Esempio:
    # metadata_qualt5_msmarco_passage_MP.cache
    cache_file_override = config["experiment"].get("cache_file", None)

    scorers_to_run = _resolve_scorers_to_run(scorer_config)

    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Scorers da eseguire: {scorers_to_run}")
    print(f"[INFO] Cache dir: {cache_dir}")
    print(f"[INFO] Indexes dir: {indexes_dir}")
    print(f"[INFO] Threshold quantile: {threshold_value}")

    if scorer_output_name is not None:
        print(f"[INFO] scorer_output_name: {scorer_output_name}")

    if cache_file_override is not None:
        print(f"[INFO] cache_file override: {cache_file_override}")

    loader = DatasetLoader(dataset_name)

    # --------------------------------------------------
    # Inizializzazione modelli neurali usati per creare gli indici
    # --------------------------------------------------
    print("[INFO] Inizializzazione modelli neurali in corso...")

    print("[INFO] Caricamento SPLADE...")
    splade_model = pyt_splade.Splade("naver/efficient-splade-VI-BT-large-doc")
    splade_model = _move_model_to_cuda(splade_model, "SPLADE")

    print("[INFO] Caricamento TAS-B...")
    tasb_model = pyterrier_dr.TasB(
        "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    )
    tasb_model = _move_model_to_cuda(tasb_model, "TAS-B")

    # --------------------------------------------------
    # Esecuzione per ogni scorer
    # --------------------------------------------------
    for scorer_name in scorers_to_run:
        print("\n" + "=" * 70)
        print(f"[INFO] AVVIO PIPELINE PER SCORER: {scorer_name}")
        print("=" * 70)

        # Se esegui un solo scorer, puoi usare un nome di output custom.
        # Questo serve per distinguere, per esempio:
        # metadata_qualt5_MP
        # metadata_qualt5_concat
        # metadata_qualt5_pooled
        if len(scorers_to_run) == 1 and scorer_output_name is not None:
            current_output_name = scorer_output_name
        else:
            current_output_name = scorer_name

        base_path = os.path.join(
            indexes_dir,
            f"{current_output_name}_pruned_{threshold_value}",
        )

        bm25_path = os.path.join(base_path, "pisa_bm25")
        splade_path = os.path.join(base_path, "pisa_splade")
        tasb_path = os.path.join(base_path, "tasb.flex")

        print(f"[INFO] Output name: {current_output_name}")
        print(f"[INFO] Base path: {base_path}")
        print(f"[INFO] BM25 path: {bm25_path}")
        print(f"[INFO] SPLADE path: {splade_path}")
        print(f"[INFO] TAS-B path: {tasb_path}")

        # Se TAS-B esiste, assumiamo che tutta la pipeline sia già completa,
        # perché TAS-B è l'ultimo indice creato.
        if _is_index_complete(tasb_path):
            print(
                f"[SKIP] L'indice TAS-B per {current_output_name} esiste già "
                f"in {tasb_path}. Procedo al prossimo scorer."
            )
            continue

        # --------------------------------------------------
        # Cache path
        # --------------------------------------------------
        if cache_file_override is not None and len(scorers_to_run) == 1:
            cache_path = os.path.join(cache_dir, cache_file_override)
        else:
            cache_path = os.path.join(
                cache_dir,
                f"{scorer_name}_{dataset_name}.cache",
            )

        if not os.path.exists(cache_path):
            print(f"[ERROR] Cache non trovata: {cache_path}")
            print("[ERROR] Salto questo scorer e passo al prossimo.")
            continue

        print(f"[INFO] Cache usata: {cache_path}")

        # --------------------------------------------------
        # Caricamento cache e soglia
        # --------------------------------------------------
        quality_cache = QualCache(cache_path)
        threshold = quality_cache.quantile(threshold_value)

        print(f"[INFO] Quantile richiesto: {threshold_value}")
        print(f"[INFO] Soglia calcolata: {threshold}")

        # --------------------------------------------------
        # Creazione cartelle e indexer
        # --------------------------------------------------
        os.makedirs(base_path, exist_ok=True)

        bm25_indexer = pyterrier_pisa.PisaIndex(
            bm25_path,
            stemmer="porter2",
        )

        splade_indexer = pyterrier_pisa.PisaIndex(
            splade_path,
            stemmer="none",
        )

        tasb_index = pyterrier_dr.FlexIndex(tasb_path)
        tasb_indexer = tasb_index.indexer()

        # --------------------------------------------------
        # Pipeline di pruning
        # --------------------------------------------------
        pruner = quality_cache.scorer() >> Filter(threshold)

        print(f"[INFO] Avvio indicizzazione prunata in: {base_path}")

        # --------------------------------------------------
        # Indicizzazione BM25
        # --------------------------------------------------
        print(f"[INFO] [{current_output_name}] Indicizzazione BM25...")
        (
            pruner
            >> bm25_indexer
        ).index(loader.get_corpus_iter())

        print(f"[INFO] [{current_output_name}] BM25 completato.")

        # --------------------------------------------------
        # Indicizzazione SPLADE
        # --------------------------------------------------
        print(f"[INFO] [{current_output_name}] Indicizzazione SPLADE...")
        (
            pruner
            >> splade_model.doc_encoder()
            >> splade_indexer.toks_indexer()
        ).index(loader.get_corpus_iter())

        print(f"[INFO] [{current_output_name}] SPLADE completato.")

        # --------------------------------------------------
        # Indicizzazione TAS-B
        # --------------------------------------------------
        print(f"[INFO] [{current_output_name}] Indicizzazione TAS-B...")
        (
            pruner
            >> tasb_model.doc_encoder()
            >> tasb_indexer
        ).index(loader.get_corpus_iter())

        print(f"[INFO] [{current_output_name}] TAS-B completato.")

        print(f"[INFO] Indici per {current_output_name} completati con successo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_pruning_metadata_mp.yaml",
        help="Path al file YAML di configurazione.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_full_pruning_and_indexing(config)
    
    