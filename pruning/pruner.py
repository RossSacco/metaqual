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
from pyterrier_pisa import PisaStemmer


def _move_model_to_cuda(model, model_name="model"):
    """
    Prova a spostare il modello su GPU.
    Alcuni wrapper espongono .to(...), altri tengono il modello PyTorch interno.
    """
    moved = False

    if hasattr(model, "to"):
        try:
            model = model.to("cuda")
            moved = True
        except Exception:
            pass

    # Tentativo su attributi interni comuni
    for attr in ["model", "_model", "encoder", "_encoder"]:
        if not moved and hasattr(model, attr):
            inner = getattr(model, attr)
            if hasattr(inner, "to"):
                try:
                    inner.to("cuda")
                    moved = True
                except Exception:
                    pass

    if not moved:
        print(f"[WARNING] Non sono riuscita a forzare {model_name} su CUDA tramite .to('cuda').")
        print(f"[WARNING] Verifica l'API della tua versione di pyterrier_dr.")
    else:
        print(f"[INFO] {model_name} spostato su GPU.")

    return model


def run_full_pruning_and_indexing(config):
    # --------------------------------------------------
    # Check GPU & Init
    # --------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: questo script è configurato per forzare l'uso della GPU.")

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

    # Gestione del parametro "all"
    if scorer_config == "all":
        # Inserisci qui l'elenco esatto degli scorer che vuoi eseguire
        scorers_to_run = ["qualt5", "tasb", "perplexity", "itn", "cdd"]
    elif isinstance(scorer_config, list):
        scorers_to_run = scorer_config
    else:
        scorers_to_run = [scorer_config]

    loader = DatasetLoader(dataset_name)

    print("[INFO] Inizializzazione modelli Neurali in corso...")
    
    # SPLADE
    splade_model = pyt_splade.Splade("naver/efficient-splade-VI-BT-large-doc")
    splade_model = _move_model_to_cuda(splade_model, "SPLADE")

    # TAS-B
    tasb_model = pyterrier_dr.TasB("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    tasb_model = _move_model_to_cuda(tasb_model, "TAS-B")
    # --------------------------------------------------
    # Esecuzione per ogni scorer
    # --------------------------------------------------
    for scorer_name in scorers_to_run:
        print(f"\n{'='*50}")
        print(f"[INFO] AVVIO PIPELINE PER SCORER: {scorer_name}")
        print(f"{'='*50}")
        
        base_path = os.path.join(indexes_dir, f"{scorer_name}_pruned_{threshold_value}")
        bm25_path = os.path.join(base_path, "pisa_bm25")
        splade_path = os.path.join(base_path, "pisa_splade")
        tasb_path = os.path.join(base_path, "tasb.flex")

         # Se esiste il file pt_meta.json, l'indice è considerato completo
        if os.path.exists(os.path.join(tasb_path, "pt_meta.json")):
            print(f"[SKIP] L'indice TAS-B per {scorer_name} esiste già in {tasb_path}. Procedo al prossimo.")
            continue

        # 1. Caricamento cache e calcolo soglia
        cache_path = os.path.join(cache_dir, f"{scorer_name}_{dataset_name}.cache")
        if not os.path.exists(cache_path):
            print(f"[ERROR] Cache non trovata: {cache_path}. Salto questo scorer e passo al prossimo.")
            continue
        

        quality_cache = QualCache(cache_path)
        threshold = quality_cache.quantile(threshold_value)

        print(f"[INFO] Dataset: {dataset_name}")
        print(f"[INFO] Quantile richiesto: {threshold_value} -> Soglia calcolata: {threshold}")

        # 2. Cartella di output per questo specifico scorer
        
        os.makedirs(base_path, exist_ok=True)

        # 3. Definizione degli indexer per la cartella corrente
        
        bm25_indexer = pyterrier_pisa.PisaIndex(bm25_path, stemmer="porter2")

        
        splade_indexer = pyterrier_pisa.PisaIndex(splade_path, stemmer="none")

        # 3. Definizione degli indexer
        # Creiamo l'oggetto indice
        tasb_index = pyterrier_dr.FlexIndex(tasb_path)
        # Otteniamo l'indexer dall'oggetto indice
        tasb_indexer = tasb_index.indexer()

        # 4. Pipeline di pruning
        pruner = quality_cache.scorer() >> Filter(threshold)
        print(f"[INFO] Avvio indicizzazione prunata in: {base_path}")

        # 5. Esecuzione indicizzazioni
        print(f"[INFO] [{scorer_name}] Indicizzazione BM25...")
        (pruner >> bm25_indexer).index(loader.get_corpus_iter())

        print(f"[INFO] [{scorer_name}] Indicizzazione SPLADE...")
        (pruner >> splade_model.doc_encoder() >> splade_indexer.toks_indexer()).index(loader.get_corpus_iter())
        
        print(f"[INFO] [{scorer_name}] Indicizzazione TAS-B...")
        (pruner >> tasb_model.doc_encoder() >> tasb_indexer).index(loader.get_corpus_iter())

        print(f"[INFO] Indici per {scorer_name} completati con successo!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_full_pruning_and_indexing(config)