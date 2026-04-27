import os
import argparse
import shutil
import pyterrier as pt
from pyterrier_quality import QualCache
from itertools import islice
import yaml

from metaqual.models.base_scorer import get_scorer
from metaqual.data.loaders.msmarco.dataset_loader import DatasetLoader

def checkpointed_iter(corpus_iter, scorer_name, chunk_size=50000, checkpoint_dir="./cache"):
    """
    Wraps the corpus iterator. Saves progress to a text file
    and skips already processed documents on restart.
    """
    ckpt_file = os.path.join(checkpoint_dir, f"{scorer_name}_checkpoint.txt")
    processed_docs = 0
    
    # 1. Restore from checkpoint
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'r') as f:
            processed_docs = int(f.read().strip())
        print(f"🔄 Checkpoint trovato! Salto i primi {processed_docs} documenti già elaborati...", flush=True)
        # Advance the iterator to the point where processing stopped
        corpus_iter = islice(corpus_iter, processed_docs, None)
    
    # 2. Yield items and save periodically
    count = 0
    for doc in corpus_iter:
        yield doc
        count += 1
        
        # Save state every 'chunk_size' documents
        if count % chunk_size == 0:
            current_total = processed_docs + count
            with open(ckpt_file, 'w') as f:
                f.write(str(current_total))
            print(f" Checkpoint salvato: {current_total} documenti elaborati su 8.8M.")

    # 3. Final write
    with open(ckpt_file, 'w') as f:
        f.write(str(processed_docs + count))

def cache_gen(name_scorer, dataset_name, path_output_base="./cache", resume=True):
    if not pt.started():
        pt.init()

    print(f"STARTING SCORER PROCESSING: {name_scorer.upper()}")

    # In metaqual/experiments/phase1_qualityscores.py
    print("DEBUG: Avvio DatasetLoader...") # Aggiungi questo
    loader = DatasetLoader(dataset_name)
    print("DEBUG: DatasetLoader pronto. Recupero iteratore...") # Aggiungi questo
    corpus_iter = loader.get_corpus_iter()
    
    kwargs = {}
    if name_scorer == 'cdd':
        kwargs['background_corpus'] = loader.get_corpus_iter()

    scorer_transformer = get_scorer(name_scorer, **kwargs)

    # Output folder setup
    percorso_cache = os.path.join(path_output_base, f"{name_scorer}_{dataset_name}.cache")
    ckpt_file = os.path.join(path_output_base, f"{name_scorer}_{dataset_name}_checkpoint.txt")
    
    os.makedirs(path_output_base, exist_ok=True)

    # --- RESUME LOGIC ---
    if os.path.exists(percorso_cache):
        if resume and os.path.exists(ckpt_file):
            print(f"⚡ Ripresa dell'elaborazione dalla cache esistente: {percorso_cache}")
        else:
            print(f"WARNING: Riavvio pulito. Rimuovo la cache e i checkpoint precedenti.")
            shutil.rmtree(percorso_cache)
            if os.path.exists(ckpt_file):
                os.remove(ckpt_file)
                
    mia_cache = QualCache(percorso_cache)

    # Create checkpoint-protected iterator (saves every 100k docs)
    safe_iter = checkpointed_iter(corpus_iter, name_scorer, chunk_size=100000, checkpoint_dir=path_output_base)

    # Caching pipeline
    pipeline = scorer_transformer >> mia_cache.indexer()

    print(f"Indexing in progress for {name_scorer.upper()}... (this may take a long time)")
    pipeline.index(safe_iter)
    
    print(f"Cache for {name_scorer.upper()} successfully saved to: {percorso_cache}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    scorer_scelto = config['experiment']['scorer']
    restart = config['experiment']['restart_cache']
    cache_dir = config['paths']['cache_dir']
    dataset_name = config['dataset']['name']
    
    ALL_SCORER = ['qualt5', 'tasb', 'perplexity', 'itn', 'cdd']
    resume_flag = not restart
    scorers_da_eseguire = ALL_SCORER if scorer_scelto == 'all' else [scorer_scelto]
    
    for s in scorers_da_eseguire:
        cache_gen(s, dataset_name=dataset_name, path_output_base=cache_dir, resume=resume_flag)
        
        
        
#CUDA_VISIBLE_DEVICES=1 nohup python -m metaqual.experiments.phase1_qualityscores --config metaqual/config.yaml &