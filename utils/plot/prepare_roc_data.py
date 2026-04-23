import os
import pandas as pd
import numpy as np
import pickle
from metaqual.data.loaders.dataset_loader import DatasetLoader
from pyterrier_quality import QualCache

def prepare_data(cache_dir="./cache", output_file="roc_data.pkl"):
    loader = DatasetLoader("msmarco_passage")
    
    # 1. Aggregazione Qrels (Dev, test-2019, test-2020)
    print("Caricamento Qrels di valutazione...")
    eval_qrels = pd.concat([
        loader.get_qrels("dev"),
        loader.get_qrels("test-2019"),
        loader.get_qrels("test-2020")
    ])
    rel_eval = set(eval_qrels[eval_qrels['label'] > 0]['docno'].unique())
    
    # 2. Esclusione Train
    print("Esclusione documenti di Train...")
    train_qrels = loader.get_qrels("train")
    rel_train = set(train_qrels[train_qrels['label'] > 0]['docno'].unique())
    final_rel_docs = rel_eval - rel_train
    
    # 3. Mappatura Label (1 = Rilevante, 0 = Non Rilevante)
    print("Generazione etichette binarie (questa operazione richiederà qualche minuto)...")
    labels = []
    for doc in loader.get_corpus_iter():
        labels.append(1 if doc['docno'] in final_rel_docs else 0)
    labels = np.array(labels)
    expected_length = len(labels)
    print(f"Corpus analizzato. Documenti totali aspettati: {expected_length}")
    
    # 4. Raccolta punteggi dalle cache
    results = {"labels": labels, "scorers": {}}
    for file in os.listdir(cache_dir):
        if file.endswith(".cache"):
            scorer_name = file.split("_")[0]
            print(f"\nCaricamento punteggi per: {scorer_name}...")
            cache = QualCache(os.path.join(cache_dir, file))
            scores = cache.quality_scores()
            
            # --- SANITY CHECK ---
            if len(scores) != expected_length:
                print(f" [ERRORE] La cache '{file}' ha {len(scores)} documenti invece di {expected_length}!")
                print(f" -> Lo scorer '{scorer_name}' verrà ignorato per evitare errori nel plot.")
                continue
                
            results["scorers"][scorer_name] = scores
            print(f" [OK] {len(scores)} punteggi caricati correttamente.")
            
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nDati salvati con successo in {output_file}")

if __name__ == "__main__":
    prepare_data()