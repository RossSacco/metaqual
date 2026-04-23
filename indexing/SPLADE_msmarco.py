import os
import pyterrier as pt
import pyterrier_pisa
import pyterrier_caching  


def build_cached_splade():
    # Usiamo java.init() per eliminare il DeprecationWarning
    if not pt.java.started():
        pt.java.init()

    # 1. Cartella di output
    index_dir = "./indexes/splade_full"
    os.makedirs(index_dir, exist_ok=True)

    # 2. Scarica e carica la cache di SPLADE da Hugging Face
    print("[INFO] Scaricamento/Caricamento della cache SPLADE da Hugging Face...")
    splade_cache = pt.Artifact.from_hf('macavaney/msmarco-passage.splade-lg.cache')

    # 3. Prepara l'indicizzatore PISA (senza stemmer)
    print("[INFO] Inizializzazione indicizzatore PISA...")
    pisa_indexer = pyterrier_pisa.PisaIndex(index_dir, stemmer="none").toks_indexer()

    # 4. Avvia l'indicizzazione passando direttamente la cache
    print(f"[INFO] Costruzione dell'indice PISA in {index_dir}...")
    pisa_indexer.index(splade_cache)
    
    print("[INFO] Finito! L'indice completo è pronto all'uso.")

if __name__ == "__main__":
    build_cached_splade()  # Rimosso il punto finale che causava errore di sintassi