import os
import pyterrier_pisa
import pyterrier_dr

# Inserisci il percorso esatto della cartella che contiene i 3 indici
base_path = "./indexes/qualt5_pruned_0.3"

print(f"Verifica degli indici in: {base_path}\n" + "-"*40)

# 1. Check BM25 (PISA)
bm25_path = os.path.join(base_path, "pisa_bm25")
try:
    bm25_idx = pyterrier_pisa.PisaIndex(bm25_path)
    print(f"[BM25] Indice costruito correttamente: {bm25_idx.built()}")
    # Leggiamo il numero di righe dal file dei nomi dei documenti
    with open(os.path.join(bm25_path, "fwd.docnames"), "r") as f:
        bm25_docs = sum(1 for _ in f)
    print(f"[BM25] Numero documenti: {bm25_docs}")
except Exception as e:
    print(f"[BM25] Errore nella lettura: {e}")

print("-" * 40)

# 2. Check SPLADE (PISA)
splade_path = os.path.join(base_path, "pisa_splade")
try:
    splade_idx = pyterrier_pisa.PisaIndex(splade_path)
    print(f"[SPLADE] Indice costruito correttamente: {splade_idx.built()}")
    with open(os.path.join(splade_path, "fwd.docnames"), "r") as f:
        splade_docs = sum(1 for _ in f)
    print(f"[SPLADE] Numero documenti: {splade_docs}")
except Exception as e:
    print(f"[SPLADE] Errore nella lettura: {e}")

print("-" * 40)

# 3. Check TAS-B (FlexIndex)
tasb_path = os.path.join(base_path, "tasb.flex")
try:
    tasb_idx = pyterrier_dr.FlexIndex(tasb_path)
    print(f"[TAS-B] Indice caricato correttamente.")
    print(f"[TAS-B] Numero documenti: {len(tasb_idx)}")
    print(f"[TAS-B] Dimensione dei vettori: {tasb_idx.payload(0).shape}")
except Exception as e:
    print(f"[TAS-B] Errore nella lettura: {e}")