import os
import argparse
import numpy as np
import pyterrier as pt
from scipy.stats import pearsonr, spearmanr  # Rimasto solo spearmanr e pearsonr
from pyterrier_quality import QualCache

def compare_reproduction(my_cache_path, official_url, model_name):
    """
    Compares a local reproduced cache against the official library cache downloaded from HF.
    """
    if not pt.started():
        pt.init()

    print(f" REPRODUCTION VALIDATION: {model_name}")

    try:
        print(f"Loading YOUR reproduced cache from: {my_cache_path}...")
        my_cache = QualCache(my_cache_path)
        my_scores = my_cache.quality_scores()

        print(f"Downloading/Loading OFFICIAL cache from: {official_url}...")
        official_cache = QualCache.from_url(official_url)
        official_scores = official_cache.quality_scores()
        
    except Exception as e:
        print(f" Error during loading: {e}")
        return

    if len(my_scores) != len(official_scores):
        print(f"ERROR: Length mismatch! Mine: {len(my_scores)}, Official: {len(official_scores)}")
        return
    else:
        print(f" Alignment confirmed: {len(my_scores)} documents compared.")

    # --- Calcoli Statistici ---
    mae = np.mean(np.abs(my_scores - official_scores))
    p_corr, _ = pearsonr(my_scores, official_scores)
    
    # Calcolo della correlazione di Spearman (ranking)
    s_corr, s_p_value = spearmanr(my_scores, official_scores)

    # --- Report ---
    print(f"\n--- STATISTICAL RESULTS ---")
    print(f"Mean Absolute Error (MAE): {mae:.8f}")
    print(f"Pearson Correlation      : {p_corr:.6f}")
    print(f"Spearman Rank Correlation: {s_corr:.6f}")
    print("-" * 30)

    # Interpretazione combinata aggiornata
    if p_corr > 0.99 and s_corr > 0.99:
        print(" SUCCESS: Valori lineari e ordinamento (ranking) identici!")
    elif p_corr > 0.99 and s_corr <= 0.99:
        print(" CAUTION: Alta correlazione lineare, ma il ranking presenta delle piccole variazioni.")
    elif p_corr > 0.90 or s_corr > 0.90:
        print(" CLOSE: Correlazione alta, ma differenze numeriche o di ordinamento presenti.")
    else:
        print(" WARNING: Differenze significative. Controlla tokenizzazione o versioni.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate local cache against official library scores.")

    # Path to your local folder, e.g., ./cache/qualt5_msmarco.cache
    parser.add_argument("--my_cache", type=str, required=True)

    # Official URL, e.g., hf:pyterrier-quality/qt5-small.msmarco-passage.cache
    parser.add_argument("--official_url", type=str, required=True)

    parser.add_argument("--model", type=str, default="Scorer")

    args = parser.parse_args()

    compare_reproduction(args.my_cache, args.official_url, args.model)


"""
Esempio di esecuzione:
python -m metaqual.utils.qualityscore_resultscheck \
  --my_cache "./cache/itn_msmarco.cache" \
  --official_url "hf:pyterrier-quality/itn.msmarco-passage.cache" \
  --model "itn"
"""