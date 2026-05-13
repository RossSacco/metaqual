import argparse
import numpy as np
import pyterrier as pt
from scipy.stats import pearsonr, spearmanr
from pyterrier_quality import QualCache


def safe_corr(x, y, name="correlation"):
    """
    Calcola una correlazione solo se gli array non sono costanti.
    """
    if len(x) < 2:
        print(f"{name}: impossibile, meno di 2 valori validi.")
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        print(f"{name}: impossibile, almeno uno dei due array è costante.")
        return np.nan

    if name.lower().startswith("pearson"):
        return pearsonr(x, y)[0]
    elif name.lower().startswith("spearman"):
        return spearmanr(x, y)[0]
    else:
        raise ValueError("Unknown correlation type.")


def compare_reproduction(my_cache_path, official_url, model_name):
    """
    Compares a local reproduced cache against the official library cache downloaded from HF.
    """

    if not pt.java.started():
        pt.java.init()

    print(f"\nREPRODUCTION VALIDATION: {model_name}")

    try:
        print(f"Loading YOUR reproduced cache from: {my_cache_path}...")
        my_cache = QualCache(my_cache_path)
        my_scores = np.asarray(my_cache.quality_scores(), dtype=np.float32)

        print(f"Downloading/Loading OFFICIAL cache from: {official_url}...")
        official_cache = QualCache.from_url(official_url)
        official_scores = np.asarray(official_cache.quality_scores(), dtype=np.float32)

    except Exception as e:
        print(f"Error during loading: {e}")
        return

    if len(my_scores) != len(official_scores):
        print(f"ERROR: Length mismatch! Mine: {len(my_scores)}, Official: {len(official_scores)}")
        return

    print(f"Length confirmed: {len(my_scores)} documents compared.")

    # -----------------------------
    # Check NaN / Inf
    # -----------------------------
    my_finite = np.isfinite(my_scores)
    official_finite = np.isfinite(official_scores)
    both_finite = my_finite & official_finite

    print("\n--- FINITE VALUES CHECK ---")
    print(f"My cache finite values       : {my_finite.sum()} / {len(my_scores)}")
    print(f"My cache NaN/Inf values      : {(~my_finite).sum()}")

    print(f"Official cache finite values : {official_finite.sum()} / {len(official_scores)}")
    print(f"Official cache NaN/Inf values: {(~official_finite).sum()}")

    print(f"Comparable finite pairs      : {both_finite.sum()} / {len(my_scores)}")
    print("---------------------------")

    if both_finite.sum() == 0:
        print("ERROR: Nessuna coppia valida su cui calcolare le metriche.")
        return

    if both_finite.sum() < len(my_scores):
        bad_idx = np.where(~both_finite)[0][:20]
        print("\nEsempi di indici non validi:")
        print(bad_idx)
        print("Probabile cache incompleta o score falliti durante la generazione.")

    my_valid = my_scores[both_finite]
    official_valid = official_scores[both_finite]

    # -----------------------------
    # Debug distribuzioni
    # -----------------------------
    print("\n--- SCORE DISTRIBUTION ---")
    print(f"My scores min/max/mean       : {my_valid.min():.6f} / {my_valid.max():.6f} / {my_valid.mean():.6f}")
    print(f"Official scores min/max/mean : {official_valid.min():.6f} / {official_valid.max():.6f} / {official_valid.mean():.6f}")
    print(f"My scores std               : {my_valid.std():.6f}")
    print(f"Official scores std          : {official_valid.std():.6f}")
    print("--------------------------")

    # -----------------------------
    # Metrics
    # -----------------------------
    mae = np.mean(np.abs(my_valid - official_valid))
    p_corr = safe_corr(my_valid, official_valid, name="Pearson")
    s_corr = safe_corr(my_valid, official_valid, name="Spearman")

    print("\n--- STATISTICAL RESULTS ---")
    print(f"Mean Absolute Error (MAE): {mae:.8f}")

    if np.isfinite(p_corr):
        print(f"Pearson Correlation      : {p_corr:.6f}")
    else:
        print("Pearson Correlation      : nan")

    if np.isfinite(s_corr):
        print(f"Spearman Rank Correlation: {s_corr:.6f}")
    else:
        print("Spearman Rank Correlation: nan")

    print("---------------------------")

    # -----------------------------
    # Interpretation
    # -----------------------------
    if np.isfinite(p_corr) and np.isfinite(s_corr):
        if p_corr > 0.99 and s_corr > 0.99:
            print("SUCCESS: valori e ranking quasi identici.")
        elif p_corr > 0.99 and s_corr <= 0.99:
            print("CAUTION: alta correlazione lineare, ma ranking diverso.")
        elif p_corr > 0.90 or s_corr > 0.90:
            print("CLOSE: correlazione alta, ma differenze presenti.")
        else:
            print("WARNING: differenze significative.")
    else:
        print("WARNING: correlazioni non calcolabili. Controlla NaN, Inf o array costanti.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate local cache against official library scores.")

    parser.add_argument("--my_cache", type=str, required=True)
    parser.add_argument("--official_url", type=str, required=True)
    parser.add_argument("--model", type=str, default="Scorer")

    args = parser.parse_args()

    compare_reproduction(args.my_cache, args.official_url, args.model)


"""
Esempio di esecuzione:
python -m metaqual.utils.qualityscore_resultscheck \
  --my_cache "/data/data-sacco/cache/finetuned_qualt5_msmarco.cache" \
  --official_url "hf:pyterrier-quality/qt5-base.msmarco-passage.cache" \
  --model "finetuned_qualt5"
"""