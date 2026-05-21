import os
import pandas as pd
import numpy as np
import pickle

from metaqual.data.loaders.msmarco.dataset_loader import DatasetLoader
from pyterrier_quality import QualCache


def get_scorer_name_from_cache(filename, dataset_name):
    """
    Esempi:
    finetuned_qualt5_msmarco_passage.cache        -> finetuned_qualt5
    metadata_qualt5_msmarco_passage_nuovo.cache  -> finetuned_qualt5_nuovo
    qualt5_msmarco_passage.cache                  -> qualt5
    tasb_msmarco_passage.cache                    -> tasb
    """

    suffix_nuovo = f"_{dataset_name}_nuovo.cache"
    suffix = f"_{dataset_name}.cache"

    if filename.endswith(suffix_nuovo):
        base_name = filename[:-len(suffix_nuovo)]
        return f"{base_name}_nuovo"

    if filename.endswith(suffix):
        return filename[:-len(suffix)]

    return filename.replace(".cache", "")


def prepare_data(
    cache_dir="./cache",
    output_file="roc_data.pkl",
    dataset_name="msmarco_passage",
):
    loader = DatasetLoader(dataset_name)

    # 1. Aggregazione Qrels: Dev, test-2019, test-2020
    print("Caricamento Qrels di valutazione...")

    eval_qrels = pd.concat([
        loader.get_qrels("dev"),
        loader.get_qrels("test-2019"),
        loader.get_qrels("test-2020"),
    ])

    rel_eval = set(
        eval_qrels[eval_qrels["label"] > 0]["docno"].unique()
    )

    # 2. Esclusione Train
    print("Esclusione documenti di Train...")

    train_qrels = loader.get_qrels("train")
    rel_train = set(
        train_qrels[train_qrels["label"] > 0]["docno"].unique()
    )

    final_rel_docs = rel_eval - rel_train

    print(f"Documenti rilevanti eval      : {len(rel_eval)}")
    print(f"Documenti rilevanti train     : {len(rel_train)}")
    print(f"Documenti rilevanti finali    : {len(final_rel_docs)}")

    # 3. Mappatura Label
    print("Generazione etichette binarie...")

    labels = []

    for doc in loader.get_corpus_iter():
        labels.append(1 if doc["docno"] in final_rel_docs else 0)

    labels = np.array(labels, dtype=np.int8)
    expected_length = len(labels)

    print(f"Corpus analizzato. Documenti totali aspettati: {expected_length}")
    print(f"Positivi: {labels.sum()}")
    print(f"Negativi: {expected_length - labels.sum()}")

    # 4. Raccolta punteggi dalle cache
    results = {
        "labels": labels,
        "scorers": {},
    }

    for file in sorted(os.listdir(cache_dir)):
        if not file.endswith(".cache"):
            continue

        scorer_name = get_scorer_name_from_cache(file, dataset_name)

        # Se vuoi usare il fine-tuned al posto del QualT5 normale,
        # puoi ignorare qualt5 quando esiste finetuned_qualt5.
        if scorer_name == "qualt5":
            print(f"\nSkippo {file}, perché vuoi usare finetuned_qualt5 al posto di qualt5.")
            continue

        print(f"\nCaricamento punteggi per: {scorer_name}")
        print(f"File cache: {file}")

        cache_path = os.path.join(cache_dir, file)
        cache = QualCache(cache_path)

        scores = np.asarray(cache.quality_scores(), dtype=np.float32)

        # Sanity check lunghezza
        if len(scores) != expected_length:
            print(
                f"[ERRORE] La cache '{file}' ha {len(scores)} documenti "
                f"invece di {expected_length}."
            )
            print(f"-> Lo scorer '{scorer_name}' verrà ignorato.")
            continue

        # Sanity check NaN / Inf
        finite_mask = np.isfinite(scores)
        n_bad = (~finite_mask).sum()

        if n_bad > 0:
            print(f"[ERRORE] La cache '{file}' contiene {n_bad} valori NaN/Inf.")
            print(f"-> Lo scorer '{scorer_name}' verrà ignorato.")
            continue

        results["scorers"][scorer_name] = scores

        print(f"[OK] {len(scores)} punteggi caricati correttamente.")
        print(f"Min/Max/Mean: {scores.min():.6f} / {scores.max():.6f} / {scores.mean():.6f}")

    print("\nScorers salvati nel file:")
    print(list(results["scorers"].keys()))

    if "finetuned_qualt5" not in results["scorers"]:
        print(
            "\n[WARNING] Non è stato trovato 'finetuned_qualt5'. "
            "Controlla che la cache si chiami esattamente "
            f"'finetuned_qualt5_{dataset_name}.cache'."
        )

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nDati salvati con successo in {output_file}")


if __name__ == "__main__":
    prepare_data(
        cache_dir="/data/data-sacco/cache",
        output_file="roc_data.pkl",
        dataset_name="msmarco_passage",
    )