import os
import pandas as pd
import numpy as np
import pickle

from metaqual.data.loaders.msmarco.dataset_loader import DatasetLoader
from pyterrier_quality import QualCache

from metaqual.utils.plot.scorer_config import (
    get_scorer_name_from_cache,
    is_scorer_enabled,
    get_enabled_scorers,
)


def prepare_data(
    cache_dir="./cache",
    output_file="roc_data.pkl",
    dataset_name="msmarco_passage",
):
    loader = DatasetLoader(dataset_name)

    print("Scorers abilitati da scorer_config.py:")
    print(get_enabled_scorers())

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

        if scorer_name is None:
            print(f"\nSkippo {file}, perché non è definito in scorer_config.py.")
            continue

        if not is_scorer_enabled(scorer_name):
            print(f"\nSkippo {file}, perché '{scorer_name}' ha enabled=False.")
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
        print(
            f"Min/Max/Mean: "
            f"{scores.min():.6f} / {scores.max():.6f} / {scores.mean():.6f}"
        )

    print("\nScorers salvati nel file:")
    print(list(results["scorers"].keys()))

    enabled_scorers = set(get_enabled_scorers())
    found_scorers = set(results["scorers"].keys())
    missing_scorers = enabled_scorers - found_scorers

    if missing_scorers:
        print("\n[WARNING] Alcuni scorer enabled=True non sono stati trovati/caricati:")
        for scorer in sorted(missing_scorers):
            print(f"- {scorer}")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nDati salvati con successo in {output_file}")


if __name__ == "__main__":
    prepare_data(
        cache_dir="/data/data-sacco/cache",
        output_file="roc_data.pkl",
        dataset_name="msmarco_passage",
    )