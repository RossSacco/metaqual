import os
import gc
import shutil
import hashlib
from pathlib import Path
from itertools import islice

import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier_quality import QualCache

from sklearn.metrics import roc_auc_score, accuracy_score

from metaqual.models.base_scorer import get_scorer
from metaqual.data.loaders.msmarco.dataset_loader import DatasetLoader


# =============================================================================
# CONFIGURAZIONE DA MODIFICARE A MANO
# =============================================================================

DATASET_NAME = "msmarco_passage"

# Deve essere il CSV salvato dal diagnostic.
# Idealmente contiene: docno,text,label
# Se contiene solo docno,label, metti HYDRATE_TEXT = True.
SAMPLE_CSV ="/home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/diagnostic_checkpoint_18000_examination/diagnostic_sample_20000.csv"
HYDRATE_TEXT = False

CACHE_DIR = "/data/data-sacco/cache"
RESULTS_DIR = "./results2"

RESTART_CACHE = True
RESUME_CACHE = False

CHECKPOINT_CHUNK_SIZE = 100000

PRUNE_FRACTIONS = [0.15, 0.25, 0.30, 0.45]


# -----------------------------------------------------------------------------
# Qui definisci direttamente gli scorer da testare.
# Puoi lasciarne uno solo oppure più di uno.
# -----------------------------------------------------------------------------

RUNS = [
    {
        "run_name": "finetuned_qualt5_baseline_diagnostic_sample",
        "scorer_name": "finetuned_qualt5",
        "cache_file": "finetuned_qualt5_msmarco_passage_DIAGNOSTIC_SAMPLE.cache",
        "kwargs": {
            "model_name_or_path": "/home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000",
            "batch_size": 100,
            "max_length": 512,
            "device": "cuda",
        },
    },

    {
        "run_name": "concat_noffn_check18000_diagnostic_sample",
        "scorer_name": "metadata_qualt5",
        "cache_file": "metadata_qualt5_msmarco_passage_CONCAT_NOFFN_CHECK18000_DIAGNOSTIC_SAMPLE.cache",
        "kwargs": {
            # CAMBIA QUESTO PATH con quello reale del tuo concat-noffn-check18000.
            "model_name_or_path": "/home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/scorer-checkpoint-18000",

            "base_model_name_or_path": "/home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000",

            "metadata_path": "/home/sacco/data/msmarco_passage/msmarco_passage_lexical_metadata.parquet",

            # CAMBIA QUESTI PATH se il training concat-noffn-check18000 ha un output_dir diverso.
            "lexical_scaler_path": "/home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/lexical_metadata_scaler.pkl",
            "metadata_scaler_path": "/home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/metadata_metadata_scaler.pkl",
            "embedding_scaler_path": "/home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/embedding_metadata_scaler.pkl",
            "token_scaler_path": "/home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/token_metadata_scaler.pkl",

            "batch_size": 100,
            "max_length": 512,
            "device": "cuda",

            "scoring_mode": "true_logprob",
            "allow_missing_metadata": False,

            # IMPORTANTE: per concat-noffn questi valori devono essere quelli reali del modello.
            # Se nel checkpoint c'è metadata_qualt5_config.json, il tuo base_scorer dovrebbe leggerlo
            # e dare priorità alla config salvata.
            "metadata_fusion_mode": "concat_tokens",
            "metadata_dropout": 0.1,
            "metadata_mlp_hidden_dim": 768,
            "attention_heads": 8,
            "use_meta_ffn": False,
            "normalize_metadata_features": False,
            "metadata_projection_type": "linear",
            "metadata_normalization_mode": "feature_aware",
            "decoder_trainable_scope": "full_decoder",
            "unfreeze_lm_head": True,
            "unfreeze_shared_embeddings": False,

            "metadata_feature_groups": {
                "lexical": [
                    "length_tokens",
                    "avg_token_length",
                    "unique_token_ratio",
                    "repetition_ratio",
                    "lexical_entropy",
                    "stopword_ratio",
                    "content_word_ratio",
                ],
                "embedding": [
                    "embedding_l1_norm",
                    "embedding_l2_norm",
                    "embedding_linf_norm",
                    "embedding_mean",
                    "embedding_variance",
                    "near_zero_fraction",
                ],
                "token": [
                    "mean_token_norm",
                    "std_token_norm",
                    "max_token_norm",
                    "token_norm_entropy",
                    "token_to_passage_similarity_mean",
                ],
            },
        },
    },
]


# =============================================================================
# CODICE
# =============================================================================

POSSIBLE_DOCNO_COLS = [
    "docno",
    "docid",
    "doc_id",
    "pid",
    "passage_id",
]

POSSIBLE_TEXT_COLS = [
    "text",
    "passage",
    "body",
    "contents",
]

POSSIBLE_LABEL_COLS = [
    "label",
    "labels",
    "target",
    "y",
    "is_relevant",
    "relevance",
]


def init_pyterrier():
    if not pt.started():
        pt.init()


def read_table(path):
    path = str(path)

    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)

    return pd.read_csv(path)


def find_column(df, candidates, required=True, name="column"):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(
            f"Non riesco a trovare la colonna {name}. "
            f"Colonne disponibili: {list(df.columns)}. "
            f"Candidati provati: {candidates}"
        )

    return None


def normalize_sample_df(df):
    docno_col = find_column(df, POSSIBLE_DOCNO_COLS, required=True, name="docno")
    label_col = find_column(df, POSSIBLE_LABEL_COLS, required=True, name="label")
    text_col = find_column(df, POSSIBLE_TEXT_COLS, required=False, name="text")

    cols = [docno_col, label_col]
    rename_map = {
        docno_col: "docno",
        label_col: "label",
    }

    if text_col is not None:
        cols.append(text_col)
        rename_map[text_col] = "text"

    out = df[cols].copy()
    out = out.rename(columns=rename_map)

    out["docno"] = out["docno"].astype(str)
    out["label"] = out["label"].astype(int)

    if "text" in out.columns:
        out["text"] = out["text"].astype(str)

    return out


def hydrate_text_from_corpus(sample_df):
    print("[HYDRATE] Il sample non contiene text. Recupero testi dal corpus...", flush=True)

    target_docnos = set(sample_df["docno"].astype(str).tolist())
    found = {}

    loader = DatasetLoader(DATASET_NAME)
    corpus_iter = loader.get_corpus_iter()

    for i, doc in enumerate(corpus_iter):
        docno = str(doc.get("docno", doc.get("docid", doc.get("pid", ""))))

        if docno in target_docnos:
            found[docno] = str(doc.get("text", ""))

            if len(found) % 1000 == 0:
                print(
                    f"[HYDRATE] Trovati {len(found)}/{len(target_docnos)} testi",
                    flush=True,
                )

            if len(found) == len(target_docnos):
                break

        if i > 0 and i % 500000 == 0:
            print(f"[HYDRATE] Scansionati {i} documenti...", flush=True)

    missing = [d for d in target_docnos if d not in found]

    if missing:
        raise RuntimeError(
            f"Non ho trovato il testo per {len(missing)} docno. "
            f"Esempi: {missing[:20]}"
        )

    out = sample_df.copy()
    out["text"] = out["docno"].map(found).astype(str)

    return out


def sample_to_iter(sample_df):
    for row in sample_df[["docno", "text"]].itertuples(index=False):
        yield {
            "docno": str(row.docno),
            "text": str(row.text),
        }


def checkpointed_iter(corpus_iter, checkpoint_file, chunk_size):
    processed_docs = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            raw = f.read().strip()

        if raw:
            processed_docs = int(raw)

        print(
            f"🔄 Checkpoint trovato. Salto i primi {processed_docs} documenti del sample...",
            flush=True,
        )

        corpus_iter = islice(corpus_iter, processed_docs, None)

    count = 0

    for doc in corpus_iter:
        yield doc
        count += 1

        if count % chunk_size == 0:
            current_total = processed_docs + count

            with open(checkpoint_file, "w") as f:
                f.write(str(current_total))

            print(
                f"Checkpoint salvato: {current_total} documenti del sample elaborati.",
                flush=True,
            )

    final_total = processed_docs + count

    with open(checkpoint_file, "w") as f:
        f.write(str(final_total))

    print(
        f"Checkpoint finale salvato: {final_total} documenti del sample elaborati.",
        flush=True,
    )


def sample_fingerprint(sample_df):
    joined = "\n".join(sample_df["docno"].astype(str).tolist())
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def save_sample_manifest(sample_df):
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "diagnostic_sample_manifest_used_by_phase1_fixed_scorer.csv"
    fp_path = cache_dir / "diagnostic_sample_manifest_used_by_phase1_fixed_scorer.fingerprint.txt"

    sample_df[["docno", "label"]].to_csv(manifest_path, index=False)

    fp = sample_fingerprint(sample_df)

    with open(fp_path, "w") as f:
        f.write(f"rows={len(sample_df)}\n")
        f.write(f"fingerprint_md5_docno_order={fp}\n")
        f.write("label_counts:\n")
        f.write(sample_df["label"].value_counts().sort_index().to_string())
        f.write("\nfirst_20_docno:\n")

        for docno in sample_df["docno"].head(20).tolist():
            f.write(str(docno) + "\n")

    print()
    print("=" * 100)
    print("[SAMPLE MANIFEST]")
    print("=" * 100)
    print(f"Manifest salvato in: {manifest_path}")
    print(f"Fingerprint salvato in: {fp_path}")
    print(f"rows: {len(sample_df)}")
    print("label distribution:")
    print(sample_df["label"].value_counts().sort_index())
    print("first 10 docno:")
    print(sample_df["docno"].head(10).tolist())
    print(f"md5 docno order: {fp}")


def build_scorer(run_cfg):
    scorer_name = run_cfg["scorer_name"]
    kwargs = dict(run_cfg.get("kwargs", {}))

    if scorer_name == "cdd":
        loader = DatasetLoader(DATASET_NAME)
        kwargs["background_corpus"] = loader.get_corpus_iter()

    print()
    print("=" * 100)
    print("[BUILD SCORER]")
    print("=" * 100)
    print(f"run_name: {run_cfg['run_name']}")
    print(f"scorer_name: {scorer_name}")
    print("kwargs:")
    for k, v in kwargs.items():
        if k != "metadata_feature_groups":
            print(f"  {k}: {v}")

    return get_scorer(scorer_name, **kwargs)


def create_cache_for_run(run_cfg, sample_df):
    init_pyterrier()

    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / run_cfg["cache_file"]
    checkpoint_file = cache_dir / f"{cache_path.stem}_checkpoint.txt"

    print()
    print("=" * 100)
    print("[CACHE RUN]")
    print("=" * 100)
    print(f"run_name: {run_cfg['run_name']}")
    print(f"scorer_name: {run_cfg['scorer_name']}")
    print(f"cache_path: {cache_path}")
    print(f"checkpoint_file: {checkpoint_file}")

    if cache_path.exists():
        if RESTART_CACHE:
            print("[CACHE] RESTART_CACHE=True. Rimuovo cache e checkpoint precedenti.")

            shutil.rmtree(cache_path)

            if checkpoint_file.exists():
                checkpoint_file.unlink()

        elif RESUME_CACHE and checkpoint_file.exists():
            print("[CACHE] RESUME_CACHE=True. Riprendo da checkpoint.")

        else:
            raise RuntimeError(
                f"La cache esiste già: {cache_path}\n"
                f"Imposta RESTART_CACHE=True oppure RESUME_CACHE=True."
            )

    scorer_transformer = build_scorer(run_cfg)
    qual_cache = QualCache(str(cache_path))

    corpus_iter = sample_to_iter(sample_df)

    safe_iter = checkpointed_iter(
        corpus_iter,
        checkpoint_file=str(checkpoint_file),
        chunk_size=CHECKPOINT_CHUNK_SIZE,
    )

    pipeline = scorer_transformer >> qual_cache.indexer()

    print()
    print("=" * 100)
    print("[INDEXING]")
    print("=" * 100)
    print(
        f"Eseguo: get_scorer('{run_cfg['scorer_name']}') >> QualCache(...).indexer()",
        flush=True,
    )

    pipeline.index(safe_iter)

    print()
    print("=" * 100)
    print("[CACHE SAVED]")
    print("=" * 100)
    print(f"cache_path: {cache_path}")

    del scorer_transformer
    gc.collect()

    return str(cache_path)


def try_read_scores_from_cache(cache_path, sample_df, score_col):
    """
    Prova a leggere gli score dalla QualCache appena creata.
    Non ricalcola lo scorer.
    """
    print()
    print("=" * 100)
    print("[READ SCORES FROM CACHE]")
    print("=" * 100)
    print(f"cache_path: {cache_path}")

    qual_cache = QualCache(str(cache_path))

    query_df = sample_df[["docno", "text"]].copy()

    scored = qual_cache.transform(query_df)

    if "quality" not in scored.columns:
        raise RuntimeError(
            f"QualCache.transform non ha prodotto quality. "
            f"Colonne ottenute: {list(scored.columns)}"
        )

    out = sample_df[["docno", "label"]].copy()
    out[score_col] = scored["quality"].astype(float).values

    return out


def compute_basic_metrics(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    auc = roc_auc_score(labels, scores)

    threshold = float(np.median(scores))
    preds = (scores >= threshold).astype(int)

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    return {
        "auc": float(auc),
        "accuracy_median_threshold": float(accuracy_score(labels, preds)),
        "threshold_median": threshold,
        "positive_mean": float(pos.mean()),
        "positive_std": float(pos.std()),
        "positive_min": float(pos.min()),
        "positive_max": float(pos.max()),
        "negative_mean": float(neg.mean()),
        "negative_std": float(neg.std()),
        "negative_min": float(neg.min()),
        "negative_max": float(neg.max()),
        "mean_gap_pos_minus_neg": float(pos.mean() - neg.mean()),
    }


def pruning_stats(labels, scores, prune_fraction):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    total = len(labels)
    n_pruned = int(round(total * prune_fraction))

    order = np.argsort(scores)

    pruned = np.zeros(total, dtype=bool)

    if n_pruned > 0:
        pruned[order[:n_pruned]] = True

    kept = ~pruned

    pos = labels == 1
    neg = labels == 0

    total_positive = int(pos.sum())
    total_negative = int(neg.sum())

    positives_kept = int((kept & pos).sum())
    positives_pruned = int((pruned & pos).sum())
    negatives_kept = int((kept & neg).sum())
    negatives_pruned = int((pruned & neg).sum())

    if n_pruned > 0:
        threshold = float(scores[order[n_pruned - 1]])
    else:
        threshold = float("-inf")

    return {
        "prune_fraction": float(prune_fraction),
        "threshold": threshold,
        "total": int(total),
        "n_pruned": int(n_pruned),
        "n_kept": int(kept.sum()),
        "total_positive": total_positive,
        "total_negative": total_negative,
        "positives_kept": positives_kept,
        "positives_pruned": positives_pruned,
        "negatives_kept": negatives_kept,
        "negatives_pruned": negatives_pruned,
        "kept_positive_rate": positives_kept / total_positive if total_positive else np.nan,
        "removed_negative_rate": negatives_pruned / total_negative if total_negative else np.nan,
        "positive_loss": positives_pruned / total_positive if total_positive else np.nan,
        "kept_negative_rate": negatives_kept / total_negative if total_negative else np.nan,
    }


def save_metrics_and_pruning(scores_df):
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = results_dir / "phase1_fixed_scorer_on_diagnostic_sample"

    scores_path = out_prefix.with_suffix(".scores.csv")
    metrics_path = out_prefix.with_suffix(".metrics.csv")
    pruning_path = out_prefix.with_suffix(".pruning.csv")

    labels = scores_df["label"].to_numpy(dtype=np.int64)

    metrics_rows = []
    pruning_rows = []

    for run_cfg in RUNS:
        score_col = f"score_{run_cfg['run_name']}"

        if score_col not in scores_df.columns:
            continue

        scores = scores_df[score_col].to_numpy(dtype=np.float64)

        metrics = compute_basic_metrics(labels, scores)

        metrics_rows.append(
            {
                "run_name": run_cfg["run_name"],
                "scorer_name": run_cfg["scorer_name"],
                "score_col": score_col,
                **metrics,
            }
        )

        print()
        print("=" * 100)
        print(f"[METRICS] {run_cfg['run_name']}")
        print("=" * 100)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.9f}")
            else:
                print(f"{k}: {v}")

        for prune_fraction in PRUNE_FRACTIONS:
            row = pruning_stats(labels, scores, prune_fraction)
            row["run_name"] = run_cfg["run_name"]
            row["scorer_name"] = run_cfg["scorer_name"]
            row["score_col"] = score_col
            pruning_rows.append(row)

    scores_df.to_csv(scores_path, index=False)
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    pruning_df = pd.DataFrame(pruning_rows)

    pruning_df = pruning_df[
        [
            "prune_fraction",
            "run_name",
            "scorer_name",
            "score_col",
            "threshold",
            "total",
            "n_pruned",
            "n_kept",
            "total_positive",
            "total_negative",
            "positives_kept",
            "positives_pruned",
            "negatives_kept",
            "negatives_pruned",
            "kept_positive_rate",
            "removed_negative_rate",
            "positive_loss",
            "kept_negative_rate",
        ]
    ]

    pruning_df.to_csv(pruning_path, index=False)

    print()
    print("=" * 100)
    print("[OUTPUT]")
    print("=" * 100)
    print(f"scores: {scores_path}")
    print(f"metrics: {metrics_path}")
    print(f"pruning: {pruning_path}")


def main():
    print()
    print("=" * 100)
    print("[LOAD SAMPLE]")
    print("=" * 100)
    print(f"SAMPLE_CSV: {SAMPLE_CSV}")

    sample_raw = read_table(SAMPLE_CSV)
    sample_df = normalize_sample_df(sample_raw)

    if "text" not in sample_df.columns:
        if not HYDRATE_TEXT:
            raise ValueError(
                "Il sample non contiene la colonna text. "
                "Metti HYDRATE_TEXT=True oppure salva text nel diagnostic_sample_20000.csv."
            )

        sample_df = hydrate_text_from_corpus(sample_df)

    sample_df = sample_df.dropna(subset=["docno", "text", "label"]).copy()
    sample_df["docno"] = sample_df["docno"].astype(str)
    sample_df["text"] = sample_df["text"].astype(str)
    sample_df["label"] = sample_df["label"].astype(int)

    print(f"righe sample: {len(sample_df)}")
    print("label distribution:")
    print(sample_df["label"].value_counts().sort_index())
    print("first 10 docno:")
    print(sample_df["docno"].head(10).tolist())
    print(f"fingerprint md5 docno order: {sample_fingerprint(sample_df)}")

    save_sample_manifest(sample_df)

    combined_scores = sample_df[["docno", "label"]].copy()

    cache_paths = {}

    for run_cfg in RUNS:
        cache_path = create_cache_for_run(run_cfg, sample_df)
        cache_paths[run_cfg["run_name"]] = cache_path

        score_col = f"score_{run_cfg['run_name']}"

        scored_df = try_read_scores_from_cache(
            cache_path=cache_path,
            sample_df=sample_df,
            score_col=score_col,
        )

        combined_scores[score_col] = scored_df[score_col].values

    print()
    print("=" * 100)
    print("[CACHE PATHS]")
    print("=" * 100)
    for run_name, cache_path in cache_paths.items():
        print(f"{run_name}: {cache_path}")

    save_metrics_and_pruning(combined_scores)

    print()
    print("=" * 100)
    print("[FINAL CHECK]")
    print("=" * 100)
    print("Questo script NON prende lo scorer dal config YAML.")
    print("Lo scorer è definito direttamente nella lista RUNS.")
    print("La cache viene creata usando esattamente:")
    print("  get_scorer(...) >> QualCache(...).indexer()")
    print("Quindi è lo stesso meccanismo della phase1_qualityscores.py.")


if __name__ == "__main__":
    main()