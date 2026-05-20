from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from metaqual.data.loaders.msmarco.metadata_triples_dataset import (
        MetadataQualT5TriplesIterableDataset,
    )
    from metaqual.models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        load_metadata_qualt5_config,
    )
except ImportError:
    from data.loaders.msmarco.metadata_triples_dataset import (
        MetadataQualT5TriplesIterableDataset,
    )
    from models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        load_metadata_qualt5_config,
    )


LOGGER = logging.getLogger("evaluate_metadata_qualt5_diagnostic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic evaluation: compare text-only QualT5 vs MetadataQualT5 "
            "on balanced positive/negative passages, optionally skipping raw train examples."
        )
    )

    parser.add_argument("--text_model_path", type=str, required=True)
    parser.add_argument("--metadata_model_path", type=str, required=True)

    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--metadata_scaler_path", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--triples_source", type=str, choices=["irds", "file"], default="irds")
    parser.add_argument("--irds_dataset_id", type=str, default="msmarco-passage/train/triples-small")
    parser.add_argument("--irds_cache_dir", type=str, default=None)

    parser.add_argument("--triples_path", type=str, default=None)
    parser.add_argument("--triples_format", type=str, choices=["text", "id"], default="id")
    parser.add_argument("--collection_path", type=str, default=None)

    parser.add_argument("--sample_per_class", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--allow_missing_metadata", action="store_true")
    parser.add_argument("--prompt_template", type=str, default="Document: {text} Relevant:")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--metadata_dropout", type=float, default=0.1)
    parser.add_argument("--metadata_mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--disable_meta_ffn", action="store_true")
    parser.add_argument(
        "--scoring_mode",
        type=str,
        choices=["true_logprob", "true_prob"],
        default="true_logprob",
    )

    parser.add_argument(
        "--skip_first_raw_examples",
        type=int,
        default=0,
        help=(
            "Number of raw examples to skip before collecting the diagnostic sample. "
            "Useful to avoid evaluating on examples probably already seen during training. "
            "For example: max_steps * batch_size * gradient_accumulation_steps."
        ),
    )

    parser.add_argument(
        "--max_raw_scan",
        type=int,
        default=None,
        help=(
            "Optional maximum number of raw examples to scan after skipping. "
            "If None, scan until enough positives/negatives are collected."
        ),
    )

    parser.add_argument(
        "--prune_fractions",
        type=str,
        default="0.15,0.25,0.30,0.45",
        help=(
            "Comma-separated pruning fractions. "
            "Example: 0.15,0.25,0.30,0.45"
        ),
    )

    parser.add_argument(
        "--no_normalize_metadata_features",
        action="store_true",
        help=(
            "Disable LayerNorm before each metadata-group MLP. "
            "Use this when evaluating old checkpoints trained without metadata feature normalization."
        ),
    )

    parser.add_argument(
        "--no_unfreeze_last_decoder_block",
        action="store_true",
        help=(
            "Keep the last decoder block frozen when instantiating the model. "
            "This must match the checkpoint architecture/training config."
        ),
    )

    parser.add_argument(
        "--no_unfreeze_lm_head",
        action="store_true",
        help=(
            "Keep the LM head frozen when instantiating the model. "
            "This must match the checkpoint architecture/training config."
        ),
    )

    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_prune_fractions(value: str) -> list[float]:
    fractions = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        frac = float(part)
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"Invalid prune fraction: {frac}")
        fractions.append(frac)

    if not fractions:
        raise ValueError("No valid prune fractions provided.")

    return fractions


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_true_false_token_ids(tokenizer) -> tuple[int, int]:
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)

    if not true_ids or not false_ids:
        raise ValueError("Unable to encode true/false labels.")

    LOGGER.info("true token ids: %s", true_ids)
    LOGGER.info("false token ids: %s", false_ids)

    return true_ids[0], false_ids[0]


def validate_scaler_features(
    scaler: MetadataFeatureScaler,
    lexical_store: LexicalMetadataStore,
) -> None:
    scaler_features = list(scaler.feature_names)
    store_features = list(lexical_store.feature_names)

    if scaler_features != store_features:
        raise ValueError(
            "Mismatch tra feature dello scaler e feature del lexical store.\n"
            f"Scaler features: {scaler_features}\n"
            f"Metadata features: {store_features}"
        )


def load_or_init_identity_scaler(lexical_store: LexicalMetadataStore) -> MetadataFeatureScaler:
    feature_dim = len(lexical_store.feature_names)

    return MetadataFeatureScaler(
        feature_names=lexical_store.feature_names,
        mean=np.zeros(feature_dim, dtype=np.float32),
        std=np.ones(feature_dim, dtype=np.float32),
    )


def collect_balanced_examples(
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
) -> pd.DataFrame:
    LOGGER.info(
        "Campiono %d positive e %d negative da %s...",
        args.sample_per_class,
        args.sample_per_class,
        args.irds_dataset_id if args.triples_source == "irds" else args.triples_path,
    )

    LOGGER.info(
        "Salto i primi %d raw examples prima del campionamento.",
        args.skip_first_raw_examples,
    )

    if args.max_raw_scan is not None:
        LOGGER.info(
            "Limiterò la scansione a max_raw_scan=%d esempi dopo lo skip.",
            args.max_raw_scan,
        )

    identity_scaler = load_or_init_identity_scaler(lexical_store)

    dataset = MetadataQualT5TriplesIterableDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        triples_source=args.triples_source,
        triples_path=args.triples_path,
        triples_format=args.triples_format,
        collection_path=args.collection_path,
        irds_dataset_id=args.irds_dataset_id,
        max_irds_triples=None,
        lexical_store=lexical_store,
        lexical_scaler=identity_scaler,
        allow_missing_metadata=args.allow_missing_metadata,
    )

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []

    scanned_after_skip = 0

    for raw_idx, (docno, passage, label) in enumerate(dataset.iter_raw_examples(), 1):
        if raw_idx <= args.skip_first_raw_examples:
            if raw_idx % 50_000 == 0:
                LOGGER.info(
                    "Skipping train-overlap examples: skipped=%d/%d",
                    raw_idx,
                    args.skip_first_raw_examples,
                )
            continue

        scanned_after_skip += 1

        if args.max_raw_scan is not None and scanned_after_skip > args.max_raw_scan:
            LOGGER.info(
                "Stop per max_raw_scan=%d dopo lo skip.",
                args.max_raw_scan,
            )
            break

        label_int = int(label)

        row = {
            "docno": str(docno),
            "text": str(passage),
            "label": label_int,
            "text_length_chars": len(str(passage)),
            "raw_example_index": int(raw_idx),
            "scanned_after_skip_index": int(scanned_after_skip),
        }

        if label_int == 1 and len(positives) < args.sample_per_class:
            positives.append(row)
        elif label_int == 0 and len(negatives) < args.sample_per_class:
            negatives.append(row)

        if scanned_after_skip % 50_000 == 0:
            LOGGER.info(
                "Sampling progress: raw_idx=%d | scanned_after_skip=%d | pos=%d | neg=%d",
                raw_idx,
                scanned_after_skip,
                len(positives),
                len(negatives),
            )

        if len(positives) >= args.sample_per_class and len(negatives) >= args.sample_per_class:
            LOGGER.info(
                "Raggiunto campione bilanciato: pos=%d | neg=%d",
                len(positives),
                len(negatives),
            )
            break

    if len(positives) < args.sample_per_class or len(negatives) < args.sample_per_class:
        raise RuntimeError(
            f"Campionamento incompleto: pos={len(positives)}, neg={len(negatives)}. "
            f"Richiesti {args.sample_per_class} per classe. "
            f"Prova a ridurre --sample_per_class, ridurre --skip_first_raw_examples, "
            f"oppure aumentare/rimuovere --max_raw_scan."
        )

    df = pd.DataFrame(positives + negatives)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    LOGGER.info("Campione finale: %d righe", len(df))
    LOGGER.info("Label distribution:\n%s", df["label"].value_counts())

    LOGGER.info(
        "Raw index range nel campione: min=%d | max=%d",
        int(df["raw_example_index"].min()),
        int(df["raw_example_index"].max()),
    )

    LOGGER.info(
        "Scanned-after-skip range nel campione: min=%d | max=%d",
        int(df["scanned_after_skip_index"].min()),
        int(df["scanned_after_skip_index"].max()),
    )

    return df


def make_prompts(texts: list[str], template: str) -> list[str]:
    return [template.format(text=text) for text in texts]


def score_text_only_model(
    model,
    tokenizer,
    texts: list[str],
    args: argparse.Namespace,
    device: torch.device,
    true_token_id: int,
    false_token_id: int,
) -> np.ndarray:
    LOGGER.info("Scoring text-only model...")

    model.eval()
    scores: list[float] = []

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        prompts = make_prompts(batch_texts, args.prompt_template)

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        decoder_input_ids = torch.full(
            (encoded["input_ids"].shape[0], 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

            first_step_logits = outputs.logits[:, 0, :]
            true_false_logits = torch.stack(
                [
                    first_step_logits[:, true_token_id],
                    first_step_logits[:, false_token_id],
                ],
                dim=-1,
            )

            log_probs = torch.log_softmax(true_false_logits, dim=-1)
            batch_scores = log_probs[:, 0]

        scores.extend(batch_scores.detach().float().cpu().tolist())

        if (start // args.batch_size) % 20 == 0:
            LOGGER.info(
                "Text-only scored %d/%d",
                min(start + args.batch_size, len(texts)),
                len(texts),
            )

    return np.asarray(scores, dtype=np.float32)


def scale_metadata_features(
    scaler: MetadataFeatureScaler,
    values: np.ndarray,
) -> np.ndarray:
    if hasattr(scaler, "transform"):
        return scaler.transform(values).astype(np.float32)

    mean = np.asarray(scaler.mean, dtype=np.float32)
    std = np.asarray(scaler.std, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    return ((values.astype(np.float32) - mean) / std).astype(np.float32)


def _safe_load_metadata_config(model_path: str | Path) -> Dict[str, Any]:
    try:
        cfg = load_metadata_qualt5_config(model_path)
        if cfg:
            LOGGER.info("metadata_qualt5_config.json caricato da %s", model_path)
            LOGGER.info("Metadata config: %s", cfg)
        else:
            LOGGER.warning(
                "metadata_qualt5_config.json non trovato o vuoto in %s. "
                "Uso i valori da CLI.",
                model_path,
            )
        return cfg or {}
    except Exception as exc:
        LOGGER.warning(
            "Impossibile caricare metadata_qualt5_config.json da %s: %s. "
            "Uso i valori da CLI.",
            model_path,
            exc,
        )
        return {}


def load_metadata_model(
    args: argparse.Namespace,
    device: torch.device,
    true_token_id: int,
    false_token_id: int,
    lexical_feature_dim: int,
):
    LOGGER.info("Carico MetadataEnrichedQualT5 da: %s", args.metadata_model_path)

    metadata_cfg = _safe_load_metadata_config(args.metadata_model_path)

    scoring_mode = metadata_cfg.get("scoring_mode", args.scoring_mode)
    metadata_dropout = metadata_cfg.get("metadata_dropout", args.metadata_dropout)
    metadata_mlp_hidden_dim = metadata_cfg.get(
        "metadata_mlp_hidden_dim",
        args.metadata_mlp_hidden_dim,
    )
    attention_heads = metadata_cfg.get("attention_heads", args.attention_heads)
    use_meta_ffn = metadata_cfg.get("use_meta_ffn", not args.disable_meta_ffn)

    normalize_metadata_features = metadata_cfg.get(
        "normalize_metadata_features",
        not args.no_normalize_metadata_features,
    )
    unfreeze_last_decoder_block = metadata_cfg.get(
        "unfreeze_last_decoder_block",
        not args.no_unfreeze_last_decoder_block,
    )
    unfreeze_lm_head = metadata_cfg.get(
        "unfreeze_lm_head",
        not args.no_unfreeze_lm_head,
    )

    LOGGER.info(
        "Metadata model init config | scoring_mode=%s | dropout=%s | hidden_dim=%s | "
        "attention_heads=%s | use_meta_ffn=%s | normalize_metadata_features=%s | "
        "unfreeze_last_decoder_block=%s | unfreeze_lm_head=%s",
        scoring_mode,
        metadata_dropout,
        metadata_mlp_hidden_dim,
        attention_heads,
        use_meta_ffn,
        normalize_metadata_features,
        unfreeze_last_decoder_block,
        unfreeze_lm_head,
    )

    try:
        model = MetadataEnrichedQualT5(
            model_name_or_path=args.metadata_model_path,
            lexical_feature_dim=lexical_feature_dim,
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            scoring_mode=scoring_mode,
            metadata_mlp_hidden_dim=metadata_mlp_hidden_dim,
            metadata_dropout=metadata_dropout,
            attention_heads=attention_heads,
            use_meta_ffn=use_meta_ffn,
            normalize_metadata_features=normalize_metadata_features,
            unfreeze_last_decoder_block=unfreeze_last_decoder_block,
            unfreeze_lm_head=unfreeze_lm_head,
        )
    except TypeError as exc:
        LOGGER.warning(
            "Costruttore MetadataEnrichedQualT5 non accetta i nuovi flag. "
            "Riprovo con la vecchia firma. Errore originale: %s",
            exc,
        )
        model = MetadataEnrichedQualT5(
            model_name_or_path=args.metadata_model_path,
            lexical_feature_dim=lexical_feature_dim,
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            scoring_mode=scoring_mode,
            metadata_mlp_hidden_dim=metadata_mlp_hidden_dim,
            metadata_dropout=metadata_dropout,
            attention_heads=attention_heads,
            use_meta_ffn=use_meta_ffn,
        )

    loaded_metadata_modules = False

    if hasattr(model, "load_metadata_modules"):
        LOGGER.info("Trovato metodo load_metadata_modules(...). Carico moduli metadata.")
        model.load_metadata_modules(args.metadata_model_path)
        loaded_metadata_modules = True
    else:
        LOGGER.warning(
            "Il modello non espone load_metadata_modules(...). "
            "Assumo che MetadataEnrichedQualT5 carichi i moduli metadata automaticamente."
        )

    model.to(device)
    model.eval()

    try:
        first_param = next(model.parameters())
        LOGGER.info("Metadata model device=%s dtype=%s", first_param.device, first_param.dtype)
    except StopIteration:
        LOGGER.warning("Metadata model senza parametri?")

    LOGGER.info("Metadata modules loaded explicitly: %s", loaded_metadata_modules)

    return model


def extract_scores_from_metadata_output(
    outputs: Any,
    true_token_id: int,
    false_token_id: int,
) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs.view(-1)

    if isinstance(outputs, dict):
        for key in [
            "scores",
            "score",
            "quality_scores",
            "quality_score",
            "logprob_true",
            "true_logprob",
        ]:
            if key in outputs:
                return outputs[key].view(-1)

        if "logits" in outputs:
            logits = outputs["logits"]
        elif "decoder_logits" in outputs:
            logits = outputs["decoder_logits"]
        else:
            raise RuntimeError(
                f"Output dict senza chiave score/logits. Keys: {list(outputs.keys())}"
            )

    elif hasattr(outputs, "logits"):
        logits = outputs.logits

    elif isinstance(outputs, tuple) and len(outputs) > 0:
        logits = outputs[0]

    else:
        raise RuntimeError(f"Formato output metadata model non supportato: {type(outputs)}")

    if logits.ndim == 3:
        logits = logits[:, 0, :]

    if logits.ndim == 2 and logits.shape[-1] == 2:
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[:, 0]

    if logits.ndim == 2 and logits.shape[-1] > max(true_token_id, false_token_id):
        true_false_logits = torch.stack(
            [
                logits[:, true_token_id],
                logits[:, false_token_id],
            ],
            dim=-1,
        )
        log_probs = torch.log_softmax(true_false_logits, dim=-1)
        return log_probs[:, 0]

    raise RuntimeError(f"Shape logits non supportata: {tuple(logits.shape)}")


def score_metadata_model(
    model,
    tokenizer,
    lexical_store: LexicalMetadataStore,
    scaler: MetadataFeatureScaler,
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    true_token_id: int,
    false_token_id: int,
) -> np.ndarray:
    LOGGER.info("Scoring metadata model...")

    scores: list[float] = []
    texts = df["text"].tolist()
    docnos = df["docno"].tolist()

    for start in range(0, len(df), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        batch_docnos = docnos[start : start + args.batch_size]

        prompts = make_prompts(batch_texts, args.prompt_template)

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        raw_metadata = lexical_store.lookup(
            batch_docnos,
            allow_missing_metadata=args.allow_missing_metadata,
        )
        raw_metadata = np.asarray(raw_metadata, dtype=np.float32)
        scaled_metadata = scale_metadata_features(scaler, raw_metadata)

        batch = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "lexical_features": torch.tensor(
                scaled_metadata,
                dtype=torch.float32,
                device=device,
            ),
        }

        with torch.no_grad():
            outputs = model(**batch)
            batch_scores = extract_scores_from_metadata_output(
                outputs,
                true_token_id=true_token_id,
                false_token_id=false_token_id,
            )

        scores.extend(batch_scores.detach().float().cpu().tolist())

        if (start // args.batch_size) % 20 == 0:
            LOGGER.info(
                "Metadata model scored %d/%d",
                min(start + args.batch_size, len(df)),
                len(df),
            )

    return np.asarray(scores, dtype=np.float32)


def summarize_scores(labels: np.ndarray, scores: np.ndarray, name: str) -> dict[str, Any]:
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    auc = roc_auc_score(labels, scores)

    threshold = float(np.median(scores))
    preds = (scores >= threshold).astype(int)
    acc_median_threshold = accuracy_score(labels, preds)

    summary = {
        "model": name,
        "auc": float(auc),
        "accuracy_median_threshold": float(acc_median_threshold),
        "threshold_median": threshold,
        "positive_mean": float(np.mean(pos_scores)),
        "positive_std": float(np.std(pos_scores)),
        "positive_min": float(np.min(pos_scores)),
        "positive_max": float(np.max(pos_scores)),
        "negative_mean": float(np.mean(neg_scores)),
        "negative_std": float(np.std(neg_scores)),
        "negative_min": float(np.min(neg_scores)),
        "negative_max": float(np.max(neg_scores)),
        "mean_gap_pos_minus_neg": float(np.mean(pos_scores) - np.mean(neg_scores)),
    }

    return summary


def compute_pruning_metrics(
    df: pd.DataFrame,
    score_col: str,
    model_name: str,
    prune_fractions: list[float],
) -> list[dict[str, Any]]:
    """
    Simulates static pruning by removing the lowest-scored passages.

    Assumption:
    - lower score = lower estimated passage quality
    - label = 1 means positive/relevant
    - label = 0 means negative/non-relevant
    """

    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    if score_col not in df.columns:
        raise ValueError(f"DataFrame must contain score column '{score_col}'.")

    labels = df["label"].to_numpy(dtype=np.int32)
    scores = df[score_col].to_numpy(dtype=np.float32)

    total = len(df)
    total_pos = int((labels == 1).sum())
    total_neg = int((labels == 0).sum())

    if total_pos == 0 or total_neg == 0:
        raise ValueError(
            f"Need both positive and negative examples. Found pos={total_pos}, neg={total_neg}."
        )

    rows: list[dict[str, Any]] = []

    # Sort ascending: lowest quality first, therefore pruned first.
    order = np.argsort(scores)

    for prune_fraction in prune_fractions:
        n_pruned = int(round(total * prune_fraction))

        pruned_idx = order[:n_pruned]
        kept_idx = order[n_pruned:]

        pruned_labels = labels[pruned_idx]
        kept_labels = labels[kept_idx]

        positives_kept = int((kept_labels == 1).sum())
        positives_pruned = int((pruned_labels == 1).sum())

        negatives_kept = int((kept_labels == 0).sum())
        negatives_pruned = int((pruned_labels == 0).sum())

        kept_positive_rate = positives_kept / total_pos
        positive_loss = positives_pruned / total_pos

        removed_negative_rate = negatives_pruned / total_neg
        kept_negative_rate = negatives_kept / total_neg

        threshold = float(scores[order[n_pruned - 1]]) if n_pruned > 0 else float("-inf")

        rows.append(
            {
                "prune_fraction": float(prune_fraction),
                "model": model_name,
                "score_col": score_col,
                "threshold": threshold,
                "total": int(total),
                "n_pruned": int(n_pruned),
                "n_kept": int(total - n_pruned),
                "total_positive": int(total_pos),
                "total_negative": int(total_neg),
                "positives_kept": int(positives_kept),
                "positives_pruned": int(positives_pruned),
                "negatives_kept": int(negatives_kept),
                "negatives_pruned": int(negatives_pruned),
                "kept_positive_rate": float(kept_positive_rate),
                "removed_negative_rate": float(removed_negative_rate),
                "positive_loss": float(positive_loss),
                "kept_negative_rate": float(kept_negative_rate),
            }
        )

    return rows


def save_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        for col in ["score_text_only", "score_metadata"]:
            plt.figure()
            plt.hist(df.loc[df["label"] == 1, col], bins=50, alpha=0.6, label="positive")
            plt.hist(df.loc[df["label"] == 0, col], bins=50, alpha=0.6, label="negative")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.title(col)
            plt.legend()
            plt.tight_layout()
            out = output_dir / f"{col}_hist.png"
            plt.savefig(out, dpi=150)
            plt.close()
            LOGGER.info("Istogramma salvato: %s", out)
    except Exception as exc:
        LOGGER.warning("Non riesco a salvare gli istogrammi: %s", exc)


def main() -> None:
    setup_logging()
    args = parse_args()

    prune_fractions = parse_prune_fractions(args.prune_fractions)

    if args.irds_cache_dir:
        os.environ["IR_DATASETS_HOME"] = args.irds_cache_dir
        LOGGER.info("IR_DATASETS_HOME impostato a: %s", args.irds_cache_dir)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args)

    LOGGER.info("============================================================")
    LOGGER.info("Avvio diagnostic_metadata")
    LOGGER.info("Device: %s", device)
    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Args: %s", vars(args))
    LOGGER.info("Prune fractions: %s", prune_fractions)
    LOGGER.info("============================================================")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, use_fast=True)
    true_token_id, false_token_id = get_true_false_token_ids(tokenizer)

    LOGGER.info("Carico lexical metadata store...")
    lexical_store = LexicalMetadataStore.from_path(args.metadata_path)
    LOGGER.info("Lexical features: %s", lexical_store.feature_names)

    LOGGER.info("Carico metadata scaler...")
    scaler = MetadataFeatureScaler.load(args.metadata_scaler_path)
    validate_scaler_features(scaler, lexical_store)
    LOGGER.info("Scaler caricato e validato.")

    df = collect_balanced_examples(
        args=args,
        tokenizer=tokenizer,
        lexical_store=lexical_store,
    )

    LOGGER.info("Carico text-only model: %s", args.text_model_path)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.text_model_path,
        torch_dtype=dtype,
    ).to(device)
    text_model.eval()

    df["score_text_only"] = score_text_only_model(
        model=text_model,
        tokenizer=tokenizer,
        texts=df["text"].tolist(),
        args=args,
        device=device,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
    )

    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metadata_model = load_metadata_model(
        args=args,
        device=device,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        lexical_feature_dim=len(lexical_store.feature_names),
    )

    df["score_metadata"] = score_metadata_model(
        model=metadata_model,
        tokenizer=tokenizer,
        lexical_store=lexical_store,
        scaler=scaler,
        df=df,
        args=args,
        device=device,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
    )

    labels = df["label"].to_numpy(dtype=np.int32)

    summaries = [
        summarize_scores(labels, df["score_text_only"].to_numpy(), "text_only_qualt5"),
        summarize_scores(labels, df["score_metadata"].to_numpy(), "metadata_qualt5"),
    ]

    summary_df = pd.DataFrame(summaries)

    pruning_rows: list[dict[str, Any]] = []
    pruning_rows.extend(
        compute_pruning_metrics(
            df=df,
            score_col="score_text_only",
            model_name="text_only_qualt5",
            prune_fractions=prune_fractions,
        )
    )
    pruning_rows.extend(
        compute_pruning_metrics(
            df=df,
            score_col="score_metadata",
            model_name="metadata_qualt5",
            prune_fractions=prune_fractions,
        )
    )

    pruning_df = pd.DataFrame(pruning_rows)

    scores_path = output_dir / "diagnostic_scores.csv"
    summary_path = output_dir / "diagnostic_summary.csv"
    summary_json_path = output_dir / "diagnostic_summary.json"
    pruning_path = output_dir / "diagnostic_pruning_metrics.csv"

    df.to_csv(scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pruning_df.to_csv(pruning_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    save_histogram(df, output_dir)

    LOGGER.info("Scores salvati in: %s", scores_path)
    LOGGER.info("Summary salvata in: %s", summary_path)
    LOGGER.info("Summary JSON salvata in: %s", summary_json_path)
    LOGGER.info("Pruning metrics salvate in: %s", pruning_path)

    LOGGER.info("\nSummary:\n%s", summary_df.to_string(index=False))

    pruning_view_cols = [
        "prune_fraction",
        "model",
        "kept_positive_rate",
        "removed_negative_rate",
        "positive_loss",
        "threshold",
        "positives_pruned",
        "negatives_pruned",
    ]

    LOGGER.info(
        "\nPruning metrics:\n%s",
        pruning_df[pruning_view_cols].to_string(index=False),
    )

    LOGGER.info("Fine diagnostica.")
    LOGGER.info("============================================================")


if __name__ == "__main__":
    main()
    
'''
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
python -u -m metaqual.utils.diagnostic_metadata \
  --text_model_path /home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000 \
  --metadata_model_path /home/sacco/metaqual/outputs/metadata-qualt5lastdec-lmhead-lr5e5-10k \
  --metadata_path /home/sacco/data/msmarco_passage/msmarco_passage_lexical_metadata.parquet \
  --metadata_scaler_path /home/sacco/metaqual/outputs/metadata-qualt5lastdec-lmhead-lr5e5-10k/metadata_scaler.pkl \
  --output_dir /home/sacco/metaqual/outputs/metadata-qualt5lastdec-lmhead-lr5e5-10k/diagnostic_eval_skip_train \
  --triples_source irds \
  --irds_dataset_id msmarco-passage/train/triples-small \
  --sample_per_class 10000 \
  --skip_first_raw_examples 160000 \
  --metadata_mlp_hidden_dim 256 \
  --batch_size 16 \
  --max_length 512 \
  --bf16
'''