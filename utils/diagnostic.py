from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import hashlib


def export_diagnostic_sample(
    df,
    output_path,
    *,
    docno_col="docno",
    text_col="text",
    label_col="label",
    extra_score_cols=None,
):
    """
    Salva esattamente il sample usato dal diagnostic.

    Va chiamata DOPO che il diagnostic ha costruito il sample finale,
    cioè dopo eventuale bilanciamento 10000 positivi / 10000 negativi,
    shuffle, dropna, filtri, ecc.

    Salva:
      - docno
      - text, se presente
      - label
      - eventuali score già calcolati, se passati in extra_score_cols

    Produce anche un file .fingerprint.txt per verificare che due run usino
    lo stesso identico sample.
    """
    if df is None or len(df) == 0:
        raise ValueError("[export_diagnostic_sample] DataFrame vuoto.")

    if docno_col not in df.columns:
        raise ValueError(
            f"[export_diagnostic_sample] Colonna docno non trovata: {docno_col}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    if label_col not in df.columns:
        raise ValueError(
            f"[export_diagnostic_sample] Colonna label non trovata: {label_col}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    cols = [docno_col]

    if text_col in df.columns:
        cols.append(text_col)
    else:
        print(
            f"[export_diagnostic_sample][WARNING] Colonna text '{text_col}' non presente. "
            "Salvo solo docno,label. Nel replay dovrai usare --hydrate_text."
        )

    cols.append(label_col)

    if extra_score_cols:
        for c in extra_score_cols:
            if c in df.columns and c not in cols:
                cols.append(c)

    out = df[cols].copy()

    rename_map = {
        docno_col: "docno",
        label_col: "label",
    }

    if text_col in out.columns:
        rename_map[text_col] = "text"

    out = out.rename(columns=rename_map)

    out["docno"] = out["docno"].astype(str)
    out["label"] = out["label"].astype(int)

    if "text" in out.columns:
        out["text"] = out["text"].astype(str)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)

    fingerprint = hashlib.md5(
        "\n".join(out["docno"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()

    fp_path = output_path.with_suffix(output_path.suffix + ".fingerprint.txt")

    with open(fp_path, "w") as f:
        f.write(f"rows={len(out)}\n")
        f.write(f"fingerprint_md5_docno_order={fingerprint}\n")
        f.write("label_counts:\n")
        f.write(out["label"].value_counts().sort_index().to_string())
        f.write("\nfirst_20_docno:\n")
        for d in out["docno"].head(20).tolist():
            f.write(str(d) + "\n")

    print()
    print("=" * 100)
    print("[DIAGNOSTIC SAMPLE EXPORTED]")
    print("=" * 100)
    print(f"file: {output_path}")
    print(f"fingerprint: {fp_path}")
    print(f"rows: {len(out)}")
    print("label distribution:")
    print(out["label"].value_counts().sort_index())
    print("first 10 docno:")
    print(out["docno"].head(10).tolist())
    print(f"md5 docno order: {fingerprint}")
    print("=" * 100)

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


VALID_NEW_FUSION_MODES = {
    "concat_tokens",
    "att_fusion",
    "pooled_concat_projection",
    "allmeta_token_projection",
    "meta_prefix",
}

VALID_DECODER_TRAINABLE_SCOPES = {
    "cross_attention",
    "last_n_blocks",
    "full_decoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic evaluation: compare text-only QualT5 vs MetadataQualT5 "
            "on balanced positive/negative passages."
        )
    )

    parser.add_argument("--text_model_path", type=str, required=True)
    parser.add_argument("--metadata_model_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Backward-compatible lexical scaler name.
    parser.add_argument("--metadata_scaler_path", type=str, default=None)

    # Explicit scaler paths.
    parser.add_argument("--lexical_scaler_path", type=str, default=None)
    parser.add_argument("--embedding_scaler_path", type=str, default=None)
    parser.add_argument("--token_scaler_path", type=str, default=None)

    parser.add_argument(
        "--triples_source",
        type=str,
        choices=["irds", "file"],
        default="irds",
    )
    parser.add_argument(
        "--irds_dataset_id",
        type=str,
        default="msmarco-passage/train/triples-small",
    )
    parser.add_argument("--irds_cache_dir", type=str, default=None)

    parser.add_argument("--triples_path", type=str, default=None)
    parser.add_argument(
        "--triples_format",
        type=str,
        choices=["text", "id"],
        default="id",
    )
    parser.add_argument("--collection_path", type=str, default=None)

    parser.add_argument("--sample_per_class", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--allow_missing_metadata", action="store_true")
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Document: {text} Relevant:",
    )

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--metadata_dropout", type=float, default=0.0)
    parser.add_argument("--metadata_mlp_hidden_dim", type=int, default=None)
    parser.add_argument(
        "--metadata_projection_type",
        type=str,
        choices=["linear", "mlp"],
        default="linear",
    )
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
            "Useful to avoid evaluating on examples probably already seen during training."
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
        "--fixed_sample_path",
        type=str,
        default=None,
        help=(
            "CSV con colonne docno,text,label da usare come sample fisso. "
            "Se fornito, non viene fatto nuovo campionamento."
        ),
    )

    parser.add_argument(
        "--save_sample_path",
        type=str,
        default=None,
        help=(
            "Path dove salvare il sample creato prima dello scoring. "
            "Utile per riusarlo identico su altri checkpoint."
        ),
    )

    parser.add_argument(
        "--exclude_docnos_path",
        type=str,
        default=None,
        help=(
            "File .txt o .csv contenente docno da escludere dal diagnostic sample, "
            "ad esempio docno probabilmente visti in training."
        ),
    )

    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Seed usato per lo shuffle finale del diagnostic sample.",
    )

    parser.add_argument(
        "--deduplicate_docnos",
        action="store_true",
        help="Se attivo, evita che lo stesso docno appaia più volte nel sample diagnostic.",
    )

    parser.add_argument(
        "--prune_fractions",
        type=str,
        default="0.15,0.25,0.30,0.45",
        help="Comma-separated pruning fractions. Example: 0.15,0.25,0.30,0.45",
    )

    parser.add_argument(
        "--normalize_metadata_features",
        action="store_true",
        help=(
            "Enable LayerNorm before each metadata-group projection. "
            "Default is disabled because features are standardized with scalers."
        ),
    )

    parser.add_argument(
        "--no_normalize_metadata_features",
        action="store_true",
        help="Deprecated. Kept for backward compatibility. Forces normalization off.",
    )

    parser.add_argument(
        "--unfreeze_last_n_decoder_blocks",
        type=int,
        default=1,
        help=(
            "Number of final decoder blocks to unfreeze when instantiating the model. "
            "Used only if the checkpoint config does not provide the value."
        ),
    )

    parser.add_argument(
        "--decoder_trainable_scope",
        type=str,
        choices=["cross_attention", "last_n_blocks", "full_decoder"],
        default="last_n_blocks",
        help=(
            "Decoder training scope used by the checkpoint. "
            "The value from metadata_qualt5_config.json is preferred when available."
        ),
    )

    parser.add_argument(
        "--unfreeze_shared_embeddings",
        action="store_true",
        help=(
            "Whether shared T5 embeddings were trainable. "
            "The value from metadata_qualt5_config.json is preferred when available."
        ),
    )

    parser.add_argument(
        "--no_unfreeze_lm_head",
        action="store_true",
        help="Keep LM head frozen when instantiating the model.",
    )

    parser.add_argument(
        "--metadata_fusion_mode",
        type=str,
        choices=[
            "concat_tokens",
            "att_fusion",
            "pooled_concat_projection",
            "allmeta_token_projection",
            "meta_prefix",
        ],
        default="pooled_concat_projection",
        help=(
            "Metadata fusion mode. "
            "If metadata_qualt5_config.json exists, its value is preferred."
        ),
    )

    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_prune_fractions(value: str) -> list[float]:
    fractions: list[float] = []

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

    return int(true_ids[0]), int(false_ids[0])


def _is_trainer_checkpoint(path: str | Path) -> bool:
    p = Path(path)
    return p.name.startswith("checkpoint-")


def _metadata_asset_dir(model_path: str | Path) -> Path:
    """
    Returns the directory where metadata_qualt5_config.json, scalers and
    metadata_qualt5_modules.pt are expected.

    If metadata_model_path is a Trainer checkpoint, these files are usually
    in the parent output directory.
    """
    p = Path(model_path)

    if _is_trainer_checkpoint(p):
        return p.parent

    return p


def _safe_load_metadata_config(model_path: str | Path) -> Dict[str, Any]:
    candidates = []

    p = Path(model_path)
    candidates.append(p)

    if _is_trainer_checkpoint(p):
        candidates.append(p.parent)

    for candidate in candidates:
        try:
            cfg = load_metadata_qualt5_config(candidate)

            if cfg:
                LOGGER.info("metadata_qualt5_config.json caricato da %s", candidate)
                LOGGER.info("Metadata config: %s", cfg)
                return cfg

        except Exception as exc:
            LOGGER.warning(
                "Impossibile caricare metadata_qualt5_config.json da %s: %s",
                candidate,
                exc,
            )

    LOGGER.warning(
        "metadata_qualt5_config.json non trovato. Uso valori CLI dove possibile."
    )
    return {}


def _resolve_model_relative_path(
    path_value: Optional[str],
    base_dir: str | Path,
) -> Optional[str]:
    if path_value is None:
        return None

    p = Path(path_value)

    if p.is_absolute():
        return str(p)

    candidate = Path(base_dir) / p
    return str(candidate)


def _get_cfg_path(
    args_value: Optional[str],
    metadata_cfg: Dict[str, Any],
    cfg_keys: list[str],
    base_dir: str | Path,
) -> Optional[str]:
    if args_value is not None:
        return args_value

    for key in cfg_keys:
        value = metadata_cfg.get(key)
        if value:
            return _resolve_model_relative_path(str(value), base_dir)

    return None


def _checkpoint_state_dict_path(model_path: str | Path) -> Optional[Path]:
    """
    Returns the Trainer checkpoint state_dict path if metadata_model_path
    points to a checkpoint-* directory.
    """
    p = Path(model_path)

    if not _is_trainer_checkpoint(p):
        return None

    candidates = [
        p / "pytorch_model.bin",
        p / "model.safetensors",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _load_checkpoint_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    LOGGER.info("Carico state_dict checkpoint da: %s", path)

    if path.name.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Il checkpoint è in formato safetensors, ma safetensors non è installato."
            ) from exc

        state_dict = load_file(str(path))
    else:
        state_dict = torch.load(path, map_location="cpu")

    # In alcuni casi il checkpoint può contenere wrapper tipo {"model": state_dict}.
    if isinstance(state_dict, dict):
        for key in ["model", "state_dict", "module"]:
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Formato checkpoint non supportato: {type(state_dict)}")

    # Rimuove eventuale prefisso DataParallel/Accelerate.
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value

    return cleaned


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


def load_identity_scaler(lexical_store: LexicalMetadataStore) -> MetadataFeatureScaler:
    feature_dim = len(lexical_store.feature_names)

    return MetadataFeatureScaler(
        feature_names=lexical_store.feature_names,
        mean=np.zeros(feature_dim, dtype=np.float32),
        std=np.ones(feature_dim, dtype=np.float32),
    )


def load_docno_set(path: Optional[str]) -> set[str]:
    """
    Carica un insieme di docno da escludere.

    Supporta:
      - .txt: un docno per riga
      - .csv: colonna 'docno' se presente, altrimenti prima colonna
    """
    if path is None:
        return set()

    p = Path(path).expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(f"exclude_docnos_path non trovato: {p}")

    if p.suffix.lower() == ".txt":
        with p.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    df = pd.read_csv(p)

    if df.empty:
        return set()

    if "docno" in df.columns:
        col = "docno"
    else:
        col = df.columns[0]

    return set(df[col].astype(str).tolist())


def load_fixed_diagnostic_sample(
    path: str,
    *,
    excluded_docnos: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Carica un diagnostic sample già salvato.

    Il CSV deve contenere almeno:
      - docno
      - text
      - label

    Se excluded_docnos è fornito, fallisce se il fixed sample contiene docno vietati.
    Questo è voluto: evita di valutare per sbaglio su documenti già visti in training.
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(f"fixed_sample_path non trovato: {p}")

    df = pd.read_csv(p)

    required = {"docno", "text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Il sample fisso deve contenere colonne {sorted(required)}. "
            f"Mancano: {sorted(missing)}. Colonne presenti: {list(df.columns)}"
        )

    df = df.copy()
    df["docno"] = df["docno"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if excluded_docnos:
        overlap = set(df["docno"]) & excluded_docnos
        if overlap:
            raise ValueError(
                f"Il fixed sample contiene {len(overlap)} docno presenti "
                f"in exclude_docnos_path. Esempi: {list(overlap)[:10]}"
            )

    LOGGER.info("Fixed diagnostic sample caricato da: %s", p)
    LOGGER.info("Righe: %d", len(df))
    LOGGER.info("Label distribution:\n%s", df["label"].value_counts().sort_index())

    return df.reset_index(drop=True)


def collect_balanced_examples(
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
    excluded_docnos: Optional[set[str]] = None,
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

    excluded_docnos = excluded_docnos or set()
    seen_docnos: set[str] = set()

    if excluded_docnos:
        LOGGER.info(
            "Escluderò %d docno dal diagnostic sample.",
            len(excluded_docnos),
        )

    if args.deduplicate_docnos:
        LOGGER.info("Deduplica docno attiva per il diagnostic sample.")

    identity_scaler = load_identity_scaler(lexical_store)

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
            LOGGER.info("Stop per max_raw_scan=%d dopo lo skip.", args.max_raw_scan)
            break

        docno_str = str(docno)

        if docno_str in excluded_docnos:
            continue

        label_int = int(label)

        row = {
            "docno": docno_str,
            "text": str(passage),
            "label": label_int,
            "text_length_chars": len(str(passage)),
            "raw_example_index": int(raw_idx),
            "scanned_after_skip_index": int(scanned_after_skip),
        }

        accepted = False

        if label_int == 1 and len(positives) < args.sample_per_class:
            accepted = True
        elif label_int == 0 and len(negatives) < args.sample_per_class:
            accepted = True

        if not accepted:
            continue

        if args.deduplicate_docnos and docno_str in seen_docnos:
            continue

        if label_int == 1:
            positives.append(row)
        else:
            negatives.append(row)

        if args.deduplicate_docnos:
            seen_docnos.add(docno_str)

        if scanned_after_skip % 50_000 == 0:
            LOGGER.info(
                "Sampling progress: raw_idx=%d | scanned_after_skip=%d | pos=%d | neg=%d",
                raw_idx,
                scanned_after_skip,
                len(positives),
                len(negatives),
            )

        if (
            len(positives) >= args.sample_per_class
            and len(negatives) >= args.sample_per_class
        ):
            LOGGER.info(
                "Raggiunto campione bilanciato: pos=%d | neg=%d",
                len(positives),
                len(negatives),
            )
            break

    if len(positives) < args.sample_per_class or len(negatives) < args.sample_per_class:
        raise RuntimeError(
            f"Campionamento incompleto: pos={len(positives)}, neg={len(negatives)}. "
            f"Richiesti {args.sample_per_class} per classe."
        )

    df = pd.DataFrame(positives + negatives)
    df = df.sample(frac=1.0, random_state=args.sample_seed).reset_index(drop=True)

    LOGGER.info("Campione finale: %d righe", len(df))
    LOGGER.info("Label distribution:\n%s", df["label"].value_counts())

    if "raw_example_index" in df.columns:
        LOGGER.info(
            "Raw index range nel campione: min=%d | max=%d",
            int(df["raw_example_index"].min()),
            int(df["raw_example_index"].max()),
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


def _resolve_base_init_path(
    args: argparse.Namespace,
    metadata_cfg: Dict[str, Any],
    metadata_asset_dir: Path,
    checkpoint_state_path: Optional[Path],
) -> str:
    """
    Path used only to instantiate the T5 architecture before loading metadata modules
    or the Trainer checkpoint state_dict.
    """
    if checkpoint_state_path is not None:
        # For Trainer checkpoints, avoid calling AutoModelForSeq2SeqLM.from_pretrained
        # directly on checkpoint-* because its keys may belong to the custom wrapper.
        base_from_cfg = metadata_cfg.get("base_model_name_or_path")
        if base_from_cfg:
            return str(base_from_cfg)

        return str(args.text_model_path)

    # For final output_dir, the base_model was saved with save_pretrained(output_dir).
    return str(args.metadata_model_path)


def load_metadata_model(
    args: argparse.Namespace,
    device: torch.device,
    true_token_id: int,
    false_token_id: int,
    lexical_feature_dim: int,
    metadata_cfg: Dict[str, Any],
    metadata_asset_dir: Path,
    embedding_scaler_path: Optional[str],
    token_scaler_path: Optional[str],
):
    LOGGER.info("Carico MetadataEnrichedQualT5 da: %s", args.metadata_model_path)

    checkpoint_state_path = _checkpoint_state_dict_path(args.metadata_model_path)

    scoring_mode = metadata_cfg.get("scoring_mode", args.scoring_mode)

    metadata_dropout = metadata_cfg.get(
        "metadata_dropout",
        args.metadata_dropout,
    )

    metadata_mlp_hidden_dim = metadata_cfg.get(
        "metadata_mlp_hidden_dim",
        args.metadata_mlp_hidden_dim,
    )

    metadata_projection_type = metadata_cfg.get(
        "metadata_projection_type",
        args.metadata_projection_type,
    )

    attention_heads = metadata_cfg.get(
        "attention_heads",
        args.attention_heads,
    )

    use_meta_ffn = metadata_cfg.get(
        "use_meta_ffn",
        not args.disable_meta_ffn,
    )

    normalize_metadata_features = metadata_cfg.get(
        "normalize_metadata_features",
        bool(args.normalize_metadata_features)
        and not bool(args.no_normalize_metadata_features),
    )

    unfreeze_last_n_decoder_blocks = metadata_cfg.get(
        "unfreeze_last_n_decoder_blocks",
        args.unfreeze_last_n_decoder_blocks,
    )

    unfreeze_lm_head = metadata_cfg.get(
        "unfreeze_lm_head",
        not args.no_unfreeze_lm_head,
    )

    decoder_trainable_scope = metadata_cfg.get(
        "decoder_trainable_scope",
        args.decoder_trainable_scope,
    )

    unfreeze_shared_embeddings = metadata_cfg.get(
        "unfreeze_shared_embeddings",
        bool(args.unfreeze_shared_embeddings),
    )

    metadata_fusion_mode = metadata_cfg.get(
        "metadata_fusion_mode",
        args.metadata_fusion_mode,
    )

    if metadata_fusion_mode not in VALID_NEW_FUSION_MODES:
        raise ValueError(
            f"Il checkpoint/config indica metadata_fusion_mode={metadata_fusion_mode!r}, "
            f"ma questo script supporta solo {sorted(VALID_NEW_FUSION_MODES)}."
        )

    if decoder_trainable_scope not in VALID_DECODER_TRAINABLE_SCOPES:
        raise ValueError(
            f"decoder_trainable_scope={decoder_trainable_scope!r} non supportato. "
            f"Valori ammessi: {sorted(VALID_DECODER_TRAINABLE_SCOPES)}"
        )

    base_init_path = _resolve_base_init_path(
        args=args,
        metadata_cfg=metadata_cfg,
        metadata_asset_dir=metadata_asset_dir,
        checkpoint_state_path=checkpoint_state_path,
    )

    LOGGER.info(
        "Metadata model init config | base_init_path=%s | scoring_mode=%s | "
        "dropout=%s | hidden_dim=%s | projection_type=%s | attention_heads=%s | "
        "use_meta_ffn=%s | normalize_metadata_features=%s | "
        "unfreeze_last_n_decoder_blocks=%s | unfreeze_lm_head=%s | "
        "decoder_trainable_scope=%s | unfreeze_shared_embeddings=%s | "
        "metadata_fusion_mode=%s | embedding_scaler_path=%s | token_scaler_path=%s | "
        "checkpoint_state_path=%s",
        base_init_path,
        scoring_mode,
        metadata_dropout,
        metadata_mlp_hidden_dim,
        metadata_projection_type,
        attention_heads,
        use_meta_ffn,
        normalize_metadata_features,
        unfreeze_last_n_decoder_blocks,
        unfreeze_lm_head,
        decoder_trainable_scope,
        unfreeze_shared_embeddings,
        metadata_fusion_mode,
        embedding_scaler_path,
        token_scaler_path,
        checkpoint_state_path,
    )

    model = MetadataEnrichedQualT5(
        model_name_or_path=base_init_path,
        lexical_feature_dim=lexical_feature_dim,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        scoring_mode=scoring_mode,
        metadata_mlp_hidden_dim=metadata_mlp_hidden_dim,
        metadata_dropout=float(metadata_dropout),
        metadata_projection_type=str(metadata_projection_type),
        attention_heads=int(attention_heads),
        use_meta_ffn=bool(use_meta_ffn),
        normalize_metadata_features=bool(normalize_metadata_features),
        unfreeze_last_n_decoder_blocks=int(unfreeze_last_n_decoder_blocks),
        unfreeze_lm_head=bool(unfreeze_lm_head),
        decoder_trainable_scope=str(decoder_trainable_scope),
        unfreeze_shared_embeddings=bool(unfreeze_shared_embeddings),
        metadata_fusion_mode=str(metadata_fusion_mode),
        lexical_feature_scaler_path=None,
        embedding_feature_scaler_path=embedding_scaler_path,
        token_feature_scaler_path=token_scaler_path,
    )

    if checkpoint_state_path is not None:
        LOGGER.info("Metadata model path è un Trainer checkpoint. Carico state_dict completo.")
        state_dict = _load_checkpoint_state_dict(checkpoint_state_path)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        LOGGER.info(
            "Checkpoint state_dict caricato. missing_keys=%d | unexpected_keys=%d",
            len(missing),
            len(unexpected),
        )

        if missing:
            LOGGER.warning("Missing keys preview: %s", missing[:30])

        if unexpected:
            LOGGER.warning("Unexpected keys preview: %s", unexpected[:30])

    else:
        if hasattr(model, "load_metadata_modules"):
            LOGGER.info("Carico moduli metadata da: %s", metadata_asset_dir)
            model.load_metadata_modules(metadata_asset_dir)
        else:
            LOGGER.warning("Il modello non espone load_metadata_modules(...).")

    model.to(device)
    model.eval()

    # For diagnostic comparison, force score = log P(true | true,false).
    model.scoring_mode = "true_logprob"

    try:
        first_param = next(model.parameters())
        LOGGER.info(
            "Metadata model device=%s dtype=%s",
            first_param.device,
            first_param.dtype,
        )
    except StopIteration:
        LOGGER.warning("Metadata model senza parametri?")

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
            "quality_score",
            "quality_scores",
            "scores",
            "score",
            "logprob_true",
            "true_logprob",
        ]:
            if key in outputs and outputs[key] is not None:
                return outputs[key].view(-1)

        if "pair_logits" in outputs:
            pair_logits = outputs["pair_logits"]
            log_probs = torch.log_softmax(pair_logits, dim=-1)
            return log_probs[:, 0]

        if "decoder_logits" in outputs:
            logits = outputs["decoder_logits"]
        elif "logits" in outputs:
            logits = outputs["logits"]
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
    lexical_scaler: MetadataFeatureScaler,
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    true_token_id: int,
    false_token_id: int,
) -> np.ndarray:
    LOGGER.info("Scoring metadata model...")

    scores: list[float] = []
    texts = df["text"].astype(str).tolist()
    docnos = df["docno"].astype(str).tolist()

    decoder_start_token_id = model.base_model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

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
        scaled_metadata = scale_metadata_features(lexical_scaler, raw_metadata)

        batch_size = len(batch_texts)

        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        batch = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "lexical_features": torch.tensor(
                scaled_metadata,
                dtype=torch.float32,
                device=device,
            ),
            "decoder_input_ids": decoder_input_ids,
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

    return {
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


def compute_pruning_metrics(
    df: pd.DataFrame,
    score_col: str,
    model_name: str,
    prune_fractions: list[float],
) -> list[dict[str, Any]]:
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

    metadata_model_path = Path(args.metadata_model_path)
    metadata_asset_dir = _metadata_asset_dir(metadata_model_path)

    LOGGER.info("metadata_model_path: %s", metadata_model_path)
    LOGGER.info("metadata_asset_dir: %s", metadata_asset_dir)

    metadata_cfg = _safe_load_metadata_config(args.metadata_model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, use_fast=True)
    true_token_id, false_token_id = get_true_false_token_ids(tokenizer)

    LOGGER.info("Carico lexical metadata store...")

    lexical_feature_names = metadata_cfg.get("lexical_feature_names")
    lexical_store = LexicalMetadataStore.from_path(
        args.metadata_path,
        feature_names=lexical_feature_names,
    )

    LOGGER.info("Lexical features: %s", lexical_store.feature_names)

    lexical_scaler_path = _get_cfg_path(
        args.lexical_scaler_path or args.metadata_scaler_path,
        metadata_cfg,
        ["lexical_scaler_path", "metadata_scaler_path"],
        metadata_asset_dir,
    )

    embedding_scaler_path = _get_cfg_path(
        args.embedding_scaler_path,
        metadata_cfg,
        ["embedding_feature_scaler_path", "embedding_scaler_path"],
        metadata_asset_dir,
    )

    token_scaler_path = _get_cfg_path(
        args.token_scaler_path,
        metadata_cfg,
        ["token_feature_scaler_path", "token_scaler_path"],
        metadata_asset_dir,
    )

    if lexical_scaler_path is None:
        raise ValueError(
            "Lexical scaler path non trovato. Passa --metadata_scaler_path oppure "
            "--lexical_scaler_path, oppure assicurati che metadata_qualt5_config.json "
            "contenga lexical_scaler_path/metadata_scaler_path."
        )

    if embedding_scaler_path is None:
        raise ValueError(
            "Embedding scaler path non trovato. Passa --embedding_scaler_path oppure "
            "assicurati che metadata_qualt5_config.json contenga embedding_scaler_path."
        )

    if token_scaler_path is None:
        raise ValueError(
            "Token scaler path non trovato. Passa --token_scaler_path oppure "
            "assicurati che metadata_qualt5_config.json contenga token_scaler_path."
        )

    LOGGER.info("Lexical scaler path: %s", lexical_scaler_path)
    LOGGER.info("Embedding scaler path: %s", embedding_scaler_path)
    LOGGER.info("Token scaler path: %s", token_scaler_path)

    LOGGER.info("Carico lexical scaler...")
    lexical_scaler = MetadataFeatureScaler.load(lexical_scaler_path)

    if list(lexical_store.feature_names) != list(lexical_scaler.feature_names):
        LOGGER.info(
            "Ricarico lexical_store usando feature_names dello scaler: %s",
            lexical_scaler.feature_names,
        )
        lexical_store = LexicalMetadataStore.from_path(
            args.metadata_path,
            feature_names=lexical_scaler.feature_names,
        )

    validate_scaler_features(lexical_scaler, lexical_store)
    LOGGER.info("Lexical scaler caricato e validato.")

    excluded_docnos = load_docno_set(args.exclude_docnos_path)

    if excluded_docnos:
        LOGGER.info("Docno caricati da exclude_docnos_path: %d", len(excluded_docnos))

    if args.fixed_sample_path is not None:
        df = load_fixed_diagnostic_sample(
            args.fixed_sample_path,
            excluded_docnos=excluded_docnos,
        )
    else:
        df = collect_balanced_examples(
            args=args,
            tokenizer=tokenizer,
            lexical_store=lexical_store,
            excluded_docnos=excluded_docnos,
        )

        clean_sample_path = (
            Path(args.save_sample_path).expanduser().resolve()
            if args.save_sample_path is not None
            else output_dir / f"diagnostic_sample_clean_{len(df)}.csv"
        )

        export_diagnostic_sample(
            df,
            clean_sample_path,
            docno_col="docno",
            text_col="text",
            label_col="label",
            extra_score_cols=[
                "raw_example_index",
                "scanned_after_skip_index",
                "text_length_chars",
            ],
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
        metadata_cfg=metadata_cfg,
        metadata_asset_dir=metadata_asset_dir,
        embedding_scaler_path=embedding_scaler_path,
        token_scaler_path=token_scaler_path,
    )

    df["score_metadata"] = score_metadata_model(
        model=metadata_model,
        tokenizer=tokenizer,
        lexical_store=lexical_store,
        lexical_scaler=lexical_scaler,
        df=df,
        args=args,
        device=device,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
    )

    labels = df["label"].to_numpy(dtype=np.int32)
    
    diagnostic_sample_path = output_dir / "diagnostic_sample_scored.csv"

    export_diagnostic_sample(
        df,
        diagnostic_sample_path,
        docno_col="docno",
        text_col="text",
        label_col="label",
        extra_score_cols=[
            "raw_example_index",
            "scanned_after_skip_index",
            "text_length_chars",
            "score_text_only",
            "score_metadata",
        ],
        
    )
    summaries = [
        summarize_scores(
            labels,
            df["score_text_only"].to_numpy(),
            "text_only_qualt5",
        ),
        summarize_scores(
            labels,
            df["score_metadata"].to_numpy(),
            "metadata_qualt5",
        ),
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


"""
# 1) Prima crea il sample held-out fisso
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
python -u -m metaqual.utils.diagnostic \
  --text_model_path /home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000 \
  --metadata_model_path /home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/scorer-checkpoint-18000 \
  --metadata_path /home/sacco/data/msmarco_passage/msmarco_passage_lexical_metadata.parquet \
  --output_dir /home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/diagnostic/checkpoint-18000 \
  --triples_source irds \
  --irds_dataset_id msmarco-passage/train/triples-small \
  --sample_per_class 10000 \
  --skip_first_raw_examples 1000000 \
  --max_raw_scan 3000000 \
  --save_sample_path /home/sacco/metaqual/outputs/diagnostics/samples/heldout_20k_skip1M.csv \
  --deduplicate_docnos \
  --metadata_projection_type linear \
  --metadata_fusion_mode concat_tokens \
  --decoder_trainable_scope full_decoder \
  --disable_meta_ffn \
  --batch_size 16 \
  --max_length 512 \
  --bf16

"""

"""
for CKPT in 8000 9000 10000 11000 12000 
do
  CUDA_VISIBLE_DEVICES=1 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  python -u -m metaqual.utils.diagnostic \
    --text_model_path /home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000 \
    --metadata_model_path /home/sacco/metaqual/outputs/metadata-qualt5-allmeta_token_projection-featureaware-noffnpostatt-fullDec-lr5e5-12k-V2/scorer-checkpoint-${CKPT} \
    --metadata_path /home/sacco/data/msmarco_passage/msmarco_passage_lexical_metadata.parquet \
    --output_dir /home/sacco/metaqual/outputs/metadata-qualt5-allmeta_token_projection-featureaware-noffnpostatt-fullDec-lr5e5-12k-V2/diagnostic/checkpoint-${CKPT} \
    --fixed_sample_path /home/sacco/metaqual/outputs/diagnostics/samples/heldout_20k_skip1M.csv \
    --metadata_projection_type linear \
    --metadata_fusion_mode allmeta_token_projection  \
    --decoder_trainable_scope full_decoder \
    --disable_meta_ffn \
    --batch_size 16 \
    --max_length 512 \
    --bf16
done

"""


"""
hf download RossSacco/metadata-qualt5-att_fusion-featureaware-noffnpostatt-fullDec-lr5e5-15k \
  --repo-type model \
  --include "scorer-checkpoint-12000/*" \
  --local-dir /home/sacco/metaqual/outputs/metadata-qualt5-att_fusion-featureaware-noffnpostatt-fullDec-lr5e5-15k
"""

"""
set -euo pipefail
REPO_ID="RossSacco/models-metaqual"
OUT="/home/sacco/metaqual/outputs/"
BASE_TMP="/tmp/hf_upload_checkpoints"
ROOT_TMP="$BASE_TMP/root_files"
export HF_XET_HIGH_PERFORMANCE=1
echo "=============================="
echo "Preparing root files"
echo "=============================="
rm -rf "$ROOT_TMP"
mkdir -p "$ROOT_TMP"
rsync -av "$OUT/" "$ROOT_TMP/" \
  --exclude "runs/" \
  --exclude "__pycache__/" \
  --exclude "*.log"
echo "=============================="
echo "Uploading root files"
echo "=============================="
hf upload-large-folder "$REPO_ID" "$ROOT_TMP" \
  --repo-type model \
  --num-workers 8
echo "Finished root files upload."
"""