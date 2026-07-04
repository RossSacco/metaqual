from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
import shutil
from transformers import TrainerCallback

try:
    from metaqual.data.loaders.msmarco.metadata_triples_dataset import (
        MetadataQualT5TriplesIterableDataset,
    )
    from metaqual.models.metadata_qualt5 import (
        DEFAULT_EMBEDDING_FEATURE_TRANSFORMS,
        DEFAULT_LEXICAL_FEATURE_TRANSFORMS,
        DEFAULT_TOKEN_FEATURE_TRANSFORMS,
        EMBEDDING_FEATURE_NAMES,
        TOKEN_FEATURE_NAMES,
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        RunningMoments,
        build_feature_transform_map,
        fit_online_metadata_scalers,
    )
except ImportError:
    from data.loaders.msmarco.metadata_triples_dataset import (
        MetadataQualT5TriplesIterableDataset,
    )
    from models.metadata_qualt5 import (
        DEFAULT_EMBEDDING_FEATURE_TRANSFORMS,
        DEFAULT_LEXICAL_FEATURE_TRANSFORMS,
        DEFAULT_TOKEN_FEATURE_TRANSFORMS,
        EMBEDDING_FEATURE_NAMES,
        TOKEN_FEATURE_NAMES,
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        RunningMoments,
        build_feature_transform_map,
        fit_online_metadata_scalers,
    )


LOGGER = logging.getLogger("train_metadata_qualt5")


class MetadataQualT5Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        text_examples = [
            {
                "input_ids": ex["input_ids"],
                "attention_mask": ex["attention_mask"],
            }
            for ex in examples
        ]

        encoded = self.tokenizer.pad(
            text_examples,
            padding=True,
            return_tensors="pt",
        )

        lexical = torch.tensor(
            [ex["lexical_features"] for ex in examples],
            dtype=torch.float32,
        )

        labels = torch.tensor(
            [ex["binary_labels"] for ex in examples],
            dtype=torch.long,
        )

        encoded["lexical_features"] = lexical
        encoded["binary_labels"] = labels

        return encoded


class MetadataScorerCheckpointCallback(TrainerCallback):
    """
    A ogni save del Trainer crea una cartella parallela:
        scorer-checkpoint-STEP/

    Questa cartella è pensata per inference/upload, NON per resume training.
    """

    def __init__(
        self,
        *,
        tokenizer,
        metadata_config_template: dict,
        lexical_scaler_path: str,
        embedding_scaler_path: str,
        token_scaler_path: str,
        metadata_feature_config_path: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.metadata_config_template = dict(metadata_config_template)
        self.lexical_scaler_path = lexical_scaler_path
        self.embedding_scaler_path = embedding_scaler_path
        self.token_scaler_path = token_scaler_path
        self.metadata_feature_config_path = metadata_feature_config_path

    def _copy_asset(self, src: str | None, dst_dir: Path) -> str | None:
        if src is None:
            return None

        src_path = Path(src)

        if not src_path.exists():
            raise FileNotFoundError(f"Asset non trovato: {src_path}")

        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)

        return dst_path.name

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")

        if model is None:
            return control

        output_root = Path(args.output_dir)
        export_dir = output_root / f"scorer-checkpoint-{state.global_step}"
        export_dir.mkdir(parents=True, exist_ok=True)

        print(f"[MetadataScorerCheckpointCallback] Salvo modello inference in: {export_dir}")

        # 1. Salva la parte T5 in formato HuggingFace standard.
        model.base_model.save_pretrained(export_dir)

        # 2. Salva tokenizer.
        self.tokenizer.save_pretrained(export_dir)

        # 3. Copia scaler nella cartella esportata.
        lexical_scaler_name = self._copy_asset(self.lexical_scaler_path, export_dir)
        embedding_scaler_name = self._copy_asset(self.embedding_scaler_path, export_dir)
        token_scaler_name = self._copy_asset(self.token_scaler_path, export_dir)

        # 4. Prepara config con path relativi locali.
        metadata_config = dict(self.metadata_config_template)

        metadata_config["lexical_scaler_path"] = lexical_scaler_name
        metadata_config["metadata_scaler_path"] = lexical_scaler_name

        metadata_config["embedding_scaler_path"] = embedding_scaler_name
        metadata_config["embedding_feature_scaler_path"] = embedding_scaler_name

        metadata_config["token_scaler_path"] = token_scaler_name
        metadata_config["token_feature_scaler_path"] = token_scaler_name

        # 5. Salva moduli metadata + metadata_qualt5_config.json.
        model.save_metadata_modules(export_dir, metadata_config)

        # 6. Copia anche metadata_feature_config.yaml, se esiste già.
        if self.metadata_feature_config_path is not None:
            src = Path(self.metadata_feature_config_path)
            if src.exists():
                shutil.copy2(src, export_dir / src.name)

        print(f"[MetadataScorerCheckpointCallback] Export completato: {export_dir}")

        return control
    
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _get_rss_gb() -> Optional[float]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    except Exception:
        return None


def _log_memory(prefix: str) -> None:
    rss_gb = _get_rss_gb()

    if rss_gb is not None:
        LOGGER.info("%s | RAM RSS: %.2f GB", prefix, rss_gb)
    else:
        LOGGER.info("%s | RAM RSS: non disponibile", prefix)


def _log_cuda_status(prefix: str) -> None:
    LOGGER.info("%s | CUDA_VISIBLE_DEVICES=%s", prefix, os.environ.get("CUDA_VISIBLE_DEVICES"))
    LOGGER.info("%s | torch.cuda.is_available=%s", prefix, torch.cuda.is_available())
    LOGGER.info("%s | torch.cuda.device_count=%d", prefix, torch.cuda.device_count())
    LOGGER.info("%s | torch.version.cuda=%s", prefix, torch.version.cuda)

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            LOGGER.info(
                "%s | CUDA device %d: %s",
                prefix,
                idx,
                torch.cuda.get_device_name(idx),
            )

        try:
            LOGGER.info("%s | bf16 supported=%s", prefix, torch.cuda.is_bf16_supported())
        except Exception:
            LOGGER.info("%s | bf16 supported=non verificabile", prefix)


def _log_path_status(path: str | Path, description: str) -> None:
    p = Path(path).expanduser()
    exists = p.exists()

    LOGGER.info("%s: %s", description, p)
    LOGGER.info("%s exists=%s", description, exists)

    if exists and p.is_file():
        size_gb = p.stat().st_size / (1024**3)
        LOGGER.info("%s size=%.3f GB", description, size_gb)


def _default_lexical_scaler_path(output_dir: str | Path) -> str:
    return str(Path(output_dir) / "lexical_metadata_scaler.pkl")


def _default_embedding_scaler_path(output_dir: str | Path) -> str:
    return str(Path(output_dir) / "embedding_metadata_scaler.pkl")


def _default_token_scaler_path(output_dir: str | Path) -> str:
    return str(Path(output_dir) / "token_metadata_scaler.pkl")


def _relative_if_inside_output(path: str | Path, output_dir: str | Path) -> str:
    path = Path(path)
    output_dir = Path(output_dir)

    try:
        if path.parent.resolve() == output_dir.resolve():
            return path.name
    except Exception:
        pass

    return str(path)



def _identity_transforms(feature_names) -> Dict[str, str]:
    return {name: "identity" for name in feature_names}


def _zscore_transforms(feature_names) -> Dict[str, str]:
    return {name: "zscore" for name in feature_names}


def _resolve_feature_transforms(
    *,
    feature_names,
    default_map: Dict[str, str],
    normalization_mode: str,
) -> Dict[str, str]:
    """
    Resolve the transform used by the scaler for each feature.

    normalization_mode:
    - zscore:        old behaviour, all features use z-score;
    - feature_aware: feature-specific transforms;
    - none:          no offline/online scaling, raw values are used.
    """
    normalization_mode = str(normalization_mode)

    if normalization_mode == "zscore":
        return _zscore_transforms(feature_names)

    if normalization_mode == "none":
        return _identity_transforms(feature_names)

    if normalization_mode == "feature_aware":
        return build_feature_transform_map(
            feature_names,
            default_map,
            fallback="zscore",
        )

    raise ValueError(
        "metadata_normalization_mode must be one of: zscore, feature_aware, none. "
        f"Got {normalization_mode!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training metadata-aware QualT5 from a fine-tuned QualT5 checkpoint."
    )

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--metadata_scaler_path", type=str, default=None)
    parser.add_argument("--lexical_scaler_path", type=str, default=None)

    parser.add_argument("--embedding_scaler_path", type=str, default=None)
    parser.add_argument("--token_scaler_path", type=str, default=None)

    parser.add_argument("--triples_source", type=str, choices=["file", "irds"], default="file")
    parser.add_argument("--triples_path", type=str, default=None)
    parser.add_argument("--triples_format", type=str, choices=["text", "id"], default="id")
    parser.add_argument("--collection_path", type=str, default=None)

    parser.add_argument("--irds_dataset_id", type=str, default="msmarco-passage/train/triples-small")
    parser.add_argument("--max_irds_triples", type=int, default=None)
    parser.add_argument("--irds_cache_dir", type=str, default=None)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--allow_missing_metadata", action="store_true")


    parser.add_argument(
        "--metadata_normalization_mode",
        type=str,
        choices=["zscore", "feature_aware", "none"],
        default="feature_aware",
        help=(
            "How metadata scalers are fitted. "
            "'zscore' reproduces the old behaviour. "
            "'feature_aware' uses log1p+zscore for count/norm features, "
            "identity for ratio features, and zscore for compact statistics. "
            "'none' disables scaling and keeps raw values."
        ),
    )

    parser.add_argument("--max_scaler_examples", type=int, default=500000)
    parser.add_argument(
        "--scaler_log_every",
        type=int,
        default=50_000,
        help="Log progress every N examples during lexical scaler fitting.",
    )

    parser.add_argument(
        "--max_online_scaler_examples",
        type=int,
        default=100000,
        help=(
            "Maximum number of tokenized training examples used to fit x_emb and x_tok scalers. "
            "Use -1 to scan all available examples."
        ),
    )
    parser.add_argument(
        "--online_scaler_batch_size",
        type=int,
        default=16,
        help="Batch size used only for fitting x_emb and x_tok scalers.",
    )

    parser.add_argument("--metadata_dropout", type=float, default=0.0)
    parser.add_argument("--metadata_mlp_hidden_dim", type=int, default=None)
    parser.add_argument(
        "--metadata_projection_type",
        type=str,
        choices=["linear", "mlp"],
        default="linear",
        help=(
            "How each metadata group is projected into the T5 hidden space. "
            "'linear' uses a single Linear(in_dim, d_model) without activation. "
            "'mlp' keeps the previous two-layer ReLU MLP for backward-compatible experiments."
        ),
    )
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--disable_meta_ffn", action="store_true")

    parser.add_argument(
        "--normalize_metadata_features",
        action="store_true",
        help=(
            "Enable LayerNorm before each metadata-group MLP. "
            "Default is disabled because metadata are standardized with scalers."
        ),
    )

    parser.add_argument(
        "--no_normalize_metadata_features",
        action="store_true",
        help="Deprecated. Kept for backward compatibility. Forces metadata input normalization off.",
    )

    parser.add_argument(
        "--unfreeze_last_n_decoder_blocks",
        type=int,
        default=1,
        help=(
            "Number of final decoder blocks to unfreeze. "
            "0 means only decoder cross-attention layers are trainable; "
            "1 means unfreeze the last decoder block; "
            "2 means unfreeze the last two decoder blocks, etc."
        ),
    )

    parser.add_argument(
        "--decoder_trainable_scope",
        type=str,
        choices=["cross_attention", "last_n_blocks", "full_decoder"],
        default="last_n_blocks",
        help=(
            "Which part of the decoder to train. "
            "'cross_attention' trains only decoder cross-attention layers. "
            "'last_n_blocks' trains decoder cross-attention plus the last N decoder blocks. "
            "'full_decoder' trains the whole decoder stack."
        ),
    )

    parser.add_argument(
        "--unfreeze_full_decoder",
        action="store_true",
        help="Shortcut for --decoder_trainable_scope full_decoder.",
    )

    parser.add_argument(
        "--unfreeze_shared_embeddings",
        action="store_true",
        help=(
            "When using full_decoder, also train T5 shared input embeddings. "
            "By default they stay frozen because they are shared with the encoder."
        ),
    )

    parser.add_argument(
        "--no_unfreeze_lm_head",
        action="store_true",
        help="Keep the LM head frozen. By default it is trainable.",
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
            "How to fuse metadata after Uni-Attention. "
            "'concat_tokens': decoder attends to [z_lex ; z_emb ; z_tok ; H_text]. "
            "'att_fusion': decoder attends only to Z_meta_fused. "
            "'pooled_concat_projection': H_fused = LayerNorm(H_text + MLP([H_text ; mean(Z_meta_fused)])). "
            "'allmeta_token_projection': H_fused = LayerNorm(H_text + ReLU-MLP([H_text ; z_lex ; z_emb ; z_tok])). "
            "'meta_prefix': decoder attends to [mean(Z_meta_fused) ; H_text]."
        ),
    )

    parser.add_argument(
        "--scoring_mode",
        type=str,
        choices=["true_logprob", "true_prob"],
        default="true_logprob",
    )

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def _fit_lexical_scaler_on_training_docnos(
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
    lexical_feature_transforms: Dict[str, str],
) -> MetadataFeatureScaler:
    LOGGER.info("Inizio fit scaler sui metadata lessicali dei docno del training...")
    _log_memory("Prima del fit lexical scaler")

    feature_dim = len(lexical_store.feature_names)

    LOGGER.info("Lexical feature dim: %d", feature_dim)
    LOGGER.info("Lexical feature names: %s", lexical_store.feature_names)

    identity_scaler = MetadataFeatureScaler(
        feature_names=lexical_store.feature_names,
        mean=np.zeros(feature_dim, dtype=np.float32),
        std=np.ones(feature_dim, dtype=np.float32),
        feature_transforms=_identity_transforms(lexical_store.feature_names),
    )

    dataset_for_fit = MetadataQualT5TriplesIterableDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        triples_source=args.triples_source,
        triples_path=args.triples_path,
        triples_format=args.triples_format,
        collection_path=args.collection_path,
        irds_dataset_id=args.irds_dataset_id,
        max_irds_triples=args.max_irds_triples,
        lexical_store=lexical_store,
        lexical_scaler=identity_scaler,
        allow_missing_metadata=args.allow_missing_metadata,
    )

    moments = RunningMoments(
        dim=feature_dim,
        feature_names=lexical_store.feature_names,
        feature_transforms=lexical_feature_transforms,
    )
    start_time = time.time()

    for idx, (docno, _passage, _label) in enumerate(dataset_for_fit.iter_raw_examples(), 1):
        values = lexical_store.lookup(
            [docno],
            allow_missing_metadata=args.allow_missing_metadata,
        )

        moments.update(values)

        if args.scaler_log_every > 0 and idx % args.scaler_log_every == 0:
            elapsed = time.time() - start_time
            examples_per_sec = idx / elapsed if elapsed > 0 else 0.0

            LOGGER.info(
                "Lexical scaler fit progress: %d esempi | %.2f ex/s | elapsed %.1f min",
                idx,
                examples_per_sec,
                elapsed / 60.0,
            )
            _log_memory("Durante fit lexical scaler")

        if args.max_scaler_examples is not None and idx >= args.max_scaler_examples:
            LOGGER.info(
                "Lexical scaler fit stop anticipato su max_scaler_examples=%d",
                args.max_scaler_examples,
            )
            break

    scaler = moments.finalize()

    elapsed = time.time() - start_time
    LOGGER.info(
        "Lexical scaler fit completato su %d esempi in %.1f min.",
        moments.count,
        elapsed / 60.0,
    )
    _log_memory("Dopo fit lexical scaler")

    return scaler


def _fit_and_save_online_scalers(
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
    lexical_scaler: MetadataFeatureScaler,
    embedding_scaler_path: str,
    token_scaler_path: str,
    embedding_feature_transforms: Dict[str, str],
    token_feature_transforms: Dict[str, str],
) -> tuple[MetadataFeatureScaler, MetadataFeatureScaler]:
    LOGGER.info("Inizio fit scaler online per x_emb e x_tok...")
    LOGGER.info("Embedding scaler path: %s", embedding_scaler_path)
    LOGGER.info("Token scaler path: %s", token_scaler_path)

    max_online_examples = args.max_online_scaler_examples

    if max_online_examples is not None and max_online_examples < 0:
        max_online_examples = None

    max_batches = None

    if max_online_examples is not None:
        max_batches = math.ceil(max_online_examples / args.online_scaler_batch_size)

    dataset_for_online_scaler = MetadataQualT5TriplesIterableDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        triples_source=args.triples_source,
        triples_path=args.triples_path,
        triples_format=args.triples_format,
        collection_path=args.collection_path,
        irds_dataset_id=args.irds_dataset_id,
        max_irds_triples=args.max_irds_triples,
        lexical_store=lexical_store,
        lexical_scaler=lexical_scaler,
        allow_missing_metadata=args.allow_missing_metadata,
    )

    dataloader_for_online_scaler = DataLoader(
        dataset_for_online_scaler,
        batch_size=args.online_scaler_batch_size,
        collate_fn=MetadataQualT5Collator(tokenizer),
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_time = time.time()

    emb_scaler, tok_scaler = fit_online_metadata_scalers(
        model_name_or_path=args.model_name_or_path,
        dataloader=dataloader_for_online_scaler,
        device=device,
        input_ids_key="input_ids",
        attention_mask_key="attention_mask",
        max_batches=max_batches,
        embedding_feature_transforms=embedding_feature_transforms,
        token_feature_transforms=token_feature_transforms,
    )

    emb_scaler.save(embedding_scaler_path)
    tok_scaler.save(token_scaler_path)

    elapsed = time.time() - start_time

    LOGGER.info(
        "Online scalers fit completato in %.1f min. max_online_examples=%s max_batches=%s",
        elapsed / 60.0,
        max_online_examples,
        max_batches,
    )
    LOGGER.info("Embedding scaler salvato in: %s", embedding_scaler_path)
    LOGGER.info("Token scaler salvato in: %s", token_scaler_path)

    _log_memory("Dopo fit online scalers")

    return emb_scaler, tok_scaler


def _save_reproducibility_files(args: argparse.Namespace, output_dir: str | Path) -> None:
    LOGGER.info("Salvo file di riproducibilità...")

    args_dict = vars(args)
    json_path = Path(output_dir) / "training_args.json"
    yaml_path = Path(output_dir) / "training_args.yaml"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=2, ensure_ascii=False)

    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(args_dict, handle, sort_keys=True, allow_unicode=True)

    LOGGER.info("File di riproducibilità salvati: %s, %s", json_path, yaml_path)


def _log_model_device(model: torch.nn.Module, prefix: str) -> None:
    try:
        first_param = next(model.parameters())
        LOGGER.info(
            "%s | first parameter device=%s dtype=%s requires_grad=%s",
            prefix,
            first_param.device,
            first_param.dtype,
            first_param.requires_grad,
        )
    except StopIteration:
        LOGGER.info("%s | modello senza parametri?", prefix)


def main() -> None:
    _setup_logging()
    args = parse_args()

    LOGGER.info("============================================================")
    LOGGER.info("Avvio train_metadata_qualt5")
    LOGGER.info("PID: %s", os.getpid())
    LOGGER.info("Output dir: %s", args.output_dir)
    LOGGER.info("Args: %s", vars(args))
    LOGGER.info("============================================================")

    _log_cuda_status("Startup")
    _log_memory("Startup")

    if args.bf16 and args.fp16:
        raise ValueError("Scegli solo una tra --bf16 e --fp16")

    if args.triples_source == "file":
        if not args.triples_path:
            raise ValueError("Con --triples_source file devi specificare --triples_path")

        if args.triples_format == "id" and not args.collection_path:
            raise ValueError(
                "Con triples_source=file e triples_format=id devi specificare --collection_path"
            )

    if args.irds_cache_dir:
        os.environ["IR_DATASETS_HOME"] = args.irds_cache_dir
        LOGGER.info("IR_DATASETS_HOME impostato a: %s", args.irds_cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Output directory pronta: %s", output_dir.resolve())

    _log_path_status(args.model_name_or_path, "Model/checkpoint path")
    _log_path_status(args.metadata_path, "Metadata parquet path")

    if args.triples_path:
        _log_path_status(args.triples_path, "Triples path")

    if args.collection_path:
        _log_path_status(args.collection_path, "Collection path")

    LOGGER.info("Carico tokenizer da: %s", args.model_name_or_path)
    tokenizer_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    LOGGER.info(
        "Tokenizer caricato in %.2f sec. vocab_size=%s pad_token=%s eos_token=%s",
        time.time() - tokenizer_start,
        getattr(tokenizer, "vocab_size", None),
        tokenizer.pad_token,
        tokenizer.eos_token,
    )

    _log_memory("Dopo caricamento tokenizer")

    LOGGER.info("Carico lexical metadata da: %s", args.metadata_path)
    metadata_start = time.time()

    lexical_store = LexicalMetadataStore.from_path(args.metadata_path)

    LOGGER.info(
        "Lexical metadata caricati in %.2f sec. feature_dim=%d",
        time.time() - metadata_start,
        len(lexical_store.feature_names),
    )
    LOGGER.info("Lexical feature names: %s", lexical_store.feature_names)

    lexical_feature_transforms = _resolve_feature_transforms(
        feature_names=lexical_store.feature_names,
        default_map=DEFAULT_LEXICAL_FEATURE_TRANSFORMS,
        normalization_mode=args.metadata_normalization_mode,
    )
    embedding_feature_transforms = _resolve_feature_transforms(
        feature_names=EMBEDDING_FEATURE_NAMES,
        default_map=DEFAULT_EMBEDDING_FEATURE_TRANSFORMS,
        normalization_mode=args.metadata_normalization_mode,
    )
    token_feature_transforms = _resolve_feature_transforms(
        feature_names=TOKEN_FEATURE_NAMES,
        default_map=DEFAULT_TOKEN_FEATURE_TRANSFORMS,
        normalization_mode=args.metadata_normalization_mode,
    )

    LOGGER.info("metadata_normalization_mode=%s", args.metadata_normalization_mode)
    LOGGER.info("Lexical feature transforms: %s", lexical_feature_transforms)
    LOGGER.info("Embedding feature transforms: %s", embedding_feature_transforms)
    LOGGER.info("Token feature transforms: %s", token_feature_transforms)

    _log_memory("Dopo caricamento lexical metadata")

    lexical_scaler_path = (
        args.lexical_scaler_path
        or args.metadata_scaler_path
        or _default_lexical_scaler_path(args.output_dir)
    )

    if Path(lexical_scaler_path).exists():
        LOGGER.info("Carico lexical scaler già esistente da: %s", lexical_scaler_path)
        lexical_scaler = MetadataFeatureScaler.load(lexical_scaler_path)
        if getattr(lexical_scaler, "feature_transforms", None) != lexical_feature_transforms:
            LOGGER.warning(
                "Il lexical scaler esistente usa transforms diversi da quelli richiesti. "
                "Se vuoi rifittare la feature-aware normalization, elimina lo scaler o usa un nuovo output_dir. "
                "loaded=%s requested=%s",
                getattr(lexical_scaler, "feature_transforms", None),
                lexical_feature_transforms,
            )
    else:
        LOGGER.info("Lexical scaler non trovato. Lo calcolo ora...")
        lexical_scaler = _fit_lexical_scaler_on_training_docnos(
            args=args,
            tokenizer=tokenizer,
            lexical_store=lexical_store,
            lexical_feature_transforms=lexical_feature_transforms,
        )

        LOGGER.info("Salvo lexical scaler in: %s", lexical_scaler_path)
        lexical_scaler.save(lexical_scaler_path)
        LOGGER.info("Lexical scaler salvato correttamente.")

    _log_memory("Dopo lexical scaler")

    embedding_scaler_path = args.embedding_scaler_path or _default_embedding_scaler_path(
        args.output_dir
    )
    token_scaler_path = args.token_scaler_path or _default_token_scaler_path(args.output_dir)

    embedding_exists = Path(embedding_scaler_path).exists()
    token_exists = Path(token_scaler_path).exists()

    if embedding_exists and token_exists:
        LOGGER.info("Online scalers già esistenti. Skip fit.")
        LOGGER.info("Embedding scaler: %s", embedding_scaler_path)
        LOGGER.info("Token scaler: %s", token_scaler_path)

        loaded_emb_scaler = MetadataFeatureScaler.load(embedding_scaler_path)
        loaded_tok_scaler = MetadataFeatureScaler.load(token_scaler_path)

        if getattr(loaded_emb_scaler, "feature_transforms", None) != embedding_feature_transforms:
            LOGGER.warning(
                "L'embedding scaler esistente usa transforms diversi da quelli richiesti. "
                "loaded=%s requested=%s",
                getattr(loaded_emb_scaler, "feature_transforms", None),
                embedding_feature_transforms,
            )

        if getattr(loaded_tok_scaler, "feature_transforms", None) != token_feature_transforms:
            LOGGER.warning(
                "Il token scaler esistente usa transforms diversi da quelli richiesti. "
                "loaded=%s requested=%s",
                getattr(loaded_tok_scaler, "feature_transforms", None),
                token_feature_transforms,
            )
    else:
        LOGGER.info("Almeno uno tra embedding/token scaler non esiste. Calcolo online scalers...")
        _fit_and_save_online_scalers(
            args=args,
            tokenizer=tokenizer,
            lexical_store=lexical_store,
            lexical_scaler=lexical_scaler,
            embedding_scaler_path=embedding_scaler_path,
            token_scaler_path=token_scaler_path,
            embedding_feature_transforms=embedding_feature_transforms,
            token_feature_transforms=token_feature_transforms,
        )

    _log_memory("Dopo online scalers")

    LOGGER.info("Calcolo token id per true/false...")

    true_token_ids = tokenizer.encode("true", add_special_tokens=False)
    false_token_ids = tokenizer.encode("false", add_special_tokens=False)

    LOGGER.info("true_token_ids=%s", true_token_ids)
    LOGGER.info("false_token_ids=%s", false_token_ids)

    if len(true_token_ids) == 0 or len(false_token_ids) == 0:
        raise ValueError("Non riesco a ricavare true_token_id o false_token_id.")

    true_token_id = true_token_ids[0]
    false_token_id = false_token_ids[0]

    LOGGER.info("true_token_id=%s | false_token_id=%s", true_token_id, false_token_id)

    normalize_metadata_features = (
        bool(args.normalize_metadata_features)
        and not bool(args.no_normalize_metadata_features)
    )

    decoder_trainable_scope = (
        "full_decoder" if bool(args.unfreeze_full_decoder) else args.decoder_trainable_scope
    )

    LOGGER.info("Creo modello MetadataEnrichedQualT5...")
    LOGGER.info("metadata_dropout=%s", args.metadata_dropout)
    LOGGER.info("metadata_projection_type=%s", args.metadata_projection_type)
    LOGGER.info("decoder_trainable_scope=%s", decoder_trainable_scope)
    LOGGER.info("unfreeze_shared_embeddings=%s", args.unfreeze_shared_embeddings)
    LOGGER.info("normalize_metadata_features=%s", normalize_metadata_features)
    LOGGER.info("metadata_fusion_mode=%s", args.metadata_fusion_mode)
    LOGGER.info("embedding_feature_scaler_path=%s", embedding_scaler_path)
    LOGGER.info("token_feature_scaler_path=%s", token_scaler_path)
    LOGGER.info("lexical_features are already scaled by the dataset; no lexical scaler is passed to model.")

    model_start = time.time()

    model = MetadataEnrichedQualT5(
        model_name_or_path=args.model_name_or_path,
        lexical_feature_dim=len(lexical_store.feature_names),
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        scoring_mode=args.scoring_mode,
        metadata_mlp_hidden_dim=args.metadata_mlp_hidden_dim,
        metadata_dropout=args.metadata_dropout,
        metadata_projection_type=args.metadata_projection_type,
        attention_heads=args.attention_heads,
        use_meta_ffn=not args.disable_meta_ffn,
        normalize_metadata_features=normalize_metadata_features,
        unfreeze_last_n_decoder_blocks=args.unfreeze_last_n_decoder_blocks,
        unfreeze_lm_head=not args.no_unfreeze_lm_head,
        decoder_trainable_scope=decoder_trainable_scope,
        unfreeze_shared_embeddings=args.unfreeze_shared_embeddings,
        metadata_fusion_mode=args.metadata_fusion_mode,
        lexical_feature_scaler_path=None,
        embedding_feature_scaler_path=embedding_scaler_path,
        token_feature_scaler_path=token_scaler_path,
    )

    LOGGER.info("Modello creato in %.2f sec.", time.time() - model_start)

    _log_model_device(model, "Dopo creazione modello")
    _log_memory("Dopo creazione modello")
    _log_cuda_status("Dopo creazione modello")

    stats = model.get_trainable_parameter_stats()
    LOGGER.info("Trainable parameters: %s / %s", stats["trainable"], stats["total"])

    if hasattr(model, "get_trainable_parameter_summary"):
        summary = model.get_trainable_parameter_summary(max_items=160)
        LOGGER.info(
            "Trainable tensors: %d | total_trainable=%d",
            summary["num_trainable_tensors"],
            summary["total_trainable"],
        )

        for item in summary["trainable_preview"]:
            LOGGER.info(
                "TRAINABLE | %s | shape=%s | params=%d",
                item["name"],
                item["shape"],
                item["num_params"],
            )

    LOGGER.info("Creo train_dataset...")
    dataset_start = time.time()

    train_dataset = MetadataQualT5TriplesIterableDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        triples_source=args.triples_source,
        triples_path=args.triples_path,
        triples_format=args.triples_format,
        collection_path=args.collection_path,
        irds_dataset_id=args.irds_dataset_id,
        max_irds_triples=args.max_irds_triples,
        lexical_store=lexical_store,
        lexical_scaler=lexical_scaler,
        allow_missing_metadata=args.allow_missing_metadata,
    )

    LOGGER.info("train_dataset creato in %.2f sec.", time.time() - dataset_start)
    _log_memory("Dopo creazione train_dataset")

    LOGGER.info("Creo TrainingArguments...")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    LOGGER.info("TrainingArguments creati.")

    LOGGER.info("Creo Trainer...")
    trainer_start = time.time()
    
    lexical_scaler_cfg_value = Path(lexical_scaler_path).name
    embedding_scaler_cfg_value = Path(embedding_scaler_path).name
    token_scaler_cfg_value = Path(token_scaler_path).name

    metadata_config_template = {
        "model_type": "metadata_qualt5",
        "base_model_name_or_path": args.model_name_or_path,

        "lexical_feature_names": lexical_store.feature_names,
        "embedding_feature_names": EMBEDDING_FEATURE_NAMES,
        "token_feature_names": TOKEN_FEATURE_NAMES,

        "scoring_mode": args.scoring_mode,
        "metadata_dropout": args.metadata_dropout,
        "metadata_mlp_hidden_dim": args.metadata_mlp_hidden_dim,
        "metadata_projection_type": args.metadata_projection_type,
        "attention_heads": args.attention_heads,
        "use_meta_ffn": not args.disable_meta_ffn,
        "normalize_metadata_features": normalize_metadata_features,

        "unfreeze_last_n_decoder_blocks": args.unfreeze_last_n_decoder_blocks,
        "unfreeze_lm_head": not args.no_unfreeze_lm_head,
        "decoder_trainable_scope": decoder_trainable_scope,
        "unfreeze_full_decoder": bool(args.unfreeze_full_decoder),
        "unfreeze_shared_embeddings": bool(args.unfreeze_shared_embeddings),

        "metadata_fusion_mode": args.metadata_fusion_mode,
        "metadata_normalization_mode": args.metadata_normalization_mode,

        "lexical_feature_transforms": lexical_feature_transforms,
        "embedding_feature_transforms": embedding_feature_transforms,
        "token_feature_transforms": token_feature_transforms,

        "lexical_scaler_path": lexical_scaler_cfg_value,
        "metadata_scaler_path": lexical_scaler_cfg_value,

        "embedding_scaler_path": embedding_scaler_cfg_value,
        "embedding_feature_scaler_path": embedding_scaler_cfg_value,

        "token_scaler_path": token_scaler_cfg_value,
        "token_feature_scaler_path": token_scaler_cfg_value,
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MetadataQualT5Collator(tokenizer),
        tokenizer=tokenizer,
        callbacks=[
            MetadataScorerCheckpointCallback(
                tokenizer=tokenizer,
                metadata_config_template=metadata_config_template,
                lexical_scaler_path=lexical_scaler_path,
                embedding_scaler_path=embedding_scaler_path,
                token_scaler_path=token_scaler_path,
                metadata_feature_config_path=str(output_dir / "metadata_feature_config.yaml"),
            )
        ],
    )

    LOGGER.info("Trainer creato in %.2f sec.", time.time() - trainer_start)

    _log_model_device(model, "Dopo creazione Trainer")
    _log_memory("Dopo creazione Trainer")
    _log_cuda_status("Dopo creazione Trainer")

    LOGGER.info("============================================================")
    LOGGER.info("Inizio training metadata_qualt5...")
    LOGGER.info("resume_from_checkpoint=%s", args.resume_from_checkpoint)
    LOGGER.info("============================================================")

    train_start = time.time()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    LOGGER.info("Training finito in %.1f min.", (time.time() - train_start) / 60.0)

    _log_model_device(model, "Dopo training")
    _log_memory("Dopo training")

    LOGGER.info("Salvo checkpoint finale base_model + tokenizer...")

    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    LOGGER.info("Base model e tokenizer salvati.")

    lexical_scaler_cfg_value = _relative_if_inside_output(lexical_scaler_path, output_dir)
    embedding_scaler_cfg_value = _relative_if_inside_output(embedding_scaler_path, output_dir)
    token_scaler_cfg_value = _relative_if_inside_output(token_scaler_path, output_dir)

    metadata_config = {
        "model_type": "metadata_qualt5",
        "base_model_name_or_path": args.model_name_or_path,
        "lexical_feature_names": lexical_store.feature_names,
        "embedding_feature_names": EMBEDDING_FEATURE_NAMES,
        "token_feature_names": TOKEN_FEATURE_NAMES,
        "scoring_mode": args.scoring_mode,
        "metadata_dropout": args.metadata_dropout,
        "metadata_mlp_hidden_dim": args.metadata_mlp_hidden_dim,
        "metadata_projection_type": args.metadata_projection_type,
        "attention_heads": args.attention_heads,
        "use_meta_ffn": not args.disable_meta_ffn,
        "normalize_metadata_features": normalize_metadata_features,
        "unfreeze_last_n_decoder_blocks": args.unfreeze_last_n_decoder_blocks,
        "unfreeze_lm_head": not args.no_unfreeze_lm_head,
        "decoder_trainable_scope": decoder_trainable_scope,
        "unfreeze_full_decoder": bool(args.unfreeze_full_decoder),
        "unfreeze_shared_embeddings": bool(args.unfreeze_shared_embeddings),
        "metadata_fusion_mode": args.metadata_fusion_mode,
        "metadata_normalization_mode": args.metadata_normalization_mode,
        "lexical_feature_transforms": lexical_feature_transforms,
        "embedding_feature_transforms": embedding_feature_transforms,
        "token_feature_transforms": token_feature_transforms,
        "lexical_scaler_path": lexical_scaler_cfg_value,
        "metadata_scaler_path": lexical_scaler_cfg_value,
        "embedding_scaler_path": embedding_scaler_cfg_value,
        "token_scaler_path": token_scaler_cfg_value,
        "embedding_feature_scaler_path": embedding_scaler_cfg_value,
        "token_feature_scaler_path": token_scaler_cfg_value,
    }

    LOGGER.info("Salvo metadata modules...")

    model.save_metadata_modules(output_dir, metadata_config)

    LOGGER.info("Metadata modules salvati.")

    _save_reproducibility_files(args, output_dir)

    meta_yaml_path = output_dir / "metadata_feature_config.yaml"
    LOGGER.info("Salvo metadata_feature_config.yaml in: %s", meta_yaml_path)

    with meta_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "scorers": {
                    "metadata_qualt5": {
                        "metadata_feature_groups": {
                            "lexical": lexical_store.feature_names,
                            "embedding": EMBEDDING_FEATURE_NAMES,
                            "token": TOKEN_FEATURE_NAMES,
                        },
                        "scalers": {
                            "lexical": lexical_scaler_cfg_value,
                            "embedding": embedding_scaler_cfg_value,
                            "token": token_scaler_cfg_value,
                        },
                        "normalization": {
                            "mode": args.metadata_normalization_mode,
                            "lexical_feature_transforms": lexical_feature_transforms,
                            "embedding_feature_transforms": embedding_feature_transforms,
                            "token_feature_transforms": token_feature_transforms,
                        },
                    }
                }
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    LOGGER.info("metadata_feature_config.yaml salvato.")
    LOGGER.info("Training completato. Output: %s", output_dir)
    LOGGER.info("============================================================")


if __name__ == "__main__":
    main()
    
    
'''
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup python -u -m metaqual.models.train_metadata_qualt5 \
  --model_name_or_path /home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000 \
  --metadata_path /home/sacco/data/msmarco_passage/msmarco_passage_lexical_metadata.parquet \
  --output_dir /home/sacco/metaqual/outputs/metadata-qualt5-allmeta_token_projection-featureaware-noffnpostatt-fullDec-lr5e5-12k-V2 \
  --triples_source irds \
  --irds_dataset_id msmarco-passage/train/triples-small \
  --max_steps 12000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --max_length 512 \
  --save_steps 1000 \
  --save_total_limit 12 \
  --logging_steps 50 \
  --metadata_dropout 0.1 \
  --metadata_projection_type linear \
  --metadata_fusion_mode allmeta_token_projection \
  --metadata_normalization_mode feature_aware \
  --disable_meta_ffn \
  --decoder_trainable_scope full_decoder \
  --max_scaler_examples 500000 \
  --max_online_scaler_examples 100000 \
  --online_scaler_batch_size 16 \
  --bf16 \
  > metadata_qualt5_3.log 2>&1 &

'''


"""
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup python -u -m metaqual.models.train_metadata_qualt5 \
  --model_name_or_path /home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000 \
  --metadata_path /home/sacco/data/msmarco_passage/msmarco_passage_lexical_metadata.parquet \
  --output_dir /home/sacco/metaqual/outputs/metadata-qualt5-allmeta_token_projection-featureaware-noffnpostatt-fullDec-lr5e5-12k-V2 \
  --triples_source irds \
  --irds_dataset_id msmarco-passage/train/triples-small \
  --max_steps 12000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --max_length 512 \
  --save_steps 1000 \
  --save_total_limit 10 \
  --logging_steps 50 \
  --metadata_dropout 0.1 \
  --metadata_projection_type linear \
  --metadata_fusion_mode allmeta_token_projection \
  --metadata_normalization_mode feature_aware \
  --disable_meta_ffn \
  --decoder_trainable_scope full_decoder \
  --max_scaler_examples 500000 \
  --max_online_scaler_examples 100000 \
  --online_scaler_batch_size 16 \
  --bf16 \
  --resume_from_checkpoint /home/sacco/metaqual/outputs/metadata-qualt5-concat-featureaware-fullDec-noffnatt-lr5e5-13k/checkpoint-15000 \
  > finetuning.log 2>&1 &
"""