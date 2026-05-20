from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer

try:
    from metaqual.data.loaders.msmarco.metadata_triples_dataset import MetadataQualT5TriplesIterableDataset
    from metaqual.models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        RunningMoments,
        EMBEDDING_FEATURE_NAMES,
        TOKEN_FEATURE_NAMES,
    )
except ImportError:
    from data.loaders.msmarco.metadata_triples_dataset import MetadataQualT5TriplesIterableDataset
    from models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        RunningMoments,
        EMBEDDING_FEATURE_NAMES,
        TOKEN_FEATURE_NAMES,
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


def _default_scaler_path(output_dir: str) -> str:
    return str(Path(output_dir) / "metadata_scaler.pkl")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training metadata_qualt5 a partire da un checkpoint QualT5 fine-tuned."
    )

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_scaler_path", type=str, default=None)

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
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--allow_missing_metadata", action="store_true")
    parser.add_argument("--max_scaler_examples", type=int, default=500000)
    parser.add_argument(
        "--scaler_log_every",
        type=int,
        default=50_000,
        help="Log progress every N examples during scaler fitting.",
    )

    parser.add_argument("--metadata_dropout", type=float, default=0.1)
    parser.add_argument("--metadata_mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--disable_meta_ffn", action="store_true")
    parser.add_argument(
        "--no_normalize_metadata_features",
        action="store_true",
        help="Disable LayerNorm before each metadata-group MLP.",
    )

    parser.add_argument(
        "--no_unfreeze_last_decoder_block",
        action="store_true",
        help="Keep the last decoder block frozen. By default it is trainable.",
    )

    parser.add_argument(
        "--no_unfreeze_lm_head",
        action="store_true",
        help="Keep the LM head frozen. By default it is trainable.",
    )
    parser.add_argument(
        "--metadata_fusion_mode",
        type=str,
        choices=["concat_tokens", "gated_residual"],
        default="gated_residual",
        help=(
            "How to fuse metadata after Uni-Attention. "
            "'concat_tokens' keeps the old behaviour; "
            "'gated_residual' injects pooled metadata into text hidden states."
        ),
    )

    parser.add_argument(
        "--metadata_gate_init_bias",
        type=float,
        default=-2.0,
        help=(
            "Initial bias for the metadata gate. "
            "With -2.0, sigmoid(-2) is about 0.12, so metadata correction starts small."
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


def _fit_scaler_on_training_docnos(
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
) -> MetadataFeatureScaler:
    LOGGER.info("Inizio fit scaler sui docno del training...")
    _log_memory("Prima del fit scaler")

    feature_dim = len(lexical_store.feature_names)
    LOGGER.info("Feature dim scaler: %d", feature_dim)
    LOGGER.info("Feature names: %s", lexical_store.feature_names)

    identity_scaler = MetadataFeatureScaler(
        feature_names=lexical_store.feature_names,
        mean=np.zeros(feature_dim, dtype=np.float32),
        std=np.ones(feature_dim, dtype=np.float32),
    )

    LOGGER.info("Creo dataset temporaneo per fit scaler...")
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
    LOGGER.info("Dataset temporaneo per fit scaler creato.")

    moments = RunningMoments(dim=feature_dim)
    start_time = time.time()

    LOGGER.info("Inizio iterazione raw examples per fit scaler...")

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
                "Scaler fit progress: %d esempi | %.2f ex/s | elapsed %.1f min",
                idx,
                examples_per_sec,
                elapsed / 60.0,
            )
            _log_memory("Durante fit scaler")

        if args.max_scaler_examples is not None and idx >= args.max_scaler_examples:
            LOGGER.info(
                "Scaler fit stop anticipato su max_scaler_examples=%d",
                args.max_scaler_examples,
            )
            break

    scaler = moments.finalize(feature_names=lexical_store.feature_names)

    elapsed = time.time() - start_time
    LOGGER.info(
        "Scaler fit completato su %d esempi in %.1f min.",
        moments.count,
        elapsed / 60.0,
    )
    _log_memory("Dopo fit scaler")

    return scaler


def _save_reproducibility_files(args: argparse.Namespace, output_dir: str) -> None:
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
    _log_memory("Dopo caricamento lexical metadata")

    LOGGER.info("Inizio calcolo scaler metadata...")
    scaler = _fit_scaler_on_training_docnos(
        args=args,
        tokenizer=tokenizer,
        lexical_store=lexical_store,
    )

    scaler_path = args.metadata_scaler_path or _default_scaler_path(args.output_dir)
    LOGGER.info("Salvo scaler in: %s", scaler_path)
    scaler.save(scaler_path)
    LOGGER.info("Scaler salvato correttamente.")
    _log_memory("Dopo salvataggio scaler")

    LOGGER.info("Calcolo token id per true/false...")
    true_token_ids = tokenizer.encode("true", add_special_tokens=False)
    false_token_ids = tokenizer.encode("false", add_special_tokens=False)

    LOGGER.info("true_token_ids=%s", true_token_ids)
    LOGGER.info("false_token_ids=%s", false_token_ids)

    true_token_id = true_token_ids[0]
    false_token_id = false_token_ids[0]

    LOGGER.info("true_token_id=%s | false_token_id=%s", true_token_id, false_token_id)

    LOGGER.info("Creo modello MetadataEnrichedQualT5...")
    model_start = time.time()
    model = MetadataEnrichedQualT5(
        model_name_or_path=args.model_name_or_path,
        lexical_feature_dim=len(lexical_store.feature_names),
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        scoring_mode=args.scoring_mode,
        metadata_mlp_hidden_dim=args.metadata_mlp_hidden_dim,
        metadata_dropout=args.metadata_dropout,
        attention_heads=args.attention_heads,
        use_meta_ffn=not args.disable_meta_ffn,
        normalize_metadata_features=not args.no_normalize_metadata_features,
        unfreeze_last_decoder_block=not args.no_unfreeze_last_decoder_block,
        unfreeze_lm_head=not args.no_unfreeze_lm_head,
        metadata_fusion_mode=args.metadata_fusion_mode,
        metadata_gate_init_bias=args.metadata_gate_init_bias,
    )
    LOGGER.info("Modello creato in %.2f sec.", time.time() - model_start)
    _log_model_device(model, "Dopo creazione modello")
    _log_memory("Dopo creazione modello")
    _log_cuda_status("Dopo creazione modello")

    stats = model.get_trainable_parameter_stats()
    LOGGER.info("Trainable parameters: %s / %s", stats["trainable"], stats["total"])

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
        lexical_scaler=scaler,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MetadataQualT5Collator(tokenizer),
        tokenizer=tokenizer,
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

    scaler_cfg_value = str(scaler_path)
    if Path(scaler_path).parent.resolve() == output_dir.resolve():
        scaler_cfg_value = Path(scaler_path).name

    metadata_config = {
        "model_type": "metadata_qualt5",
        "base_model_name_or_path": args.model_name_or_path,
        "lexical_feature_names": lexical_store.feature_names,
        "embedding_feature_names": EMBEDDING_FEATURE_NAMES,
        "token_feature_names": TOKEN_FEATURE_NAMES,
        "scoring_mode": args.scoring_mode,
        "metadata_dropout": args.metadata_dropout,
        "metadata_mlp_hidden_dim": args.metadata_mlp_hidden_dim,
        "attention_heads": args.attention_heads,
        "use_meta_ffn": not args.disable_meta_ffn,
        "normalize_metadata_features": not args.no_normalize_metadata_features,
        "unfreeze_last_decoder_block": not args.no_unfreeze_last_decoder_block,
        "unfreeze_lm_head": not args.no_unfreeze_lm_head,
        "metadata_fusion_mode": args.metadata_fusion_mode,
        "metadata_gate_init_bias": args.metadata_gate_init_bias,
        "metadata_scaler_path": scaler_cfg_value,
    }
    LOGGER.info("Salvo metadata modules...")
    model.save_metadata_modules(output_dir, metadata_config)
    LOGGER.info("Metadata modules salvati.")

    _save_reproducibility_files(args, str(output_dir))

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
                        }
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
  --output_dir /home/sacco/metaqual/outputs/metadata-qualt5-gated-lr2e5-h256-10k \
  --triples_source irds \
  --irds_dataset_id msmarco-passage/train/triples-small \
  --max_steps 10000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --max_length 512 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --logging_steps 50 \
  --metadata_mlp_hidden_dim 256 \
  --metadata_fusion_mode gated_residual \
  --metadata_gate_init_bias -2.0 \
  --bf16 \
  > metaqualt5_gated.log 2>&1 &


'''