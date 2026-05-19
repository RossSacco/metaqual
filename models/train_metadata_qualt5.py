from __future__ import annotations

import argparse
import json
import logging
import os
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
        labels = torch.tensor([ex["binary_labels"] for ex in examples], dtype=torch.long)

        encoded["lexical_features"] = lexical
        encoded["binary_labels"] = labels
        return encoded


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


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
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--allow_missing_metadata", action="store_true")
    parser.add_argument("--max_scaler_examples", type=int, default=None)

    parser.add_argument("--metadata_dropout", type=float, default=0.1)
    parser.add_argument("--metadata_mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--disable_meta_ffn", action="store_true")
    parser.add_argument("--scoring_mode", type=str, choices=["true_logprob", "true_prob"], default="true_logprob")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def _fit_scaler_on_training_docnos(
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
) -> MetadataFeatureScaler:
    feature_dim = len(lexical_store.feature_names)
    identity_scaler = MetadataFeatureScaler(
        feature_names=lexical_store.feature_names,
        mean=np.zeros(feature_dim, dtype=np.float32),
        std=np.ones(feature_dim, dtype=np.float32),
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

    moments = RunningMoments(dim=feature_dim)

    for idx, (docno, _passage, _label) in enumerate(dataset_for_fit.iter_raw_examples(), 1):
        values = lexical_store.lookup([docno], allow_missing_metadata=args.allow_missing_metadata)
        moments.update(values)

        if args.max_scaler_examples is not None and idx >= args.max_scaler_examples:
            LOGGER.info("Scaler fit stop anticipato su max_scaler_examples=%d", args.max_scaler_examples)
            break

    scaler = moments.finalize(feature_names=lexical_store.feature_names)
    LOGGER.info("Scaler fit completato su %d esempi.", moments.count)
    return scaler


def _save_reproducibility_files(args: argparse.Namespace, output_dir: str) -> None:
    args_dict = vars(args)
    json_path = Path(output_dir) / "training_args.json"
    yaml_path = Path(output_dir) / "training_args.yaml"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=2, ensure_ascii=False)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(args_dict, handle, sort_keys=True, allow_unicode=True)


def main() -> None:
    _setup_logging()
    args = parse_args()

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    lexical_store = LexicalMetadataStore.from_path(args.metadata_path)
    scaler = _fit_scaler_on_training_docnos(
        args=args,
        tokenizer=tokenizer,
        lexical_store=lexical_store,
    )

    scaler_path = args.metadata_scaler_path or _default_scaler_path(args.output_dir)
    scaler.save(scaler_path)
    LOGGER.info("Scaler salvato in: %s", scaler_path)

    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("false", add_special_tokens=False)[0]

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
    )

    stats = model.get_trainable_parameter_stats()
    LOGGER.info("Trainable parameters: %s / %s", stats["trainable"], stats["total"])

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

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        weight_decay=0.01,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MetadataQualT5Collator(tokenizer),
        tokenizer=tokenizer,
    )

    LOGGER.info("Inizio training metadata_qualt5...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Salvataggio checkpoint finale.
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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
        "metadata_scaler_path": scaler_cfg_value,
    }
    model.save_metadata_modules(output_dir, metadata_config)
    _save_reproducibility_files(args, str(output_dir))

    meta_yaml_path = output_dir / "metadata_feature_config.yaml"
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

    LOGGER.info("Training completato. Output: %s", output_dir)


if __name__ == "__main__":
    main()
