from __future__ import annotations

"""
Two-step joint training for MetadataEnrichedQualT5.

Stage 1 - metadata interaction/fusion warm-up
------------------------------------------------
The complete QualT5 backbone (encoder, decoder, LM head and shared embeddings)
is frozen. The forward pass is unchanged: three distinct metadata tokens are
projected, used as queries in Uni-Attention, and fused with H_text. The loss is
still the final true/false cross-entropy produced through the frozen decoder.

Stage 2 - joint fine-tuning
---------------------------
The encoder remains frozen. The metadata branch and fusion remain trainable,
while the requested decoder scope is unfrozen. A fresh optimizer and scheduler
are created with separate learning rates for:
  * metadata group projections;
  * Uni-Attention / metadata interaction / active fusion;
  * decoder (and optional output/shared parameters).

Each Trainer checkpoint is accompanied by an inference-ready directory:
    scorer-checkpoint-STEP/
containing the standard Hugging Face T5 weights, tokenizer, metadata modules,
metadata config and all three scalers.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

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


LOGGER = logging.getLogger("train_metadata_qualt5_two_step")
NON_POOLED_FUSION_MODES = {
    "concat_tokens",
    "att_fusion",
    "allmeta_token_projection",
}


# -----------------------------------------------------------------------------
# Data collation
# -----------------------------------------------------------------------------


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

        encoded["lexical_features"] = torch.tensor(
            [ex["lexical_features"] for ex in examples],
            dtype=torch.float32,
        )
        encoded["binary_labels"] = torch.tensor(
            [ex["binary_labels"] for ex in examples],
            dtype=torch.long,
        )
        return encoded


# -----------------------------------------------------------------------------
# Trainer and checkpoint export
# -----------------------------------------------------------------------------


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def _copy_asset(src: str | Path | None, dst_dir: str | Path) -> Optional[str]:
    if src is None:
        return None

    src_path = Path(src).expanduser().resolve()
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = (dst_dir / src_path.name).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Asset non trovato: {src_path}")

    if src_path != dst_path:
        shutil.copy2(src_path, dst_path)

    return dst_path.name


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _prepare_stage_assets(
    *,
    stage_dir: str | Path,
    tokenizer,
    metadata_config: Dict[str, Any],
    metadata_feature_config: Dict[str, Any],
    lexical_scaler_path: str,
    embedding_scaler_path: str,
    token_scaler_path: str,
) -> Dict[str, str]:
    """
    Make the stage parent directory sufficient for loading checkpoint-* directly.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    lexical_name = _copy_asset(lexical_scaler_path, stage_dir)
    embedding_name = _copy_asset(embedding_scaler_path, stage_dir)
    token_name = _copy_asset(token_scaler_path, stage_dir)

    tokenizer.save_pretrained(stage_dir)

    cfg = dict(metadata_config)
    cfg.update(
        {
            "lexical_scaler_path": lexical_name,
            "metadata_scaler_path": lexical_name,
            "embedding_scaler_path": embedding_name,
            "embedding_feature_scaler_path": embedding_name,
            "token_scaler_path": token_name,
            "token_feature_scaler_path": token_name,
        }
    )
    _write_json(stage_dir / "metadata_qualt5_config.json", cfg)
    _write_yaml(stage_dir / "metadata_feature_config.yaml", metadata_feature_config)

    return {
        "lexical": str(stage_dir / lexical_name),
        "embedding": str(stage_dir / embedding_name),
        "token": str(stage_dir / token_name),
    }


def _export_inference_model(
    *,
    model: torch.nn.Module,
    tokenizer,
    export_dir: str | Path,
    metadata_config: Dict[str, Any],
    metadata_feature_config: Dict[str, Any],
    lexical_scaler_path: str,
    embedding_scaler_path: str,
    token_scaler_path: str,
) -> None:
    model = _unwrap_model(model)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model.base_model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    lexical_name = _copy_asset(lexical_scaler_path, export_dir)
    embedding_name = _copy_asset(embedding_scaler_path, export_dir)
    token_name = _copy_asset(token_scaler_path, export_dir)

    cfg = dict(metadata_config)
    cfg.update(
        {
            "lexical_scaler_path": lexical_name,
            "metadata_scaler_path": lexical_name,
            "embedding_scaler_path": embedding_name,
            "embedding_feature_scaler_path": embedding_name,
            "token_scaler_path": token_name,
            "token_feature_scaler_path": token_name,
        }
    )

    model.save_metadata_modules(export_dir, cfg)
    _write_yaml(export_dir / "metadata_feature_config.yaml", metadata_feature_config)


def _parse_prune_fractions(value: str) -> list[float]:
    fractions: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fraction = float(item)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Invalid diagnostic prune fraction: {fraction}")
        fractions.append(fraction)
    if not fractions:
        raise ValueError("At least one diagnostic prune fraction is required")
    return fractions


def _fraction_key(fraction: float) -> str:
    return f"{int(round(fraction * 100)):03d}"


def _diagnostic_sample_fingerprint(df: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{docno}\t{label}\t{text}"
        for docno, label, text in zip(df["docno"], df["label"], df["text"])
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_diagnostic_sample(
    path: str | Path,
    *,
    max_examples: Optional[int],
    seed: int,
) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic sample not found: {path}")

    df = pd.read_csv(path)
    required = {"docno", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Diagnostic sample must contain {sorted(required)}; missing={sorted(missing)}"
        )

    df = df.copy()
    df["docno"] = df["docno"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    if max_examples is not None and max_examples > 0 and len(df) > max_examples:
        per_class = max_examples // 2
        if per_class < 1:
            raise ValueError("diagnostic_max_examples must be >= 2")
        positive = df[df["label"] == 1].sample(n=per_class, random_state=seed)
        negative = df[df["label"] == 0].sample(n=per_class, random_state=seed)
        df = pd.concat([positive, negative], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = df["label"].value_counts().to_dict()
    if 0 not in counts or 1 not in counts:
        raise ValueError(f"Diagnostic sample needs both classes; counts={counts}")

    LOGGER.info(
        "Diagnostic sample loaded: %s | rows=%d | label_counts=%s",
        path,
        len(df),
        counts,
    )
    return df


def _scale_metadata_features(
    scaler: MetadataFeatureScaler,
    values: np.ndarray,
) -> np.ndarray:
    if hasattr(scaler, "transform"):
        return scaler.transform(values).astype(np.float32)
    mean = np.asarray(scaler.mean, dtype=np.float32)
    std = np.asarray(scaler.std, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values.astype(np.float32) - mean) / std).astype(np.float32)


def _summarize_scores(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    threshold = float(np.median(scores))
    predictions = (scores >= threshold).astype(np.int32)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy_median_threshold": float(accuracy_score(labels, predictions)),
        "threshold_median": threshold,
        "positive_mean": float(np.mean(positives)),
        "positive_std": float(np.std(positives)),
        "negative_mean": float(np.mean(negatives)),
        "negative_std": float(np.std(negatives)),
        "mean_gap_pos_minus_neg": float(np.mean(positives) - np.mean(negatives)),
    }


def _compute_pruning_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    fractions: Sequence[float],
) -> Dict[float, Dict[str, float]]:
    order = np.argsort(scores)
    total = len(labels)
    total_positive = int((labels == 1).sum())
    total_negative = int((labels == 0).sum())
    output: Dict[float, Dict[str, float]] = {}

    for fraction in fractions:
        n_pruned = int(round(total * fraction))
        pruned_idx = order[:n_pruned]
        kept_idx = order[n_pruned:]
        positives_pruned = int((labels[pruned_idx] == 1).sum())
        negatives_pruned = int((labels[pruned_idx] == 0).sum())
        positives_kept = int((labels[kept_idx] == 1).sum())
        negatives_kept = int((labels[kept_idx] == 0).sum())
        threshold = (
            float(scores[order[n_pruned - 1]]) if n_pruned > 0 else float("-inf")
        )
        output[float(fraction)] = {
            "threshold": threshold,
            "n_pruned": int(n_pruned),
            "positives_kept": positives_kept,
            "positives_pruned": positives_pruned,
            "negatives_kept": negatives_kept,
            "negatives_pruned": negatives_pruned,
            "kept_positive_rate": float(positives_kept / total_positive),
            "removed_negative_rate": float(negatives_pruned / total_negative),
            "positive_loss": float(positives_pruned / total_positive),
            "kept_negative_rate": float(negatives_kept / total_negative),
        }
    return output


class DiagnosticEvaluator:
    """
    Fixed held-out diagnostic evaluated directly on the in-memory training model.

    The QualT5-Finetuned baseline is scored once and cached. At every requested
    checkpoint only MetadataEnrichedQualT5 is evaluated.
    """

    VALID_SELECTION_METRICS = {
        "auc",
        "delta_auc",
        "mean_kept_positive_rate",
        "mean_delta_kept_positive_rate",
    }

    def __init__(
        self,
        *,
        sample_path: str,
        baseline_model_path: str,
        tokenizer,
        lexical_store: LexicalMetadataStore,
        lexical_scaler: MetadataFeatureScaler,
        output_root: str | Path,
        batch_size: int,
        max_length: int,
        prompt_template: str,
        prune_fractions: Sequence[float],
        selection_metric: str,
        max_examples: Optional[int],
        sample_seed: int,
        baseline_bf16: bool,
        save_scores: bool,
    ):
        self.sample_path = str(sample_path)
        self.baseline_model_path = str(baseline_model_path)
        self.tokenizer = tokenizer
        self.lexical_store = lexical_store
        self.lexical_scaler = lexical_scaler
        self.output_root = Path(output_root)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.prompt_template = str(prompt_template)
        self.prune_fractions = [float(value) for value in prune_fractions]
        self.baseline_bf16 = bool(baseline_bf16)
        self.save_scores = bool(save_scores)

        dynamic_metrics = set()
        for fraction in self.prune_fractions:
            suffix = _fraction_key(fraction)
            dynamic_metrics.add(f"kept_positive_rate_{suffix}")
            dynamic_metrics.add(f"delta_kept_positive_rate_{suffix}")
        self.valid_selection_metrics = self.VALID_SELECTION_METRICS | dynamic_metrics
        self.selection_metric = self.validate_selection_metric(selection_metric)

        self.df = _load_diagnostic_sample(
            sample_path,
            max_examples=max_examples,
            seed=sample_seed,
        )
        self.labels = self.df["label"].to_numpy(dtype=np.int32)
        self.fingerprint = _diagnostic_sample_fingerprint(self.df)

        raw_metadata = self.lexical_store.lookup(
            self.df["docno"].tolist(),
            allow_missing_metadata=False,
        )
        self.scaled_lexical_features = _scale_metadata_features(
            self.lexical_scaler,
            np.asarray(raw_metadata, dtype=np.float32),
        )

        self.cache_dir = self.output_root / "diagnostic_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_scores, self.baseline_summary, self.baseline_pruning = (
            self._load_or_compute_baseline()
        )

    def _make_prompts(self, texts: Sequence[str]) -> list[str]:
        return [self.prompt_template.format(text=text) for text in texts]

    def _baseline_cache_paths(self) -> tuple[Path, Path]:
        return (
            self.cache_dir / "baseline_scores.csv",
            self.cache_dir / "baseline_metadata.json",
        )

    def _load_or_compute_baseline(
        self,
    ) -> tuple[np.ndarray, Dict[str, float], Dict[float, Dict[str, float]]]:
        scores_path, metadata_path = self._baseline_cache_paths()
        if scores_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("sample_fingerprint") == self.fingerprint
                and metadata.get("baseline_model_path") == self.baseline_model_path
            ):
                cached = pd.read_csv(scores_path)
                if len(cached) == len(self.df):
                    scores = cached["score_text_only"].to_numpy(dtype=np.float32)
                    summary = _summarize_scores(self.labels, scores)
                    pruning = _compute_pruning_metrics(
                        self.labels, scores, self.prune_fractions
                    )
                    LOGGER.info("Loaded diagnostic baseline cache: %s", scores_path)
                    return scores, summary, pruning

        LOGGER.info("Computing fixed QualT5-Finetuned diagnostic baseline...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (
            torch.bfloat16
            if self.baseline_bf16 and device.type == "cuda"
            else torch.float32
        )
        baseline_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.baseline_model_path,
            torch_dtype=dtype,
        ).to(device)
        baseline_model.eval()

        true_ids = self.tokenizer.encode("true", add_special_tokens=False)
        false_ids = self.tokenizer.encode("false", add_special_tokens=False)
        if not true_ids or not false_ids:
            raise ValueError("Unable to encode true/false for diagnostic baseline")
        true_token_id = int(true_ids[0])
        false_token_id = int(false_ids[0])

        decoder_start_token_id = baseline_model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id

        scores: list[float] = []
        texts = self.df["text"].tolist()
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    self._make_prompts(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                decoder_input_ids = torch.full(
                    (len(batch_texts), 1),
                    decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
                outputs = baseline_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )
                first_logits = outputs.logits[:, 0, :]
                pair_logits = torch.stack(
                    [
                        first_logits[:, true_token_id],
                        first_logits[:, false_token_id],
                    ],
                    dim=-1,
                )
                scores.extend(
                    torch.log_softmax(pair_logits, dim=-1)[:, 0]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )

        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        score_array = np.asarray(scores, dtype=np.float32)
        summary = _summarize_scores(self.labels, score_array)
        pruning = _compute_pruning_metrics(
            self.labels, score_array, self.prune_fractions
        )
        pd.DataFrame(
            {
                "docno": self.df["docno"],
                "label": self.labels,
                "score_text_only": score_array,
            }
        ).to_csv(scores_path, index=False)
        _write_json(
            metadata_path,
            {
                "sample_path": self.sample_path,
                "sample_fingerprint": self.fingerprint,
                "rows": len(self.df),
                "baseline_model_path": self.baseline_model_path,
                "summary": summary,
                "pruning": {str(k): v for k, v in pruning.items()},
            },
        )
        LOGGER.info("Saved diagnostic baseline cache: %s", scores_path)
        return score_array, summary, pruning

    def _score_metadata_model(
        self,
        model: torch.nn.Module,
    ) -> tuple[np.ndarray, float]:
        """Return true log-probabilities and held-out binary cross-entropy."""
        model = _unwrap_model(model)
        device = next(model.parameters()).device
        decoder_start_token_id = model.base_model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id

        texts = self.df["text"].tolist()
        scores: list[float] = []
        total_loss = 0.0
        total_examples = 0
        old_scoring_mode = model.scoring_mode
        was_training = model.training
        model.scoring_mode = "true_logprob"
        model.eval()

        try:
            with torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    end = min(start + self.batch_size, len(texts))
                    batch_texts = texts[start:end]
                    batch_binary_labels = torch.tensor(
                        self.labels[start:end],
                        dtype=torch.long,
                        device=device,
                    )
                    encoded = self.tokenizer(
                        self._make_prompts(batch_texts),
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    batch = {
                        "input_ids": encoded["input_ids"].to(device),
                        "attention_mask": encoded["attention_mask"].to(device),
                        "lexical_features": torch.tensor(
                            self.scaled_lexical_features[start:end],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "binary_labels": batch_binary_labels,
                        "decoder_input_ids": torch.full(
                            (end - start, 1),
                            decoder_start_token_id,
                            dtype=torch.long,
                            device=device,
                        ),
                    }
                    outputs = model(**batch)
                    if "quality_score" in outputs:
                        batch_scores = outputs["quality_score"]
                    else:
                        batch_scores = torch.log_softmax(
                            outputs["pair_logits"], dim=-1
                        )[:, 0]

                    batch_loss = outputs.get("loss")
                    if batch_loss is None:
                        target_idx = torch.where(
                            batch_binary_labels > 0,
                            torch.zeros_like(batch_binary_labels),
                            torch.ones_like(batch_binary_labels),
                        )
                        batch_loss = F.cross_entropy(
                            outputs["pair_logits"], target_idx
                        )

                    batch_n = end - start
                    total_loss += float(batch_loss.detach().float().cpu()) * batch_n
                    total_examples += batch_n
                    scores.extend(batch_scores.detach().float().cpu().tolist())
        finally:
            model.scoring_mode = old_scoring_mode
            if was_training:
                model.train()

        if total_examples == 0:
            raise RuntimeError("Diagnostic sample produced zero validation examples")

        return (
            np.asarray(scores, dtype=np.float32),
            float(total_loss / total_examples),
        )

    def validate_selection_metric(self, selection_metric: str) -> str:
        selection_metric = str(selection_metric)
        if selection_metric not in self.valid_selection_metrics:
            raise ValueError(
                f"Unsupported diagnostic selection metric={selection_metric!r}. "
                f"Supported={sorted(self.valid_selection_metrics)}"
            )
        return selection_metric

    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        stage_name: str,
        step: int,
        stage_dir: str | Path,
        scorer_checkpoint_path: str | Path,
        selection_metric: Optional[str] = None,
        recent_training_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        LOGGER.info(
            "Automatic diagnostic | stage=%s step=%d rows=%d",
            stage_name,
            step,
            len(self.df),
        )
        metadata_scores, validation_loss = self._score_metadata_model(model)
        metadata_summary = _summarize_scores(self.labels, metadata_scores)
        metadata_pruning = _compute_pruning_metrics(
            self.labels, metadata_scores, self.prune_fractions
        )

        row: Dict[str, Any] = {
            "stage": stage_name,
            "step": int(step),
            "scorer_checkpoint_path": str(scorer_checkpoint_path),
            "sample_fingerprint": self.fingerprint,
            "sample_rows": int(len(self.df)),
            "validation_loss": float(validation_loss),
            "recent_training_loss": (
                float(recent_training_loss)
                if recent_training_loss is not None
                else None
            ),
            "generalization_gap": (
                float(validation_loss - recent_training_loss)
                if recent_training_loss is not None
                else None
            ),
            "metadata_auc": metadata_summary["auc"],
            "baseline_auc": self.baseline_summary["auc"],
            "delta_auc": metadata_summary["auc"] - self.baseline_summary["auc"],
            "accuracy_median_threshold": metadata_summary["accuracy_median_threshold"],
            "threshold_median": metadata_summary["threshold_median"],
            "positive_mean": metadata_summary["positive_mean"],
            "negative_mean": metadata_summary["negative_mean"],
            "mean_gap_pos_minus_neg": metadata_summary["mean_gap_pos_minus_neg"],
        }

        kept_values = []
        delta_kept_values = []
        pruning_rows = []
        for fraction in self.prune_fractions:
            suffix = _fraction_key(fraction)
            current = metadata_pruning[fraction]
            baseline = self.baseline_pruning[fraction]
            kept = current["kept_positive_rate"]
            delta_kept = kept - baseline["kept_positive_rate"]
            row[f"kept_positive_rate_{suffix}"] = kept
            row[f"baseline_kept_positive_rate_{suffix}"] = baseline[
                "kept_positive_rate"
            ]
            row[f"delta_kept_positive_rate_{suffix}"] = delta_kept
            row[f"removed_negative_rate_{suffix}"] = current[
                "removed_negative_rate"
            ]
            kept_values.append(kept)
            delta_kept_values.append(delta_kept)
            pruning_rows.append(
                {
                    "stage": stage_name,
                    "step": int(step),
                    "prune_fraction": fraction,
                    **{f"metadata_{key}": value for key, value in current.items()},
                    **{f"baseline_{key}": value for key, value in baseline.items()},
                    "delta_kept_positive_rate": delta_kept,
                }
            )

        row["auc"] = row["metadata_auc"]
        row["mean_kept_positive_rate"] = float(np.mean(kept_values))
        row["mean_delta_kept_positive_rate"] = float(np.mean(delta_kept_values))

        effective_selection_metric = self.validate_selection_metric(
            selection_metric or self.selection_metric
        )
        row["selection_metric"] = effective_selection_metric
        row["selection_score"] = float(row[effective_selection_metric])

        diagnostic_root = Path(stage_dir) / "diagnostic"
        checkpoint_dir = diagnostic_root / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_json(checkpoint_dir / "diagnostic_summary.json", row)
        pd.DataFrame(pruning_rows).to_csv(
            checkpoint_dir / "diagnostic_pruning_metrics.csv", index=False
        )
        if self.save_scores:
            scored = self.df[["docno", "text", "label"]].copy()
            scored["score_text_only"] = self.baseline_scores
            scored["score_metadata"] = metadata_scores
            scored.to_csv(checkpoint_dir / "diagnostic_scores.csv", index=False)

        history_path = diagnostic_root / "diagnostic_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            history = history[history["step"] != int(step)]
            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        else:
            history = pd.DataFrame([row])
        history = history.sort_values("step").reset_index(drop=True)
        history.to_csv(history_path, index=False)
        _write_json(
            diagnostic_root / "diagnostic_history.json",
            {"rows": history.to_dict(orient="records")},
        )

        LOGGER.info(
            "Diagnostic result | stage=%s step=%d | val_loss=%.6f | "
            "train_loss=%s | metadata_auc=%.6f | delta_auc=%+.6f | %s=%.6f",
            stage_name,
            step,
            row["validation_loss"],
            (
                f"{row['recent_training_loss']:.6f}"
                if row["recent_training_loss"] is not None
                else "n/a"
            ),
            row["metadata_auc"],
            row["delta_auc"],
            effective_selection_metric,
            row["selection_score"],
        )
        return row


    def update_persisted_row(
        self,
        *,
        stage_dir: str | Path,
        step: int,
        row: Dict[str, Any],
    ) -> None:
        """Rewrite summary/history after callback-level overfitting checks."""
        diagnostic_root = Path(stage_dir) / "diagnostic"
        checkpoint_dir = diagnostic_root / f"checkpoint-{step}"
        _write_json(checkpoint_dir / "diagnostic_summary.json", row)

        history_path = diagnostic_root / "diagnostic_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            history = history[history["step"] != int(step)]
            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        else:
            history = pd.DataFrame([row])
        history = history.sort_values("step").reset_index(drop=True)
        history.to_csv(history_path, index=False)
        _write_json(
            diagnostic_root / "diagnostic_history.json",
            {"rows": history.where(pd.notna(history), None).to_dict(orient="records")},
        )


class StageScorerCheckpointCallback(TrainerCallback):
    """
    Export every scorer checkpoint and optionally run automatic diagnostics.

    Best-model selection and patience are stage-local. The baseline/sample cache
    is shared by the DiagnosticEvaluator across both stages.
    """

    def __init__(
        self,
        *,
        tokenizer,
        stage_name: str,
        metadata_config_template: Dict[str, Any],
        metadata_feature_config: Dict[str, Any],
        lexical_scaler_path: str,
        embedding_scaler_path: str,
        token_scaler_path: str,
        diagnostic_evaluator: Optional[DiagnosticEvaluator] = None,
        diagnostic_selection_metric: Optional[str] = None,
        diagnostic_every_steps: int = 0,
        diagnostic_patience: int = 0,
        diagnostic_min_delta: float = 0.0,
        overfitting_patience: int = 0,
        validation_loss_min_delta: float = 0.0,
        max_validation_loss_increase: float = float("inf"),
        stop_on_overfitting: bool = False,
        loss_guard_best_checkpoint: bool = True,
    ):
        self.tokenizer = tokenizer
        self.stage_name = stage_name
        self.metadata_config_template = dict(metadata_config_template)
        self.metadata_feature_config = metadata_feature_config
        self.lexical_scaler_path = lexical_scaler_path
        self.embedding_scaler_path = embedding_scaler_path
        self.token_scaler_path = token_scaler_path
        self.diagnostic_evaluator = diagnostic_evaluator
        if diagnostic_evaluator is not None:
            metric = diagnostic_selection_metric or diagnostic_evaluator.selection_metric
            self.diagnostic_selection_metric = diagnostic_evaluator.validate_selection_metric(metric)
        else:
            self.diagnostic_selection_metric = str(diagnostic_selection_metric or "auc")
        self.diagnostic_every_steps = int(diagnostic_every_steps)
        self.diagnostic_patience = int(diagnostic_patience)
        self.diagnostic_min_delta = float(diagnostic_min_delta)
        self.overfitting_patience = int(overfitting_patience)
        self.validation_loss_min_delta = float(validation_loss_min_delta)
        self.max_validation_loss_increase = float(max_validation_loss_increase)
        self.stop_on_overfitting = bool(stop_on_overfitting)
        self.loss_guard_best_checkpoint = bool(loss_guard_best_checkpoint)
        self.last_diagnostic_step = -1
        self.best_score = float("-inf")
        self.best_step: Optional[int] = None
        self.best_checkpoint_path: Optional[str] = None
        self.bad_evaluations = 0
        self.best_validation_loss = float("inf")
        self.previous_validation_loss: Optional[float] = None
        self.previous_training_loss: Optional[float] = None
        self.consecutive_overfitting_signals = 0

    def _restore_previous_best(self, output_dir: Path) -> None:
        history_path = output_dir / "diagnostic" / "diagnostic_history.csv"
        if not history_path.exists() or self.diagnostic_evaluator is None:
            return
        try:
            history = pd.read_csv(history_path)
            if history.empty or "selection_score" not in history.columns:
                return
            if "selection_metric" in history.columns:
                history = history[
                    history["selection_metric"].astype(str)
                    == self.diagnostic_selection_metric
                ]
            if history.empty:
                return
            best_idx = history["selection_score"].astype(float).idxmax()
            best = history.loc[best_idx]
            self.best_score = float(best["selection_score"])
            self.best_step = int(best["step"])
            self.best_checkpoint_path = str(best["scorer_checkpoint_path"])
            self.last_diagnostic_step = int(history["step"].max())
            if "validation_loss" in history.columns:
                valid_losses = pd.to_numeric(history["validation_loss"], errors="coerce").dropna()
                if not valid_losses.empty:
                    self.best_validation_loss = float(valid_losses.min())
                    latest = history.sort_values("step").iloc[-1]
                    self.previous_validation_loss = float(latest["validation_loss"])
                    if (
                        "recent_training_loss" in history.columns
                        and pd.notna(latest.get("recent_training_loss"))
                    ):
                        self.previous_training_loss = float(latest["recent_training_loss"])
            LOGGER.info(
                "Restored previous diagnostic best | stage=%s metric=%s step=%s score=%.6f",
                self.stage_name,
                self.diagnostic_selection_metric,
                self.best_step,
                self.best_score,
            )
        except Exception as exc:
            LOGGER.warning("Unable to restore previous diagnostic history: %s", exc)

    def on_train_begin(self, args, state, control, **kwargs):
        self._restore_previous_best(Path(args.output_dir))
        return control

    def _recent_training_loss(self, state, current_step: int) -> Optional[float]:
        losses = []
        for entry in getattr(state, "log_history", []):
            if "loss" not in entry:
                continue
            entry_step = int(entry.get("step", 0))
            if self.last_diagnostic_step < entry_step <= current_step:
                try:
                    losses.append(float(entry["loss"]))
                except (TypeError, ValueError):
                    continue
        if not losses:
            return None
        return float(np.mean(losses))

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control

        step = int(state.global_step)
        export_dir = Path(args.output_dir) / f"scorer-checkpoint-{step}"
        cfg = dict(self.metadata_config_template)
        cfg["training_stage"] = self.stage_name
        cfg["training_stage_step"] = step

        LOGGER.info("Exporting scorer checkpoint: %s", export_dir)
        _export_inference_model(
            model=model,
            tokenizer=self.tokenizer,
            export_dir=export_dir,
            metadata_config=cfg,
            metadata_feature_config=self.metadata_feature_config,
            lexical_scaler_path=self.lexical_scaler_path,
            embedding_scaler_path=self.embedding_scaler_path,
            token_scaler_path=self.token_scaler_path,
        )

        should_evaluate = (
            self.diagnostic_evaluator is not None
            and self.diagnostic_every_steps > 0
            and (
                self.last_diagnostic_step < 0
                or step - self.last_diagnostic_step >= self.diagnostic_every_steps
            )
        )
        if not should_evaluate:
            return control

        recent_training_loss = self._recent_training_loss(state, step)
        row = self.diagnostic_evaluator.evaluate(
            model=model,
            stage_name=self.stage_name,
            step=step,
            stage_dir=args.output_dir,
            scorer_checkpoint_path=export_dir,
            selection_metric=self.diagnostic_selection_metric,
            recent_training_loss=recent_training_loss,
        )
        self.last_diagnostic_step = step

        validation_loss = float(row["validation_loss"])
        best_validation_loss_before = self.best_validation_loss
        validation_loss_improved = (
            validation_loss
            < self.best_validation_loss - self.validation_loss_min_delta
        )
        if validation_loss_improved:
            self.best_validation_loss = validation_loss

        loss_increase_from_best = max(
            0.0, validation_loss - self.best_validation_loss
        )
        train_loss_decreasing = (
            recent_training_loss is not None
            and self.previous_training_loss is not None
            and recent_training_loss < self.previous_training_loss
        )
        validation_loss_rising = (
            self.previous_validation_loss is not None
            and validation_loss
            > self.previous_validation_loss + self.validation_loss_min_delta
        )
        overfitting_signal = bool(train_loss_decreasing and validation_loss_rising)
        if overfitting_signal:
            self.consecutive_overfitting_signals += 1
        else:
            self.consecutive_overfitting_signals = 0

        severe_loss_degradation = bool(
            math.isfinite(self.max_validation_loss_increase)
            and loss_increase_from_best > self.max_validation_loss_increase
        )
        overfitting_detected = bool(
            severe_loss_degradation
            or (
                self.overfitting_patience > 0
                and self.consecutive_overfitting_signals
                >= self.overfitting_patience
            )
        )
        loss_guard_passed = bool(
            not self.loss_guard_best_checkpoint
            or not severe_loss_degradation
        )

        row.update(
            {
                "best_validation_loss_so_far": float(self.best_validation_loss),
                "validation_loss_increase_from_best": float(loss_increase_from_best),
                "validation_loss_improved": bool(validation_loss_improved),
                "train_loss_decreasing": bool(train_loss_decreasing),
                "validation_loss_rising": bool(validation_loss_rising),
                "overfitting_signal": bool(overfitting_signal),
                "consecutive_overfitting_signals": int(
                    self.consecutive_overfitting_signals
                ),
                "severe_loss_degradation": bool(severe_loss_degradation),
                "overfitting_detected": bool(overfitting_detected),
                "loss_guard_passed": bool(loss_guard_passed),
                "max_validation_loss_increase": (
                    float(self.max_validation_loss_increase)
                    if math.isfinite(self.max_validation_loss_increase)
                    else None
                ),
            }
        )
        self.diagnostic_evaluator.update_persisted_row(
            stage_dir=args.output_dir,
            step=step,
            row=row,
        )

        score = float(row["selection_score"])
        improved = (
            loss_guard_passed
            and score > self.best_score + self.diagnostic_min_delta
        )

        self.previous_validation_loss = validation_loss
        if recent_training_loss is not None:
            self.previous_training_loss = recent_training_loss

        if improved:
            self.best_score = score
            self.best_step = step
            self.best_checkpoint_path = str(export_dir)
            self.bad_evaluations = 0
            _write_json(
                Path(args.output_dir) / "diagnostic" / "best_diagnostic.json",
                {
                    **row,
                    "best_checkpoint_path": self.best_checkpoint_path,
                    "diagnostic_selection_metric": self.diagnostic_selection_metric,
                    "diagnostic_min_delta": self.diagnostic_min_delta,
                },
            )
            LOGGER.info(
                "New best diagnostic checkpoint | stage=%s metric=%s step=%d score=%.6f",
                self.stage_name,
                self.diagnostic_selection_metric,
                step,
                score,
            )
        else:
            self.bad_evaluations += 1
            LOGGER.info(
                "No diagnostic improvement | stage=%s step=%d | bad=%d/%d | "
                "loss_guard=%s | val_loss=%.6f | best_val_loss=%.6f",
                self.stage_name,
                step,
                self.bad_evaluations,
                self.diagnostic_patience,
                loss_guard_passed,
                validation_loss,
                self.best_validation_loss,
            )

        if overfitting_detected:
            LOGGER.warning(
                "Overfitting detected | stage=%s step=%d | train_loss=%s | "
                "val_loss=%.6f | best_val_loss=%.6f | consecutive=%d",
                self.stage_name,
                step,
                (
                    f"{recent_training_loss:.6f}"
                    if recent_training_loss is not None
                    else "n/a"
                ),
                validation_loss,
                self.best_validation_loss,
                self.consecutive_overfitting_signals,
            )
            if self.stop_on_overfitting:
                control.should_training_stop = True

        if (
            self.diagnostic_patience > 0
            and self.bad_evaluations >= self.diagnostic_patience
        ):
            LOGGER.info(
                "Diagnostic early stopping requested | stage=%s best_step=%s best_score=%.6f",
                self.stage_name,
                self.best_step,
                self.best_score,
            )
            control.should_training_stop = True
        return control

    def finalize_best_alias(self, stage_dir: str | Path) -> Optional[str]:
        if self.best_checkpoint_path is None:
            return None
        source = Path(self.best_checkpoint_path)
        if not source.exists():
            LOGGER.warning("Best scorer checkpoint no longer exists: %s", source)
            return self.best_checkpoint_path
        destination = Path(stage_dir) / "best-scorer"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        LOGGER.info("Best scorer alias created: %s -> %s", destination, source)
        return str(destination)


class DifferentialLRTrainer(Trainer):
    """
    Trainer with explicit optimizer groups.

    In stage 1, keep_frozen_base_eval=True keeps the frozen T5 backbone in eval
    mode while metadata modules stay in train mode. This disables stochastic
    dropout in the frozen teacher-like path without blocking gradients through it.
    """

    def __init__(
        self,
        *args,
        optimizer_param_groups: Sequence[Dict[str, Any]],
        keep_frozen_base_eval: bool = False,
        keep_frozen_encoder_eval: bool = False,
        **kwargs,
    ):
        self._optimizer_param_groups = list(optimizer_param_groups)
        self.keep_frozen_base_eval = bool(keep_frozen_base_eval)
        self.keep_frozen_encoder_eval = bool(keep_frozen_encoder_eval)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self._optimizer_param_groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer

    def training_step(self, model, inputs, *args, **kwargs):
        unwrapped = _unwrap_model(model)
        if self.keep_frozen_base_eval:
            unwrapped.base_model.eval()
        elif self.keep_frozen_encoder_eval:
            unwrapped.base_model.get_encoder().eval()
        return super().training_step(model, inputs, *args, **kwargs)


# -----------------------------------------------------------------------------
# Logging and scaler utilities
# -----------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _get_rss_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return None


def _log_memory(prefix: str) -> None:
    value = _get_rss_gb()
    if value is not None:
        LOGGER.info("%s | RAM RSS: %.2f GB", prefix, value)


def _log_cuda_status(prefix: str) -> None:
    LOGGER.info("%s | CUDA_VISIBLE_DEVICES=%s", prefix, os.environ.get("CUDA_VISIBLE_DEVICES"))
    LOGGER.info("%s | torch.cuda.is_available=%s", prefix, torch.cuda.is_available())
    LOGGER.info("%s | torch.cuda.device_count=%d", prefix, torch.cuda.device_count())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            LOGGER.info("%s | CUDA device %d: %s", prefix, idx, torch.cuda.get_device_name(idx))


def _identity_transforms(feature_names: Sequence[str]) -> Dict[str, str]:
    return {name: "identity" for name in feature_names}


def _zscore_transforms(feature_names: Sequence[str]) -> Dict[str, str]:
    return {name: "zscore" for name in feature_names}


def _resolve_feature_transforms(
    *,
    feature_names: Sequence[str],
    default_map: Dict[str, str],
    normalization_mode: str,
) -> Dict[str, str]:
    if normalization_mode == "zscore":
        return _zscore_transforms(feature_names)
    if normalization_mode == "none":
        return _identity_transforms(feature_names)
    if normalization_mode == "feature_aware":
        return build_feature_transform_map(feature_names, default_map, fallback="zscore")
    raise ValueError(f"Unsupported metadata_normalization_mode={normalization_mode!r}")


def _default_lexical_scaler_path(output_dir: str | Path) -> str:
    return str(Path(output_dir) / "lexical_metadata_scaler.pkl")


def _default_embedding_scaler_path(output_dir: str | Path) -> str:
    return str(Path(output_dir) / "embedding_metadata_scaler.pkl")


def _default_token_scaler_path(output_dir: str | Path) -> str:
    return str(Path(output_dir) / "token_metadata_scaler.pkl")


def _fit_lexical_scaler_on_training_docnos(
    *,
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
    lexical_feature_transforms: Dict[str, str],
) -> MetadataFeatureScaler:
    LOGGER.info("Fit lexical scaler sui docno del training...")

    feature_dim = len(lexical_store.feature_names)
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
            LOGGER.info(
                "Lexical scaler: %d esempi | %.2f ex/s | %.1f min",
                idx,
                idx / max(elapsed, 1e-9),
                elapsed / 60.0,
            )

        if args.max_scaler_examples is not None and idx >= args.max_scaler_examples:
            break

    scaler = moments.finalize()
    LOGGER.info(
        "Lexical scaler completato su %d esempi in %.1f min",
        moments.count,
        (time.time() - start_time) / 60.0,
    )
    return scaler


def _fit_and_save_online_scalers(
    *,
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
    lexical_scaler: MetadataFeatureScaler,
    embedding_scaler_path: str,
    token_scaler_path: str,
    embedding_feature_transforms: Dict[str, str],
    token_feature_transforms: Dict[str, str],
) -> None:
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

    dataloader = DataLoader(
        dataset_for_online_scaler,
        batch_size=args.online_scaler_batch_size,
        collate_fn=MetadataQualT5Collator(tokenizer),
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_scaler, tok_scaler = fit_online_metadata_scalers(
        model_name_or_path=args.model_name_or_path,
        dataloader=dataloader,
        device=device,
        input_ids_key="input_ids",
        attention_mask_key="attention_mask",
        max_batches=max_batches,
        embedding_feature_transforms=embedding_feature_transforms,
        token_feature_transforms=token_feature_transforms,
    )
    emb_scaler.save(embedding_scaler_path)
    tok_scaler.save(token_scaler_path)


# -----------------------------------------------------------------------------
# Trainability and optimizer groups
# -----------------------------------------------------------------------------


def _set_module_trainable(module: Optional[torch.nn.Module], value: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = value


def _active_interaction_modules(model: MetadataEnrichedQualT5) -> list[tuple[str, torch.nn.Module]]:
    modules: list[tuple[str, torch.nn.Module]] = [
        ("uni_attention", model.uni_attention),
        ("meta_ln_1", model.meta_ln_1),
    ]

    if model.use_meta_ffn:
        modules.extend(
            [
                ("meta_ffn", model.meta_ffn),
                ("meta_ln_2", model.meta_ln_2),
            ]
        )

    if model.metadata_fusion_mode == "allmeta_token_projection":
        modules.extend(
            [
                ("allmeta_token_projection", model.allmeta_token_projection),
                ("allmeta_token_projection_ln", model.allmeta_token_projection_ln),
            ]
        )
    elif model.metadata_fusion_mode in {"concat_tokens", "att_fusion"}:
        pass
    else:
        raise ValueError(
            "Questo script two-step non usa mean pooling. "
            f"Fusion mode non supportata: {model.metadata_fusion_mode}"
        )

    return modules


def configure_stage1_trainability(model: MetadataEnrichedQualT5) -> None:
    """Freeze all parameters, then enable only metadata projections + interaction/fusion."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    _set_module_trainable(model.group_encoder, True)
    for _name, module in _active_interaction_modules(model):
        _set_module_trainable(module, True)


def configure_stage2_trainability(
    model: MetadataEnrichedQualT5,
    *,
    decoder_trainable_scope: str,
    unfreeze_last_n_decoder_blocks: int,
    unfreeze_lm_head: bool,
    unfreeze_shared_embeddings: bool,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    _set_module_trainable(model.group_encoder, True)
    for _name, module in _active_interaction_modules(model):
        _set_module_trainable(module, True)

    model.configure_trainable_parameters(
        decoder_trainable_scope=decoder_trainable_scope,
        unfreeze_last_n_decoder_blocks=unfreeze_last_n_decoder_blocks,
        unfreeze_lm_head=unfreeze_lm_head,
        unfreeze_shared_embeddings=unfreeze_shared_embeddings,
    )

    # T5 normally ties lm_head.weight to shared.weight. Do not silently claim that
    # one is frozen while the same Parameter is trained through the other name.
    if unfreeze_lm_head and hasattr(model.base_model, "lm_head") and hasattr(model.base_model, "shared"):
        tied = model.base_model.lm_head.weight is model.base_model.shared.weight
        if tied and not unfreeze_shared_embeddings:
            raise ValueError(
                "La LM head è weight-tied con gli shared embeddings. "
                "Non può essere sbloccata indipendentemente. Usa anche "
                "--stage2_unfreeze_shared_embeddings oppure lascia la LM head congelata."
            )


def _iter_named_module_parameters(
    module_name: str,
    module: torch.nn.Module,
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, parameter in module.named_parameters():
        yield f"{module_name}.{name}" if name else module_name, parameter


def _unique_trainable_named_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    used_ids: set[int],
) -> list[tuple[str, torch.nn.Parameter]]:
    output = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in used_ids:
            continue
        used_ids.add(id(parameter))
        output.append((name, parameter))
    return output


def _is_no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    lower = name.lower()
    return (
        parameter.ndim <= 1
        or lower.endswith("bias")
        or "layernorm" in lower
        or "layer_norm" in lower
        or ".ln" in lower
        or "relative_attention_bias" in lower
    )


def _append_lr_group(
    groups: list[Dict[str, Any]],
    *,
    label: str,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    learning_rate: float,
    weight_decay: float,
) -> None:
    decay = [p for n, p in named_parameters if not _is_no_decay(n, p)]
    no_decay = [p for n, p in named_parameters if _is_no_decay(n, p)]

    if decay:
        groups.append(
            {
                "params": decay,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": f"{label}_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "group_name": f"{label}_no_decay",
            }
        )

    LOGGER.info(
        "Optimizer group %-24s | tensors=%d | params=%d | lr=%g",
        label,
        len(named_parameters),
        sum(p.numel() for _, p in named_parameters),
        learning_rate,
    )


def build_stage1_optimizer_groups(
    model: MetadataEnrichedQualT5,
    *,
    group_encoder_lr: float,
    interaction_lr: float,
    weight_decay: float,
) -> list[Dict[str, Any]]:
    used: set[int] = set()
    groups: list[Dict[str, Any]] = []

    group_named = _unique_trainable_named_parameters(
        _iter_named_module_parameters("group_encoder", model.group_encoder), used
    )
    interaction_named: list[tuple[str, torch.nn.Parameter]] = []
    for module_name, module in _active_interaction_modules(model):
        interaction_named.extend(
            _unique_trainable_named_parameters(
                _iter_named_module_parameters(module_name, module), used
            )
        )

    _append_lr_group(
        groups,
        label="stage1_group_encoder",
        named_parameters=group_named,
        learning_rate=group_encoder_lr,
        weight_decay=weight_decay,
    )
    _append_lr_group(
        groups,
        label="stage1_interaction_fusion",
        named_parameters=interaction_named,
        learning_rate=interaction_lr,
        weight_decay=weight_decay,
    )
    return groups


def build_stage2_optimizer_groups(
    model: MetadataEnrichedQualT5,
    *,
    group_encoder_lr: float,
    interaction_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> list[Dict[str, Any]]:
    used: set[int] = set()
    groups: list[Dict[str, Any]] = []

    group_named = _unique_trainable_named_parameters(
        _iter_named_module_parameters("group_encoder", model.group_encoder), used
    )

    interaction_named: list[tuple[str, torch.nn.Parameter]] = []
    for module_name, module in _active_interaction_modules(model):
        interaction_named.extend(
            _unique_trainable_named_parameters(
                _iter_named_module_parameters(module_name, module), used
            )
        )

    # All remaining trainable base-model parameters belong to the slow LR group.
    decoder_named = _unique_trainable_named_parameters(
        (
            (f"base_model.{name}", parameter)
            for name, parameter in model.base_model.named_parameters()
        ),
        used,
    )

    _append_lr_group(
        groups,
        label="stage2_group_encoder",
        named_parameters=group_named,
        learning_rate=group_encoder_lr,
        weight_decay=weight_decay,
    )
    _append_lr_group(
        groups,
        label="stage2_interaction_fusion",
        named_parameters=interaction_named,
        learning_rate=interaction_lr,
        weight_decay=weight_decay,
    )
    _append_lr_group(
        groups,
        label="stage2_decoder_output",
        named_parameters=decoder_named,
        learning_rate=decoder_lr,
        weight_decay=weight_decay,
    )

    remaining = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad and id(p) not in used
    ]
    if remaining:
        preview = [name for name, _ in remaining[:30]]
        raise RuntimeError(f"Parametri trainable non assegnati a un optimizer group: {preview}")

    return groups


def _log_trainable_parameters(model: MetadataEnrichedQualT5, stage_name: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("%s | trainable=%d / total=%d (%.3f%%)", stage_name, trainable, total, 100 * trainable / total)

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            LOGGER.info("%s TRAINABLE | %s | shape=%s | params=%d", stage_name, name, tuple(parameter.shape), parameter.numel())


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------


def _find_state_dict_file(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    candidates = [
        checkpoint_dir / "pytorch_model.bin",
        checkpoint_dir / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Nessuno state_dict trovato in {checkpoint_dir}")


def _load_state_dict_file(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(path))
    else:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        for key in ("model", "state_dict", "module"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise RuntimeError(f"Formato checkpoint non supportato: {type(payload)}")

    cleaned = {}
    for key, value in payload.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def load_two_step_initialization(
    model: MetadataEnrichedQualT5,
    checkpoint_path: str | Path,
) -> None:
    """Load either a Trainer checkpoint-* or an inference-ready scorer/stage dir."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if checkpoint_path.name.startswith("checkpoint-"):
        state_dict = _load_state_dict_file(_find_state_dict_file(checkpoint_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        LOGGER.info(
            "Trainer checkpoint caricato da %s | missing=%d unexpected=%d",
            checkpoint_path,
            len(missing),
            len(unexpected),
        )
        if missing:
            LOGGER.warning("Missing keys preview: %s", missing[:30])
        if unexpected:
            LOGGER.warning("Unexpected keys preview: %s", unexpected[:30])
        return

    # Inference-ready directory: standard HF base model + metadata modules.
    loaded_base = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model.base_model.load_state_dict(loaded_base.state_dict(), strict=True)
    del loaded_base
    model.load_metadata_modules(checkpoint_path)
    LOGGER.info("Inference-ready checkpoint caricato da %s", checkpoint_path)


# -----------------------------------------------------------------------------
# Configuration builders
# -----------------------------------------------------------------------------


def _build_metadata_feature_config(
    *,
    lexical_feature_names: Sequence[str],
    lexical_feature_transforms: Dict[str, str],
    embedding_feature_transforms: Dict[str, str],
    token_feature_transforms: Dict[str, str],
    normalization_mode: str,
    lexical_scaler_name: str,
    embedding_scaler_name: str,
    token_scaler_name: str,
) -> Dict[str, Any]:
    return {
        "scorers": {
            "metadata_qualt5": {
                "metadata_feature_groups": {
                    "lexical": list(lexical_feature_names),
                    "embedding": list(EMBEDDING_FEATURE_NAMES),
                    "token": list(TOKEN_FEATURE_NAMES),
                },
                "scalers": {
                    "lexical": lexical_scaler_name,
                    "embedding": embedding_scaler_name,
                    "token": token_scaler_name,
                },
                "normalization": {
                    "mode": normalization_mode,
                    "lexical_feature_transforms": lexical_feature_transforms,
                    "embedding_feature_transforms": embedding_feature_transforms,
                    "token_feature_transforms": token_feature_transforms,
                },
            }
        }
    }


def _build_metadata_config(
    *,
    args: argparse.Namespace,
    lexical_feature_names: Sequence[str],
    lexical_feature_transforms: Dict[str, str],
    embedding_feature_transforms: Dict[str, str],
    token_feature_transforms: Dict[str, str],
    stage_name: str,
) -> Dict[str, Any]:
    return {
        "model_type": "metadata_qualt5",
        "base_model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.model_name_or_path,
        "lexical_feature_names": list(lexical_feature_names),
        "embedding_feature_names": list(EMBEDDING_FEATURE_NAMES),
        "token_feature_names": list(TOKEN_FEATURE_NAMES),
        "scoring_mode": args.scoring_mode,
        "metadata_dropout": args.metadata_dropout,
        "metadata_mlp_hidden_dim": args.metadata_mlp_hidden_dim,
        "metadata_projection_type": args.metadata_projection_type,
        "attention_heads": args.attention_heads,
        "use_meta_ffn": not args.disable_meta_ffn,
        "normalize_metadata_features": args.normalize_metadata_features,
        "unfreeze_last_n_decoder_blocks": args.stage2_unfreeze_last_n_decoder_blocks,
        "unfreeze_lm_head": args.stage2_unfreeze_lm_head,
        "decoder_trainable_scope": args.stage2_decoder_trainable_scope,
        "unfreeze_full_decoder": args.stage2_decoder_trainable_scope == "full_decoder",
        "unfreeze_shared_embeddings": args.stage2_unfreeze_shared_embeddings,
        "metadata_fusion_mode": args.metadata_fusion_mode,
        "metadata_normalization_mode": args.metadata_normalization_mode,
        "lexical_feature_transforms": lexical_feature_transforms,
        "embedding_feature_transforms": embedding_feature_transforms,
        "token_feature_transforms": token_feature_transforms,
        "two_step_joint_training": True,
        "training_stage": stage_name,
        "stage1_group_encoder_lr": args.stage1_group_encoder_lr,
        "stage1_interaction_lr": args.stage1_interaction_lr,
        "stage2_group_encoder_lr": args.stage2_group_encoder_lr,
        "stage2_interaction_lr": args.stage2_interaction_lr,
        "stage2_decoder_lr": args.stage2_decoder_lr,
    }


def _create_train_dataset(
    *,
    args: argparse.Namespace,
    tokenizer,
    lexical_store: LexicalMetadataStore,
    lexical_scaler: MetadataFeatureScaler,
):
    return MetadataQualT5TriplesIterableDataset(
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


def _create_training_args(
    *,
    output_dir: str | Path,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    nominal_learning_rate: float,
    save_steps: int,
    save_total_limit: int,
    logging_steps: int,
    warmup_ratio: float,
    weight_decay: float,
    bf16: bool,
    fp16: bool,
    seed: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=nominal_learning_rate,
        optim="adamw_torch",
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_safetensors=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-step joint training for metadata-enriched QualT5."
    )

    # Base paths and data.
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--metadata_scaler_path", type=str, default=None)
    parser.add_argument("--lexical_scaler_path", type=str, default=None)
    parser.add_argument("--embedding_scaler_path", type=str, default=None)
    parser.add_argument("--token_scaler_path", type=str, default=None)

    parser.add_argument("--triples_source", choices=["file", "irds"], default="file")
    parser.add_argument("--triples_path", type=str, default=None)
    parser.add_argument("--triples_format", choices=["text", "id"], default="id")
    parser.add_argument("--collection_path", type=str, default=None)
    parser.add_argument(
        "--irds_dataset_id",
        type=str,
        default="msmarco-passage/train/triples-small",
    )
    parser.add_argument("--max_irds_triples", type=int, default=None)
    parser.add_argument("--irds_cache_dir", type=str, default=None)
    parser.add_argument("--allow_missing_metadata", action="store_true")

    # Shared training settings.
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Stage 1.
    parser.add_argument("--stage1_max_steps", type=int, default=3000)
    parser.add_argument("--stage1_group_encoder_lr", type=float, default=5e-5)
    parser.add_argument("--stage1_interaction_lr", type=float, default=5e-5)
    parser.add_argument("--stage1_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--stage1_save_steps", type=int, default=500)
    parser.add_argument("--stage1_save_total_limit", type=int, default=6)
    parser.add_argument("--stage1_resume_from_checkpoint", type=str, default=None)

    # Stage 2.
    parser.add_argument("--stage2_max_steps", type=int, default=12000)
    parser.add_argument("--stage2_group_encoder_lr", type=float, default=1e-5)
    parser.add_argument("--stage2_interaction_lr", type=float, default=2e-5)
    parser.add_argument("--stage2_decoder_lr", type=float, default=5e-6)
    parser.add_argument("--stage2_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--stage2_save_steps", type=int, default=1000)
    parser.add_argument("--stage2_save_total_limit", type=int, default=12)
    parser.add_argument("--stage2_resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--stage2_decoder_trainable_scope",
        choices=["cross_attention", "last_n_blocks", "full_decoder"],
        default="full_decoder",
    )
    parser.add_argument("--stage2_unfreeze_last_n_decoder_blocks", type=int, default=1)
    parser.add_argument("--stage2_unfreeze_lm_head", action="store_true")
    parser.add_argument("--stage2_unfreeze_shared_embeddings", action="store_true")

    # Workflow control.
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip stage 1 and initialize stage 2 from --stage1_pretrained_checkpoint.",
    )
    parser.add_argument(
        "--stage1_only",
        action="store_true",
        help="Run only the metadata interaction/fusion warm-up.",
    )
    parser.add_argument(
        "--stage1_pretrained_checkpoint",
        type=str,
        default=None,
        help=(
            "Stage-1 Trainer checkpoint-* or inference-ready scorer/stage directory. "
            "Required with --skip_stage1."
        ),
    )

    # Metadata architecture. Pooled modes are deliberately excluded.
    parser.add_argument("--metadata_dropout", type=float, default=0.1)
    parser.add_argument("--metadata_mlp_hidden_dim", type=int, default=None)
    parser.add_argument(
        "--metadata_projection_type",
        choices=["linear", "mlp"],
        default="linear",
    )
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--disable_meta_ffn", action="store_true")
    parser.add_argument("--normalize_metadata_features", action="store_true")
    parser.add_argument(
        "--metadata_fusion_mode",
        choices=sorted(NON_POOLED_FUSION_MODES),
        default="allmeta_token_projection",
        help="Only non-pooled fusion modes are allowed by this script.",
    )
    parser.add_argument(
        "--scoring_mode",
        choices=["true_logprob", "true_prob"],
        default="true_logprob",
    )

    # Scalers.
    parser.add_argument(
        "--metadata_normalization_mode",
        choices=["zscore", "feature_aware", "none"],
        default="feature_aware",
    )
    parser.add_argument("--max_scaler_examples", type=int, default=500000)
    parser.add_argument("--scaler_log_every", type=int, default=50000)
    parser.add_argument("--max_online_scaler_examples", type=int, default=100000)
    parser.add_argument("--online_scaler_batch_size", type=int, default=16)

    # Automatic held-out diagnostic and model selection.
    parser.add_argument(
        "--diagnostic_sample_path",
        type=str,
        default=None,
        help="Fixed CSV with docno,text,label. If omitted, automatic diagnostic is disabled.",
    )
    parser.add_argument(
        "--diagnostic_baseline_model_path",
        type=str,
        default=None,
        help="QualT5-Finetuned baseline. Defaults to --model_name_or_path.",
    )
    parser.add_argument("--diagnostic_batch_size", type=int, default=16)
    parser.add_argument("--diagnostic_max_examples", type=int, default=None)
    parser.add_argument("--diagnostic_sample_seed", type=int, default=42)
    parser.add_argument(
        "--diagnostic_prompt_template",
        type=str,
        default="Document: {text} Relevant:",
    )
    parser.add_argument(
        "--diagnostic_prune_fractions",
        type=str,
        default="0.15,0.25,0.30,0.45",
    )
    parser.add_argument(
        "--stage1_diagnostic_selection_metric",
        type=str,
        default="auc",
        help=(
            "Metric used to select the best Stage-1 checkpoint. Default: auc. "
            "Other supported values are delta_auc and the pruning metrics."
        ),
    )
    parser.add_argument(
        "--stage2_diagnostic_selection_metric",
        type=str,
        default="mean_delta_kept_positive_rate",
        help=(
            "Metric used to select the best Stage-2 checkpoint. Default: "
            "mean_delta_kept_positive_rate. Other supported values: auc, delta_auc, "
            "mean_kept_positive_rate, kept_positive_rate_XXX or "
            "delta_kept_positive_rate_XXX (for example XXX=045)."
        ),
    )
    parser.add_argument("--stage1_diagnostic_every_steps", type=int, default=500)
    parser.add_argument("--stage2_diagnostic_every_steps", type=int, default=1000)
    parser.add_argument("--stage1_diagnostic_patience", type=int, default=0)
    parser.add_argument("--stage2_diagnostic_patience", type=int, default=0)
    parser.add_argument("--diagnostic_min_delta", type=float, default=0.0)
    parser.add_argument("--diagnostic_save_scores", action="store_true")
    parser.add_argument(
        "--stage1_overfitting_patience",
        type=int,
        default=2,
        help=(
            "Consecutive diagnostics where training loss decreases and validation "
            "loss rises before Stage 1 is marked as overfitting."
        ),
    )
    parser.add_argument(
        "--stage2_overfitting_patience",
        type=int,
        default=3,
        help=(
            "Consecutive diagnostics where training loss decreases and validation "
            "loss rises before Stage 2 is marked as overfitting."
        ),
    )
    parser.add_argument(
        "--validation_loss_min_delta",
        type=float,
        default=0.001,
        help="Minimum validation-loss increase treated as a real worsening.",
    )
    parser.add_argument(
        "--max_validation_loss_increase",
        type=float,
        default=0.05,
        help=(
            "Maximum absolute validation-loss increase from the best loss. "
            "Checkpoints above this guard cannot become best."
        ),
    )
    parser.add_argument(
        "--stop_on_overfitting",
        action="store_true",
        help="Stop the current stage when the overfitting criterion is met.",
    )
    parser.add_argument(
        "--no_loss_guard_best_checkpoint",
        dest="loss_guard_best_checkpoint",
        action="store_false",
        help="Allow a checkpoint with strongly degraded validation loss to become best.",
    )
    parser.set_defaults(loss_guard_best_checkpoint=True)
    parser.add_argument(
        "--no_use_best_stage1_for_stage2",
        dest="use_best_stage1_for_stage2",
        action="store_false",
        help="Start stage 2 from the final stage-1 weights instead of the diagnostic best.",
    )
    parser.set_defaults(use_best_stage1_for_stage2=True)

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Stage execution
# -----------------------------------------------------------------------------


def _run_stage(
    *,
    stage_name: str,
    model: MetadataEnrichedQualT5,
    tokenizer,
    train_dataset,
    stage_dir: Path,
    training_args: TrainingArguments,
    optimizer_groups: Sequence[Dict[str, Any]],
    metadata_config: Dict[str, Any],
    metadata_feature_config: Dict[str, Any],
    lexical_scaler_path: str,
    embedding_scaler_path: str,
    token_scaler_path: str,
    keep_frozen_base_eval: bool,
    keep_frozen_encoder_eval: bool,
    resume_from_checkpoint: Optional[str],
    diagnostic_evaluator: Optional[DiagnosticEvaluator],
    diagnostic_selection_metric: str,
    diagnostic_every_steps: int,
    diagnostic_patience: int,
    diagnostic_min_delta: float,
    overfitting_patience: int,
    validation_loss_min_delta: float,
    max_validation_loss_increase: float,
    stop_on_overfitting: bool,
    loss_guard_best_checkpoint: bool,
) -> tuple[Trainer, StageScorerCheckpointCallback]:
    checkpoint_callback = StageScorerCheckpointCallback(
        tokenizer=tokenizer,
        stage_name=stage_name,
        metadata_config_template=metadata_config,
        metadata_feature_config=metadata_feature_config,
        lexical_scaler_path=lexical_scaler_path,
        embedding_scaler_path=embedding_scaler_path,
        token_scaler_path=token_scaler_path,
        diagnostic_evaluator=diagnostic_evaluator,
        diagnostic_selection_metric=diagnostic_selection_metric,
        diagnostic_every_steps=diagnostic_every_steps,
        diagnostic_patience=diagnostic_patience,
        diagnostic_min_delta=diagnostic_min_delta,
        overfitting_patience=overfitting_patience,
        validation_loss_min_delta=validation_loss_min_delta,
        max_validation_loss_increase=max_validation_loss_increase,
        stop_on_overfitting=stop_on_overfitting,
        loss_guard_best_checkpoint=loss_guard_best_checkpoint,
    )

    trainer = DifferentialLRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MetadataQualT5Collator(tokenizer),
        tokenizer=tokenizer,
        callbacks=[checkpoint_callback],
        optimizer_param_groups=optimizer_groups,
        keep_frozen_base_eval=keep_frozen_base_eval,
        keep_frozen_encoder_eval=keep_frozen_encoder_eval,
    )

    LOGGER.info("============================================================")
    LOGGER.info("Starting %s | output=%s", stage_name, stage_dir)
    LOGGER.info("resume_from_checkpoint=%s", resume_from_checkpoint)
    LOGGER.info("============================================================")

    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    LOGGER.info("%s completed in %.1f min", stage_name, (time.time() - start) / 60.0)

    final_cfg = dict(metadata_config)
    final_cfg["training_stage"] = stage_name
    final_cfg["training_stage_step"] = int(trainer.state.global_step)
    _export_inference_model(
        model=model,
        tokenizer=tokenizer,
        export_dir=stage_dir,
        metadata_config=final_cfg,
        metadata_feature_config=metadata_feature_config,
        lexical_scaler_path=lexical_scaler_path,
        embedding_scaler_path=embedding_scaler_path,
        token_scaler_path=token_scaler_path,
    )

    checkpoint_callback.finalize_best_alias(stage_dir)
    trainer.save_state()
    return trainer, checkpoint_callback


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    _setup_logging()
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Scegli solo una tra --bf16 e --fp16")
    if args.skip_stage1 and args.stage1_only:
        raise ValueError("--skip_stage1 e --stage1_only sono incompatibili")
    if args.skip_stage1 and not args.stage1_pretrained_checkpoint:
        raise ValueError("Con --skip_stage1 devi fornire --stage1_pretrained_checkpoint")
    if args.metadata_fusion_mode not in NON_POOLED_FUSION_MODES:
        raise ValueError("Questo script non consente fusioni basate su mean pooling")

    if args.diagnostic_sample_path is not None:
        if args.stage1_diagnostic_every_steps <= 0 and args.stage2_diagnostic_every_steps <= 0:
            raise ValueError("Diagnostic sample provided but both diagnostic intervals are disabled")
        if args.diagnostic_batch_size <= 0:
            raise ValueError("diagnostic_batch_size must be > 0")

    if args.stage2_unfreeze_shared_embeddings:
        LOGGER.warning(
            "Gli shared embeddings influenzano anche l'encoder congelato: le feature "
            "embedding/token potrebbero spostarsi rispetto agli scaler online calcolati "
            "sul checkpoint iniziale. La configurazione raccomandata li lascia congelati."
        )

    if args.triples_source == "file":
        if not args.triples_path:
            raise ValueError("Con --triples_source file devi specificare --triples_path")
        if args.triples_format == "id" and not args.collection_path:
            raise ValueError("Con file/id devi specificare --collection_path")

    if args.irds_cache_dir:
        os.environ["IR_DATASETS_HOME"] = args.irds_cache_dir

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stage1_dir = output_root / "stage1_metadata_fusion_warmup"
    stage2_dir = output_root / "stage2_joint_finetuning"

    _log_cuda_status("Startup")
    _log_memory("Startup")
    LOGGER.info("Args: %s", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    lexical_store = LexicalMetadataStore.from_path(args.metadata_path)

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

    # Keep one canonical set of scalers in the two-step root.
    lexical_scaler_path = (
        args.lexical_scaler_path
        or args.metadata_scaler_path
        or _default_lexical_scaler_path(output_root)
    )
    embedding_scaler_path = args.embedding_scaler_path or _default_embedding_scaler_path(output_root)
    token_scaler_path = args.token_scaler_path or _default_token_scaler_path(output_root)

    if Path(lexical_scaler_path).exists():
        lexical_scaler = MetadataFeatureScaler.load(lexical_scaler_path)
    else:
        lexical_scaler = _fit_lexical_scaler_on_training_docnos(
            args=args,
            tokenizer=tokenizer,
            lexical_store=lexical_store,
            lexical_feature_transforms=lexical_feature_transforms,
        )
        lexical_scaler.save(lexical_scaler_path)

    if not (Path(embedding_scaler_path).exists() and Path(token_scaler_path).exists()):
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

    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if not true_ids or not false_ids:
        raise ValueError("Impossibile ricavare true_token_id/false_token_id")

    model = MetadataEnrichedQualT5(
        model_name_or_path=args.model_name_or_path,
        lexical_feature_dim=len(lexical_store.feature_names),
        true_token_id=int(true_ids[0]),
        false_token_id=int(false_ids[0]),
        scoring_mode=args.scoring_mode,
        metadata_mlp_hidden_dim=args.metadata_mlp_hidden_dim,
        metadata_dropout=args.metadata_dropout,
        metadata_projection_type=args.metadata_projection_type,
        attention_heads=args.attention_heads,
        use_meta_ffn=not args.disable_meta_ffn,
        normalize_metadata_features=args.normalize_metadata_features,
        unfreeze_last_n_decoder_blocks=args.stage2_unfreeze_last_n_decoder_blocks,
        unfreeze_lm_head=args.stage2_unfreeze_lm_head,
        decoder_trainable_scope=args.stage2_decoder_trainable_scope,
        unfreeze_shared_embeddings=args.stage2_unfreeze_shared_embeddings,
        metadata_fusion_mode=args.metadata_fusion_mode,
        lexical_feature_scaler_path=None,  # lexical features are scaled by dataset
        embedding_feature_scaler_path=embedding_scaler_path,
        token_feature_scaler_path=token_scaler_path,
    )

    # Optional external stage-1 initialization, useful for stage2-only runs.
    if args.stage1_pretrained_checkpoint:
        load_two_step_initialization(model, args.stage1_pretrained_checkpoint)

    diagnostic_evaluator: Optional[DiagnosticEvaluator] = None
    diagnostic_prune_fractions: list[float] = []
    if args.diagnostic_sample_path is not None:
        diagnostic_prune_fractions = _parse_prune_fractions(
            args.diagnostic_prune_fractions
        )
        diagnostic_evaluator = DiagnosticEvaluator(
            sample_path=args.diagnostic_sample_path,
            baseline_model_path=(
                args.diagnostic_baseline_model_path or args.model_name_or_path
            ),
            tokenizer=tokenizer,
            lexical_store=lexical_store,
            lexical_scaler=lexical_scaler,
            output_root=output_root,
            batch_size=args.diagnostic_batch_size,
            max_length=args.max_length,
            prompt_template=args.diagnostic_prompt_template,
            prune_fractions=diagnostic_prune_fractions,
            # Default only; each stage passes its own selection metric to evaluate().
            selection_metric=args.stage2_diagnostic_selection_metric,
            max_examples=args.diagnostic_max_examples,
            sample_seed=args.diagnostic_sample_seed,
            baseline_bf16=args.bf16,
            save_scores=args.diagnostic_save_scores,
        )

    lexical_name = Path(lexical_scaler_path).name
    embedding_name = Path(embedding_scaler_path).name
    token_name = Path(token_scaler_path).name

    feature_cfg = _build_metadata_feature_config(
        lexical_feature_names=lexical_store.feature_names,
        lexical_feature_transforms=lexical_feature_transforms,
        embedding_feature_transforms=embedding_feature_transforms,
        token_feature_transforms=token_feature_transforms,
        normalization_mode=args.metadata_normalization_mode,
        lexical_scaler_name=lexical_name,
        embedding_scaler_name=embedding_name,
        token_scaler_name=token_name,
    )

    # Save global reproducibility information immediately.
    _write_json(output_root / "two_step_training_args.json", vars(args))
    _write_yaml(output_root / "two_step_training_args.yaml", vars(args))
    manifest = {
        "model_name_or_path": args.model_name_or_path,
        "output_root": str(output_root),
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "metadata_fusion_mode": args.metadata_fusion_mode,
        "mean_pooling_used": False,
        "automatic_diagnostic_enabled": diagnostic_evaluator is not None,
        "diagnostic_sample_path": args.diagnostic_sample_path,
        "stage1_diagnostic_selection_metric": args.stage1_diagnostic_selection_metric,
        "stage2_diagnostic_selection_metric": args.stage2_diagnostic_selection_metric,
        "validation_loss_monitoring": True,
        "stage1_overfitting_patience": args.stage1_overfitting_patience,
        "stage2_overfitting_patience": args.stage2_overfitting_patience,
        "validation_loss_min_delta": args.validation_loss_min_delta,
        "max_validation_loss_increase": args.max_validation_loss_increase,
        "stop_on_overfitting": args.stop_on_overfitting,
        "loss_guard_best_checkpoint": args.loss_guard_best_checkpoint,
        "diagnostic_prune_fractions": diagnostic_prune_fractions,
        "use_best_stage1_for_stage2": args.use_best_stage1_for_stage2,
        "stage1_completed": False,
        "stage2_completed": False,
    }
    _write_json(output_root / "two_step_manifest.json", manifest)

    # ------------------------------- Stage 1 -------------------------------
    if not args.skip_stage1:
        configure_stage1_trainability(model)
        _log_trainable_parameters(model, "STAGE1")

        stage1_cfg = _build_metadata_config(
            args=args,
            lexical_feature_names=lexical_store.feature_names,
            lexical_feature_transforms=lexical_feature_transforms,
            embedding_feature_transforms=embedding_feature_transforms,
            token_feature_transforms=token_feature_transforms,
            stage_name="stage1_metadata_fusion_warmup",
        )
        stage1_assets = _prepare_stage_assets(
            stage_dir=stage1_dir,
            tokenizer=tokenizer,
            metadata_config=stage1_cfg,
            metadata_feature_config=feature_cfg,
            lexical_scaler_path=lexical_scaler_path,
            embedding_scaler_path=embedding_scaler_path,
            token_scaler_path=token_scaler_path,
        )

        stage1_groups = build_stage1_optimizer_groups(
            model,
            group_encoder_lr=args.stage1_group_encoder_lr,
            interaction_lr=args.stage1_interaction_lr,
            weight_decay=args.weight_decay,
        )
        stage1_training_args = _create_training_args(
            output_dir=stage1_dir,
            max_steps=args.stage1_max_steps,
            batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            nominal_learning_rate=max(args.stage1_group_encoder_lr, args.stage1_interaction_lr),
            save_steps=args.stage1_save_steps,
            save_total_limit=args.stage1_save_total_limit,
            logging_steps=args.logging_steps,
            warmup_ratio=args.stage1_warmup_ratio,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=args.fp16,
            seed=args.seed,
        )

        stage1_trainer, stage1_callback = _run_stage(
            stage_name="stage1_metadata_fusion_warmup",
            model=model,
            tokenizer=tokenizer,
            train_dataset=_create_train_dataset(
                args=args,
                tokenizer=tokenizer,
                lexical_store=lexical_store,
                lexical_scaler=lexical_scaler,
            ),
            stage_dir=stage1_dir,
            training_args=stage1_training_args,
            optimizer_groups=stage1_groups,
            metadata_config=stage1_cfg,
            metadata_feature_config=feature_cfg,
            lexical_scaler_path=stage1_assets["lexical"],
            embedding_scaler_path=stage1_assets["embedding"],
            token_scaler_path=stage1_assets["token"],
            keep_frozen_base_eval=True,
            keep_frozen_encoder_eval=False,
            resume_from_checkpoint=args.stage1_resume_from_checkpoint,
            diagnostic_evaluator=diagnostic_evaluator,
            diagnostic_selection_metric=args.stage1_diagnostic_selection_metric,
            diagnostic_every_steps=args.stage1_diagnostic_every_steps,
            diagnostic_patience=args.stage1_diagnostic_patience,
            diagnostic_min_delta=args.diagnostic_min_delta,
            overfitting_patience=args.stage1_overfitting_patience,
            validation_loss_min_delta=args.validation_loss_min_delta,
            max_validation_loss_increase=args.max_validation_loss_increase,
            stop_on_overfitting=args.stop_on_overfitting,
            loss_guard_best_checkpoint=args.loss_guard_best_checkpoint,
        )

        manifest["stage1_completed"] = True
        manifest["stage1_final_model"] = str(stage1_dir)
        manifest["stage1_best_selection_metric"] = args.stage1_diagnostic_selection_metric
        manifest["stage1_best_step"] = stage1_callback.best_step
        manifest["stage1_best_score"] = (
            stage1_callback.best_score
            if stage1_callback.best_checkpoint_path is not None
            else None
        )
        manifest["stage1_best_checkpoint"] = stage1_callback.best_checkpoint_path
        _write_json(output_root / "two_step_manifest.json", manifest)

    if args.stage1_only:
        LOGGER.info("Requested --stage1_only. Finished.")
        return

    if (
        not args.skip_stage1
        and args.use_best_stage1_for_stage2
        and stage1_callback.best_checkpoint_path is not None
    ):
        LOGGER.info(
            "Reloading best stage-1 diagnostic checkpoint before stage 2: %s",
            stage1_callback.best_checkpoint_path,
        )
        load_two_step_initialization(
            model,
            stage1_callback.best_checkpoint_path,
        )
        manifest["stage2_initialization"] = stage1_callback.best_checkpoint_path
        _write_json(output_root / "two_step_manifest.json", manifest)
    elif not args.skip_stage1:
        manifest["stage2_initialization"] = str(stage1_dir)
        _write_json(output_root / "two_step_manifest.json", manifest)

    # ------------------------------- Stage 2 -------------------------------
    model.zero_grad(set_to_none=True)
    configure_stage2_trainability(
        model,
        decoder_trainable_scope=args.stage2_decoder_trainable_scope,
        unfreeze_last_n_decoder_blocks=args.stage2_unfreeze_last_n_decoder_blocks,
        unfreeze_lm_head=args.stage2_unfreeze_lm_head,
        unfreeze_shared_embeddings=args.stage2_unfreeze_shared_embeddings,
    )
    _log_trainable_parameters(model, "STAGE2")

    stage2_cfg = _build_metadata_config(
        args=args,
        lexical_feature_names=lexical_store.feature_names,
        lexical_feature_transforms=lexical_feature_transforms,
        embedding_feature_transforms=embedding_feature_transforms,
        token_feature_transforms=token_feature_transforms,
        stage_name="stage2_joint_finetuning",
    )
    stage2_assets = _prepare_stage_assets(
        stage_dir=stage2_dir,
        tokenizer=tokenizer,
        metadata_config=stage2_cfg,
        metadata_feature_config=feature_cfg,
        lexical_scaler_path=lexical_scaler_path,
        embedding_scaler_path=embedding_scaler_path,
        token_scaler_path=token_scaler_path,
    )

    stage2_groups = build_stage2_optimizer_groups(
        model,
        group_encoder_lr=args.stage2_group_encoder_lr,
        interaction_lr=args.stage2_interaction_lr,
        decoder_lr=args.stage2_decoder_lr,
        weight_decay=args.weight_decay,
    )
    stage2_training_args = _create_training_args(
        output_dir=stage2_dir,
        max_steps=args.stage2_max_steps,
        batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        nominal_learning_rate=max(
            args.stage2_group_encoder_lr,
            args.stage2_interaction_lr,
            args.stage2_decoder_lr,
        ),
        save_steps=args.stage2_save_steps,
        save_total_limit=args.stage2_save_total_limit,
        logging_steps=args.logging_steps,
        warmup_ratio=args.stage2_warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
    )

    stage2_trainer, stage2_callback = _run_stage(
        stage_name="stage2_joint_finetuning",
        model=model,
        tokenizer=tokenizer,
        train_dataset=_create_train_dataset(
            args=args,
            tokenizer=tokenizer,
            lexical_store=lexical_store,
            lexical_scaler=lexical_scaler,
        ),
        stage_dir=stage2_dir,
        training_args=stage2_training_args,
        optimizer_groups=stage2_groups,
        metadata_config=stage2_cfg,
        metadata_feature_config=feature_cfg,
        lexical_scaler_path=stage2_assets["lexical"],
        embedding_scaler_path=stage2_assets["embedding"],
        token_scaler_path=stage2_assets["token"],
        keep_frozen_base_eval=False,
        keep_frozen_encoder_eval=True,
        resume_from_checkpoint=args.stage2_resume_from_checkpoint,
        diagnostic_evaluator=diagnostic_evaluator,
        diagnostic_selection_metric=args.stage2_diagnostic_selection_metric,
        diagnostic_every_steps=args.stage2_diagnostic_every_steps,
        diagnostic_patience=args.stage2_diagnostic_patience,
        diagnostic_min_delta=args.diagnostic_min_delta,
        overfitting_patience=args.stage2_overfitting_patience,
        validation_loss_min_delta=args.validation_loss_min_delta,
        max_validation_loss_increase=args.max_validation_loss_increase,
        stop_on_overfitting=args.stop_on_overfitting,
        loss_guard_best_checkpoint=args.loss_guard_best_checkpoint,
    )

    manifest["stage2_completed"] = True
    manifest["stage2_final_model"] = str(stage2_dir)
    manifest["stage2_best_selection_metric"] = args.stage2_diagnostic_selection_metric
    manifest["stage2_best_step"] = stage2_callback.best_step
    manifest["stage2_best_score"] = (
        stage2_callback.best_score
        if stage2_callback.best_checkpoint_path is not None
        else None
    )
    manifest["stage2_best_checkpoint"] = stage2_callback.best_checkpoint_path
    manifest["recommended_model"] = (
        str(stage2_dir / "best-scorer")
        if stage2_callback.best_checkpoint_path is not None
        else str(stage2_dir)
    )
    _write_json(output_root / "two_step_manifest.json", manifest)

    LOGGER.info("Two-step joint training completato: %s", output_root)


if __name__ == "__main__":
    main()
    
"""
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup python -u -m metaqual.models.two-step-joint-tr \
  --model_name_or_path /home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000 \
  --metadata_path /home/sacco/data/msmarco_passage/msmarco_passage_lexical_metadata.parquet \
  --output_dir /home/sacco/metaqual/outputs/metadata-qualt5-2step-attfus-featureaware-noffn \
  --triples_source irds \
  --irds_dataset_id msmarco-passage/train/triples-small \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --max_length 512 \
  --logging_steps 50 \
  --metadata_dropout 0.1 \
  --metadata_projection_type linear \
  --metadata_fusion_mode att_fusion \
  --metadata_normalization_mode feature_aware \
  --disable_meta_ffn \
  --max_scaler_examples 500000 \
  --max_online_scaler_examples 100000 \
  --online_scaler_batch_size 16 \
  --stage1_max_steps 5000 \
  --stage1_group_encoder_lr 5e-5 \
  --stage1_interaction_lr 5e-5 \
  --stage1_save_steps 500 \
  --stage1_save_total_limit 6 \
  --stage2_max_steps 10000 \
  --stage2_group_encoder_lr 1e-5 \
  --stage2_interaction_lr 2e-5 \
  --stage2_decoder_lr 5e-6 \
  --stage2_decoder_trainable_scope full_decoder \
  --stage2_save_steps 1000 \
  --stage2_save_total_limit 10 \
  --diagnostic_sample_path /home/sacco/metaqual/outputs/diagnostics/samples/heldout_20k_skip1M.csv \
  --diagnostic_baseline_model_path /home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000 \
  --diagnostic_batch_size 16 \
  --diagnostic_prune_fractions 0.15,0.25,0.30,0.45 \
  --stage1_diagnostic_selection_metric auc \
  --stage2_diagnostic_selection_metric mean_delta_kept_positive_rate \
  --stage1_diagnostic_every_steps 500 \
  --stage2_diagnostic_every_steps 1000 \
  --diagnostic_min_delta 0.0 \
  --stage1_overfitting_patience 2 \
  --stage2_overfitting_patience 3 \
  --validation_loss_min_delta 0.001 \
  --max_validation_loss_increase 0.05 \
  --stop_on_overfitting \
  --bf16 \
  > metadata_qualt5_two_step_auto_diagnostic.log 2>&1 &

"""