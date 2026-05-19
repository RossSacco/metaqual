from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from metaqual.models.base_scorer import MetadataEnrichedQualT5Scorer
    from metaqual.models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        REQUIRED_LEXICAL_FEATURES,
    )
except ImportError:
    from models.base_scorer import MetadataEnrichedQualT5Scorer
    from models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        REQUIRED_LEXICAL_FEATURES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test metadata_qualt5")
    parser.add_argument("--model_name_or_path", type=str, default="t5-small")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _build_dummy_metadata(path: Path) -> tuple[Path, list[str]]:
    rows = [
        {
            "docno": "d1",
            "length_tokens": 10.0,
            "avg_token_length": 4.2,
            "unique_token_ratio": 0.8,
            "repetition_ratio": 0.2,
            "lexical_entropy": 2.1,
            "stopword_ratio": 0.3,
            "content_word_ratio": 0.7,
        },
        {
            "docno": "d2",
            "length_tokens": 14.0,
            "avg_token_length": 4.8,
            "unique_token_ratio": 0.75,
            "repetition_ratio": 0.25,
            "lexical_entropy": 2.0,
            "stopword_ratio": 0.35,
            "content_word_ratio": 0.65,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path, REQUIRED_LEXICAL_FEATURES


def _fit_scaler_from_metadata(store: LexicalMetadataStore, scaler_path: Path) -> MetadataFeatureScaler:
    raw = store.lookup(["d1", "d2"], allow_missing_metadata=False)
    scaler = MetadataFeatureScaler.fit(raw, feature_names=store.feature_names)
    scaler.save(scaler_path)
    return scaler


def run_model_forward_sanity(
    model_name_or_path: str,
    tokenizer,
    store: LexicalMetadataStore,
    scaler: MetadataFeatureScaler,
    max_length: int,
) -> None:
    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("false", add_special_tokens=False)[0]

    model = MetadataEnrichedQualT5(
        model_name_or_path=model_name_or_path,
        lexical_feature_dim=len(store.feature_names),
        true_token_id=true_token_id,
        false_token_id=false_token_id,
    )
    model.eval()

    texts = [
        "This is a clear and informative passage.",
        "Noisy repeated repeated repeated tokens appear here.",
    ]
    prompts = [f"Document: {t} Relevant:" for t in texts]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    lexical_raw = store.lookup(["d1", "d2"], allow_missing_metadata=False)
    lexical_norm = scaler.transform(lexical_raw)
    lexical_tensor = torch.tensor(lexical_norm, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            lexical_features=lexical_tensor,
        )

    bsz = 2
    seq_len = int(encoded["attention_mask"].shape[1])
    d_model = model.d_model

    assert tuple(outputs["h_text"].shape) == (bsz, seq_len, d_model)
    assert tuple(outputs["z_meta"].shape) == (bsz, 3, d_model)
    assert tuple(outputs["h_fused"].shape) == (bsz, seq_len + 3, d_model)
    assert tuple(outputs["fused_attention_mask"].shape) == (bsz, seq_len + 3)
    assert tuple(outputs["quality_score"].shape) == (bsz,)
    assert torch.isfinite(outputs["quality_score"]).all()


def run_scorer_sanity(
    model_name_or_path: str,
    metadata_path: Path,
    scaler_path: Path,
    max_length: int,
    device: str | None,
) -> None:
    df = pd.DataFrame(
        [
            {"docno": "d1", "text": "This passage is coherent and factual."},
            {"docno": "d2", "text": "This one is noisy and repetitive repetitive."},
        ]
    )

    scorer = MetadataEnrichedQualT5Scorer(
        model_name_or_path=model_name_or_path,
        metadata_path=str(metadata_path),
        metadata_scaler_path=str(scaler_path),
        batch_size=2,
        max_length=max_length,
        device=device,
    )
    out = scorer.transform(df)

    assert "quality" in out.columns
    assert len(out) == 2
    assert out["quality"].map(np.isfinite).all()

    try:
        scorer.transform(pd.DataFrame([{"docno": "MISSING", "text": "x"}]))
        raise AssertionError("Expected missing metadata error was not raised.")
    except KeyError:
        pass

    scorer_allow_missing = MetadataEnrichedQualT5Scorer(
        model_name_or_path=model_name_or_path,
        metadata_path=str(metadata_path),
        metadata_scaler_path=str(scaler_path),
        batch_size=1,
        max_length=max_length,
        device=device,
        allow_missing_metadata=True,
    )
    out_missing = scorer_allow_missing.transform(
        pd.DataFrame([{"docno": "MISSING", "text": "text"}])
    )
    assert "quality" in out_missing.columns


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="metadata_qualt5_smoke_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        metadata_path, feature_names = _build_dummy_metadata(tmpdir_path / "metadata.csv")

        store = LexicalMetadataStore.from_path(metadata_path, feature_names=feature_names)
        scaler = _fit_scaler_from_metadata(store, tmpdir_path / "metadata_scaler.pkl")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

        run_model_forward_sanity(
            model_name_or_path=args.model_name_or_path,
            tokenizer=tokenizer,
            store=store,
            scaler=scaler,
            max_length=args.max_length,
        )

        run_scorer_sanity(
            model_name_or_path=args.model_name_or_path,
            metadata_path=metadata_path,
            scaler_path=tmpdir_path / "metadata_scaler.pkl",
            max_length=args.max_length,
            device=args.device,
        )

    print("Smoke test OK: metadata_qualt5 forward + scorer checks passed.")


if __name__ == "__main__":
    main()
