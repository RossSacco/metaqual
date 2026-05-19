from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

REQUIRED_LEXICAL_FEATURES = [
    "length_tokens",
    "avg_token_length",
    "unique_token_ratio",
    "repetition_ratio",
    "lexical_entropy",
    "stopword_ratio",
    "content_word_ratio",
]

OPTIONAL_LEXICAL_FEATURES = [
    "avg_idf",
    "max_idf",
    "idf_std",
]

EMBEDDING_FEATURE_NAMES = [
    "embedding_l1_norm",
    "embedding_l2_norm",
    "embedding_linf_norm",
    "embedding_mean",
    "embedding_variance",
    "near_zero_fraction",
]

TOKEN_FEATURE_NAMES = [
    "mean_token_norm",
    "std_token_norm",
    "max_token_norm",
    "token_norm_entropy",
    "token_to_passage_similarity_mean",
]


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return num / den.clamp_min(eps)


def _masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden_states: [B, L, d]
    # attention_mask: [B, L]
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    return _safe_div(summed, counts)


def extract_embedding_level_metadata(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    near_zero_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns embedding-level metadata tensor [B, 6].
    """
    pooled = _masked_mean(hidden_states, attention_mask)

    l1_norm = pooled.abs().sum(dim=-1)
    l2_norm = pooled.norm(p=2, dim=-1)
    linf_norm = pooled.abs().amax(dim=-1)
    emb_mean = pooled.mean(dim=-1)
    emb_var = pooled.var(dim=-1, unbiased=False)
    near_zero = (pooled.abs() < near_zero_eps).to(hidden_states.dtype).mean(dim=-1)

    return torch.stack(
        [l1_norm, l2_norm, linf_norm, emb_mean, emb_var, near_zero],
        dim=-1,
    )


def extract_token_level_metadata(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns token-level metadata tensor [B, 5].
    """
    mask = attention_mask.to(hidden_states.dtype)
    valid_lengths = mask.sum(dim=1).clamp_min(1.0)

    token_norms = hidden_states.norm(p=2, dim=-1)  # [B, L]
    masked_norms = token_norms * mask

    mean_norm = _safe_div(masked_norms.sum(dim=1), valid_lengths, eps=eps)

    centered = (token_norms - mean_norm.unsqueeze(1)) * mask
    var_norm = _safe_div((centered ** 2).sum(dim=1), valid_lengths, eps=eps)
    std_norm = torch.sqrt(var_norm.clamp_min(0.0))

    max_norm = masked_norms.max(dim=1).values

    norm_distribution = masked_norms.clamp_min(0.0)
    dist_sum = norm_distribution.sum(dim=1, keepdim=True).clamp_min(eps)
    probs = norm_distribution / dist_sum
    token_norm_entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=1)

    pooled = _masked_mean(hidden_states, attention_mask)
    pooled_unit = F.normalize(pooled, p=2, dim=-1, eps=eps)
    token_unit = F.normalize(hidden_states, p=2, dim=-1, eps=eps)
    token_sim = (token_unit * pooled_unit.unsqueeze(1)).sum(dim=-1)
    token_to_passage_similarity_mean = _safe_div((token_sim * mask).sum(dim=1), valid_lengths, eps=eps)

    return torch.stack(
        [
            mean_norm,
            std_norm,
            max_norm,
            token_norm_entropy,
            token_to_passage_similarity_mean,
        ],
        dim=-1,
    )


class MetadataFeatureScaler:
    """
    Lightweight standardization scaler for lexical metadata.
    """

    def __init__(self, feature_names: Sequence[str], mean: np.ndarray, std: np.ndarray):
        self.feature_names = list(feature_names)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std == 0.0] = 1.0

    @classmethod
    def fit(cls, values: np.ndarray, feature_names: Sequence[str]) -> "MetadataFeatureScaler":
        if values.ndim != 2:
            raise ValueError(f"Expected 2D array for scaler fit, got shape={values.shape}")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0.0] = 1.0
        return cls(feature_names=feature_names, mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2:
            raise ValueError(f"Expected 2D array for scaler transform, got shape={values.shape}")
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                "Feature dimension mismatch in scaler transform: "
                f"got {values.shape[1]}, expected {len(self.feature_names)}"
            )
        return (values - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MetadataFeatureScaler":
        return cls(
            feature_names=payload["feature_names"],
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.to_dict(), handle)

    @classmethod
    def load(cls, path: str | Path) -> "MetadataFeatureScaler":
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls.from_dict(payload)


class LexicalMetadataStore:
    """
    Loads and serves lexical metadata by docno.
    """

    def __init__(self, dataframe: pd.DataFrame, feature_names: Sequence[str]):
        if "docno" not in dataframe.columns:
            raise ValueError("Lexical metadata table must contain 'docno' column.")

        missing = [f for f in feature_names if f not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing lexical metadata features: {missing}")

        self.feature_names = list(feature_names)
        table = dataframe[["docno", *self.feature_names]].copy()
        table["docno"] = table["docno"].astype(str)
        self.table = table.set_index("docno")

    @classmethod
    def from_path(
        cls,
        metadata_path: str | Path,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "LexicalMetadataStore":
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        suffix = metadata_path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(metadata_path)
        elif suffix == ".csv":
            df = pd.read_csv(metadata_path)
        else:
            raise ValueError(
                f"Unsupported metadata format '{suffix}'. Supported: .parquet, .csv"
            )

        if feature_names is None:
            missing_required = [c for c in REQUIRED_LEXICAL_FEATURES if c not in df.columns]
            if missing_required:
                raise ValueError(
                    "Metadata table missing required lexical columns: "
                    f"{missing_required}"
                )
            resolved = list(REQUIRED_LEXICAL_FEATURES)
            for col in OPTIONAL_LEXICAL_FEATURES:
                if col in df.columns:
                    resolved.append(col)
            feature_names = resolved
        else:
            feature_names = [f for f in feature_names if f != "docno"]

        return cls(dataframe=df, feature_names=feature_names)

    def lookup(
        self,
        docnos: Sequence[str],
        *,
        allow_missing_metadata: bool = False,
    ) -> np.ndarray:
        docnos_str = [str(d) for d in docnos]
        indexed = self.table.reindex(docnos_str)

        missing_mask = indexed[self.feature_names].isna().all(axis=1)
        if missing_mask.any() and not allow_missing_metadata:
            missing_docnos = indexed.index[missing_mask].tolist()
            preview = missing_docnos[:20]
            raise KeyError(
                "Missing lexical metadata for docnos. "
                f"count={len(missing_docnos)}, sample={preview}"
            )

        values = indexed[self.feature_names].fillna(0.0).to_numpy(dtype=np.float32)
        return values


class GroupWiseMetadataEncoder(nn.Module):
    """
    Encodes each metadata group with a dedicated MLP.
    """

    def __init__(
        self,
        lexical_dim: int,
        embedding_dim: int,
        token_dim: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        inner = hidden_dim if hidden_dim is not None else d_model
        self.lexical_mlp = self._build_mlp(lexical_dim, inner, d_model, dropout)
        self.embedding_mlp = self._build_mlp(embedding_dim, inner, d_model, dropout)
        self.token_mlp = self._build_mlp(token_dim, inner, d_model, dropout)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        lexical_features: torch.Tensor,
        embedding_features: torch.Tensor,
        token_features: torch.Tensor,
    ) -> torch.Tensor:
        z_lex = self.lexical_mlp(lexical_features)
        z_emb = self.embedding_mlp(embedding_features)
        z_tok = self.token_mlp(token_features)

        # Z_meta: [B, G=3, d_model]
        return torch.stack([z_lex, z_emb, z_tok], dim=1)


class UniAttention(nn.Module):
    """
    Uni-attention from metadata tokens (query) to text hidden states (key/value).
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out


class MetadataEnrichedQualT5(nn.Module):
    """
    Fine-tuned QualT5 + metadata-aware fusion.
    """

    def __init__(
        self,
        model_name_or_path: str,
        lexical_feature_dim: int,
        true_token_id: int,
        false_token_id: int,
        *,
        scoring_mode: str = "true_logprob",
        metadata_mlp_hidden_dim: Optional[int] = None,
        metadata_dropout: float = 0.1,
        attention_heads: int = 8,
        use_meta_ffn: bool = True,
    ):
        super().__init__()
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.config = self.base_model.config

        self.true_token_id = int(true_token_id)
        self.false_token_id = int(false_token_id)
        self.scoring_mode = scoring_mode

        self.d_model = int(self.config.d_model)

        self.group_encoder = GroupWiseMetadataEncoder(
            lexical_dim=lexical_feature_dim,
            embedding_dim=len(EMBEDDING_FEATURE_NAMES),
            token_dim=len(TOKEN_FEATURE_NAMES),
            d_model=self.d_model,
            hidden_dim=metadata_mlp_hidden_dim,
            dropout=metadata_dropout,
        )

        self.uni_attention = UniAttention(
            d_model=self.d_model,
            num_heads=attention_heads,
            dropout=metadata_dropout,
        )

        self.meta_dropout = nn.Dropout(metadata_dropout)
        self.meta_ln_1 = nn.LayerNorm(self.d_model)
        self.use_meta_ffn = bool(use_meta_ffn)
        if self.use_meta_ffn:
            self.meta_ffn = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.GELU(),
                nn.Dropout(metadata_dropout),
                nn.Linear(self.d_model * 4, self.d_model),
            )
            self.meta_ln_2 = nn.LayerNorm(self.d_model)

        self.freeze_encoder()
        self.freeze_all_base_model_except_decoder_cross_attention()

    def freeze_encoder(self) -> None:
        for param in self.base_model.get_encoder().parameters():
            param.requires_grad = False

    def freeze_all_base_model_except_decoder_cross_attention(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

        for name, param in self.base_model.named_parameters():
            # In T5 blocks, decoder layer[1] is cross-attention.
            if "decoder.block" in name and ".layer.1." in name:
                param.requires_grad = True

    def get_trainable_parameter_stats(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": int(trainable), "total": int(total)}

    def _compute_quality_score(self, true_logits: torch.Tensor, false_logits: torch.Tensor) -> torch.Tensor:
        pair_logits = torch.stack([true_logits, false_logits], dim=1)
        if self.scoring_mode == "true_prob":
            return torch.softmax(pair_logits, dim=1)[:, 0]
        return torch.log_softmax(pair_logits, dim=1)[:, 0]

    def _compute_loss(
        self,
        true_logits: torch.Tensor,
        false_logits: torch.Tensor,
        binary_labels: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        target_idx = None
        pair_logits = torch.stack([true_logits, false_logits], dim=1)

        if binary_labels is not None:
            # binary_labels: 1=true, 0=false
            binary_labels = binary_labels.long()
            target_idx = torch.where(
                binary_labels > 0,
                torch.zeros_like(binary_labels),
                torch.ones_like(binary_labels),
            )
        elif labels is not None:
            if labels.ndim == 1:
                first_label = labels
            else:
                first_label = labels[:, 0]

            target_idx = torch.where(
                first_label == self.true_token_id,
                torch.zeros_like(first_label),
                torch.ones_like(first_label),
            )

        if target_idx is None:
            return None

        return F.cross_entropy(pair_logits, target_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lexical_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        binary_labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        # Encoder text hidden states H_text: [B, L, d]
        encoder_outputs = self.base_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h_text = encoder_outputs.last_hidden_state

        batch_size, seq_len, hidden_dim = h_text.shape
        if hidden_dim != self.d_model:
            raise ValueError(
                f"Unexpected hidden dim. got={hidden_dim}, expected={self.d_model}"
            )

        # Online metadata features.
        emb_features = extract_embedding_level_metadata(h_text, attention_mask)
        tok_features = extract_token_level_metadata(h_text, attention_mask)

        # Z_meta: [B, G=3, d]
        z_meta = self.group_encoder(lexical_features, emb_features, tok_features)
        if z_meta.shape != (batch_size, 3, self.d_model):
            raise ValueError(
                "Invalid Z_meta shape: "
                f"got={tuple(z_meta.shape)}, expected={(batch_size, 3, self.d_model)}"
            )

        key_padding_mask = attention_mask == 0
        z_att = self.uni_attention(
            query=z_meta,
            key=h_text,
            value=h_text,
            key_padding_mask=key_padding_mask,
        )

        z_meta_fused = self.meta_ln_1(z_meta + self.meta_dropout(z_att))

        if self.use_meta_ffn:
            z_meta_fused = self.meta_ln_2(
                z_meta_fused + self.meta_dropout(self.meta_ffn(z_meta_fused))
            )

        # H_fused: [B, G+L, d]
        h_fused = torch.cat([z_meta_fused, h_text], dim=1)

        meta_mask = torch.ones(
            (batch_size, z_meta_fused.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # fused_attention_mask: [B, G+L]
        fused_attention_mask = torch.cat([meta_mask, attention_mask], dim=1)

        if h_fused.shape[1] != seq_len + z_meta_fused.shape[1]:
            raise ValueError(
                "Invalid H_fused sequence length: "
                f"got={h_fused.shape[1]}, expected={seq_len + z_meta_fused.shape[1]}"
            )

        decoder_start_token_id = self.base_model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.base_model.config.pad_token_id

        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (batch_size, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )

        decoder_outputs = self.base_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=h_fused),
            attention_mask=fused_attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

        first_step_logits = decoder_outputs.logits[:, 0, :]
        logits_true = first_step_logits[:, self.true_token_id]
        logits_false = first_step_logits[:, self.false_token_id]

        quality_score = self._compute_quality_score(logits_true, logits_false)
        loss = self._compute_loss(logits_true, logits_false, binary_labels, labels)

        return {
            "loss": loss,
            "logits_true": logits_true,
            "logits_false": logits_false,
            "quality_score": quality_score,
            "decoder_logits": decoder_outputs.logits,
            "h_text": h_text,
            "z_meta": z_meta,
            "h_fused": h_fused,
            "fused_attention_mask": fused_attention_mask,
        }

    def save_metadata_modules(self, output_dir: str | Path, metadata_config: Dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        modules_payload = {
            "group_encoder": self.group_encoder.state_dict(),
            "uni_attention": self.uni_attention.state_dict(),
            "meta_ln_1": self.meta_ln_1.state_dict(),
            "use_meta_ffn": self.use_meta_ffn,
        }
        if self.use_meta_ffn:
            modules_payload["meta_ffn"] = self.meta_ffn.state_dict()
            modules_payload["meta_ln_2"] = self.meta_ln_2.state_dict()

        torch.save(modules_payload, output_dir / "metadata_qualt5_modules.pt")

        with (output_dir / "metadata_qualt5_config.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata_config, handle, indent=2, ensure_ascii=False)

    def load_metadata_modules(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        modules_path = model_dir / "metadata_qualt5_modules.pt"
        if not modules_path.exists():
            return

        payload = torch.load(modules_path, map_location="cpu")
        self.group_encoder.load_state_dict(payload["group_encoder"])
        self.uni_attention.load_state_dict(payload["uni_attention"])
        self.meta_ln_1.load_state_dict(payload["meta_ln_1"])

        if self.use_meta_ffn and "meta_ffn" in payload:
            self.meta_ffn.load_state_dict(payload["meta_ffn"])
            self.meta_ln_2.load_state_dict(payload["meta_ln_2"])


class RunningMoments:
    """
    Streaming mean/std estimator for scaler fitting.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, batch_values: np.ndarray) -> None:
        if batch_values.ndim != 2 or batch_values.shape[1] != self.dim:
            raise ValueError(
                f"Invalid batch for RunningMoments: shape={batch_values.shape}, expected [N, {self.dim}]"
            )

        for row in batch_values:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def finalize(self, feature_names: Sequence[str]) -> MetadataFeatureScaler:
        if self.count == 0:
            raise ValueError("Cannot finalize RunningMoments with count=0")

        variance = self.m2 / max(self.count, 1)
        std = np.sqrt(np.maximum(variance, 1e-12))
        return MetadataFeatureScaler(
            feature_names=feature_names,
            mean=self.mean.astype(np.float32),
            std=std.astype(np.float32),
        )


def load_metadata_qualt5_config(model_dir: str | Path) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    config_path = model_dir / "metadata_qualt5_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
