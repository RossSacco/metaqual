from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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


SUPPORTED_METADATA_TRANSFORMS = {"identity", "zscore", "log1p_zscore"}

DEFAULT_LEXICAL_FEATURE_TRANSFORMS = {
    "length_tokens": "log1p_zscore",
    "avg_token_length": "zscore",
    "unique_token_ratio": "identity",
    "repetition_ratio": "identity",
    "lexical_entropy": "zscore",
    "stopword_ratio": "identity",
    "content_word_ratio": "identity",
    "avg_idf": "zscore",
    "max_idf": "zscore",
    "idf_std": "zscore",
}

DEFAULT_EMBEDDING_FEATURE_TRANSFORMS = {
    "embedding_l1_norm": "log1p_zscore",
    "embedding_l2_norm": "log1p_zscore",
    "embedding_linf_norm": "log1p_zscore",
    "embedding_mean": "zscore",
    "embedding_variance": "log1p_zscore",
    "near_zero_fraction": "identity",
}

DEFAULT_TOKEN_FEATURE_TRANSFORMS = {
    "mean_token_norm": "log1p_zscore",
    "std_token_norm": "zscore",
    "max_token_norm": "log1p_zscore",
    "token_norm_entropy": "zscore",
    "token_to_passage_similarity_mean": "zscore",
}


def build_feature_transform_map(
    feature_names: Sequence[str],
    default_map: Optional[Dict[str, str]] = None,
    *,
    fallback: str = "zscore",
) -> Dict[str, str]:
    if fallback not in SUPPORTED_METADATA_TRANSFORMS:
        raise ValueError(f"Unsupported fallback transform: {fallback}")

    default_map = default_map or {}
    transforms: Dict[str, str] = {}

    for name in feature_names:
        transform = default_map.get(name, fallback)
        if transform not in SUPPORTED_METADATA_TRANSFORMS:
            raise ValueError(
                f"Unsupported transform {transform!r} for feature {name!r}. "
                f"Supported transforms: {sorted(SUPPORTED_METADATA_TRANSFORMS)}"
            )
        transforms[name] = transform

    return transforms


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return num / den.clamp_min(eps)


def _masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: [B, L, d]
    attention_mask: [B, L]
    returns: [B, d]
    """
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

    token_norms = hidden_states.norm(p=2, dim=-1)
    masked_norms = token_norms * mask

    mean_norm = _safe_div(masked_norms.sum(dim=1), valid_lengths, eps=eps)

    centered = (token_norms - mean_norm.unsqueeze(1)) * mask
    var_norm = _safe_div((centered**2).sum(dim=1), valid_lengths, eps=eps)
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

    token_to_passage_similarity_mean = _safe_div(
        (token_sim * mask).sum(dim=1),
        valid_lengths,
        eps=eps,
    )

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
    Feature-aware scaler for metadata features.

    Supported per-feature transforms:
    - identity:      return the raw feature unchanged;
    - zscore:        (x - mean) / std;
    - log1p_zscore:  (log(1 + max(x, 0)) - mean_log) / std_log.

    Backward compatibility:
    old scaler files containing only feature_names/mean/std are interpreted
    as standard z-score scalers for all features.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        mean: np.ndarray,
        std: np.ndarray,
        feature_transforms: Optional[Dict[str, str]] = None,
        *,
        eps: float = 1e-12,
    ):
        self.feature_names = list(feature_names)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.eps = float(eps)

        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError(
                "MetadataFeatureScaler expects 1D mean/std vectors, "
                f"got mean={self.mean.shape}, std={self.std.shape}"
            )

        if self.mean.shape != self.std.shape:
            raise ValueError(
                "MetadataFeatureScaler mean/std shape mismatch: "
                f"mean={self.mean.shape}, std={self.std.shape}"
            )

        if self.mean.shape[0] != len(self.feature_names):
            raise ValueError(
                "MetadataFeatureScaler feature dimension mismatch: "
                f"mean/std dim={self.mean.shape[0]}, features={len(self.feature_names)}"
            )

        self.std[self.std == 0.0] = 1.0
        self.std = np.maximum(self.std, self.eps).astype(np.float32)

        if feature_transforms is None:
            feature_transforms = {name: "zscore" for name in self.feature_names}

        self.feature_transforms = build_feature_transform_map(
            self.feature_names,
            feature_transforms,
            fallback="zscore",
        )

        # For identity features mean/std are intentionally ignored during transform.
        for idx, name in enumerate(self.feature_names):
            if self.feature_transforms[name] == "identity":
                self.mean[idx] = 0.0
                self.std[idx] = 1.0

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        feature_names: Sequence[str],
        feature_transforms: Optional[Dict[str, str]] = None,
    ) -> "MetadataFeatureScaler":
        if values.ndim != 2:
            raise ValueError(f"Expected 2D array for scaler fit, got shape={values.shape}")

        feature_names = list(feature_names)
        if values.shape[1] != len(feature_names):
            raise ValueError(
                "Feature dimension mismatch in scaler fit: "
                f"got {values.shape[1]}, expected {len(feature_names)}"
            )

        if feature_transforms is None:
            feature_transforms = {name: "zscore" for name in feature_names}

        transforms = build_feature_transform_map(feature_names, feature_transforms)
        transformed = cls.apply_transforms_only(values, feature_names, transforms)

        mean = transformed.mean(axis=0).astype(np.float32)
        std = transformed.std(axis=0).astype(np.float32)
        std[std == 0.0] = 1.0

        for idx, name in enumerate(feature_names):
            if transforms[name] == "identity":
                mean[idx] = 0.0
                std[idx] = 1.0

        return cls(
            feature_names=feature_names,
            mean=mean,
            std=std,
            feature_transforms=transforms,
        )

    @staticmethod
    def apply_transforms_only(
        values: np.ndarray,
        feature_names: Sequence[str],
        feature_transforms: Dict[str, str],
    ) -> np.ndarray:
        transformed = np.asarray(values, dtype=np.float32).copy()

        for idx, name in enumerate(feature_names):
            transform = feature_transforms.get(name, "zscore")

            if transform == "identity" or transform == "zscore":
                continue

            if transform == "log1p_zscore":
                transformed[:, idx] = np.log1p(np.clip(transformed[:, idx], 0.0, None))
                continue

            raise ValueError(
                f"Unsupported transform {transform!r} for feature {name!r}. "
                f"Supported transforms: {sorted(SUPPORTED_METADATA_TRANSFORMS)}"
            )

        return transformed

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2:
            raise ValueError(f"Expected 2D array for scaler transform, got shape={values.shape}")

        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                "Feature dimension mismatch in scaler transform: "
                f"got {values.shape[1]}, expected {len(self.feature_names)}"
            )

        transformed = self.apply_transforms_only(
            values,
            self.feature_names,
            self.feature_transforms,
        )

        output = transformed.copy()
        zscore_mask = np.array(
            [self.feature_transforms[name] != "identity" for name in self.feature_names],
            dtype=bool,
        )

        if zscore_mask.any():
            output[:, zscore_mask] = (
                output[:, zscore_mask] - self.mean[zscore_mask]
            ) / self.std[zscore_mask]

        return output.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_transforms": self.feature_transforms,
            "scaler_type": "feature_aware",
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MetadataFeatureScaler":
        feature_names = list(payload["feature_names"])
        feature_transforms = payload.get("feature_transforms")

        # Backward compatibility with old scaler files.
        if feature_transforms is None:
            feature_transforms = {name: "zscore" for name in feature_names}

        return cls(
            feature_names=feature_names,
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            feature_transforms=feature_transforms,
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


class TorchMetadataScaler(nn.Module):
    """
    Torch version of MetadataFeatureScaler.

    It stores mean/std and transform identifiers as buffers, so they move
    automatically to CPU/GPU together with the model.
    """

    TRANSFORM_TO_ID = {
        "identity": 0,
        "zscore": 1,
        "log1p_zscore": 2,
    }
    ID_TO_TRANSFORM = {value: key for key, value in TRANSFORM_TO_ID.items()}

    def __init__(
        self,
        mean: Sequence[float] | np.ndarray | torch.Tensor,
        std: Sequence[float] | np.ndarray | torch.Tensor,
        *,
        feature_names: Optional[Sequence[str]] = None,
        feature_transforms: Optional[Dict[str, str]] = None,
        eps: float = 1e-12,
    ):
        super().__init__()

        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).clamp_min(eps)

        if mean_tensor.ndim != 1 or std_tensor.ndim != 1:
            raise ValueError(
                "TorchMetadataScaler expects 1D mean/std vectors, "
                f"got mean={tuple(mean_tensor.shape)}, std={tuple(std_tensor.shape)}"
            )

        if mean_tensor.shape != std_tensor.shape:
            raise ValueError(
                "TorchMetadataScaler mean/std shape mismatch: "
                f"mean={tuple(mean_tensor.shape)}, std={tuple(std_tensor.shape)}"
            )

        self.feature_names = list(feature_names) if feature_names is not None else None

        if self.feature_names is None:
            self.feature_transforms = None
            transform_ids = torch.ones_like(mean_tensor, dtype=torch.long)
        else:
            if len(self.feature_names) != mean_tensor.numel():
                raise ValueError(
                    "TorchMetadataScaler feature dimension mismatch: "
                    f"features={len(self.feature_names)}, mean/std={mean_tensor.numel()}"
                )

            if feature_transforms is None:
                feature_transforms = {name: "zscore" for name in self.feature_names}

            self.feature_transforms = build_feature_transform_map(
                self.feature_names,
                feature_transforms,
                fallback="zscore",
            )

            transform_ids = torch.tensor(
                [self.TRANSFORM_TO_ID[self.feature_transforms[name]] for name in self.feature_names],
                dtype=torch.long,
            )

            for idx, name in enumerate(self.feature_names):
                if self.feature_transforms[name] == "identity":
                    mean_tensor[idx] = 0.0
                    std_tensor[idx] = 1.0

        self.eps = float(eps)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        self.register_buffer("transform_ids", transform_ids)

    @classmethod
    def from_metadata_scaler(cls, scaler: MetadataFeatureScaler) -> "TorchMetadataScaler":
        return cls(
            mean=scaler.mean,
            std=scaler.std,
            feature_names=scaler.feature_names,
            feature_transforms=scaler.feature_transforms,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "TorchMetadataScaler":
        return cls.from_metadata_scaler(MetadataFeatureScaler.load(path))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(
                f"Expected metadata tensor with shape [B, F], got {tuple(values.shape)}"
            )

        if values.shape[1] != self.mean.numel():
            raise ValueError(
                "Feature dimension mismatch in TorchMetadataScaler: "
                f"got {values.shape[1]}, expected {self.mean.numel()}"
            )

        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        transform_ids = self.transform_ids.to(device=values.device)

        transformed = values

        log_mask = transform_ids == self.TRANSFORM_TO_ID["log1p_zscore"]
        if bool(log_mask.any()):
            transformed = transformed.clone()
            transformed[:, log_mask] = torch.log1p(transformed[:, log_mask].clamp_min(0.0))

        zscore_mask = transform_ids != self.TRANSFORM_TO_ID["identity"]
        if bool(zscore_mask.any()):
            output = transformed.clone()
            output[:, zscore_mask] = (
                output[:, zscore_mask] - mean[zscore_mask]
            ) / std[zscore_mask]
            return output

        return transformed


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
    Encodes each metadata group into the T5 hidden space.

    projection_type:
    - "linear": simple projection Linear(in_dim, d_model), without activation;
    - "mlp": backward-compatible two-layer projection
             Linear(in_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, d_model).

    Expected input:
    - lexical_features:   [B, lexical_dim]
    - embedding_features: [B, 6]
    - token_features:     [B, 5]

    Output:
    - z_meta: [B, 3, d_model]
    """

    VALID_PROJECTION_TYPES = {"linear", "mlp"}

    def __init__(
        self,
        lexical_dim: int,
        embedding_dim: int,
        token_dim: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        normalize_inputs: bool = False,
        projection_type: str = "linear",
    ):
        super().__init__()

        if projection_type not in self.VALID_PROJECTION_TYPES:
            raise ValueError(
                f"projection_type must be one of {sorted(self.VALID_PROJECTION_TYPES)}, "
                f"got {projection_type!r}"
            )

        inner = hidden_dim if hidden_dim is not None else d_model
        self.normalize_inputs = bool(normalize_inputs)
        self.projection_type = projection_type

        if self.normalize_inputs:
            self.lexical_norm = nn.LayerNorm(lexical_dim)
            self.embedding_norm = nn.LayerNorm(embedding_dim)
            self.token_norm = nn.LayerNorm(token_dim)
        else:
            self.lexical_norm = nn.Identity()
            self.embedding_norm = nn.Identity()
            self.token_norm = nn.Identity()

        self.lexical_mlp = self._build_projection(
            lexical_dim,
            inner,
            d_model,
            dropout,
            projection_type=projection_type,
        )
        self.embedding_mlp = self._build_projection(
            embedding_dim,
            inner,
            d_model,
            dropout,
            projection_type=projection_type,
        )
        self.token_mlp = self._build_projection(
            token_dim,
            inner,
            d_model,
            dropout,
            projection_type=projection_type,
        )

    @staticmethod
    def _build_projection(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        *,
        projection_type: str,
    ) -> nn.Module:
        if projection_type == "linear":
            return nn.Linear(in_dim, out_dim)

        if projection_type == "mlp":
            layers: list[nn.Module] = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ]

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_dim, out_dim))

            return nn.Sequential(*layers)

        raise ValueError(f"Unsupported projection_type={projection_type!r}")

    def forward(
        self,
        lexical_features: torch.Tensor,
        embedding_features: torch.Tensor,
        token_features: torch.Tensor,
    ) -> torch.Tensor:
        lexical_features = self.lexical_norm(lexical_features)
        embedding_features = self.embedding_norm(embedding_features)
        token_features = self.token_norm(token_features)

        z_lex = self.lexical_mlp(lexical_features)
        z_emb = self.embedding_mlp(embedding_features)
        z_tok = self.token_mlp(token_features)

        return torch.stack([z_lex, z_emb, z_tok], dim=1)


class UniAttention(nn.Module):
    """
    Uni-attention from metadata tokens to text hidden states.

    Q = metadata tokens
    K = H_text
    V = H_text
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
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

    Fusion modes:
    - concat_tokens:
        decoder attends to [z_lex ; z_emb ; z_tok ; H_text].

    - pooled_concat_projection:
        meta_vec = mean(Z_meta_fused)
        metadata_update = MLP([H_text ; repeat(meta_vec)])
        H_fused = LayerNorm(H_text + metadata_update)

    - direct_concat_projection:
        meta_vec = mean(Z_meta_fused)
        H_fused = LayerNorm(MLP([H_text ; repeat(meta_vec)]))

    - meta_prefix:
        meta_vec = mean(Z_meta_fused)
        decoder attends to [meta_vec ; H_text].
    """

    VALID_FUSION_MODES = {
        "concat_tokens",
        "pooled_concat_projection",
        "direct_concat_projection",
        "meta_prefix",
    }

    VALID_DECODER_TRAINABLE_SCOPES = {
        "cross_attention",
        "last_n_blocks",
        "full_decoder",
    }

    def __init__(
        self,
        model_name_or_path: str,
        lexical_feature_dim: int,
        true_token_id: int,
        false_token_id: int,
        *,
        scoring_mode: str = "true_logprob",
        metadata_mlp_hidden_dim: Optional[int] = None,
        metadata_dropout: float = 0.0,
        metadata_projection_type: str = "linear",
        attention_heads: int = 8,
        use_meta_ffn: bool = True,
        normalize_metadata_features: bool = False,
        unfreeze_last_n_decoder_blocks: int = 1,
        unfreeze_lm_head: bool = True,
        decoder_trainable_scope: str = "last_n_blocks",
        unfreeze_shared_embeddings: bool = False,
        metadata_fusion_mode: str = "pooled_concat_projection",
        lexical_feature_scaler_path: Optional[str | Path] = None,
        embedding_feature_scaler_path: Optional[str | Path] = None,
        token_feature_scaler_path: Optional[str | Path] = None,
    ):
        super().__init__()

        if metadata_fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(
                f"metadata_fusion_mode must be one of {sorted(self.VALID_FUSION_MODES)}, "
                f"got {metadata_fusion_mode}"
            )

        decoder_trainable_scope = str(decoder_trainable_scope)
        if decoder_trainable_scope not in self.VALID_DECODER_TRAINABLE_SCOPES:
            raise ValueError(
                "decoder_trainable_scope must be one of "
                f"{sorted(self.VALID_DECODER_TRAINABLE_SCOPES)}, "
                f"got {decoder_trainable_scope}"
            )

        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.config = self.base_model.config

        self.true_token_id = int(true_token_id)
        self.false_token_id = int(false_token_id)
        self.scoring_mode = scoring_mode

        self.d_model = int(self.config.d_model)

        self.normalize_metadata_features = bool(normalize_metadata_features)
        self.metadata_projection_type = str(metadata_projection_type)
        self.unfreeze_last_n_decoder_blocks = int(unfreeze_last_n_decoder_blocks)
        self.unfreeze_lm_head_flag = bool(unfreeze_lm_head)
        self.decoder_trainable_scope = decoder_trainable_scope
        self.unfreeze_shared_embeddings = bool(unfreeze_shared_embeddings)
        self.metadata_fusion_mode = metadata_fusion_mode

        self.lexical_feature_scaler_path = (
            str(lexical_feature_scaler_path) if lexical_feature_scaler_path is not None else None
        )
        self.embedding_feature_scaler_path = (
            str(embedding_feature_scaler_path) if embedding_feature_scaler_path is not None else None
        )
        self.token_feature_scaler_path = (
            str(token_feature_scaler_path) if token_feature_scaler_path is not None else None
        )

        if lexical_feature_scaler_path is not None:
            self.lexical_feature_scaler = TorchMetadataScaler.from_path(lexical_feature_scaler_path)
        else:
            self.lexical_feature_scaler = nn.Identity()

        if embedding_feature_scaler_path is not None:
            self.embedding_feature_scaler = TorchMetadataScaler.from_path(embedding_feature_scaler_path)
        else:
            self.embedding_feature_scaler = nn.Identity()

        if token_feature_scaler_path is not None:
            self.token_feature_scaler = TorchMetadataScaler.from_path(token_feature_scaler_path)
        else:
            self.token_feature_scaler = nn.Identity()

        if self.unfreeze_last_n_decoder_blocks < 0:
            raise ValueError(
                "unfreeze_last_n_decoder_blocks must be >= 0, "
                f"got {self.unfreeze_last_n_decoder_blocks}"
            )

        self.group_encoder = GroupWiseMetadataEncoder(
            lexical_dim=lexical_feature_dim,
            embedding_dim=len(EMBEDDING_FEATURE_NAMES),
            token_dim=len(TOKEN_FEATURE_NAMES),
            d_model=self.d_model,
            hidden_dim=metadata_mlp_hidden_dim,
            dropout=metadata_dropout,
            normalize_inputs=self.normalize_metadata_features,
            projection_type=self.metadata_projection_type,
        )

        self.uni_attention = UniAttention(
            d_model=self.d_model,
            num_heads=attention_heads,
            dropout=metadata_dropout,
        )

        self.meta_ln_1 = nn.LayerNorm(self.d_model)

        self.use_meta_ffn = bool(use_meta_ffn)
        if self.use_meta_ffn:
            meta_ffn_layers: list[nn.Module] = [
                nn.Linear(self.d_model, self.d_model * 4),
                nn.ReLU(),
            ]

            if metadata_dropout > 0.0:
                meta_ffn_layers.append(nn.Dropout(metadata_dropout))

            meta_ffn_layers.append(nn.Linear(self.d_model * 4, self.d_model))

            self.meta_ffn = nn.Sequential(*meta_ffn_layers)
            self.meta_ln_2 = nn.LayerNorm(self.d_model)

        pooled_projection_layers: list[nn.Module] = [
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
        ]

        if metadata_dropout > 0.0:
            pooled_projection_layers.append(nn.Dropout(metadata_dropout))

        pooled_projection_layers.append(nn.Linear(self.d_model, self.d_model))

        self.pooled_concat_projection = nn.Sequential(*pooled_projection_layers)
        self.pooled_concat_ln = nn.LayerNorm(self.d_model)

        # Used only by metadata_fusion_mode="meta_prefix".
        self.meta_prefix_ln = nn.LayerNorm(self.d_model)

        self.configure_trainable_parameters(
            decoder_trainable_scope=self.decoder_trainable_scope,
            unfreeze_last_n_decoder_blocks=self.unfreeze_last_n_decoder_blocks,
            unfreeze_lm_head=self.unfreeze_lm_head_flag,
            unfreeze_shared_embeddings=self.unfreeze_shared_embeddings,
        )

    def configure_trainable_parameters(
        self,
        *,
        decoder_trainable_scope: str = "last_n_blocks",
        unfreeze_last_n_decoder_blocks: int = 1,
        unfreeze_lm_head: bool = True,
        unfreeze_shared_embeddings: bool = False,
    ) -> None:
        """
        Freezes the T5 backbone, then enables one of three decoder training scopes:

        - cross_attention:
            train only decoder cross-attention layers in all decoder blocks.
        - last_n_blocks:
            train decoder cross-attention layers in all blocks plus the last N
            decoder blocks.
        - full_decoder:
            train the whole decoder stack. By default, T5 shared embeddings are
            kept frozen because they are shared with the frozen encoder; pass
            unfreeze_shared_embeddings=True to train them too.
        """
        decoder_trainable_scope = str(decoder_trainable_scope)
        unfreeze_last_n_decoder_blocks = int(unfreeze_last_n_decoder_blocks)

        if decoder_trainable_scope not in self.VALID_DECODER_TRAINABLE_SCOPES:
            raise ValueError(
                "decoder_trainable_scope must be one of "
                f"{sorted(self.VALID_DECODER_TRAINABLE_SCOPES)}, "
                f"got {decoder_trainable_scope}"
            )

        if unfreeze_last_n_decoder_blocks < 0:
            raise ValueError(
                "unfreeze_last_n_decoder_blocks must be >= 0, "
                f"got {unfreeze_last_n_decoder_blocks}"
            )

        # Start from a fully frozen T5 backbone.
        for param in self.base_model.parameters():
            param.requires_grad = False

        decoder = self.base_model.get_decoder()

        if decoder_trainable_scope == "cross_attention":
            # In T5 decoder blocks, layer.1 is the encoder-decoder cross-attention.
            for name, param in self.base_model.named_parameters():
                if "decoder.block" in name and ".layer.1." in name:
                    param.requires_grad = True

        elif decoder_trainable_scope == "last_n_blocks":
            # Cross-attention remains trainable in all decoder blocks.
            for name, param in self.base_model.named_parameters():
                if "decoder.block" in name and ".layer.1." in name:
                    param.requires_grad = True

            if hasattr(decoder, "block") and len(decoder.block) > 0:
                n_blocks = len(decoder.block)
                n_to_unfreeze = min(unfreeze_last_n_decoder_blocks, n_blocks)

                if n_to_unfreeze > 0:
                    for block in decoder.block[-n_to_unfreeze:]:
                        for param in block.parameters():
                            param.requires_grad = True

        elif decoder_trainable_scope == "full_decoder":
            # Train the full decoder stack: self-attention, cross-attention,
            # feed-forward layers, layer norms, and optionally shared embeddings.
            for param in decoder.parameters():
                param.requires_grad = True

            if not unfreeze_shared_embeddings:
                # T5 uses shared input embeddings. Freezing them keeps the encoder
                # input embedding space fixed even when the decoder is fully trainable.
                if hasattr(decoder, "embed_tokens"):
                    for param in decoder.embed_tokens.parameters():
                        param.requires_grad = False

                if hasattr(self.base_model, "shared"):
                    for param in self.base_model.shared.parameters():
                        param.requires_grad = False

                encoder = self.base_model.get_encoder()
                if hasattr(encoder, "embed_tokens"):
                    for param in encoder.embed_tokens.parameters():
                        param.requires_grad = False

        if unfreeze_lm_head and hasattr(self.base_model, "lm_head"):
            for param in self.base_model.lm_head.parameters():
                param.requires_grad = True

    def freeze_all_base_model_except_decoder_cross_attention(
        self,
        *,
        unfreeze_last_n_decoder_blocks: int = 1,
        unfreeze_lm_head: bool = True,
    ) -> None:
        """
        Backward-compatible wrapper for the previous training setup:
        decoder cross-attention in all blocks + optionally the last N decoder blocks.
        """
        self.configure_trainable_parameters(
            decoder_trainable_scope="last_n_blocks",
            unfreeze_last_n_decoder_blocks=unfreeze_last_n_decoder_blocks,
            unfreeze_lm_head=unfreeze_lm_head,
            unfreeze_shared_embeddings=False,
        )

    def get_trainable_parameter_stats(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"trainable": int(trainable), "total": int(total)}

    def get_trainable_parameter_summary(self, max_items: int = 120) -> Dict[str, Any]:
        trainable = []
        total_trainable = 0

        for name, param in self.named_parameters():
            if param.requires_grad:
                n_params = param.numel()
                total_trainable += n_params
                trainable.append(
                    {
                        "name": name,
                        "shape": list(param.shape),
                        "num_params": int(n_params),
                    }
                )

        return {
            "total_trainable": int(total_trainable),
            "num_trainable_tensors": len(trainable),
            "trainable_preview": trainable[:max_items],
        }

    def _pair_logits_to_quality_score(self, pair_logits: torch.Tensor) -> torch.Tensor:
        """
        pair_logits: [B, 2], order = [true, false]
        """
        if self.scoring_mode == "true_prob":
            return torch.softmax(pair_logits, dim=1)[:, 0]

        return torch.log_softmax(pair_logits, dim=1)[:, 0]

    def _labels_to_target_idx(
        self,
        binary_labels: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if binary_labels is not None:
            binary_labels = binary_labels.long()
            return torch.where(
                binary_labels > 0,
                torch.zeros_like(binary_labels),
                torch.ones_like(binary_labels),
            )

        if labels is not None:
            if labels.ndim == 1:
                first_label = labels
            else:
                first_label = labels[:, 0]

            return torch.where(
                first_label == self.true_token_id,
                torch.zeros_like(first_label),
                torch.ones_like(first_label),
            )

        return None

    def _compute_loss_from_pair_logits(
        self,
        pair_logits: torch.Tensor,
        binary_labels: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        target_idx = self._labels_to_target_idx(binary_labels, labels)

        if target_idx is None:
            return None

        return F.cross_entropy(pair_logits, target_idx)

    def _apply_metadata_fusion(
        self,
        h_text: torch.Tensor,
        attention_mask: torch.Tensor,
        z_meta_fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
        - h_fused
        - fused_attention_mask
        - debug tensors
        """
        batch_size, seq_len, _ = h_text.shape

        if self.metadata_fusion_mode == "concat_tokens":
            # Decoder attends to [z_lex ; z_emb ; z_tok ; H_text].
            h_fused = torch.cat([z_meta_fused, h_text], dim=1)

            meta_mask = torch.ones(
                (batch_size, z_meta_fused.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            fused_attention_mask = torch.cat([meta_mask, attention_mask], dim=1)

            debug = {
                "meta_vec": z_meta_fused.mean(dim=1),
            }

            return h_fused, fused_attention_mask, debug

        if self.metadata_fusion_mode == "meta_prefix":
            # Decoder attends to [meta_vec ; H_text].
            meta_vec = z_meta_fused.mean(dim=1)
            meta_token = self.meta_prefix_ln(meta_vec).unsqueeze(1)

            h_fused = torch.cat([meta_token, h_text], dim=1)

            meta_mask = torch.ones(
                (batch_size, 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            fused_attention_mask = torch.cat([meta_mask, attention_mask], dim=1)

            debug = {
                "meta_vec": meta_vec,
                "metadata_update": meta_token,
                "metadata_update_norm": meta_token.norm(p=2, dim=-1).squeeze(1),
            }

            return h_fused, fused_attention_mask, debug

        if self.metadata_fusion_mode in {
            "pooled_concat_projection",
            "direct_concat_projection",
        }:
            # z_meta_fused: [B, 3, d]
            meta_vec = z_meta_fused.mean(dim=1)

            # meta_seq: [B, L, d]
            meta_seq = meta_vec.unsqueeze(1).expand(-1, seq_len, -1)

            # token_meta_pair: [B, L, 2d]
            token_meta_pair = torch.cat([h_text, meta_seq], dim=-1)

            # projected: [B, L, d]
            projected = self.pooled_concat_projection(token_meta_pair)

            if self.metadata_fusion_mode == "pooled_concat_projection":
                # H_fused = LayerNorm(H_text + Linear([H_text ; meta_vec]))
                h_fused = self.pooled_concat_ln(h_text + projected)
            else:
                # H_fused = LayerNorm(Linear([H_text ; meta_vec]))
                h_fused = self.pooled_concat_ln(projected)

            fused_attention_mask = attention_mask

            debug = {
                "meta_vec": meta_vec,
                "metadata_update": projected,
                "metadata_update_norm": projected.norm(p=2, dim=-1).mean(dim=1),
            }

            return h_fused, fused_attention_mask, debug

        raise ValueError(f"Unsupported metadata_fusion_mode={self.metadata_fusion_mode}")

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
        encoder_outputs = self.base_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        h_text = encoder_outputs.last_hidden_state

        batch_size, _seq_len, hidden_dim = h_text.shape

        if hidden_dim != self.d_model:
            raise ValueError(
                f"Unexpected hidden dim. got={hidden_dim}, expected={self.d_model}"
            )

        emb_features = extract_embedding_level_metadata(h_text, attention_mask)
        tok_features = extract_token_level_metadata(h_text, attention_mask)

        # Standardization.
        # lexical_features may already be standardized by the dataset.
        # If lexical_feature_scaler_path is None, this is Identity().
        lexical_features = self.lexical_feature_scaler(lexical_features.to(h_text.dtype))
        emb_features = self.embedding_feature_scaler(emb_features)
        tok_features = self.token_feature_scaler(tok_features)

        z_meta = self.group_encoder(
            lexical_features=lexical_features,
            embedding_features=emb_features,
            token_features=tok_features,
        )

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

        z_meta_fused = self.meta_ln_1(z_meta + z_att)

        if self.use_meta_ffn:
            z_meta_fused = self.meta_ln_2(
                z_meta_fused + self.meta_ffn(z_meta_fused)
            )

        h_fused, fused_attention_mask, fusion_debug = self._apply_metadata_fusion(
            h_text=h_text,
            attention_mask=attention_mask,
            z_meta_fused=z_meta_fused,
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

        pair_logits = torch.stack([logits_true, logits_false], dim=1)
        quality_score = self._pair_logits_to_quality_score(pair_logits)
        loss = self._compute_loss_from_pair_logits(pair_logits, binary_labels, labels)

        output = {
            "loss": loss,
            "logits_true": logits_true,
            "logits_false": logits_false,
            "pair_logits": pair_logits,
            "quality_score": quality_score,
            "decoder_logits": decoder_outputs.logits,
            "h_text": h_text,
            "z_meta": z_meta,
            "z_meta_fused": z_meta_fused,
            "h_fused": h_fused,
            "fused_attention_mask": fused_attention_mask,
        }

        output.update(fusion_debug)

        return output

    def save_metadata_modules(self, output_dir: str | Path, metadata_config: Dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        modules_payload = {
            "group_encoder": self.group_encoder.state_dict(),
            "uni_attention": self.uni_attention.state_dict(),
            "meta_ln_1": self.meta_ln_1.state_dict(),
            "meta_prefix_ln": self.meta_prefix_ln.state_dict(),
            "use_meta_ffn": self.use_meta_ffn,
            "normalize_metadata_features": self.normalize_metadata_features,
            "metadata_projection_type": self.metadata_projection_type,
            "unfreeze_last_n_decoder_blocks": self.unfreeze_last_n_decoder_blocks,
            "unfreeze_lm_head": self.unfreeze_lm_head_flag,
            "decoder_trainable_scope": self.decoder_trainable_scope,
            "unfreeze_shared_embeddings": self.unfreeze_shared_embeddings,
            "metadata_fusion_mode": self.metadata_fusion_mode,
            "lexical_feature_scaler_path": self.lexical_feature_scaler_path,
            "embedding_feature_scaler_path": self.embedding_feature_scaler_path,
            "token_feature_scaler_path": self.token_feature_scaler_path,
            "lexical_feature_scaler": self.lexical_feature_scaler.state_dict(),
            "embedding_feature_scaler": self.embedding_feature_scaler.state_dict(),
            "token_feature_scaler": self.token_feature_scaler.state_dict(),
            "pooled_concat_projection": self.pooled_concat_projection.state_dict(),
            "pooled_concat_ln": self.pooled_concat_ln.state_dict(),
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

        if "meta_prefix_ln" in payload:
            self.meta_prefix_ln.load_state_dict(payload["meta_prefix_ln"])

        if "lexical_feature_scaler" in payload:
            self.lexical_feature_scaler.load_state_dict(
                payload["lexical_feature_scaler"],
                strict=False,
            )

        if "embedding_feature_scaler" in payload:
            self.embedding_feature_scaler.load_state_dict(
                payload["embedding_feature_scaler"],
                strict=False,
            )

        if "token_feature_scaler" in payload:
            self.token_feature_scaler.load_state_dict(
                payload["token_feature_scaler"],
                strict=False,
            )

        if self.use_meta_ffn and "meta_ffn" in payload:
            self.meta_ffn.load_state_dict(payload["meta_ffn"])
            self.meta_ln_2.load_state_dict(payload["meta_ln_2"])

        if "pooled_concat_projection" in payload:
            self.pooled_concat_projection.load_state_dict(payload["pooled_concat_projection"])

        if "pooled_concat_ln" in payload:
            self.pooled_concat_ln.load_state_dict(payload["pooled_concat_ln"])


class RunningMoments:
    """
    Streaming mean/std estimator for scaler fitting.

    The estimator first applies the configured per-feature transform
    and then computes mean/std on the transformed values.
    """

    def __init__(
        self,
        dim: int,
        *,
        feature_names: Optional[Sequence[str]] = None,
        feature_transforms: Optional[Dict[str, str]] = None,
    ):
        self.dim = int(dim)
        self.count = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.dim)]

        if len(feature_names) != self.dim:
            raise ValueError(
                f"RunningMoments feature_names length mismatch: {len(feature_names)} != {self.dim}"
            )

        self.feature_names = list(feature_names)

        if feature_transforms is None:
            feature_transforms = {name: "zscore" for name in self.feature_names}

        self.feature_transforms = build_feature_transform_map(
            self.feature_names,
            feature_transforms,
            fallback="zscore",
        )

    def update(self, batch_values: np.ndarray) -> None:
        if batch_values.ndim != 2 or batch_values.shape[1] != self.dim:
            raise ValueError(
                f"Invalid batch for RunningMoments: shape={batch_values.shape}, "
                f"expected [N, {self.dim}]"
            )

        transformed = MetadataFeatureScaler.apply_transforms_only(
            batch_values,
            self.feature_names,
            self.feature_transforms,
        ).astype(np.float64)

        for row in transformed:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def finalize(self, feature_names: Optional[Sequence[str]] = None) -> MetadataFeatureScaler:
        if self.count == 0:
            raise ValueError("Cannot finalize RunningMoments with count=0")

        if feature_names is not None and list(feature_names) != self.feature_names:
            raise ValueError(
                "RunningMoments.finalize received feature_names different from those used at init. "
                f"init={self.feature_names}, finalize={list(feature_names)}"
            )

        variance = self.m2 / max(self.count, 1)
        std = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
        mean = self.mean.astype(np.float32)

        for idx, name in enumerate(self.feature_names):
            if self.feature_transforms[name] == "identity":
                mean[idx] = 0.0
                std[idx] = 1.0

        return MetadataFeatureScaler(
            feature_names=self.feature_names,
            mean=mean,
            std=std,
            feature_transforms=self.feature_transforms,
        )


def load_metadata_qualt5_config(model_dir: str | Path) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    config_path = model_dir / "metadata_qualt5_config.json"

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def fit_online_metadata_scalers(
    model_name_or_path: str,
    dataloader: Any,
    *,
    device: str | torch.device = "cuda",
    input_ids_key: str = "input_ids",
    attention_mask_key: str = "attention_mask",
    max_batches: Optional[int] = None,
    embedding_feature_transforms: Optional[Dict[str, str]] = None,
    token_feature_transforms: Optional[Dict[str, str]] = None,
) -> tuple[MetadataFeatureScaler, MetadataFeatureScaler]:
    """
    Fit feature-aware scaler statistics for x_emb and x_tok computed online from H_text.

    The dataloader must yield dictionaries containing:
    - input_ids
    - attention_mask
    """
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
    base_model.eval()

    encoder = base_model.get_encoder()

    if embedding_feature_transforms is None:
        embedding_feature_transforms = build_feature_transform_map(
            EMBEDDING_FEATURE_NAMES,
            DEFAULT_EMBEDDING_FEATURE_TRANSFORMS,
            fallback="zscore",
        )

    if token_feature_transforms is None:
        token_feature_transforms = build_feature_transform_map(
            TOKEN_FEATURE_NAMES,
            DEFAULT_TOKEN_FEATURE_TRANSFORMS,
            fallback="zscore",
        )

    emb_moments = RunningMoments(
        len(EMBEDDING_FEATURE_NAMES),
        feature_names=EMBEDDING_FEATURE_NAMES,
        feature_transforms=embedding_feature_transforms,
    )
    tok_moments = RunningMoments(
        len(TOKEN_FEATURE_NAMES),
        feature_names=TOKEN_FEATURE_NAMES,
        feature_transforms=token_feature_transforms,
    )

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch[input_ids_key].to(device)
        attention_mask = batch[attention_mask_key].to(device)

        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        h_text = encoder_outputs.last_hidden_state

        emb_features = extract_embedding_level_metadata(h_text, attention_mask)
        tok_features = extract_token_level_metadata(h_text, attention_mask)

        emb_moments.update(emb_features.detach().cpu().float().numpy())
        tok_moments.update(tok_features.detach().cpu().float().numpy())

    emb_scaler = emb_moments.finalize()
    tok_scaler = tok_moments.finalize()

    return emb_scaler, tok_scaler

