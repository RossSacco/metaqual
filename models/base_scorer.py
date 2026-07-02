import pyterrier as pt
import pandas as pd
import torch
import nltk
import math
from pathlib import Path
from collections import Counter
import re
import lmppl
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pyterrier_dr
import numpy as np
from pyterrier_quality import QualT5, Filter
from torch.nn import functional as F

from pathlib import Path
from typing import Any, Dict, Optional






try:
    from metaqual.models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        load_metadata_qualt5_config,
    )
except ImportError:
    from models.metadata_qualt5 import (
        LexicalMetadataStore,
        MetadataEnrichedQualT5,
        MetadataFeatureScaler,
        load_metadata_qualt5_config,
    )

class TasbMagnitudeScorer(pt.Transformer):
    """
    Adds a 'quality' column based on TAS-B embedding magnitude,
    using the official pyterrier_dr implementation for exact reproducibility.
    """
    def __init__(self, batch_size=32):
        print("Inizializzazione TAS-B Magnitude Scorer (via pyterrier_dr)...")
        # Inizializza il modello TAS-B ufficiale di pyterrier_dr
        self.tasb_model = pyterrier_dr.TasB()
        
        # doc_encoder() crea un transformer di PyTerrier che prende in input il testo 
        # e restituisce gli embedding. Gestisce GPU e batch_size in automatico.
        self.encoder = self.tasb_model.doc_encoder(batch_size=batch_size)

    def transform(self, df):
        # 1. Passiamo il DataFrame all'encoder di pyterrier_dr
        # Questo calcola gli embedding e aggiunge una colonna chiamata 'doc_vec' 
        # che contiene gli array numpy dei vettori per ogni documento.
        encoded_df = self.encoder.transform(df)
        
        # 2. Estraiamo la colonna dei vettori e li impiliamo in una matrice 2D
        vectors = np.stack(encoded_df['doc_vec'].values)
        
        # 3. Calcoliamo la magnitudo (Norma L2) lungo l'asse 1 (per ogni documento)
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        # 4. Assegniamo i punteggi alla colonna 'quality' nel DataFrame originale
        res = df.copy()
        res['quality'] = -magnitudes
        
        return res


class PerplexityScorer(pt.Transformer):
    def __init__(self, model_name="t5-base", batch_size=16):
        """
        Computes perplexity (or pseudo-perplexity) using the asahi417/lmppl library.
        Supports autoregressive (GPT), masked (BERT), and seq2seq (T5/BART) models.
        """
        self.batch_size = batch_size
        self.model_name = model_name.lower()
        
        # lmppl requires instantiating the correct class based on model architecture
        print(f"Inizializzazione LM-PPL con modello '{model_name}'...")
        if "t5" in self.model_name or "bart" in self.model_name:
            self.scorer = lmppl.EncoderDecoderLM(model_name)
            self.is_enc_dec = True
        elif "bert" in self.model_name or "roberta" in self.model_name:
            self.scorer = lmppl.MaskedLM(model_name)
            self.is_enc_dec = False
        else:
            self.scorer = lmppl.LM(model_name) # For GPT-2, GPT-Neo, etc.
            self.is_enc_dec = False

    def transform(self, df):
        res = df.copy()
        scores = []
        
        # Batch processing to avoid saturating VRAM
        for i in range(0, len(res), self.batch_size):
            batch_texts = res['text'].iloc[i:i+self.batch_size].tolist()
            
            # Perplexity computation via lmppl
            if self.is_enc_dec:
                # For encoder-decoder models (such as T5), perplexity is computed
                # on the output. We pass an empty input to evaluate only the text.
                vuoti = [""] * len(batch_texts)
                ppl_values = self.scorer.get_perplexity(input_texts=vuoti, output_texts=batch_texts, batch_size=self.batch_size)
            else:
                # For standard models (GPT or BERT), passing the text list is enough
                ppl_values = self.scorer.get_perplexity(batch_texts, batch_size=self.batch_size)
            # INVERT THE SIGN:
            # A LOW perplexity indicates better, more fluent text.
            # By inverting the sign (e.g., PPL 15 -> -15), PyTerrier's Filter(threshold)
            # keeps texts with scores above the threshold.
            # Example: Filter(-50) keeps all documents with perplexity <= 50.
            neg_ppl = [-p for p in ppl_values]
            scores.extend(neg_ppl)
            
        # Assign score to the 'quality' column (to be compatible with QualCache)
        res['quality'] = scores
        return res


class ITNScorer(pt.Transformer):
    """
    Computes the Information-To-Noise ratio using the Porter Stemmer.
    """
    def __init__(self):
        # Initialize the stemmer only once
        self.stemmer = PorterStemmer()

    def transform(self, df):
        res = df.copy()
        scores = []
        for text in res['text']:
            # 1. Base token extraction
            raw_tokens = re.findall(r'\b\w+\b', str(text).lower())
            
            # 2. Apply Porter Stemmer
            stemmed_tokens = [self.stemmer.stem(token) for token in raw_tokens]
            
            total_terms = len(stemmed_tokens)
            
            if total_terms == 0:
                scores.append(0.0)
            else:
                unique_terms = len(set(stemmed_tokens))
                scores.append(unique_terms / total_terms)
                
        res['quality'] = scores
        return res


class CDDScorer(pt.Transformer):
    """
    Computes CDD (KL-Divergence) over terms extracted with the Porter Stemmer.
    """
    def __init__(self, background_corpus_iter, sample_limit=50000):
        self.stemmer = PorterStemmer()
        self.collection_probs = {}
        self.default_prob = 1e-6 
        self._build_background_model(background_corpus_iter, sample_limit)
            
    def _build_background_model(self, corpus_iter, limit):
        print(f"Costruzione Background Model con Porter Stemmer su {limit} doc...")
        term_counts = Counter()
        total_collection_terms = 0
        
        for i, doc in enumerate(corpus_iter):
            if i >= limit: break
            raw_tokens = re.findall(r'\b\w+\b', str(doc['text']).lower())
            stemmed_tokens = [self.stemmer.stem(t) for t in raw_tokens]
            
            term_counts.update(stemmed_tokens)
            total_collection_terms += len(stemmed_tokens)
            
        for term, count in term_counts.items():
            self.collection_probs[term] = count / total_collection_terms
        print(f"Modello completato. {len(self.collection_probs)} radici (stems) apprese.")

    def transform(self, df):
        res = df.copy()
        scores = []
        # Parametro di smoothing indicato nel paper per la CDD
        lambda_param = 0.99
        
        # Il set di tutti i termini nel lessico (T) della collezione
        all_terms = set(self.collection_probs.keys())
        
        for text in res['text']:
            raw_tokens = re.findall(r'\b\w+\b', str(text).lower())
            stemmed_tokens = [self.stemmer.stem(t) for t in raw_tokens]
            doc_len = len(stemmed_tokens)
            
            if doc_len == 0:
                scores.append(-999.0)
                continue
                
            doc_counts = Counter(stemmed_tokens)
            cdd_dist = 0.0
            
            # La formula CDD del paper: \sum_{t \in T} Pr(t|P) * log(...) 
            # P_t_P è Pr(t|P), ovvero la probabilità del termine nella collezione
            for term, p_t_p in self.collection_probs.items():
                # Pr(t|p_?) è la probabilità del termine nel passaggio (documento)
                p_t_doc = doc_counts.get(term, 0) / doc_len
                
                # Calcolo del denominatore con smoothing: \lambda*Pr(t|p) + (1-\lambda)*Pr(t|P) 
                denom = (lambda_param * p_t_doc) + ((1 - lambda_param) * p_t_p)
                
                # Sommatoria della KL-Divergenza
                cdd_dist += p_t_p * math.log2(p_t_p / denom)
            
            # Il paper usa il negativo della distanza come misura di qualità [cite: 160]
            scores.append(-cdd_dist)
            
        res['quality'] = scores
        return res


class FinetunedQualT5Scorer(pt.Transformer):
    """
    QualT5-style passage quality scorer.

    Score computed:
        quality = log P(true | true, false)

    Prompt:
        Document: {passage} Relevant:

    The model predicts only between the two target tokens:
        true / false

    This implementation follows the optimization used in pyterrier-quality:
    the lm_head is restricted to the rows corresponding to the target tokens.
    """

    def __init__(
        self,
        model_name_or_path,
        *,
        batch_size=100,
        max_length=512,
        prompt="Document: {} Relevant:",
        device=None,
        verbose=False,
    ):
        if not model_name_or_path:
            raise ValueError(
                "Per 'finetuned_qualt5' devi specificare 'model_name_or_path'."
            )

        self.model_name_or_path = model_name_or_path
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.prompt = prompt
        self.verbose = verbose
        self.device = self._resolve_device(device)

        print(
            f"Inizializzazione FinetunedQualT5Scorer "
            f"(model={self.model_name_or_path}, device={self.device}, "
            f"batch_size={self.batch_size}, max_length={self.max_length})..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path
        )

        self.model.to(self.device)
        self.model.eval()

        # Se il modello ha targets nel config, usa quelli.
        # Altrimenti usa lo standard QualT5: ["true", "false"].
        targets = (
            self.model.config.targets
            if hasattr(self.model.config, "targets")
            else ["true", "false"]
        )

        if len(targets) != 2:
            raise ValueError(
                f"I targets devono essere esattamente 2, trovati: {targets}"
            )

        self.targets = targets

        true_token_id = self.tokenizer.encode(
            targets[0],
            add_special_tokens=False,
        )[0]

        false_token_id = self.tokenizer.encode(
            targets[1],
            add_special_tokens=False,
        )[0]

        self.true_token_id = true_token_id
        self.false_token_id = false_token_id

        # Ottimizzazione pyterrier-quality:
        # sostituisce la lm_head completa con una lm_head contenente solo
        # i pesi dei token "true" e "false".
        #
        # Dopo questa modifica, il modello produce solo 2 logits:
        #   posizione 0 -> true
        #   posizione 1 -> false
        with torch.no_grad():
            restricted_lm_head_weight = self.model.lm_head.weight[
                [self.true_token_id, self.false_token_id]
            ].clone()

        self.model.lm_head.weight = torch.nn.Parameter(
            restricted_lm_head_weight
        )

    def _resolve_device(self, requested_device):
        if requested_device:
            if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
                print("[WARNING] CUDA richiesta ma non disponibile. Fallback su CPU.")
                return torch.device("cpu")
            return torch.device(requested_device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_prompt(self, passage_text):
        return self.prompt.format(passage_text)

    def _score_batch(self, batch_texts):
        prompts = [self._build_prompt(t) for t in batch_texts]

        encoded = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        batch_size = len(batch_texts)

        decoder_start_token_id = self.model.config.decoder_start_token_id

        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id

        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

        encoded["decoder_input_ids"] = decoder_input_ids

        use_autocast = self.device.type == "cuda"

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            enabled=use_autocast,
        ):
            outputs = self.model(**encoded)

            # Prima posizione del decoder:
            # predizione del primo token dopo "Relevant:"
            logits = outputs.logits[:, 0]

            # Dopo la riduzione della lm_head, logits ha dimensione:
            # [batch_size, 2]
            #
            # logits[:, 0] = score per "true"
            # logits[:, 1] = score per "false"
            quality_scores = F.log_softmax(logits, dim=1)[:, 0]

        return quality_scores.detach().cpu().tolist()

    def transform(self, df):
        res = df.copy()

        if "text" not in res.columns:
            raise ValueError(
                "FinetunedQualT5Scorer richiede una colonna 'text'."
            )

        texts = res["text"].astype(str).tolist()

        quality_scores = []

        iterator = range(0, len(texts), self.batch_size)

        if self.verbose:
            iterator = pt.tqdm(
                iterator,
                desc="FinetunedQualT5Scorer",
                unit="batches",
            )

        for start_idx in iterator:
            batch_texts = texts[start_idx:start_idx + self.batch_size]
            batch_scores = self._score_batch(batch_texts)
            quality_scores.extend(float(score) for score in batch_scores)

        res["quality"] = quality_scores

        return res

def _is_trainer_checkpoint(path: str | Path) -> bool:
    p = Path(path)
    return p.name.startswith("checkpoint-")


def _metadata_asset_dir(model_path: str | Path) -> Path:
    """
    Directory da cui leggere metadata_qualt5_config.json, scaler e metadata_qualt5_modules.pt.

    Se model_path è un checkpoint Trainer tipo checkpoint-12000,
    questi file di solito stanno nella parent output_dir.
    """
    p = Path(model_path)

    if _is_trainer_checkpoint(p):
        return p.parent

    return p


def _safe_load_metadata_config(model_path: str | Path) -> Dict[str, Any]:
    """
    Stessa logica del diagnostic:
    - prova a caricare metadata_qualt5_config.json da model_path
    - se model_path è checkpoint-*, prova anche dalla parent directory
    """
    candidates = []

    p = Path(model_path)
    candidates.append(p)

    if _is_trainer_checkpoint(p):
        candidates.append(p.parent)

    for candidate in candidates:
        try:
            cfg = load_metadata_qualt5_config(candidate)

            if cfg:
                print(f"[MetadataEnrichedQualT5Scorer] metadata_qualt5_config.json caricato da {candidate}")
                print(f"[MetadataEnrichedQualT5Scorer] Metadata config: {cfg}")
                return cfg

        except Exception as exc:
            print(
                f"[MetadataEnrichedQualT5Scorer][WARNING] "
                f"Impossibile caricare metadata_qualt5_config.json da {candidate}: {exc}"
            )

    print(
        "[MetadataEnrichedQualT5Scorer][WARNING] "
        "metadata_qualt5_config.json non trovato. Uso valori YAML/constructor."
    )
    return {}


def _resolve_model_relative_path(
    path_value: Optional[str],
    base_dir: str | Path,
) -> Optional[str]:
    if path_value is None:
        return None

    p = Path(str(path_value))

    if p.is_absolute():
        return str(p)

    return str(Path(base_dir) / p)


def _checkpoint_state_dict_path(model_path: str | Path) -> Optional[Path]:
    """
    Se model_path punta a checkpoint-*, cerca il file dei pesi del Trainer.
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
    print(f"[MetadataEnrichedQualT5Scorer] Carico state_dict checkpoint da: {path}")

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

    if isinstance(state_dict, dict):
        for key in ["model", "state_dict", "module"]:
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Formato checkpoint non supportato: {type(state_dict)}")

    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        cleaned[new_key] = value

    return cleaned


def _scale_metadata_features(scaler, values: np.ndarray) -> np.ndarray:
    """
    Stessa logica del diagnostic: usa scaler.transform se disponibile.
    """
    if hasattr(scaler, "transform"):
        return scaler.transform(values).astype(np.float32)

    mean = np.asarray(scaler.mean, dtype=np.float32)
    std = np.asarray(scaler.std, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    return ((values.astype(np.float32) - mean) / std).astype(np.float32)


def _extract_scores_from_metadata_output(
    outputs: Any,
    true_token_id: int,
    false_token_id: int,
) -> torch.Tensor:
    """
    Copia coerente con il diagnostic.
    Ritorna sempre:
        log P(true | true,false)
    """
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

        if "logits_true" in outputs and "logits_false" in outputs:
            pair_logits = torch.stack(
                [outputs["logits_true"], outputs["logits_false"]],
                dim=-1,
            )
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


class MetadataEnrichedQualT5Scorer(pt.Transformer):
    """
    Metadata-enriched QualT5 scorer coerente con evaluate_metadata_qualt5_diagnostic.py.

    Input richiesti:
      - docno
      - text

    Output:
      - quality = log P(true | true,false)
    """

    VALID_FUSION_MODES = {
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

    def __init__(
        self,
        model_name_or_path,
        *,
        metadata_path,
        metadata_scaler_path=None,
        lexical_scaler_path=None,
        embedding_scaler_path=None,
        token_scaler_path=None,
        lexical_feature_names=None,
        base_model_name_or_path=None,
        text_model_path=None,
        batch_size=100,
        max_length=512,
        prompt="Document: {} Relevant:",
        device=None,
        scoring_mode=None,
        allow_missing_metadata=False,
        metadata_dropout=None,
        metadata_mlp_hidden_dim=None,
        metadata_projection_type=None,
        attention_heads=None,
        use_meta_ffn=None,
        metadata_fusion_mode=None,
        normalize_metadata_features=None,
        unfreeze_last_n_decoder_blocks=None,
        unfreeze_lm_head=None,
        decoder_trainable_scope=None,
        unfreeze_shared_embeddings=None,
        verbose=False,
    ):
        if not model_name_or_path:
            raise ValueError(
                "Per 'metadata_qualt5' devi specificare 'model_name_or_path'."
            )

        if not metadata_path:
            raise ValueError(
                "Per 'metadata_qualt5' devi specificare 'metadata_path'."
            )

        self.model_name_or_path = str(model_name_or_path)
        self.metadata_path = str(metadata_path)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.prompt = prompt
        self.verbose = bool(verbose)
        self.allow_missing_metadata = bool(allow_missing_metadata)
        self.device = self._resolve_device(device)

        self.model_path = Path(self.model_name_or_path)
        self.metadata_asset_dir = _metadata_asset_dir(self.model_path)

        # Stessa logica del diagnostic: cerca config sia nel checkpoint sia nella parent.
        self.saved_meta_cfg = _safe_load_metadata_config(self.model_path)

        # ------------------------------------------------------------
        # Feature lexical.
        # ------------------------------------------------------------
        if lexical_feature_names is None:
            lexical_feature_names = self.saved_meta_cfg.get("lexical_feature_names")

        self.lexical_store = LexicalMetadataStore.from_path(
            self.metadata_path,
            feature_names=lexical_feature_names,
        )

        # ------------------------------------------------------------
        # Scaler paths.
        # ------------------------------------------------------------
        self.lexical_scaler_path = self._resolve_config_path(
            explicit_path=lexical_scaler_path or metadata_scaler_path,
            config_keys=["lexical_scaler_path", "metadata_scaler_path"],
            fallback_filenames=[
                "lexical_metadata_scaler.pkl",
                "metadata_scaler.pkl",
            ],
            required=True,
            label="lexical/metadata scaler",
        )

        self.embedding_scaler_path = self._resolve_config_path(
            explicit_path=embedding_scaler_path,
            config_keys=["embedding_feature_scaler_path", "embedding_scaler_path"],
            fallback_filenames=[
                "embedding_metadata_scaler.pkl",
                "embedding_scaler.pkl",
            ],
            required=True,
            label="embedding scaler",
        )

        self.token_scaler_path = self._resolve_config_path(
            explicit_path=token_scaler_path,
            config_keys=["token_feature_scaler_path", "token_scaler_path"],
            fallback_filenames=[
                "token_metadata_scaler.pkl",
                "token_scaler.pkl",
            ],
            required=True,
            label="token scaler",
        )

        print(f"[MetadataEnrichedQualT5Scorer] lexical_scaler_path={self.lexical_scaler_path}")
        print(f"[MetadataEnrichedQualT5Scorer] embedding_scaler_path={self.embedding_scaler_path}")
        print(f"[MetadataEnrichedQualT5Scorer] token_scaler_path={self.token_scaler_path}")

        self.scaler = MetadataFeatureScaler.load(self.lexical_scaler_path)

        # Stessa sicurezza del diagnostic: l'ordine delle feature deve essere quello dello scaler.
        if list(self.lexical_store.feature_names) != list(self.scaler.feature_names):
            print(
                "[MetadataEnrichedQualT5Scorer] Re-loading lexical metadata store "
                "con feature_names dallo scaler."
            )
            self.lexical_store = LexicalMetadataStore.from_path(
                self.metadata_path,
                feature_names=self.scaler.feature_names,
            )

        if list(self.lexical_store.feature_names) != list(self.scaler.feature_names):
            raise ValueError(
                "Mismatch tra lexical_store.feature_names e lexical scaler feature_names.\n"
                f"store={self.lexical_store.feature_names}\n"
                f"scaler={self.scaler.feature_names}"
            )

        # ------------------------------------------------------------
        # Tokenizer.
        # Per coerenza: usa model_name_or_path se possibile.
        # Se è un Trainer checkpoint non compatibile, usa base_model_name_or_path/text_model_path.
        # ------------------------------------------------------------
        tokenizer_path = self._resolve_tokenizer_path(
            base_model_name_or_path=base_model_name_or_path,
            text_model_path=text_model_path,
        )

        print(
            f"Inizializzazione MetadataEnrichedQualT5Scorer "
            f"(model={self.model_name_or_path}, tokenizer={tokenizer_path}, "
            f"device={self.device}, batch_size={self.batch_size}, max_length={self.max_length})..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
        )

        true_ids = self.tokenizer.encode("true", add_special_tokens=False)
        false_ids = self.tokenizer.encode("false", add_special_tokens=False)

        if not true_ids or not false_ids:
            raise ValueError("Impossibile codificare i token true/false.")

        self.true_token_id = int(true_ids[0])
        self.false_token_id = int(false_ids[0])

        print(f"[MetadataEnrichedQualT5Scorer] true token ids={true_ids}")
        print(f"[MetadataEnrichedQualT5Scorer] false token ids={false_ids}")

        # ------------------------------------------------------------
        # Config effettivo: sempre priorità al metadata_qualt5_config.json,
        # poi fallback a YAML.
        # ------------------------------------------------------------
        scoring_mode = self.saved_meta_cfg.get(
            "scoring_mode",
            scoring_mode if scoring_mode is not None else "true_logprob",
        )

        metadata_dropout = self.saved_meta_cfg.get(
            "metadata_dropout",
            metadata_dropout if metadata_dropout is not None else 0.0,
        )

        metadata_mlp_hidden_dim = self.saved_meta_cfg.get(
            "metadata_mlp_hidden_dim",
            metadata_mlp_hidden_dim,
        )

        metadata_projection_type = self.saved_meta_cfg.get(
            "metadata_projection_type",
            metadata_projection_type if metadata_projection_type is not None else "linear",
        )

        attention_heads = self.saved_meta_cfg.get(
            "attention_heads",
            attention_heads if attention_heads is not None else 8,
        )

        use_meta_ffn = self.saved_meta_cfg.get(
            "use_meta_ffn",
            use_meta_ffn if use_meta_ffn is not None else True,
        )

        metadata_fusion_mode = self.saved_meta_cfg.get(
            "metadata_fusion_mode",
            metadata_fusion_mode if metadata_fusion_mode is not None else "pooled_concat_projection",
        )

        normalize_metadata_features = self.saved_meta_cfg.get(
            "normalize_metadata_features",
            normalize_metadata_features if normalize_metadata_features is not None else False,
        )

        unfreeze_last_n_decoder_blocks = self.saved_meta_cfg.get(
            "unfreeze_last_n_decoder_blocks",
            unfreeze_last_n_decoder_blocks if unfreeze_last_n_decoder_blocks is not None else 1,
        )

        unfreeze_lm_head = self.saved_meta_cfg.get(
            "unfreeze_lm_head",
            unfreeze_lm_head if unfreeze_lm_head is not None else True,
        )

        decoder_trainable_scope = self.saved_meta_cfg.get(
            "decoder_trainable_scope",
            decoder_trainable_scope if decoder_trainable_scope is not None else "last_n_blocks",
        )

        unfreeze_shared_embeddings = self.saved_meta_cfg.get(
            "unfreeze_shared_embeddings",
            unfreeze_shared_embeddings if unfreeze_shared_embeddings is not None else False,
        )

        if metadata_fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(
                f"metadata_fusion_mode={metadata_fusion_mode!r} non supportato. "
                f"Valori ammessi: {sorted(self.VALID_FUSION_MODES)}"
            )

        if decoder_trainable_scope not in self.VALID_DECODER_TRAINABLE_SCOPES:
            raise ValueError(
                f"decoder_trainable_scope={decoder_trainable_scope!r} non supportato. "
                f"Valori ammessi: {sorted(self.VALID_DECODER_TRAINABLE_SCOPES)}"
            )

        checkpoint_state_path = _checkpoint_state_dict_path(self.model_path)

        base_init_path = self._resolve_base_init_path(
            checkpoint_state_path=checkpoint_state_path,
            explicit_base_model_name_or_path=base_model_name_or_path,
            explicit_text_model_path=text_model_path,
        )

        print(
            "[MetadataEnrichedQualT5Scorer] Effective model config | "
            f"base_init_path={base_init_path} | "
            f"checkpoint_state_path={checkpoint_state_path} | "
            f"metadata_asset_dir={self.metadata_asset_dir} | "
            f"scoring_mode={scoring_mode} | "
            f"metadata_dropout={metadata_dropout} | "
            f"metadata_mlp_hidden_dim={metadata_mlp_hidden_dim} | "
            f"metadata_projection_type={metadata_projection_type} | "
            f"attention_heads={attention_heads} | "
            f"use_meta_ffn={use_meta_ffn} | "
            f"metadata_fusion_mode={metadata_fusion_mode} | "
            f"normalize_metadata_features={normalize_metadata_features} | "
            f"unfreeze_last_n_decoder_blocks={unfreeze_last_n_decoder_blocks} | "
            f"unfreeze_lm_head={unfreeze_lm_head} | "
            f"decoder_trainable_scope={decoder_trainable_scope} | "
            f"unfreeze_shared_embeddings={unfreeze_shared_embeddings}"
        )

        self.model = MetadataEnrichedQualT5(
            model_name_or_path=base_init_path,
            lexical_feature_dim=len(self.lexical_store.feature_names),
            true_token_id=self.true_token_id,
            false_token_id=self.false_token_id,
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

            # Come nel diagnostic: le lexical sono scalate fuori dal modello.
            lexical_feature_scaler_path=None,

            # Embedding/token features vengono calcolate online nel forward.
            embedding_feature_scaler_path=self.embedding_scaler_path,
            token_feature_scaler_path=self.token_scaler_path,
        )

        if checkpoint_state_path is not None:
            print(
                "[MetadataEnrichedQualT5Scorer] model_name_or_path è un Trainer checkpoint. "
                "Carico state_dict completo."
            )

            state_dict = _load_checkpoint_state_dict(checkpoint_state_path)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

            print(
                "[MetadataEnrichedQualT5Scorer] Checkpoint caricato. "
                f"missing_keys={len(missing)} | unexpected_keys={len(unexpected)}"
            )

            if missing:
                print(f"[MetadataEnrichedQualT5Scorer][WARNING] Missing keys preview: {missing[:30]}")

            if unexpected:
                print(f"[MetadataEnrichedQualT5Scorer][WARNING] Unexpected keys preview: {unexpected[:30]}")

        else:
            print(f"[MetadataEnrichedQualT5Scorer] Carico moduli metadata da: {self.metadata_asset_dir}")
            self.model.load_metadata_modules(self.metadata_asset_dir)

        self.model.to(self.device)
        self.model.eval()

        # Coerenza finale con diagnostic:
        # quality = log P(true | true,false)
        self.model.scoring_mode = "true_logprob"

    def _resolve_device(self, requested_device):
        if requested_device:
            if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
                print("[WARNING] CUDA richiesta ma non disponibile. Fallback su CPU.")
                return torch.device("cpu")

            return torch.device(requested_device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_config_path(
        self,
        *,
        explicit_path,
        config_keys,
        fallback_filenames,
        required,
        label,
    ):
        candidates = []

        if explicit_path is not None:
            candidates.append(explicit_path)

        for key in config_keys:
            value = self.saved_meta_cfg.get(key)
            if value:
                candidates.append(value)

        for filename in fallback_filenames:
            candidates.append(filename)

        for candidate in candidates:
            resolved = _resolve_model_relative_path(candidate, self.metadata_asset_dir)

            if resolved is not None and Path(resolved).exists():
                return resolved

        if required:
            raise FileNotFoundError(
                f"Non riesco a trovare {label}. "
                f"Candidati provati: {candidates}. "
                f"Metadata asset dir: {self.metadata_asset_dir}"
            )

        return None

    def _resolve_tokenizer_path(
        self,
        *,
        base_model_name_or_path=None,
        text_model_path=None,
    ) -> str:
        candidates = [
            self.model_path,
            self.saved_meta_cfg.get("tokenizer_name_or_path"),
            self.saved_meta_cfg.get("base_model_name_or_path"),
            base_model_name_or_path,
            text_model_path,
        ]

        for candidate in candidates:
            if not candidate:
                continue

            try:
                AutoTokenizer.from_pretrained(str(candidate), use_fast=True)
                return str(candidate)
            except Exception:
                continue

        raise FileNotFoundError(
            "Non riesco a caricare il tokenizer. "
            "Aggiungi nel config YAML: base_model_name_or_path oppure text_model_path."
        )

    def _resolve_base_init_path(
        self,
        *,
        checkpoint_state_path: Optional[Path],
        explicit_base_model_name_or_path=None,
        explicit_text_model_path=None,
    ) -> str:
        """
        Stessa logica del diagnostic.

        Se model_path è checkpoint-*, NON inizializzare direttamente da checkpoint-*,
        perché può contenere chiavi del wrapper custom.
        Inizializza da base model e poi carica lo state_dict completo.
        """
        if checkpoint_state_path is not None:
            base_from_cfg = self.saved_meta_cfg.get("base_model_name_or_path")

            if base_from_cfg:
                return str(base_from_cfg)

            if explicit_base_model_name_or_path:
                return str(explicit_base_model_name_or_path)

            if explicit_text_model_path:
                return str(explicit_text_model_path)

            raise ValueError(
                "model_name_or_path punta a un Trainer checkpoint, ma non so da quale "
                "base model inizializzare l'architettura. Aggiungi nel config YAML:\n"
                "base_model_name_or_path: \"/home/sacco/metaqual/outputs/qt5-supervised-t5-base/checkpoint-10000\""
            )

        return str(self.model_path)

    def _build_prompt(self, passage_text: str) -> str:
        return self.prompt.format(passage_text)

    def _score_batch(self, batch_docnos, batch_texts):
        prompts = [self._build_prompt(t) for t in batch_texts]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        raw_metadata = self.lexical_store.lookup(
            batch_docnos,
            allow_missing_metadata=self.allow_missing_metadata,
        )
        raw_metadata = np.asarray(raw_metadata, dtype=np.float32)

        scaled_metadata = _scale_metadata_features(
            self.scaler,
            raw_metadata,
        )

        batch_size = len(batch_texts)

        decoder_start_token_id = self.model.base_model.config.decoder_start_token_id

        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id

        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

        batch = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "lexical_features": torch.tensor(
                scaled_metadata,
                dtype=torch.float32,
                device=self.device,
            ),
            "decoder_input_ids": decoder_input_ids,
        }

        # Niente autocast: così è identico alla diagnostica.
        with torch.no_grad():
            outputs = self.model(**batch)

            quality_scores = _extract_scores_from_metadata_output(
                outputs,
                true_token_id=self.true_token_id,
                false_token_id=self.false_token_id,
            )

        return quality_scores.detach().float().cpu().tolist()

    def transform(self, df):
        res = df.copy()

        if "text" not in res.columns:
            raise ValueError(
                "MetadataEnrichedQualT5Scorer richiede una colonna 'text'."
            )

        if "docno" not in res.columns:
            raise ValueError(
                "MetadataEnrichedQualT5Scorer richiede una colonna 'docno'."
            )

        texts = res["text"].astype(str).tolist()
        docnos = res["docno"].astype(str).tolist()

        quality_scores = []

        iterator = range(0, len(texts), self.batch_size)

        if self.verbose:
            iterator = pt.tqdm(
                iterator,
                desc="MetadataEnrichedQualT5Scorer",
                unit="batches",
            )

        for start_idx in iterator:
            end_idx = start_idx + self.batch_size

            batch_texts = texts[start_idx:end_idx]
            batch_docnos = docnos[start_idx:end_idx]

            batch_scores = self._score_batch(batch_docnos, batch_texts)
            quality_scores.extend(float(score) for score in batch_scores)

        res["quality"] = quality_scores

        return res
    
    
def get_scorer(nome_scorer, **kwargs):
    """
    Initializes and returns the requested scorer ready for the PyTerrier pipeline.
    
    Arguments:
    - nome_scorer: string ('qualt5', 'finetuned_qualt5', 'metadata_qualt5',
      'tasb', 'perplexity', 'itn', 'cdd')
    - kwargs: extra arguments (e.g., background_corpus for CDD)
    """
    nome_scorer = nome_scorer.lower()
    
    if nome_scorer == 'qualt5':
        # You can also pass model size as a kwarg to test qt5-base or qt5-small
        model_size = kwargs.get('model_size', 'qt5-small')
        print(f"Inizializzazione QualT5 ({model_size})...")
        return QualT5(f'pyterrier-quality/{model_size}')

    elif nome_scorer == 'finetuned_qualt5':
        return FinetunedQualT5Scorer(
            model_name_or_path=kwargs.get("model_name_or_path"),
            batch_size=kwargs.get("batch_size", 100),
            max_length=kwargs.get("max_length", 512),
            device=kwargs.get("device"),
        )

    elif nome_scorer in ("metadata_qualt5", "metadata_enriched_qualt5"):
        lexical_feature_names = kwargs.get("lexical_feature_names")

        if lexical_feature_names is None:
            groups = kwargs.get("metadata_feature_groups", {})
            lexical_feature_names = groups.get("lexical")
        return MetadataEnrichedQualT5Scorer(
            model_name_or_path=kwargs.get("model_name_or_path"),
            metadata_path=kwargs.get("metadata_path"),

            # Serve se model_name_or_path punta a checkpoint-* Trainer.
            base_model_name_or_path=kwargs.get("base_model_name_or_path"),
            text_model_path=kwargs.get("text_model_path"),

        # Backward-compatible lexical scaler.
            metadata_scaler_path=kwargs.get("metadata_scaler_path"),
            lexical_scaler_path=kwargs.get("lexical_scaler_path"),

        # Online feature scalers.
            embedding_scaler_path=kwargs.get("embedding_scaler_path"),
            token_scaler_path=kwargs.get("token_scaler_path"),

            lexical_feature_names=lexical_feature_names,
            batch_size=kwargs.get("batch_size", 100),
            max_length=kwargs.get("max_length", 512),
            device=kwargs.get("device"),
            scoring_mode=kwargs.get("scoring_mode"),
            allow_missing_metadata=kwargs.get("allow_missing_metadata", False),

            metadata_dropout=kwargs.get("metadata_dropout"),
            metadata_mlp_hidden_dim=kwargs.get("metadata_mlp_hidden_dim"),
            metadata_projection_type=kwargs.get("metadata_projection_type"),
            attention_heads=kwargs.get("attention_heads"),
            use_meta_ffn=kwargs.get("use_meta_ffn"),

            metadata_fusion_mode=kwargs.get("metadata_fusion_mode"),
            normalize_metadata_features=kwargs.get("normalize_metadata_features"),
            unfreeze_last_n_decoder_blocks=kwargs.get("unfreeze_last_n_decoder_blocks"),
            unfreeze_lm_head=kwargs.get("unfreeze_lm_head"),

        # Questi prima mancavano: sono necessari per essere coerenti col diagnostic.
            decoder_trainable_scope=kwargs.get("decoder_trainable_scope"),
            unfreeze_shared_embeddings=kwargs.get("unfreeze_shared_embeddings"),

            verbose=kwargs.get("verbose", False),
        )
        
    elif nome_scorer == 'tasb':
        print("Inizializzazione TAS-B Magnitude Scorer...")
        return TasbMagnitudeScorer()
        
    elif nome_scorer == 'perplexity':
        print("Inizializzazione T5-base Perplexity Scorer...")
        return PerplexityScorer()
        
    elif nome_scorer == 'itn':
        print("Inizializzazione ITN Scorer (Porter Stemmer)...")
        return ITNScorer()
        
    elif nome_scorer == 'cdd':
        print("Inizializzazione CDD Scorer...")
        if 'background_corpus' not in kwargs:
            raise ValueError("Il CDD richiede il passaggio di 'background_corpus'")
        return CDDScorer(kwargs['background_corpus'])
        
    else:
        raise ValueError(f"Scorer '{nome_scorer}' non riconosciuto.")



"""
ROOT="/home/sacco/metaqual/outputs/metadata-qualt5-att_fusion-featureaware-fullDec-lr5e5-15k"
CKPT="$ROOT/checkpoint-12000"
OUT="$ROOT/scorer-checkpoint-12000"

rm -rf "$OUT"
mkdir -p "$OUT"

rsync -av "$ROOT/" "$OUT/" \
  --exclude "checkpoint-*" \
  --exclude "diagnostic_checkpoint_*" \
  --exclude "model.safetensors" \
  --exclude "pytorch_model.bin" \
  --exclude "metadata_qualt5_modules.pt"

python - "$CKPT/pytorch_model.bin" "$OUT" <<'PY'
import sys
import torch
from pathlib import Path

ckpt_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

sd = torch.load(ckpt_path, map_location="cpu")

if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]

base_sd = {}
meta_sd = {}

for k, v in sd.items():
    if k.startswith("base_model."):
        base_sd[k[len("base_model."):]] = v
    else:
        meta_sd[k] = v

print("Chiavi totali:", len(sd))
print("Chiavi T5/base_model:", len(base_sd))
print("Chiavi metadata:", len(meta_sd))

torch.save(base_sd, out_dir / "pytorch_model.bin")
torch.save(meta_sd, out_dir / "metadata_qualt5_modules.pt")

print("Creato:", out_dir)
PY

"""

"""
ROOT="/home/sacco/metaqual/outputs/metadata-qualt5-att_fusion-featureaware-fullDec-lr5e5-15k"
CKPT="$ROOT/checkpoint-12000"
OUT="$ROOT/scorer-checkpoint-12000"

python - "$CKPT/pytorch_model.bin" "$OUT" <<'PY'
import sys
import torch
from pathlib import Path

ckpt_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

print(f"Carico checkpoint: {ckpt_path}")

sd = torch.load(ckpt_path, map_location="cpu")

if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]

# 1. Parte T5: rimuovo prefisso base_model.
base_sd = {}
for k, v in sd.items():
    if k.startswith("base_model."):
        base_sd[k[len("base_model."):]] = v

# 2. Parte metadata: creo payload annidato come vuole load_metadata_modules()
module_prefixes = [
    "group_encoder",
    "uni_attention",
    "meta_ln_1",
    "meta_ffn",
    "meta_ln_2",
    "pooled_concat_projection",
    "pooled_concat_ln",
    "allmeta_token_projection",
    "allmeta_token_projection_ln",
    "meta_prefix_ln",
    "embedding_feature_scaler",
    "token_feature_scaler",
]

payload = {}

for prefix in module_prefixes:
    sub_sd = {}
    prefix_dot = prefix + "."

    for k, v in sd.items():
        if k.startswith(prefix_dot):
            sub_key = k[len(prefix_dot):]
            sub_sd[sub_key] = v

    if sub_sd:
        payload[prefix] = sub_sd

print("Chiavi totali checkpoint:", len(sd))
print("Chiavi T5/base_model:", len(base_sd))
print("Moduli metadata trovati:")
for k, v in payload.items():
    print(f"  {k}: {len(v)} chiavi")

if "group_encoder" not in payload:
    raise RuntimeError("ERRORE: group_encoder non trovato nel checkpoint.")

if "uni_attention" not in payload:
    raise RuntimeError("ERRORE: uni_attention non trovato nel checkpoint.")

torch.save(base_sd, out_dir / "pytorch_model.bin")
torch.save(payload, out_dir / "metadata_qualt5_modules.pt")

print()
print(f"Salvato T5: {out_dir / 'pytorch_model.bin'}")
print(f"Salvato metadata modules: {out_dir / 'metadata_qualt5_modules.pt'}")
PY

"""