import pyterrier as pt
import pandas as pd
import torch
import nltk
import math
from collections import Counter
import re
import lmppl
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pyterrier_dr
import numpy as np
from pyterrier_quality import QualT5, Filter

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
    Supervised QT5 passage quality scorer (query-independent) trained from
    MSMARCO triples where query is intentionally ignored and:
      - p+ -> "true"
      - p- -> "false"

    Prompt format:
      Document: {passage} Relevant:

    Final scoring, aligned with pyterrier-quality QualT5:
      quality = log P(true | true,false)

    This is computed as:
      log_softmax([logit_true, logit_false])[0]
    """

    def __init__(
        self,
        model_name_or_path,
        batch_size=64,
        max_length=256,
        device=None,
    ):
        if not model_name_or_path:
            raise ValueError(
                "Per 'finetuned_qualt5' devi specificare 'model_name_or_path' "
                "nei kwargs o in config scorers.finetuned_qualt5."
            )

        self.model_name_or_path = model_name_or_path
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = self._resolve_device(device)

        print(
            f"Inizializzazione FinetunedQualT5Scorer "
            f"(model={self.model_name_or_path}, device={self.device}, "
            f"batch_size={self.batch_size}, max_length={self.max_length})..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        self.true_token_id = self.tokenizer.encode(
            "true", add_special_tokens=False
        )[0]
        self.false_token_id = self.tokenizer.encode(
            "false", add_special_tokens=False
        )[0]

        if self.true_token_id is None or self.false_token_id is None:
            raise RuntimeError(
                "Tokenizzazione di 'true'/'false' non valida per questo tokenizer."
            )

    def _resolve_device(self, requested_device):
        if requested_device:
            if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
                print("[WARNING] CUDA richiesta ma non disponibile. Fallback su CPU.")
                return torch.device("cpu")
            return torch.device(requested_device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_prompt(self, passage_text):
        return f"Document: {passage_text} Relevant:"

    def _score_batch(self, batch_texts):
        prompts = [self._build_prompt(t) for t in batch_texts]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        batch_size = input_ids.size(0)

        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id

        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

            # Prima posizione del decoder: predizione del token dopo "Relevant:"
            first_logits = outputs.logits[:, 0, :]

            true_logits = first_logits[:, self.true_token_id]
            false_logits = first_logits[:, self.false_token_id]

            true_false_logits = torch.stack(
                [true_logits, false_logits],
                dim=1,
            )

            # Score finale coerente a pyterrier-quality:
            # log P(true | true,false)
            quality_scores = torch.log_softmax(
                true_false_logits,
                dim=1,
            )[:, 0]

        return quality_scores.detach().cpu().tolist()

    def transform(self, df):
        res = df.copy()

        if "text" not in res.columns:
            raise ValueError(
                "FinetunedQualT5Scorer richiede una colonna 'text'."
            )

        texts = res["text"].astype(str).tolist()
        quality_scores = []

        for start_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[start_idx:start_idx + self.batch_size]
            scores = self._score_batch(batch_texts)
            quality_scores.extend(float(s) for s in scores)

        res["quality"] = quality_scores
        return res
    
    
def get_scorer(nome_scorer, **kwargs):
    """
    Initializes and returns the requested scorer ready for the PyTerrier pipeline.
    
    Arguments:
    - nome_scorer: string ('qualt5', 'finetuned_qualt5', 'tasb', 'perplexity', 'itn', 'cdd')
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
            batch_size=kwargs.get("batch_size", 64),
            max_length=kwargs.get("max_length", 256),
            device=kwargs.get("device"),
            scoring_mode=kwargs.get("scoring_mode", "true_prob"),
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
