import argparse
import csv
import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ============================================================
# Utils
# ============================================================

def clean_model_name(path_or_id: str) -> str:
    if path_or_id.startswith("pyterrier-quality/"):
        name = "official_" + path_or_id.split("/")[-1]
    else:
        name = os.path.basename(path_or_id.rstrip("/"))

    name = name.replace("-", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return name


def finite_mask(*arrays):
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Ranking con media sui ties, simile a scipy.stats.rankdata(method='average').
    """
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1

        avg_rank = (i + j + 2) / 2.0  # rank da 1
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = finite_mask(a, b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return float("nan")

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = finite_mask(a, b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return float("nan")

    return pearson_corr(rankdata_average_ties(a), rankdata_average_ties(b))


def regression_metrics(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    mask = finite_mask(candidate, reference)
    candidate = candidate[mask]
    reference = reference[mask]

    if len(candidate) == 0:
        return {
            "n_valid": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "max_abs_error": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    diff = candidate - reference

    return {
        "n_valid": int(len(candidate)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "pearson": pearson_corr(candidate, reference),
        "spearman": spearman_corr(candidate, reference),
    }


# ============================================================
# Dataset loading
# ============================================================

def load_eval_pairs_from_ir_datasets(
    irds_dataset_id: str,
    num_triples: int,
    skip_triples: int,
) -> List[Tuple[str, str]]:
    """
    Carica coppie (p+, p-) da ir_datasets.
    La query viene ignorata perché il modello è query-independent.
    """
    import ir_datasets

    print(f"[INFO] Loading dataset: {irds_dataset_id}")
    dataset = ir_datasets.load(irds_dataset_id)
    docs_store = dataset.docs_store()

    pairs: List[Tuple[str, str]] = []
    iterator = dataset.docpairs_iter()

    print(f"[INFO] Skipping first {skip_triples} triples...")
    for i, _ in enumerate(iterator):
        if i + 1 >= skip_triples:
            break

    print(f"[INFO] Collecting {num_triples} triples...")
    for docpair in tqdm(iterator, total=num_triples):
        if len(pairs) >= num_triples:
            break

        pos_id = getattr(docpair, "doc_id_a", None)
        neg_id = getattr(docpair, "doc_id_b", None)

        if pos_id is None or neg_id is None:
            continue

        pos_doc = docs_store.get(pos_id)
        neg_doc = docs_store.get(neg_id)

        if pos_doc is None or neg_doc is None:
            continue

        pos_text = getattr(pos_doc, "text", None)
        neg_text = getattr(neg_doc, "text", None)

        if not pos_text or not neg_text:
            continue

        pairs.append((str(pos_text), str(neg_text)))

    print(f"[INFO] Collected pairs: {len(pairs)}")
    return pairs


# ============================================================
# Official QualT5-style scoring
# ============================================================

@torch.no_grad()
def score_passages_official_quality(
    model,
    tokenizer,
    passages: List[str],
    device: str,
    max_length: int,
    true_token_id: int,
    false_token_id: int,
    generate: bool = False,
):
    """
    Score identico alla logica di pyterrier-quality QualT5:

        quality = log_softmax([logit_true, logit_false])[0]

    cioè:

        quality = log P(true | true,false)

    Nota:
    - questo score è <= 0;
    - più è alto, migliore è il passaggio;
    - exp(quality) restituisce P(true | true,false).
    """
    prompts = [f"Document: {p} Relevant:" for p in passages]

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    batch_size = encoded["input_ids"].shape[0]

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    decoder_input_ids = torch.full(
        (batch_size, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )

    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        decoder_input_ids=decoder_input_ids,
    )

    first_logits = outputs.logits[:, 0, :].float()

    true_logits = first_logits[:, true_token_id]
    false_logits = first_logits[:, false_token_id]

    true_false_logits = torch.stack([true_logits, false_logits], dim=1)

    quality = torch.log_softmax(true_false_logits, dim=1)[:, 0]
    true_prob = torch.exp(quality)

    generated = [""] * len(passages)

    if generate:
        generated_ids = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=4,
        )
        generated = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

    rows = []
    for i in range(len(passages)):
        rows.append(
            {
                "quality": float(quality[i].detach().cpu()),
                "true_prob": float(true_prob[i].detach().cpu()),
                "true_logit": float(true_logits[i].detach().cpu()),
                "false_logit": float(false_logits[i].detach().cpu()),
                "logit_diff": float((true_logits[i] - false_logits[i]).detach().cpu()),
                "generated": generated[i],
            }
        )

    return rows


def evaluate_model(
    model_path_or_id: str,
    model_name: str,
    rows: List[Dict],
    pairs: List[Tuple[str, str]],
    batch_size: int,
    max_length: int,
    device: str,
    bf16: bool,
    generate: bool,
):
    print(f"\n[INFO] Evaluating model {model_name}: {model_path_or_id}")

    dtype = torch.bfloat16 if bf16 and device == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path_or_id,
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    true_token_ids = tokenizer.encode("true", add_special_tokens=False)
    false_token_ids = tokenizer.encode("false", add_special_tokens=False)

    if not true_token_ids or not false_token_ids:
        raise RuntimeError("Tokenizzazione true/false non valida.")

    true_token_id = true_token_ids[0]
    false_token_id = false_token_ids[0]

    pos_quality = []
    neg_quality = []

    correct = 0
    valid_pairs = 0
    invalid_pairs = 0

    for start in tqdm(range(0, len(pairs), batch_size), desc=f"Scoring {model_name}"):
        batch_pairs = pairs[start:start + batch_size]

        pos_passages = [x[0] for x in batch_pairs]
        neg_passages = [x[1] for x in batch_pairs]

        pos_rows = score_passages_official_quality(
            model=model,
            tokenizer=tokenizer,
            passages=pos_passages,
            device=device,
            max_length=max_length,
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            generate=generate,
        )

        neg_rows = score_passages_official_quality(
            model=model,
            tokenizer=tokenizer,
            passages=neg_passages,
            device=device,
            max_length=max_length,
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            generate=generate,
        )

        for offset, (pos_row, neg_row) in enumerate(zip(pos_rows, neg_rows)):
            row_idx = start + offset

            pq = pos_row["quality"]
            nq = neg_row["quality"]
            margin = pq - nq

            valid = math.isfinite(pq) and math.isfinite(nq)

            rows[row_idx][f"{model_name}_pos_quality"] = pq
            rows[row_idx][f"{model_name}_neg_quality"] = nq
            rows[row_idx][f"{model_name}_margin_quality"] = margin if valid else float("nan")

            rows[row_idx][f"{model_name}_pos_true_prob"] = pos_row["true_prob"]
            rows[row_idx][f"{model_name}_neg_true_prob"] = neg_row["true_prob"]

            rows[row_idx][f"{model_name}_pos_true_logit"] = pos_row["true_logit"]
            rows[row_idx][f"{model_name}_pos_false_logit"] = pos_row["false_logit"]
            rows[row_idx][f"{model_name}_neg_true_logit"] = neg_row["true_logit"]
            rows[row_idx][f"{model_name}_neg_false_logit"] = neg_row["false_logit"]

            rows[row_idx][f"{model_name}_pos_logit_diff"] = pos_row["logit_diff"]
            rows[row_idx][f"{model_name}_neg_logit_diff"] = neg_row["logit_diff"]

            rows[row_idx][f"{model_name}_pos_generated"] = pos_row["generated"]
            rows[row_idx][f"{model_name}_neg_generated"] = neg_row["generated"]

            rows[row_idx][f"{model_name}_valid_score"] = valid

            if not valid:
                invalid_pairs += 1
                pos_quality.append(float("nan"))
                neg_quality.append(float("nan"))
                continue

            if pq > nq:
                correct += 1

            valid_pairs += 1
            pos_quality.append(pq)
            neg_quality.append(nq)

    pos_quality = np.asarray(pos_quality, dtype=float)
    neg_quality = np.asarray(neg_quality, dtype=float)
    margin_quality = pos_quality - neg_quality

    pairwise_accuracy = correct / valid_pairs if valid_pairs > 0 else float("nan")

    summary = {
        "model": model_name,
        "model_path": model_path_or_id,
        "type": "official_hf" if model_path_or_id.startswith("pyterrier-quality/") else "hf_checkpoint",
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "pairwise_accuracy_vs_labels": pairwise_accuracy,
        "mean_pos_quality": float(np.nanmean(pos_quality)),
        "mean_neg_quality": float(np.nanmean(neg_quality)),
        "mean_margin_quality": float(np.nanmean(margin_quality)),
    }

    print(f"[RESULT] {model_name}")
    print(f"valid_pairs: {valid_pairs}")
    print(f"invalid_pairs: {invalid_pairs}")
    print(f"pairwise_accuracy_vs_labels: {pairwise_accuracy:.4f}")
    print(f"mean_pos_quality: {summary['mean_pos_quality']:.4f}")
    print(f"mean_neg_quality: {summary['mean_neg_quality']:.4f}")
    print(f"mean_margin_quality: {summary['mean_margin_quality']:.4f}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return summary, pos_quality, neg_quality


# ============================================================
# Fidelity comparison
# ============================================================

def compare_checkpoint_to_reference(
    ckpt_name: str,
    ckpt_pos: np.ndarray,
    ckpt_neg: np.ndarray,
    ref_name: str,
    ref_pos: np.ndarray,
    ref_neg: np.ndarray,
):
    ckpt_scores = np.concatenate([ckpt_pos, ckpt_neg])
    ref_scores = np.concatenate([ref_pos, ref_neg])

    ckpt_margins = ckpt_pos - ckpt_neg
    ref_margins = ref_pos - ref_neg

    score_metrics = regression_metrics(ckpt_scores, ref_scores)
    margin_metrics = regression_metrics(ckpt_margins, ref_margins)

    mask = finite_mask(ckpt_margins, ref_margins)

    if mask.sum() > 0:
        ckpt_decisions = ckpt_margins[mask] > 0
        ref_decisions = ref_margins[mask] > 0
        pairwise_agreement = float(np.mean(ckpt_decisions == ref_decisions))
    else:
        pairwise_agreement = float("nan")

    return {
        "checkpoint": ckpt_name,
        "reference_model": ref_name,

        "score_n_valid": score_metrics["n_valid"],
        "score_mae_vs_reference": score_metrics["mae"],
        "score_rmse_vs_reference": score_metrics["rmse"],
        "score_max_abs_error_vs_reference": score_metrics["max_abs_error"],
        "score_pearson_vs_reference": score_metrics["pearson"],
        "score_spearman_vs_reference": score_metrics["spearman"],

        "margin_n_valid": margin_metrics["n_valid"],
        "margin_mae_vs_reference": margin_metrics["mae"],
        "margin_rmse_vs_reference": margin_metrics["rmse"],
        "margin_pearson_vs_reference": margin_metrics["pearson"],
        "margin_spearman_vs_reference": margin_metrics["spearman"],

        "pairwise_agreement_with_reference": pairwise_agreement,
    }


def add_difference_columns(
    rows: List[Dict],
    checkpoint_names: List[str],
    reference_name: str,
):
    for row in rows:
        ref_pos = row.get(f"{reference_name}_pos_quality")
        ref_neg = row.get(f"{reference_name}_neg_quality")
        ref_margin = row.get(f"{reference_name}_margin_quality")

        for ckpt_name in checkpoint_names:
            ckpt_pos = row.get(f"{ckpt_name}_pos_quality")
            ckpt_neg = row.get(f"{ckpt_name}_neg_quality")
            ckpt_margin = row.get(f"{ckpt_name}_margin_quality")

            row[f"{ckpt_name}_pos_quality_minus_{reference_name}"] = (
                ckpt_pos - ref_pos
                if ckpt_pos is not None and ref_pos is not None
                else float("nan")
            )

            row[f"{ckpt_name}_neg_quality_minus_{reference_name}"] = (
                ckpt_neg - ref_neg
                if ckpt_neg is not None and ref_neg is not None
                else float("nan")
            )

            row[f"{ckpt_name}_margin_minus_{reference_name}"] = (
                ckpt_margin - ref_margin
                if ckpt_margin is not None and ref_margin is not None
                else float("nan")
            )


# ============================================================
# CSV saving
# ============================================================

def save_csv(rows: List[Dict], output_csv: str):
    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] CSV saved to: {output_csv}")


def save_summary_csv(summary_rows: List[Dict], output_csv: str):
    path = output_csv.replace(".csv", "_model_summary.csv")
    save_csv(summary_rows, path)


def save_fidelity_csv(fidelity_rows: List[Dict], output_csv: str):
    path = output_csv.replace(".csv", "_fidelity_summary.csv")
    save_csv(fidelity_rows, path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Confronta checkpoint fine-tuned QualT5 con pyterrier-quality/qt5-base "
            "usando lo score ufficiale: log_softmax([logit_true, logit_false])[0]."
        )
    )

    parser.add_argument(
        "--checkpoints",
        nargs=3,
        required=True,
        help="Tre checkpoint path da confrontare.",
    )

    parser.add_argument(
        "--official_model_id",
        default="pyterrier-quality/qt5-base",
        help="Modello ufficiale di riferimento.",
    )

    parser.add_argument(
        "--irds_dataset_id",
        default="msmarco-passage/train/triples-small",
    )

    parser.add_argument("--num_triples", type=int, default=1000)
    parser.add_argument("--skip_triples", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--output_csv", default="qualt5_fidelity_to_official_wide.csv")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--generate", action="store_true")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    pairs = load_eval_pairs_from_ir_datasets(
        irds_dataset_id=args.irds_dataset_id,
        num_triples=args.num_triples,
        skip_triples=args.skip_triples,
    )

    if not pairs:
        raise RuntimeError("No validation pairs collected.")

    rows: List[Dict] = []

    for i, (pos_text, neg_text) in enumerate(pairs):
        rows.append(
            {
                "example_id": i,
                "pos_passage": pos_text,
                "neg_passage": neg_text,
            }
        )

    reference_name = clean_model_name(args.official_model_id)
    checkpoint_names = [clean_model_name(path) for path in args.checkpoints]

    model_summaries = []
    model_outputs = {}

    # 1. Valuta riferimento ufficiale
    ref_summary, ref_pos, ref_neg = evaluate_model(
        model_path_or_id=args.official_model_id,
        model_name=reference_name,
        rows=rows,
        pairs=pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        bf16=args.bf16,
        generate=args.generate,
    )

    model_summaries.append(ref_summary)
    model_outputs[reference_name] = {
        "pos": ref_pos,
        "neg": ref_neg,
    }

    # 2. Valuta checkpoint
    for ckpt_path, ckpt_name in zip(args.checkpoints, checkpoint_names):
        summary, pos_scores, neg_scores = evaluate_model(
            model_path_or_id=ckpt_path,
            model_name=ckpt_name,
            rows=rows,
            pairs=pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            bf16=args.bf16,
            generate=args.generate,
        )

        model_summaries.append(summary)
        model_outputs[ckpt_name] = {
            "pos": pos_scores,
            "neg": neg_scores,
        }

    # 3. Colonne differenza nel CSV wide
    add_difference_columns(
        rows=rows,
        checkpoint_names=checkpoint_names,
        reference_name=reference_name,
    )

    # 4. Summary fidelity checkpoint vs official
    fidelity_rows = []

    for ckpt_name in checkpoint_names:
        fid = compare_checkpoint_to_reference(
            ckpt_name=ckpt_name,
            ckpt_pos=model_outputs[ckpt_name]["pos"],
            ckpt_neg=model_outputs[ckpt_name]["neg"],
            ref_name=reference_name,
            ref_pos=model_outputs[reference_name]["pos"],
            ref_neg=model_outputs[reference_name]["neg"],
        )
        fidelity_rows.append(fid)

    # 5. Salva CSV
    save_csv(rows, args.output_csv)
    save_summary_csv(model_summaries, args.output_csv)
    save_fidelity_csv(fidelity_rows, args.output_csv)

    # 6. Stampa risultati
    print("\n========== MODEL SUMMARY ==========")
    for row in model_summaries:
        print(
            f"{row['model']} ({row['type']}): "
            f"pairwise_acc_vs_labels={row['pairwise_accuracy_vs_labels']:.4f}, "
            f"mean_pos_quality={row['mean_pos_quality']:.4f}, "
            f"mean_neg_quality={row['mean_neg_quality']:.4f}, "
            f"mean_margin_quality={row['mean_margin_quality']:.4f}"
        )

    print("\n========== FIDELITY TO OFFICIAL ==========")
    for row in fidelity_rows:
        print(
            f"{row['checkpoint']} vs {row['reference_model']}: "
            f"score_MAE={row['score_mae_vs_reference']:.6f}, "
            f"score_RMSE={row['score_rmse_vs_reference']:.6f}, "
            f"score_Pearson={row['score_pearson_vs_reference']:.4f}, "
            f"score_Spearman={row['score_spearman_vs_reference']:.4f}, "
            f"margin_MAE={row['margin_mae_vs_reference']:.6f}, "
            f"pairwise_agreement={row['pairwise_agreement_with_reference']:.4f}"
        )

    valid_fidelity = [
        r for r in fidelity_rows
        if math.isfinite(r["score_mae_vs_reference"])
    ]

    if valid_fidelity:
        best_by_mae = min(
            valid_fidelity,
            key=lambda x: x["score_mae_vs_reference"],
        )

        best_by_spearman = max(
            valid_fidelity,
            key=lambda x: (
                -1 if not math.isfinite(x["score_spearman_vs_reference"])
                else x["score_spearman_vs_reference"]
            ),
        )

        print("\n========== BEST CHECKPOINT BY FIDELITY ==========")
        print(
            f"Best by lowest score MAE: {best_by_mae['checkpoint']} "
            f"(MAE={best_by_mae['score_mae_vs_reference']:.6f})"
        )
        print(
            f"Best by highest score Spearman: {best_by_spearman['checkpoint']} "
            f"(Spearman={best_by_spearman['score_spearman_vs_reference']:.4f})"
        )
        print("=================================================")
    else:
        print("\n[WARNING] Nessun checkpoint valido per il confronto.")


if __name__ == "__main__":
    main()