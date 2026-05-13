import argparse
import gzip
import json
import logging
import os
from typing import Dict, Iterator, Optional

import torch
import yaml
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from data.loaders.msmarco.dataset_loader import DatasetLoader


logger = logging.getLogger("qualt5_ft")


class CheckpointLoggingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        logger.info("Checkpoint salvato allo step %s in %s", state.global_step, args.output_dir)
        return control


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _iter_collection_tsv(path: str) -> Iterator[tuple[str, str, str]]:
    with _open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                yield ("", "", f"collection_malformed:{line_no}")
                continue
            yield (parts[0], parts[1], "")


def _load_collection_map_from_tsv(collection_path: str, stats: Dict[str, int]) -> Dict[str, str]:
    logger.info("Carico collection in RAM da file: %s", collection_path)
    logger.info("Nota: operazione memory-intensive (pid -> passage).")

    collection: Dict[str, str] = {}
    for pid, passage, err in _iter_collection_tsv(collection_path):
        if err:
            stats["collection_malformed"] += 1
            if stats["collection_malformed"] <= 5:
                logger.warning("Collection riga malformata, skip (%s)", err)
            continue
        collection[pid] = passage

    logger.info("Passaggi caricati in RAM dalla collection: %d", len(collection))
    return collection


def _load_collection_map_from_dataset_loader(dataset_name: str) -> Dict[str, str]:
    logger.info("Carico collection in RAM via DatasetLoader (%s)...", dataset_name)
    logger.info("Nota: operazione memory-intensive (pid -> passage).")

    loader = DatasetLoader(dataset_name)
    collection: Dict[str, str] = {}
    for row in loader.get_corpus_iter():
        pid = row.get("pid", row.get("docno"))
        text = row.get("text")
        if pid is None or text is None:
            continue
        pid_str = str(pid).strip()
        if not pid_str:
            continue
        collection[pid_str] = str(text)

    logger.info("Passaggi caricati in RAM da DatasetLoader: %d", len(collection))
    return collection


def _iter_triples_text(triples_path: str, stats: Dict[str, int]) -> Iterator[tuple[str, str]]:
    with _open_text(triples_path) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                stats["triples_malformed"] += 1
                if stats["triples_malformed"] <= 5:
                    logger.warning("Triple text malformata (linea %d), skip.", line_no)
                continue

            # query ignorata volutamente: apprendimento query-independent.
            _, pos_passage, neg_passage = parts[0], parts[1], parts[2]
            stats["triples_valid"] += 1
            yield pos_passage, "true"
            yield neg_passage, "false"


def _iter_triples_id(
    triples_path: str,
    collection: Dict[str, str],
    stats: Dict[str, int],
) -> Iterator[tuple[str, str]]:
    with _open_text(triples_path) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                stats["triples_malformed"] += 1
                if stats["triples_malformed"] <= 5:
                    logger.warning("Triple id malformata (linea %d), skip.", line_no)
                continue

            # qid ignorato volutamente: apprendimento query-independent.
            _, pos_pid, neg_pid = parts[0], parts[1], parts[2]
            pos_passage = collection.get(pos_pid)
            neg_passage = collection.get(neg_pid)

            if pos_passage is None or neg_passage is None:
                stats["triples_skipped"] += 1
                stats["missing_pid"] += 1
                if stats["missing_pid"] <= 5:
                    logger.warning(
                        "PID mancante in collection (linea %d: pos=%s, neg=%s), skip.",
                        line_no,
                        pos_pid,
                        neg_pid,
                    )
                continue

            stats["triples_valid"] += 1
            yield pos_passage, "true"
            yield neg_passage, "false"


def _iter_triples_irds(
    irds_dataset_id: str,
    max_irds_triples: Optional[int],
    stats: Dict[str, int],
) -> Iterator[tuple[str, str]]:
    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError(
            "ir_datasets non installato. Installa con: pip install ir_datasets"
        ) from exc

    logger.info("Caricamento ir_datasets: %s", irds_dataset_id)
    dataset = ir_datasets.load(irds_dataset_id)

    if "triples-v2" in irds_dataset_id:
        logger.warning(
            "Stai usando '%s': triples-v2 può essere ordinato per ID; "
            "per training robusto considera shuffling/randomizzazione esterna.",
            irds_dataset_id,
        )

    docs_store = dataset.docs_store()
    if docs_store is None:
        raise RuntimeError(f"docs_store non disponibile per dataset {irds_dataset_id}")

    for idx, docpair in enumerate(dataset.docpairs_iter(), 1):
        if max_irds_triples is not None and idx > max_irds_triples:
            break

        # query_id ignorato volutamente (query-independent).
        pos_id = getattr(docpair, "doc_id_a", None)
        neg_id = getattr(docpair, "doc_id_b", None)
        if pos_id is None or neg_id is None:
            stats["triples_malformed"] += 1
            stats["triples_skipped"] += 1
            if stats["triples_malformed"] <= 5:
                logger.warning("Docpair malformato in ir_datasets (idx=%d), skip.", idx)
            continue

        pos_doc = docs_store.get(pos_id)
        neg_doc = docs_store.get(neg_id)
        if pos_doc is None or neg_doc is None:
            stats["triples_skipped"] += 1
            stats["missing_pid"] += 1
            if stats["missing_pid"] <= 5:
                logger.warning(
                    "Doc id mancante in docs_store (idx=%d, pos=%s, neg=%s), skip.",
                    idx,
                    pos_id,
                    neg_id,
                )
            continue

        pos_text = getattr(pos_doc, "text", None)
        neg_text = getattr(neg_doc, "text", None)
        if not pos_text or not neg_text:
            stats["triples_skipped"] += 1
            stats["triples_malformed"] += 1
            if stats["triples_malformed"] <= 5:
                logger.warning("Doc senza text in docs_store (idx=%d), skip.", idx)
            continue

        stats["triples_valid"] += 1
        yield str(pos_text), "true"
        yield str(neg_text), "false"


class QualT5TriplesIterableDataset(IterableDataset):
    """
    Dataset streaming per QT5 supervisionato su MSMARCO triples.

    Ogni tripla (q, p+, p-) genera due esempi:
    - Document: p+ Relevant: -> "true"
    - Document: p- Relevant: -> "false"

    La query è ignorata intenzionalmente per apprendere uno scorer query-independent.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        triples_source: str,
        triples_path: Optional[str] = None,
        triples_format: Optional[str] = None,
        collection_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        irds_dataset_id: str = "msmarco-passage/train/triples-small",
        max_irds_triples: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.triples_source = triples_source
        self.triples_path = triples_path
        self.triples_format = triples_format
        self.collection_path = collection_path
        self.dataset_name = dataset_name

        self.irds_dataset_id = irds_dataset_id
        self.max_irds_triples = max_irds_triples

        self.stats: Dict[str, int] = {
            "triples_valid": 0,
            "triples_malformed": 0,
            "triples_skipped": 0,
            "missing_pid": 0,
            "collection_malformed": 0,
            "examples_emitted": 0,
        }

        self.collection: Optional[Dict[str, str]] = None
        if self.triples_source == "file" and self.triples_format == "id":
            if self.collection_path:
                self.collection = _load_collection_map_from_tsv(self.collection_path, self.stats)
            elif self.dataset_name:
                self.collection = _load_collection_map_from_dataset_loader(self.dataset_name)
            else:
                raise ValueError(
                    "Con triples_source='file' e triples_format='id' devi fornire "
                    "--collection_path oppure --dataset_name"
                )

    def _iter_examples(self) -> Iterator[tuple[str, str]]:
        if self.triples_source == "file":
            if self.triples_format == "text":
                yield from _iter_triples_text(self.triples_path, self.stats)
            elif self.triples_format == "id":
                if self.collection is None:
                    raise RuntimeError("Collection non caricata per triples_format='id'.")
                yield from _iter_triples_id(self.triples_path, self.collection, self.stats)
            else:
                raise ValueError(f"triples_format non supportato: {self.triples_format}")
        elif self.triples_source == "irds":
            yield from _iter_triples_irds(
                irds_dataset_id=self.irds_dataset_id,
                max_irds_triples=self.max_irds_triples,
                stats=self.stats,
            )
        else:
            raise ValueError(f"triples_source non supportato: {self.triples_source}")

    def __iter__(self):
        for passage, label in self._iter_examples():
            prompt = f"Document: {passage} Relevant:"
            model_inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
            )
            label_ids = self.tokenizer(
                text_target=label,
                truncation=True,
                max_length=4,
            )
            model_inputs["labels"] = label_ids["input_ids"]
            self.stats["examples_emitted"] += 1
            yield model_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning T5 query-independent per passage quality estimation (stile QualT5)."
    )
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")

    # Modalita file (retrocompatibile)
    parser.add_argument("--triples_source", type=str, choices=["file", "irds"], default="file")
    parser.add_argument("--triples_path", type=str, default=None)
    parser.add_argument("--triples_format", type=str, choices=["text", "id"], default="text")
    parser.add_argument("--collection_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)

    # Modalita ir_datasets
    parser.add_argument("--irds_dataset_id", type=str, default="msmarco-passage/train/triples-small")
    parser.add_argument("--max_irds_triples", type=int, default=None)
    parser.add_argument("--irds_cache_dir", type=str, default=None)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")

    return parser.parse_args()


def _save_reproducibility_files(args: argparse.Namespace, output_dir: str) -> None:
    args_dict = vars(args)

    json_path = os.path.join(output_dir, "training_args.json")
    yaml_path = os.path.join(output_dir, "training_args.yaml")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(args_dict, f, sort_keys=True, allow_unicode=True)

    logger.info("Argomenti training salvati in: %s e %s", json_path, yaml_path)


def main() -> None:
    _setup_logging()
    args = parse_args()

    if args.triples_source == "file":
        if not args.triples_path:
            raise ValueError("Con --triples_source file devi specificare --triples_path")
        if args.triples_format == "id" and not (args.collection_path or args.dataset_name):
            raise ValueError(
                "Con triples_source=file e triples_format=id devi specificare "
                "--collection_path oppure --dataset_name"
            )
    else:
        if args.irds_cache_dir:
            os.environ["IR_DATASETS_HOME"] = args.irds_cache_dir
            logger.info("IR_DATASETS_HOME impostata a: %s", args.irds_cache_dir)
        logger.info("Modalita ir_datasets attiva. Dataset: %s", args.irds_dataset_id)

    if args.bf16 and args.fp16:
        raise ValueError("Scegli solo una tra --bf16 e --fp16")

    if args.bf16 and not torch.cuda.is_available():
        logger.warning("bf16 richiesto ma CUDA non disponibile. bf16 verrà disattivato.")
        args.bf16 = False

    if args.fp16 and not torch.cuda.is_available():
        logger.warning("fp16 richiesto ma CUDA non disponibile. fp16 verrà disattivato.")
        args.fp16 = False

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    logger.info(
        "Config training: model=%s max_steps=%d lr=%g effective_batch_size=%d",
        args.model_name_or_path,
        args.max_steps,
        args.learning_rate,
        effective_batch_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Modello base da caricare: %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    train_dataset = QualT5TriplesIterableDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        triples_source=args.triples_source,
        triples_path=args.triples_path,
        triples_format=args.triples_format,
        collection_path=args.collection_path,
        dataset_name=args.dataset_name,
        irds_dataset_id=args.irds_dataset_id,
        max_irds_triples=args.max_irds_triples,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
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
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CheckpointLoggingCallback()],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"[DEBUG] Trainable parameters: {trainable:,} / {total:,}")
    
    
    logger.info("Inizio training...")
    
    tracked_name = None
    tracked_before = None

    for name, param in model.named_parameters():
        if param.requires_grad:
            tracked_name = name
            tracked_before = param.detach().flatten()[:1000].float().cpu().clone()
        break

    print(f"[DEBUG] Tracking parameter: {tracked_name}")
    
    if args.resume_from_checkpoint:
        logger.info("Resume da checkpoint: %s", args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    for name, param in model.named_parameters():
        if name == tracked_name:
            tracked_after = param.detach().flatten()[:1000].float().cpu().clone()
            diff = (tracked_after - tracked_before).abs().mean().item()
            print(f"[DEBUG] Mean abs parameter change for {tracked_name}: {diff:.12e}")
            break
    logger.info("Training completato. Salvataggio modello finale in: %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _save_reproducibility_files(args, args.output_dir)

    logger.info(
        "Statistiche dataset: triples_valid=%d triples_malformed=%d triples_skipped=%d "
        "missing_pid=%d collection_malformed=%d examples_emitted=%d",
        train_dataset.stats["triples_valid"],
        train_dataset.stats["triples_malformed"],
        train_dataset.stats["triples_skipped"],
        train_dataset.stats["missing_pid"],
        train_dataset.stats["collection_malformed"],
        train_dataset.stats["examples_emitted"],
    )

    if args.push_to_hub:
        logger.info("Push su Hugging Face Hub in corso...")
        trainer.push_to_hub()
        logger.info("Push su Hugging Face Hub completato.")

    logger.info("Fine.")


if __name__ == "__main__":
    main()
