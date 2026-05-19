from __future__ import annotations

import gzip
import hashlib
import logging
from typing import Dict, Iterator, Optional

from torch.utils.data import IterableDataset

try:
    from metaqual.models.metadata_qualt5 import LexicalMetadataStore, MetadataFeatureScaler
except ImportError:
    from models.metadata_qualt5 import LexicalMetadataStore, MetadataFeatureScaler

LOGGER = logging.getLogger(__name__)


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _iter_collection_tsv(path: str) -> Iterator[tuple[str, str, str]]:
    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                yield ("", "", f"collection_malformed:{line_no}")
                continue
            yield (parts[0], parts[1], "")


def _load_collection_map_from_tsv(collection_path: str, stats: Dict[str, int]) -> Dict[str, str]:
    LOGGER.info("Carico collection in RAM da file: %s", collection_path)
    collection: Dict[str, str] = {}
    for pid, passage, err in _iter_collection_tsv(collection_path):
        if err:
            stats["collection_malformed"] += 1
            continue
        collection[pid] = passage
    LOGGER.info("Collection caricata in RAM. passages=%d", len(collection))
    return collection


def _iter_triples_text(triples_path: str, stats: Dict[str, int]) -> Iterator[tuple[str, str, int]]:
    with _open_text(triples_path) as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                stats["triples_malformed"] += 1
                continue

            _, pos_passage, neg_passage = parts[0], parts[1], parts[2]
            stats["triples_valid"] += 1

            # In assenza di PID, usiamo un docno stabile derivato dal testo.
            pos_docno = "txt_" + hashlib.md5(pos_passage.encode("utf-8", errors="ignore")).hexdigest()
            neg_docno = "txt_" + hashlib.md5(neg_passage.encode("utf-8", errors="ignore")).hexdigest()

            yield (pos_docno, pos_passage, 1)
            yield (neg_docno, neg_passage, 0)


def _iter_triples_id(
    triples_path: str,
    collection: Dict[str, str],
    stats: Dict[str, int],
) -> Iterator[tuple[str, str, int]]:
    with _open_text(triples_path) as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                stats["triples_malformed"] += 1
                continue

            _, pos_pid, neg_pid = parts[0], parts[1], parts[2]
            pos_passage = collection.get(pos_pid)
            neg_passage = collection.get(neg_pid)

            if pos_passage is None or neg_passage is None:
                stats["triples_skipped"] += 1
                stats["missing_pid"] += 1
                continue

            stats["triples_valid"] += 1
            yield (str(pos_pid), pos_passage, 1)
            yield (str(neg_pid), neg_passage, 0)


def _iter_triples_irds(
    irds_dataset_id: str,
    max_irds_triples: Optional[int],
    stats: Dict[str, int],
) -> Iterator[tuple[str, str, int]]:
    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError(
            "ir_datasets non installato. Installa con: pip install ir_datasets"
        ) from exc

    dataset = ir_datasets.load(irds_dataset_id)
    docs_store = dataset.docs_store()
    if docs_store is None:
        raise RuntimeError(f"docs_store non disponibile per dataset {irds_dataset_id}")

    for idx, docpair in enumerate(dataset.docpairs_iter(), 1):
        if max_irds_triples is not None and idx > max_irds_triples:
            break

        pos_id = getattr(docpair, "doc_id_a", None)
        neg_id = getattr(docpair, "doc_id_b", None)
        if pos_id is None or neg_id is None:
            stats["triples_malformed"] += 1
            stats["triples_skipped"] += 1
            continue

        pos_doc = docs_store.get(pos_id)
        neg_doc = docs_store.get(neg_id)
        if pos_doc is None or neg_doc is None:
            stats["triples_skipped"] += 1
            stats["missing_pid"] += 1
            continue

        pos_text = getattr(pos_doc, "text", None)
        neg_text = getattr(neg_doc, "text", None)
        if not pos_text or not neg_text:
            stats["triples_skipped"] += 1
            stats["triples_malformed"] += 1
            continue

        stats["triples_valid"] += 1
        yield (str(pos_id), str(pos_text), 1)
        yield (str(neg_id), str(neg_text), 0)


class MetadataQualT5TriplesIterableDataset(IterableDataset):
    """
    Streaming dataset per metadata_qualt5.
    Ogni tripla produce:
      - (docno_pos, passage_pos, label=1)
      - (docno_neg, passage_neg, label=0)
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        triples_source: str,
        lexical_store: LexicalMetadataStore,
        lexical_scaler: MetadataFeatureScaler,
        allow_missing_metadata: bool,
        triples_path: Optional[str] = None,
        triples_format: Optional[str] = None,
        collection_path: Optional[str] = None,
        irds_dataset_id: str = "msmarco-passage/train/triples-small",
        max_irds_triples: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.triples_source = triples_source
        self.triples_path = triples_path
        self.triples_format = triples_format
        self.collection_path = collection_path
        self.irds_dataset_id = irds_dataset_id
        self.max_irds_triples = max_irds_triples

        self.lexical_store = lexical_store
        self.lexical_scaler = lexical_scaler
        self.allow_missing_metadata = bool(allow_missing_metadata)

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
            if not self.collection_path:
                raise ValueError(
                    "Con triples_source='file' e triples_format='id' serve --collection_path"
                )
            self.collection = _load_collection_map_from_tsv(self.collection_path, self.stats)

    def iter_raw_examples(self) -> Iterator[tuple[str, str, int]]:
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
        for docno, passage, binary_label in self.iter_raw_examples():
            prompt = f"Document: {passage} Relevant:"
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
            )

            lexical_raw = self.lexical_store.lookup(
                [docno],
                allow_missing_metadata=self.allow_missing_metadata,
            )
            lexical_norm = self.lexical_scaler.transform(lexical_raw)[0].astype("float32")

            encoded["lexical_features"] = lexical_norm.tolist()
            encoded["binary_labels"] = int(binary_label)
            encoded["docno"] = str(docno)

            self.stats["examples_emitted"] += 1
            yield encoded
