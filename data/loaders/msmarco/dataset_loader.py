# metaqual/data/loaders/msmarco/dataset_loader.py
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

import pyterrier as pt

LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_SPLITS: tuple[str, ...] = (
    "train",
    "dev",
    "dev.small",
    "test-2019",
    "test-2020",
)


class DatasetLoader:
    """Gestisce il caricamento dinamico di un dataset PyTerrier."""

    def __init__(self, dataset_name: str = "msmarco_passage"):
        if not pt.started():
            pt.init()

        print(f"Inizializzazione loader per il dataset '{dataset_name}'...")
        self.dataset_name = dataset_name
        self.dataset = pt.get_dataset(dataset_name)

    def get_corpus_iter(self):
        return self.dataset.get_corpus_iter()

    def get_topics(self, variant: str = "test"):
        return self.dataset.get_topics(variant)

    def get_qrels(self, variant: str = "test"):
        return self.dataset.get_qrels(variant)

    def get_prebuilt_index(self, index_type: str):
        print(f"Recupero indice pre-costruito ({index_type}) per {self.dataset_name}...")
        return self.dataset.get_index(index_type)

    def build_targeted_collection(
        self,
        output_path: Optional[Path] = None,
        force: bool = False,
        splits: Sequence[str] = DEFAULT_TARGET_SPLITS,
        include_header: bool = True,
        progress_every: int = 500_000,
    ) -> Dict[str, Any]:
        """Build a TSV corpus with a binary `target` based on qrels PID membership."""
        destination = self._resolve_output_path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not force:
            LOGGER.info("Targeted collection already exists at %s; skipping build.", destination)
            stats = self.get_target_stats(destination, has_header=include_header)
            return {
                "built": False,
                "output_path": str(destination),
                "splits": list(splits),
                "qrels_union_size": None,
                "stats": stats,
            }

        pid_union, split_stats = self._collect_qrels_pid_union(splits)
        LOGGER.info("Building targeted collection at %s", destination)
        LOGGER.info("Qrels union size: %d", len(pid_union))

        total = 0
        positives = 0

        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            if include_header:
                writer.writerow(["pid", "text", "target"])

            for pid, text in self._iter_corpus_records():
                target = 1 if pid in pid_union else 0
                writer.writerow([pid, text, target])
                total += 1
                positives += target

                if progress_every > 0 and total % progress_every == 0:
                    LOGGER.info("Processed %d passages...", total)

        negatives = total - positives
        stats = self._format_stats(total, positives, negatives)
        LOGGER.info(
            "Targeted collection completed. total=%d target_1=%d target_0=%d",
            total,
            positives,
            negatives,
        )

        return {
            "built": True,
            "output_path": str(destination),
            "splits": list(splits),
            "qrels_union_size": len(pid_union),
            "qrels_split_unique_counts": split_stats,
            "stats": stats,
        }

    def get_collection_with_target(
        self,
        output_path: Optional[Path] = None,
        force: bool = False,
        splits: Sequence[str] = DEFAULT_TARGET_SPLITS,
        include_header: bool = True,
    ) -> Path:
        """Return the path of the corpus with target, building it if needed."""
        result = self.build_targeted_collection(
            output_path=output_path,
            force=force,
            splits=splits,
            include_header=include_header,
        )
        return Path(result["output_path"])

    def get_target_stats(self, path: Path, has_header: bool = True) -> Dict[str, float]:
        """Return binary target statistics from a targeted collection TSV."""
        if not path.exists():
            raise FileNotFoundError(f"Targeted collection file not found: {path}")

        total = 0
        positives = 0

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            if has_header:
                header = next(reader, None)
                if not header:
                    raise ValueError("Targeted collection is empty; missing header.")
                if "target" not in header:
                    raise ValueError("Missing 'target' column in targeted collection.")

            for row in reader:
                if not row:
                    continue
                if len(row) < 3:
                    raise ValueError(f"Invalid targeted row with fewer than 3 columns: {row}")

                target = int(row[2])
                if target not in (0, 1):
                    raise ValueError(f"Invalid target value {target}; expected 0 or 1.")

                total += 1
                positives += target

        negatives = total - positives
        return self._format_stats(total, positives, negatives)

    def validate_targeted_collection(self, path: Path, has_header: bool = True) -> Dict[str, Any]:
        """Run basic sanity checks for a targeted collection TSV file."""
        if not path.exists():
            raise FileNotFoundError(f"Targeted collection file not found: {path}")

        allowed_values = {0, 1}
        observed_values = set()
        data_rows = 0
        positives = 0

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None) if has_header else None

            if has_header:
                if not header:
                    raise ValueError("Missing header in targeted collection.")
                if "target" not in header:
                    raise ValueError("Missing 'target' column in targeted collection header.")

            for row in reader:
                if not row:
                    continue
                if len(row) < 3:
                    raise ValueError(f"Invalid targeted row with fewer than 3 columns: {row}")

                target = int(row[2])
                observed_values.add(target)
                positives += target
                data_rows += 1

        if not observed_values.issubset(allowed_values):
            raise ValueError(f"Invalid target values found: {sorted(observed_values)}")

        negatives = data_rows - positives
        if positives == 0:
            raise ValueError("Sanity check failed: no positive (`target=1`) passages found.")
        if negatives == 0:
            raise ValueError("Sanity check failed: no negative (`target=0`) passages found.")

        corpus_rows = sum(1 for _ in self._iter_corpus_records())
        if data_rows != corpus_rows:
            raise ValueError(
                "Sanity check failed: targeted collection size differs from original corpus "
                f"({data_rows} != {corpus_rows})."
            )

        result = {
            "valid": True,
            "path": str(path),
            "rows": data_rows,
            "corpus_rows": corpus_rows,
            "observed_target_values": sorted(observed_values),
            "stats": self._format_stats(data_rows, positives, negatives),
        }
        LOGGER.info("Validation successful for %s", path)
        return result

    def _resolve_output_path(self, output_path: Optional[Path]) -> Path:
        if output_path is not None:
            return output_path.expanduser().resolve()
        return Path("data") / self.dataset_name / f"{self.dataset_name}_with_target.tsv"

    def _collect_qrels_pid_union(self, splits: Sequence[str]) -> tuple[set[str], Dict[str, int]]:
        if not splits:
            raise ValueError("At least one qrels split must be provided.")

        union: set[str] = set()
        split_counts: Dict[str, int] = {}

        for split in splits:
            try:
                qrels = self.get_qrels(split)
            except Exception as exc:
                raise ValueError(
                    f"Unable to load qrels split '{split}' for dataset '{self.dataset_name}'."
                ) from exc

            if qrels is None or qrels.empty:
                raise ValueError(
                    f"Qrels split '{split}' is empty or unavailable for dataset '{self.dataset_name}'."
                )

            id_column = self._find_identifier_column(
                qrels.columns, context=f"qrels split '{split}'"
            )
            split_ids = {
                str(doc_id).strip()
                for doc_id in qrels[id_column].dropna().tolist()
                if str(doc_id).strip()
            }

            if not split_ids:
                raise ValueError(f"No valid ids found in qrels split '{split}'.")

            union.update(split_ids)
            split_counts[split] = len(split_ids)
            LOGGER.info(
                "Loaded split '%s': unique_ids=%d cumulative_union=%d",
                split,
                len(split_ids),
                len(union),
            )

        return union, split_counts

    def _iter_corpus_records(self) -> Iterator[tuple[str, str]]:
        """Yield normalized (pid, text) pairs from corpus iterator."""
        for idx, row in enumerate(self.get_corpus_iter(), start=1):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Unexpected corpus row type at position {idx}: {type(row).__name__}"
                )

            pid_value = row.get("pid", row.get("docno"))
            text_value = row.get("text")

            if pid_value is None:
                raise ValueError(
                    f"Missing pid/docno field in corpus row {idx}. Available keys: {list(row.keys())}"
                )
            if text_value is None:
                raise ValueError(f"Missing text field in corpus row {idx}.")

            pid = str(pid_value).strip()
            if not pid:
                raise ValueError(f"Empty pid/docno in corpus row {idx}.")

            yield pid, str(text_value)

    @staticmethod
    def _find_identifier_column(columns: Sequence[str], context: str) -> str:
        for candidate in ("pid", "docno"):
            if candidate in columns:
                return candidate
        raise ValueError(
            f"Missing identifier column in {context}. Expected one of ['pid', 'docno'], "
            f"found: {list(columns)}"
        )

    @staticmethod
    def _format_stats(total: int, positives: int, negatives: int) -> Dict[str, float]:
        if total < 0 or positives < 0 or negatives < 0:
            raise ValueError("Statistics counts cannot be negative.")
        if positives + negatives != total:
            raise ValueError("Inconsistent statistics: positives + negatives != total")

        if total == 0:
            return {
                "total_passages": 0,
                "target_1": 0,
                "target_0": 0,
                "target_1_pct": 0.0,
                "target_0_pct": 0.0,
            }

        return {
            "total_passages": total,
            "target_1": positives,
            "target_0": negatives,
            "target_1_pct": round((positives / total) * 100.0, 6),
            "target_0_pct": round((negatives / total) * 100.0, 6),
        }
