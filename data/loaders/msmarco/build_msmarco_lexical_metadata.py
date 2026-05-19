from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders.msmarco.dataset_loader import DatasetLoader

LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "docno",
    "length_tokens",
    "avg_token_length",
    "unique_token_ratio",
    "repetition_ratio",
    "lexical_entropy",
    "stopword_ratio",
    "content_word_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MSMARCO offline lexical metadata with columns: "
            "docno, length_tokens, avg_token_length, unique_token_ratio, repetition_ratio, "
            "lexical_entropy, stopword_ratio, content_word_ratio."
        )
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="msmarco_passage",
        help="PyTerrier dataset name (default: msmarco_passage).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output parquet path. Default: data/<dataset-name>/<dataset-name>_lexical_metadata.parquet"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for the same metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if output already exists.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Log progress every N processed passages (default: 500000).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap for dry-run/testing.",
    )
    parser.add_argument(
        "--stopword-language",
        type=str,
        default="english",
        help="Language for NLTK stopwords (default: english).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity for this command.",
    )
    return parser.parse_args()


def _resolve_output_path(dataset_name: str, output_path: Optional[Path]) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    return Path("data") / dataset_name / f"{dataset_name}_lexical_metadata.parquet"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _ensure_nltk_resource(resource_path: str, *, install_hint: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError as exc:
        raise RuntimeError(
            f"NLTK resource '{resource_path}' not found.\n"
            f"Install with: {install_hint}\n"
            "Then rerun this script."
        ) from exc


def _load_stopword_set(language: str) -> set[str]:
    try:
        return set(stopwords.words(language))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords corpus not found.\n"
            "Install with: python -c \"import nltk; nltk.download('stopwords')\"\n"
            "Then rerun this script."
        ) from exc


def _extract_docno_and_text(row: Dict[str, Any], row_idx: int) -> tuple[str, str]:
    docno_value = row.get("pid", row.get("docno"))
    text_value = row.get("text")

    if docno_value is None:
        raise ValueError(
            f"Missing pid/docno in corpus row {row_idx}. Available keys: {list(row.keys())}"
        )
    if text_value is None:
        raise ValueError(f"Missing text field in corpus row {row_idx}.")

    docno = str(docno_value).strip()
    if not docno:
        raise ValueError(f"Empty pid/docno in corpus row {row_idx}.")

    return docno, str(text_value)


def _tokenize_text(text: str) -> list[str]:
    try:
        raw_tokens = word_tokenize(text.lower())
    except LookupError as exc:
        raise RuntimeError(
            "NLTK punkt tokenizer resources are missing.\n"
            "Install with:\n"
            "  python -c \"import nltk; nltk.download('punkt')\"\n"
            "If your NLTK version requires it, also install:\n"
            "  python -c \"import nltk; nltk.download('punkt_tab')\""
        ) from exc
    return [tok for tok in raw_tokens if any(ch.isalnum() for ch in tok)]


def _compute_lexical_features(tokens: list[str], stopword_set: set[str]) -> Dict[str, float]:
    total = len(tokens)
    total_f = float(total)
    if total == 0:
        return {
            "length_tokens": 0.0,
            "avg_token_length": 0.0,
            "unique_token_ratio": 0.0,
            "repetition_ratio": 0.0,
            "lexical_entropy": 0.0,
            "stopword_ratio": 0.0,
            "content_word_ratio": 0.0,
        }

    token_lengths = [len(tok) for tok in tokens]
    unique_count = float(len(set(tokens)))
    repetition_count = total_f - unique_count
    stopword_count = float(sum(1 for tok in tokens if tok in stopword_set))

    counts = Counter(tokens)
    lexical_entropy = 0.0
    for count in counts.values():
        prob = count / total_f
        lexical_entropy -= prob * math.log2(prob)

    return {
        "length_tokens": float(total),
        "avg_token_length": _safe_div(float(sum(token_lengths)), total_f),
        "unique_token_ratio": _safe_div(unique_count, total_f),
        "repetition_ratio": _safe_div(repetition_count, total_f),
        "lexical_entropy": lexical_entropy,
        "stopword_ratio": _safe_div(stopword_count, total_f),
        "content_word_ratio": _safe_div(total_f - stopword_count, total_f),
    }


def _validate_feature_row(features: Dict[str, float], docno: str) -> None:
    for key, value in features.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite value for '{key}' on docno='{docno}': {value}")


def _build_rows(
    corpus_iter: Iterable[Dict[str, Any]],
    stopword_set: set[str],
    progress_every: int,
    max_docs: Optional[int] = None,
) -> tuple[list[Dict[str, float]], Dict[str, int]]:
    rows: list[Dict[str, float]] = []
    duplicate_docno_count = 0
    seen_docnos: set[str] = set()

    for idx, row in enumerate(corpus_iter, start=1):
        docno, text = _extract_docno_and_text(row, row_idx=idx)
        if docno in seen_docnos:
            duplicate_docno_count += 1
        else:
            seen_docnos.add(docno)

        tokens = _tokenize_text(text)
        features = _compute_lexical_features(tokens, stopword_set)
        _validate_feature_row(features, docno=docno)

        rows.append(
            {
                "docno": docno,
                **features,
            }
        )

        if progress_every > 0 and idx % progress_every == 0:
            LOGGER.info("Processed %d passages...", idx)

        if max_docs is not None and idx >= max_docs:
            LOGGER.info("Reached max-docs=%d. Stopping early.", max_docs)
            break

    stats = {
        "processed_rows": len(rows),
        "unique_docno": len(seen_docnos),
        "duplicate_docno_count": duplicate_docno_count,
    }
    return rows, stats


def _validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema. Got {list(df.columns)}, expected {EXPECTED_COLUMNS}."
        )

    if df["docno"].isna().any():
        raise ValueError("Found null docno values.")

    unique_docno = int(df["docno"].nunique(dropna=True))
    total_rows = int(len(df))
    if unique_docno != total_rows:
        raise ValueError(
            f"Docno uniqueness check failed: unique_docno={unique_docno}, total_rows={total_rows}."
        )

    ratio_columns = [
        "unique_token_ratio",
        "repetition_ratio",
        "stopword_ratio",
        "content_word_ratio",
    ]
    for col in ratio_columns:
        col_values = df[col]
        if (col_values < 0.0).any() or (col_values > 1.0).any():
            raise ValueError(f"Column '{col}' contains values outside [0, 1].")

    non_negative_columns = [
        "length_tokens",
        "avg_token_length",
        "lexical_entropy",
    ]
    for col in non_negative_columns:
        if (df[col] < 0.0).any():
            raise ValueError(f"Column '{col}' contains negative values.")

    finite_df = df[EXPECTED_COLUMNS[1:]].applymap(lambda v: math.isfinite(float(v)))
    if not finite_df.to_numpy().all():
        raise ValueError("Detected non-finite values in lexical metadata output.")

    return {
        "rows": total_rows,
        "unique_docno": unique_docno,
        "columns": EXPECTED_COLUMNS,
    }


def _write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Unable to write parquet output. Ensure a parquet engine is installed "
            "(pyarrow or fastparquet)."
        ) from exc


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_path = _resolve_output_path(args.dataset_name, args.output)
    output_csv_path = args.output_csv.expanduser().resolve() if args.output_csv else None

    if output_path.exists() and not args.force:
        LOGGER.info("Output already exists at %s. Use --force to rebuild.", output_path)
        print(
            json.dumps(
                {
                    "built": False,
                    "output_path": str(output_path),
                    "reason": "already_exists",
                },
                indent=2,
            )
        )
        return

    LOGGER.info("Checking NLTK resources for tokenization and stopwords...")
    _ensure_nltk_resource(
        "tokenizers/punkt",
        install_hint='python -c "import nltk; nltk.download(\'punkt\')"',
    )
    stopword_set = _load_stopword_set(args.stopword_language)

    LOGGER.info("Loading dataset '%s' via DatasetLoader...", args.dataset_name)
    loader = DatasetLoader(args.dataset_name)

    rows, row_stats = _build_rows(
        corpus_iter=loader.get_corpus_iter(),
        stopword_set=stopword_set,
        progress_every=args.progress_every,
        max_docs=args.max_docs,
    )

    if row_stats["processed_rows"] == 0:
        raise ValueError("No passages processed. Aborting.")

    if row_stats["duplicate_docno_count"] > 0:
        raise ValueError(
            f"Found duplicated docno values: {row_stats['duplicate_docno_count']} duplicates."
        )

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    validation = _validate_dataframe(df)

    LOGGER.info("Writing parquet output to %s", output_path)
    _write_parquet(df, output_path)

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing optional CSV output to %s", output_csv_path)
        df.to_csv(output_csv_path, index=False)

    result = {
        "built": True,
        "output_path": str(output_path),
        "output_csv_path": str(output_csv_path) if output_csv_path else None,
        "stats": row_stats,
        "validation": validation,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
