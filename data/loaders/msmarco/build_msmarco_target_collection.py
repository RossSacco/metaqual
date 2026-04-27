from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders.msmarco.dataset_loader import DatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a MSMARCO Passage v1 targeted collection TSV with columns: pid, text, target."
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
            "Output TSV path. Default: data/<dataset-name>/<dataset-name>_with_target.tsv"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if the output file already exists.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run sanity validation checks after build.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity for this command.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    loader = DatasetLoader(args.dataset_name)
    build_result = loader.build_targeted_collection(
        output_path=args.output,
        force=args.force,
    )

    print("\n[BUILD RESULT]")
    print(json.dumps(build_result, indent=2, ensure_ascii=False))

    target_path = Path(build_result["output_path"])
    if args.validate:
        validation = loader.validate_targeted_collection(target_path)
        print("\n[VALIDATION RESULT]")
        print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
