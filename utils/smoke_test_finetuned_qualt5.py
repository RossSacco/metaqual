import argparse

import pandas as pd

from metaqual.models.base_scorer import get_scorer


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test per finetuned_qualt5 scorer")
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    scorer = get_scorer(
        "finetuned_qualt5",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    # Nessuna query: solo docno + text
    df = pd.DataFrame(
        [
            {"docno": "d1", "text": "This passage is clear, factual and concise."},
            {"docno": "d2", "text": "asdasd zzzz random tokens and noisy repeated repeated"},
        ]
    )

    out = scorer.transform(df)

    if "quality" not in out.columns:
        raise AssertionError("Colonna 'quality' mancante nell'output.")

    qualities = out["quality"].tolist()
    if len(qualities) != 2:
        raise AssertionError(f"Numero score inatteso: {len(qualities)}")

    for idx, score in enumerate(qualities):
        if not isinstance(score, float):
            raise AssertionError(f"Score in posizione {idx} non float: {type(score)}")
        if not (0.0 <= score <= 1.0):
            raise AssertionError(f"Score fuori range [0,1] in posizione {idx}: {score}")

    print("Smoke test OK: scorer finetuned_qualt5 restituisce float in [0,1] senza query.")


if __name__ == "__main__":
    main()
