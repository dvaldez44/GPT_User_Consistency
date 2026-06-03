from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DATA_INTERIM, PROMPT_LABELS, STAGE_PAIR_LABELS, TABLES
from .hungarian import hungarian_alignment_score, symmetrize_forward_backward
from .io import load_pickle, save_csv


def aggregate_user_matrices(path: Path, normalize: str = "max") -> pd.DataFrame:
    results = load_pickle(path)
    rows = []

    for index, pair_dict in results.items():
        for (user_a, user_b), payload in pair_dict.items():
            sym_mat = symmetrize_forward_backward(payload["forward"], payload["backward"])
            aligned = hungarian_alignment_score(sym_mat, payload["sents_a"], payload["sents_b"], normalize=normalize)
            for match in aligned["matched_pairs"]:
                rows.append({
                    "new_index": int(index),
                    "user_pair": f"{user_a}-{user_b}",
                    "similarity": match["similarity"],
                    "bucket": match["bucket"],
                    "sentence_a": match["sentence_a"],
                    "sentence_b": match["sentence_b"],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["domain"] = df["new_index"].apply(_user_domain)
    return df


def _user_domain(index: int) -> str:
    index = int(index)
    if 1 <= index <= 6:
        return "Diet"
    if 7 <= index <= 12:
        return "Physical activity"
    return "Other"


def aggregate_stage_matrices(paths: list[Path], normalize: str = "max") -> pd.DataFrame:
    rows = []
    for path in paths:
        user_id = int(path.stem.split("user")[-1]) if "user" in path.stem else None
        results = load_pickle(path)
        for index, pair_dict in results.items():
            for (stage_a, stage_b), payload in pair_dict.items():
                sym_mat = symmetrize_forward_backward(payload["forward"], payload["backward"])
                aligned = hungarian_alignment_score(sym_mat, payload["sents_a"], payload["sents_b"], normalize=normalize)
                raw_pair = f"{stage_a}-{stage_b}"
                for match in aligned["matched_pairs"]:
                    rows.append({
                        "Index": int(index),
                        "Prompt": PROMPT_LABELS.get(int(index), str(index)),
                        "User": user_id,
                        "StagePairRaw": raw_pair,
                        "StagePair": STAGE_PAIR_LABELS.get(raw_pair, raw_pair),
                        "Similarity": match["similarity"],
                        "bucket": match["bucket"],
                        "sentence_a": match["sentence_a"],
                        "sentence_b": match["sentence_b"],
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DeBERTa matrices with Hungarian alignment.")
    sub = parser.add_subparsers(dest="target", required=True)

    p_users = sub.add_parser("users")
    p_users.add_argument("--matrices", type=Path, default=DATA_INTERIM / "deberta_user_matrices.pkl")
    p_users.add_argument("--output", type=Path, default=TABLES / "sentence_similarity_users.csv")
    p_users.add_argument("--normalize", choices=["min", "max", "avg"], default="max")

    p_stages = sub.add_parser("stages")
    p_stages.add_argument("--matrices", type=Path, nargs="+", default=[
        DATA_INTERIM / "deberta_stage_matrices_user1.pkl",
        DATA_INTERIM / "deberta_stage_matrices_user2.pkl",
        DATA_INTERIM / "deberta_stage_matrices_user3.pkl",
    ])
    p_stages.add_argument("--output", type=Path, default=TABLES / "sentence_similarity_stages.csv")
    p_stages.add_argument("--normalize", choices=["min", "max", "avg"], default="max")

    args = parser.parse_args()
    if args.target == "users":
        save_csv(aggregate_user_matrices(args.matrices, args.normalize), args.output)
    else:
        save_csv(aggregate_stage_matrices(args.matrices, args.normalize), args.output)


if __name__ == "__main__":
    main()
