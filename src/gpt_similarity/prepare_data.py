from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .config import DATA_RAW
from .io import read_table, save_csv


def sentence_tokenize(text: object) -> list[str]:
    value = "" if pd.isna(text) else str(text)
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(value)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", value)

    split_lines: list[str] = []
    for sentence in sentences:
        split_lines.extend(str(sentence).splitlines())
    return [line.strip() for line in split_lines]


def is_valid_sentence(sentence: object, min_words: int = 4) -> bool:
    text = "" if pd.isna(sentence) else str(sentence).strip()
    if not text:
        return False
    if re.fullmatch(r"[\d\W_]*", text):
        return False
    if text.endswith(":"):
        return False
    return len(text.split()) >= min_words


def explode_answers(df: pd.DataFrame, answer_col: str = "Answers") -> pd.DataFrame:
    if answer_col not in df.columns:
        raise ValueError(f"Expected answer column '{answer_col}' in input data.")
    out = df.copy()
    out["Sentences"] = out[answer_col].apply(sentence_tokenize)
    out = out.explode("Sentences").reset_index(drop=True)
    out["Sentences"] = out["Sentences"].astype(str).str.replace(r"\t", "", regex=True).str.strip()
    out = out[out["Sentences"].apply(is_valid_sentence)].copy()
    out = out.drop(columns=[answer_col])
    return out.reset_index(drop=True)


def create_exploded_files(
    source: Path = DATA_RAW / "GPT_answer_filtered.xlsx",
    output_dir: Path = DATA_RAW,
) -> None:
    df = read_table(source)
    exploded = explode_answers(df)
    save_csv(exploded, output_dir / "exploded_data_filtered2.csv")

    stage_base = exploded[
        (exploded["Models"] == "ChatGpt 4o")
        & (exploded["Stage_minor"].isin([1, 2, 3, 4, 5]))
        & (exploded["Category"].isin([1, 4]))
    ].copy()

    for user_id in [1, 2, 3]:
        user_stage = stage_base[stage_base["Users"] == user_id].copy()
        save_csv(user_stage, output_dir / f"exploded_data_stages{user_id}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exploded sentence-level files from manually collected GPT answers.")
    parser.add_argument("--source", type=Path, default=DATA_RAW / "GPT_answer_filtered.xlsx")
    parser.add_argument("--output-dir", type=Path, default=DATA_RAW)
    args = parser.parse_args()
    create_exploded_files(args.source, args.output_dir)


if __name__ == "__main__":
    main()
