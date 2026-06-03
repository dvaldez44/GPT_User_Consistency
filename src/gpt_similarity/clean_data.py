from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED, STAGE_SOURCES, USER_SOURCE
from .io import read_table, save_csv


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_exploded_user_data(source: Path = USER_SOURCE) -> pd.DataFrame:
    df = read_table(source).copy()
    df = df.dropna(subset=["Sentences"])
    df["Sentences"] = df["Sentences"].map(normalize_text)
    df = df[df["Sentences"] != ""]
    if "Models" in df.columns:
        df = df[df["Models"] == "ChatGpt 4o"].copy()
    return df


def clean_stage_data(sources: dict[int, Path] = STAGE_SOURCES) -> dict[int, pd.DataFrame]:
    cleaned: dict[int, pd.DataFrame] = {}
    for user_id, source in sources.items():
        df = read_table(source).copy()
        df = df.dropna(subset=["Sentences", "Stage_minor", "Category"])
        df["Sentences"] = df["Sentences"].map(normalize_text)
        df = df[df["Sentences"] != ""]
        cleaned[user_id] = df
    return cleaned


def write_clean_outputs() -> None:
    user_df = clean_exploded_user_data()
    save_csv(user_df, DATA_PROCESSED / "clean_user_sentences.csv")

    for user_id, df in clean_stage_data().items():
        save_csv(df, DATA_PROCESSED / f"clean_stage_sentences_user{user_id}.csv")


if __name__ == "__main__":
    write_clean_outputs()
