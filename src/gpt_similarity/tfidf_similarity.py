from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .clean_data import clean_exploded_user_data, clean_stage_data, normalize_text
from .config import DATA_PROCESSED, TABLES
from .io import save_csv


def _combined_text(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({
            **dict(zip(group_cols, keys)),
            "CleanedText": normalize_text(" ".join(sub["Sentences"].astype(str))),
        })
    return pd.DataFrame(rows)


def user_tfidf_similarity(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = clean_exploded_user_data() if df is None else df
    doc_df = _combined_text(df, ["Index", "Users"])
    rows = []

    for index, group in doc_df.groupby("Index"):
        group = group.sort_values("Users")
        if group["Users"].nunique() < 2:
            continue
        matrix = TfidfVectorizer().fit_transform(group["CleanedText"])
        sim = cosine_similarity(matrix)
        users = group["Users"].tolist()
        for i, j in combinations(range(len(users)), 2):
            rows.append({
                "new_index": int(index),
                "user_pair": f"{users[i]}-{users[j]}",
                "tfidf_cosine_similarity": float(sim[i, j]),
            })
    return pd.DataFrame(rows)


def stage_tfidf_similarity(stage_dfs: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    stage_dfs = clean_stage_data() if stage_dfs is None else stage_dfs
    rows = []

    for user_id, df in stage_dfs.items():
        doc_df = _combined_text(df, ["Category", "Stage_minor"])
        for category, group in doc_df.groupby("Category"):
            group = group.sort_values("Stage_minor")
            if group["Stage_minor"].nunique() < 2:
                continue
            matrix = TfidfVectorizer().fit_transform(group["CleanedText"])
            sim = cosine_similarity(matrix)
            stages = group["Stage_minor"].tolist()
            for i, j in combinations(range(len(stages)), 2):
                rows.append({
                    "User": user_id,
                    "Index": int(category),
                    "StagePair": f"{stages[i]}-{stages[j]}",
                    "tfidf_cosine_similarity": float(sim[i, j]),
                })
    return pd.DataFrame(rows)


def write_tfidf_outputs() -> None:
    save_csv(user_tfidf_similarity(), TABLES / "tfidf_similarity_users.csv")
    save_csv(stage_tfidf_similarity(), TABLES / "tfidf_similarity_stages.csv")


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    write_tfidf_outputs()
