from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STAGE_PAIR_ORDER, TABLES
from .io import save_csv


def bootstrap_nested_mean_difference(
    df: pd.DataFrame,
    group_col: str,
    group_a,
    group_b,
    cluster_col: str,
    value_col: str,
    n_boot: int = 10000,
    ci: float = 95,
    seed: int = 20260520,
) -> dict:
    cluster_means = (
        df[df[group_col].isin([group_a, group_b])]
        .dropna(subset=[cluster_col, group_col, value_col])
        .groupby([cluster_col, group_col], observed=True)[value_col]
        .mean()
        .unstack()
    )
    if group_a not in cluster_means.columns or group_b not in cluster_means.columns:
        return _empty_result()

    paired = cluster_means[[group_a, group_b]].dropna()
    if paired.empty:
        return _empty_result()

    diffs = paired[group_a].to_numpy(dtype=float) - paired[group_b].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot[b] = np.mean(rng.choice(diffs, size=len(diffs), replace=True))

    alpha = (100 - ci) / 2
    p_two_sided = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return {
        "n_clusters": int(len(paired)),
        "mean_a": float(cluster_means[group_a].dropna().mean()),
        "mean_b": float(cluster_means[group_b].dropna().mean()),
        "mean_diff_a_minus_b": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(boot, alpha)),
        "ci95_high": float(np.percentile(boot, 100 - alpha)),
        "p_boot_two_sided": float(min(p_two_sided, 1.0)),
        "n_a": int(df.loc[df[group_col] == group_a, value_col].notna().sum()),
        "n_b": int(df.loc[df[group_col] == group_b, value_col].notna().sum()),
    }


def _empty_result() -> dict:
    return {
        "n_clusters": 0,
        "mean_a": np.nan,
        "mean_b": np.nan,
        "mean_diff_a_minus_b": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "p_boot_two_sided": np.nan,
        "n_a": 0,
        "n_b": 0,
    }


def stage_bootstrap_table(df: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    rows = []
    for prompt, sub in df.dropna(subset=["Prompt", "StagePair", "User", "Similarity"]).groupby("Prompt"):
        groups = [group for group in STAGE_PAIR_ORDER if group in set(sub["StagePair"])]
        for group_a, group_b in combinations(groups, 2):
            stats = bootstrap_nested_mean_difference(
                sub, "StagePair", group_a, group_b, "User", "Similarity", n_boot=n_boot
            )
            rows.append({"prompt": prompt, "group_a": group_a, "group_b": group_b, **stats})
    return pd.DataFrame(rows).sort_values(["prompt", "group_a", "group_b"]).reset_index(drop=True)


def user_bootstrap_table(df: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    rows = []
    data = df.dropna(subset=["domain", "new_index", "user_pair", "similarity"]).copy()
    data = data[data["domain"].isin(["Diet", "Physical activity"])].copy()
    data["new_index"] = data["new_index"].astype(int)
    for domain, sub in data.groupby("domain"):
        indices = sorted(sub["new_index"].unique())
        for group_a, group_b in combinations(indices, 2):
            stats = bootstrap_nested_mean_difference(
                sub, "new_index", group_a, group_b, "user_pair", "similarity", n_boot=n_boot
            )
            rows.append({"domain": domain, "group_a": group_a, "group_b": group_b, **stats})
    return pd.DataFrame(rows).sort_values(["domain", "group_a", "group_b"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested bootstrap mean-difference analyses.")
    sub = parser.add_subparsers(dest="target", required=True)

    p_users = sub.add_parser("users")
    p_users.add_argument("--input", type=Path, default=TABLES / "sentence_similarity_users.csv")
    p_users.add_argument("--output", type=Path, default=TABLES / "bootstrap_mean_diff_users.csv")
    p_users.add_argument("--n-boot", type=int, default=10000)

    p_stages = sub.add_parser("stages")
    p_stages.add_argument("--input", type=Path, default=TABLES / "sentence_similarity_stages.csv")
    p_stages.add_argument("--output", type=Path, default=TABLES / "bootstrap_mean_diff_stages.csv")
    p_stages.add_argument("--n-boot", type=int, default=10000)

    args = parser.parse_args()
    df = pd.read_csv(args.input)
    out = user_bootstrap_table(df, args.n_boot) if args.target == "users" else stage_bootstrap_table(df, args.n_boot)
    save_csv(out, args.output)


if __name__ == "__main__":
    main()
