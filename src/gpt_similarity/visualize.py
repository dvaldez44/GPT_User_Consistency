from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES, STAGE_PAIR_ORDER


def stage_strip_plot(df: pd.DataFrame, output: Path = FIGURES / "stage_strip_plot.png") -> None:
    data = df.dropna(subset=["Prompt", "StagePair", "User", "Similarity"]).copy()
    data["StagePair"] = data["StagePair"].astype(str)
    data["User"] = data["User"].astype(str)
    panels = [("Diet", "Diet", "a)"), ("Physical Activity", "Physical activity", "b)")]
    user_order = sorted(data["User"].unique(), key=lambda x: int(x) if x.isdigit() else x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    handles, labels = None, None
    for ax, (prompt, title, panel_label) in zip(axes, panels):
        sub = data[data["Prompt"] == prompt].copy()
        order = [label for label in STAGE_PAIR_ORDER if label in set(sub["StagePair"])]
        sns.stripplot(
            data=sub,
            x="StagePair",
            y="Similarity",
            hue="User",
            order=order,
            hue_order=user_order,
            dodge=True,
            jitter=0.12,
            alpha=0.55,
            size=3.5,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("T2D stage pair")
        ax.set_ylabel("Sentence similarity" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.text(-0.08, 1.04, panel_label, transform=ax.transAxes, fontsize=16, fontweight="bold")
        if handles is None and ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.suptitle("Sentence similarity by T2D stage pair")
    if handles:
        fig.legend(handles, labels, title="User", bbox_to_anchor=(1.02, 0.95), loc="upper left")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def user_strip_plot(df: pd.DataFrame, output: Path = FIGURES / "user_strip_plot.png") -> None:
    data = df.dropna(subset=["domain", "new_index", "user_pair", "similarity"]).copy()
    data["new_index"] = data["new_index"].astype(int)
    data["user_pair"] = data["user_pair"].astype(str)
    panels = [("Diet", "Diet (prompts 1-6)", "a)"), ("Physical activity", "Physical activity (prompts 7-12)", "b)")]
    user_pair_order = sorted(data["user_pair"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    handles, labels = None, None
    for ax, (domain, title, panel_label) in zip(axes, panels):
        sub = data[data["domain"] == domain].copy()
        order = sorted(sub["new_index"].unique())
        sns.stripplot(
            data=sub,
            x="new_index",
            y="similarity",
            hue="user_pair",
            order=order,
            hue_order=user_pair_order,
            dodge=True,
            jitter=0.22,
            alpha=0.55,
            size=3.5,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Prompt index")
        ax.set_ylabel("Sentence similarity" if ax is axes[0] else "")
        ax.text(-0.08, 1.04, panel_label, transform=ax.transAxes, fontsize=16, fontweight="bold")
        if handles is None and ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.suptitle("Sentence similarity across users by prompt")
    if handles:
        fig.legend(handles, labels, title="User pair", bbox_to_anchor=(1.02, 0.95), loc="upper left")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reviewer strip plots.")
    sub = parser.add_subparsers(dest="target", required=True)

    p_users = sub.add_parser("users")
    p_users.add_argument("--input", type=Path, required=True)
    p_users.add_argument("--output", type=Path, default=FIGURES / "user_strip_plot.png")

    p_stages = sub.add_parser("stages")
    p_stages.add_argument("--input", type=Path, required=True)
    p_stages.add_argument("--output", type=Path, default=FIGURES / "stage_strip_plot.png")

    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if args.target == "users":
        user_strip_plot(df, args.output)
    else:
        stage_strip_plot(df, args.output)


if __name__ == "__main__":
    main()
