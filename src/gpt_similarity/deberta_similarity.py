from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .clean_data import clean_exploded_user_data, clean_stage_data
from .config import DATA_INTERIM, MODEL_NAME
from .io import save_pickle


def entailment_label_index(model: AutoModelForSequenceClassification) -> int:
    labels = {idx: label.lower() for idx, label in model.config.id2label.items()}
    for idx, label in labels.items():
        if "entail" in label:
            return int(idx)
    return 0


def load_deberta(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device, entailment_label_index(model)


def entailment_matrix(
    source_sentences: list[str],
    target_sentences: list[str],
    tokenizer,
    model,
    device: str,
    entailment_idx: int,
    batch_size: int = 128,
) -> np.ndarray:
    n_source, n_target = len(source_sentences), len(target_sentences)
    pairs = [(a, b) for a in source_sentences for b in target_sentences]
    probs: list[np.ndarray] = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, entailment_idx]
        probs.append(batch_probs.detach().cpu().numpy())

    return np.concatenate(probs).reshape(n_source, n_target)


def _group_user_sentences(min_index: int | None = None, max_index: int | None = None) -> dict[int, dict[int, list[str]]]:
    df = clean_exploded_user_data()
    if min_index is not None:
        df = df[df["Index"] >= min_index]
    if max_index is not None:
        df = df[df["Index"] <= max_index]
    grouped = df.groupby(["Users", "Index"])["Sentences"].apply(list)
    out: dict[int, dict[int, list[str]]] = {}
    for (user, index), sentences in grouped.items():
        out.setdefault(int(user), {})[int(index)] = list(sentences)
    return out


def _group_stage_sentences(user_id: int) -> dict[int, dict[int, list[str]]]:
    df = clean_stage_data()[user_id]
    grouped = df.groupby(["Stage_minor", "Category"])["Sentences"].apply(list)
    out: dict[int, dict[int, list[str]]] = {}
    for (stage, category), sentences in grouped.items():
        out.setdefault(int(stage), {})[int(category)] = list(sentences)
    return out


def compute_user_matrices(
    output_path: Path,
    min_index: int | None = None,
    max_index: int | None = None,
    batch_size: int = 128,
) -> None:
    tokenizer, model, device, entailment_idx = load_deberta()
    user_sentences = _group_user_sentences(min_index, max_index)
    users = sorted(user_sentences)
    indices = sorted(set.intersection(*(set(user_sentences[u]) for u in users)))
    results = {}

    for index in indices:
        results[index] = {}
        for user_a, user_b in combinations(users, 2):
            sents_a = user_sentences[user_a][index]
            sents_b = user_sentences[user_b][index]
            forward = entailment_matrix(sents_a, sents_b, tokenizer, model, device, entailment_idx, batch_size)
            backward = entailment_matrix(sents_b, sents_a, tokenizer, model, device, entailment_idx, batch_size)
            results[index][(user_a, user_b)] = {
                "forward": forward,
                "backward": backward,
                "sents_a": sents_a,
                "sents_b": sents_b,
            }
            print(f"computed user index={index}, pair={user_a}-{user_b}", flush=True)

    save_pickle(results, output_path)


def compute_stage_matrices(user_id: int, output_path: Path, batch_size: int = 128) -> None:
    tokenizer, model, device, entailment_idx = load_deberta()
    stage_sentences = _group_stage_sentences(user_id)
    stages = sorted(stage_sentences)
    categories = sorted(set.intersection(*(set(stage_sentences[s]) for s in stages)))
    results = {}

    for category in categories:
        results[category] = {}
        for stage_a, stage_b in combinations(stages, 2):
            sents_a = stage_sentences[stage_a][category]
            sents_b = stage_sentences[stage_b][category]
            forward = entailment_matrix(sents_a, sents_b, tokenizer, model, device, entailment_idx, batch_size)
            backward = entailment_matrix(sents_b, sents_a, tokenizer, model, device, entailment_idx, batch_size)
            results[category][(stage_a, stage_b)] = {
                "forward": forward,
                "backward": backward,
                "sents_a": sents_a,
                "sents_b": sents_b,
            }
            print(f"computed stage user={user_id}, category={category}, pair={stage_a}-{stage_b}", flush=True)

    save_pickle(results, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DeBERTa entailment similarity matrices.")
    sub = parser.add_subparsers(dest="target", required=True)

    p_users = sub.add_parser("users")
    p_users.add_argument("--output", type=Path, default=DATA_INTERIM / "deberta_user_matrices.pkl")
    p_users.add_argument("--min-index", type=int, default=None)
    p_users.add_argument("--max-index", type=int, default=None)
    p_users.add_argument("--batch-size", type=int, default=128)

    p_stage = sub.add_parser("stages")
    p_stage.add_argument("--user-id", type=int, required=True, choices=[1, 2, 3])
    p_stage.add_argument("--output", type=Path, default=None)
    p_stage.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()
    if args.target == "users":
        compute_user_matrices(args.output, args.min_index, args.max_index, args.batch_size)
    else:
        output = args.output or DATA_INTERIM / f"deberta_stage_matrices_user{args.user_id}.pkl"
        compute_stage_matrices(args.user_id, output, args.batch_size)


if __name__ == "__main__":
    main()
