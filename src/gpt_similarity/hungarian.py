from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_alignment_score(
    similarity_matrix: np.ndarray,
    sents_a: list[str],
    sents_b: list[str],
    normalize: str = "max",
    threshold_low: float = 0.5,
    threshold_mid: float = 0.8,
) -> dict:
    cost_matrix = 1.0 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_similarity = float(similarity_matrix[row_ind, col_ind].sum())

    n_a, n_b = similarity_matrix.shape
    if normalize == "min":
        score = total_similarity / min(n_a, n_b)
    elif normalize == "max":
        score = total_similarity / max(n_a, n_b)
    elif normalize == "avg":
        score = total_similarity / ((n_a + n_b) / 2)
    else:
        raise ValueError("normalize must be one of: min, max, avg")

    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        sim = float(similarity_matrix[r, c])
        if sim <= threshold_low:
            bucket = "low"
        elif sim <= threshold_mid:
            bucket = "medium"
        else:
            bucket = "high"
        matched_pairs.append({
            "row": int(r),
            "col": int(c),
            "similarity": sim,
            "sentence_a": sents_a[r],
            "sentence_b": sents_b[c],
            "bucket": bucket,
        })

    return {
        "score": float(score),
        "matched_pairs": matched_pairs,
        "unmatched_rows": sorted(set(range(n_a)) - set(row_ind)),
        "unmatched_cols": sorted(set(range(n_b)) - set(col_ind)),
    }


def symmetrize_forward_backward(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    return 0.5 * (forward + backward.T)
