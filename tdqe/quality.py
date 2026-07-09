"""
Data Sample Quality Assessment (Section 2.3)
==============================================

Computes the final TDQE quality score Q(T) using Formula (2):

    Q(T) = R(V(T)) + M(T, S)

where:
    - R(V(T)) ∈ [0, 0.5]  is the robustness score  (Formula 1)
    - M(T, S)  ∈ [0, 0.5]  is the accuracy score    (Section 2.2)
    - Q(T)     ∈ [0, 1.0]  is the combined quality score
"""

from typing import List, Dict, Tuple

import numpy as np

from .robustness import robustness_score
from .accuracy import accuracy_score
from .config import NUM_DROPOUT_PASSES


def quality_score(
    model,
    tokenizer,
    text: str,
    m: int = NUM_DROPOUT_PASSES,
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """
    Compute Q(T) = R(V(T)) + M(T, S) per Formula (2).

    Returns
    -------
    (robustness, accuracy, quality) : Tuple[float, float, float]
        R ∈ [0, 0.5], M ∈ [0, 0.5], Q ∈ [0, 1.0].
    """
    r = robustness_score(model, tokenizer, text, m=m, device=device)
    a = accuracy_score(model, tokenizer, text, device=device)
    q = r + a
    return r, a, q


def compute_all_scores(
    model,
    tokenizer,
    texts: List[str],
    m: int = NUM_DROPOUT_PASSES,
    device: str = "cpu",
    progress_interval: int = 50,
) -> Dict[str, List[float]]:
    """
    Compute TDQE scores for a list of texts.

    Returns
    -------
    dict with keys "robustness", "accuracy", "quality".
    """
    R_vals, A_vals, Q_vals = [], [], []
    n = len(texts)

    for idx, text in enumerate(texts):
        r, a, q = quality_score(model, tokenizer, text, m=m, device=device)
        R_vals.append(r)
        A_vals.append(a)
        Q_vals.append(q)

        if progress_interval > 0 and (idx + 1) % progress_interval == 0:
            print(f"  [TDQE scoring] {idx + 1}/{n} samples processed")

    return {"robustness": R_vals, "accuracy": A_vals, "quality": Q_vals}
