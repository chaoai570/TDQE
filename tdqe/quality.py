"""
数据样本质量评估模块 (论文 Section 2.3)
======================================

用公式 (2) 计算最终 TDQE 质量分数 Q(T):

    Q(T) = R(V(T)) + M(T, S)

其中:
    - R(V(T)) ∈ [0, 0.5] 为鲁棒性分数 (公式 1)
    - M(T, S)  ∈ [0, 0.5] 为准确性分数 (Section 2.2)
    - Q(T)     ∈ [0, 1.0] 为综合质量分数
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
    按公式 (2) 计算 Q(T) = R(V(T)) + M(T, S)。

    Returns
    -------
    (鲁棒性, 准确性, 质量) : Tuple[float, float, float]
        R ∈ [0, 0.5], M ∈ [0, 0.5], Q ∈ [0, 1.0]。
    """
    r = robustness_score(model, tokenizer, text, m=m, device=device)
    a = accuracy_score(model, tokenizer, text, device=device)
    q = r + a
    return r, a, q


def compute_all_scores(
    model,
    tokenizer,
    texts: List[str],
    categories: List[str] = None,
    m: int = NUM_DROPOUT_PASSES,
    device: str = "cpu",
    progress_interval: int = 50,
) -> Dict[str, list]:
    """
    批量计算 TDQE 分数。

    Parameters
    ----------
    categories : List[str] or None
        若提供，则取前 len(texts) 项作为对应类别写入输出。

    Returns
    -------
    dict, 包含键 "robustness"、"accuracy"、"quality"，以及 "category"（若传入）。
    """
    R_vals, A_vals, Q_vals = [], [], []
    n = len(texts)

    for idx, text in enumerate(texts):
        r, a, q = quality_score(model, tokenizer, text, m=m, device=device)
        R_vals.append(r)
        A_vals.append(a)
        Q_vals.append(q)

        if progress_interval > 0 and (idx + 1) % progress_interval == 0:
            print(f"  [TDQE 打分进度] {idx + 1}/{n} 条已处理")

    result = {"robustness": R_vals, "accuracy": A_vals, "quality": Q_vals}
    if categories is not None:
        # 确保类别与 texts 一一对应
        result["category"] = list(categories[:n])
    return result
