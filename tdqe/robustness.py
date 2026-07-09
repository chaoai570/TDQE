"""
语义一致性检测模块 (论文 Section 2.1)
====================================

用公式 (1) 计算鲁棒性分数 R(V(T)):

    R(V(T)) = σ( − 2 / (m(m−1)) × Σ_{i<j} d(v_i, v_j) )

其中:
    - m   = 随机前向传播次数（每次 Dropout 生成不同子网络）
    - v_i = 第 i 个随机子网络输出的嵌入向量
    - d(·,·) = 欧氏距离
    - σ(·)   = Sigmoid 函数，确保 R ∈ [0, 0.5]
"""

import numpy as np
import torch

from .config import NUM_DROPOUT_PASSES


def _extract_embedding(model, tokenizer, text: str, device: str) -> np.ndarray:
    """
    对输入文本做一次前向传播，提取最后一层隐藏状态的均值池化嵌入向量。

    模型处于 train() 模式，因此 Dropout 会激活，每次调用产生不同的随机子网络。
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    if inputs["input_ids"].size(1) < 2:
        return np.zeros(768, dtype=np.float32)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 对最后一层隐藏状态在 token 维度做均值池化
        hidden = outputs.hidden_states[-1]           # (1, seq_len, hidden_size)
        embedding = hidden.mean(dim=1).cpu().numpy()[0]  # (hidden_size,)

    return embedding


def robustness_score(
    model,
    tokenizer,
    text: str,
    m: int = NUM_DROPOUT_PASSES,
    device: str = "cpu",
) -> float:
    """
    按公式 (1) 计算鲁棒性分数 R(V(T))。

    Parameters
    ----------
    model : PreTrainedModel
        带有 Dropout 层的语言模型（如 GPT-2、BART、T5）。
        需要处于 train() 模式以激活 Dropout。
    tokenizer : PreTrainedTokenizer
    text : str
        输入数据样本 T。
    m : int
        随机前向传播次数，默认 3（按论文设定）。
    device : str
        "cuda" 或 "cpu"。

    Returns
    -------
    float
        R(V(T)) ∈ [0, 0.5]。
    """
    if not text or not text.strip():
        return 0.0

    # 确保 Dropout 处于激活状态
    was_training = model.training
    model.train()

    try:
        embeddings = []
        for _ in range(m):
            emb = _extract_embedding(model, tokenizer, text, device)
            # L2 归一化，使欧氏距离有界地落在 [0, 2] 范围内
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)

        # 计算两两之间的欧氏距离
        distances = []
        for i in range(m):
            for j in range(i + 1, m):
                d = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(d)

        avg_distance = np.mean(distances) if distances else 0.0

        # 公式 (1): R(V(T)) = σ( − avg_distance )
        # 取均值即已将系数 2/(m(m−1)) 吸收，因此 sigmoid 参数为 −avg_distance
        score = 1.0 / (1.0 + np.exp(avg_distance))
        return float(score)
    finally:
        if not was_training:
            model.eval()
