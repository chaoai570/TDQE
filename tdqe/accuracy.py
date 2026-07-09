"""
匹配度检测模块 (论文 Section 2.2)
================================

计算准确性分数 M(T, S)，衡量数据样本 T 与其模型生成摘要 S 之间的匹配度。

论文使用文本相似匹配模型（参考文献 [16]，BERT）计算 M(T, S) ∈ [0, 0.5]。

实现说明:
    论文要求一个独立的相似匹配模型和一个独立的摘要生成模型，
    但当前环境只提供一个 GPT-2 模型，因此我们做如下近似:

    1. 用该语言模型从 T 生成摘要 S。
    2. 用同一个模型的均值池化隐藏状态嵌入计算 T 与 S 之间的余弦相似度
       （充当相似匹配函数）。
    3. 将相似度缩放到 [0, 0.5]。

    这保留了论文的核心设计思想: 准确度反映模型对 T 的"理解"程度，
    即 T 与其摘要之间的对齐程度。
"""

import numpy as np
import torch

from .config import MAX_INPUT_LENGTH, SUMMARY_MAX_NEW_TOKENS, SUMMARY_NUM_BEAMS


def _pool_embedding(model, tokenizer, text: str, device: str) -> np.ndarray:
    """对文本的最后一层隐藏状态做均值池化，得到固定大小向量。"""
    if not text or not text.strip():
        return np.zeros(768, dtype=np.float32)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to(device)

    if inputs["input_ids"].size(1) < 2:
        return np.zeros(768, dtype=np.float32)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]          # (1, seq_len, hidden_size)
        emb = hidden.mean(dim=1).cpu().numpy()[0]   # (hidden_size,)
    return emb


def _generate_summary(model, tokenizer, text: str, device: str) -> str:
    """用 beam search 生成输入文本的摘要/续写。"""
    if not text or not text.strip():
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to(device)

    if inputs["input_ids"].size(1) < 5:
        return text

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
            num_beams=SUMMARY_NUM_BEAMS,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只解码新生成的 token（去除输入前缀）
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    if len(full_text) > len(input_text):
        summary = full_text[len(input_text):].strip()
    else:
        summary = full_text.strip()

    return summary if summary else full_text[:200]


def accuracy_score(model, tokenizer, text: str, device: str = "cpu") -> float:
    """
    按论文 Section 2.2 计算准确性分数 M(T, S)。

    步骤:
    1. 从 T 生成摘要 S。
    2. 计算 T 和 S 的嵌入向量之间的余弦相似度。
    3. 将相似度缩放到 [0, 0.5]。

    Returns
    -------
    float
        M(T, S) ∈ [0, 0.5]。
    """
    if not text or not text.strip():
        return 0.0

    was_training = model.training
    model.eval()

    try:
        # 1. 生成摘要
        summary = _generate_summary(model, tokenizer, text, device)

        if not summary:
            return 0.0

        # 2. 在 eval 模式下计算 T 与 S 的嵌入
        emb_t = _pool_embedding(model, tokenizer, text, device)
        emb_s = _pool_embedding(model, tokenizer, summary, device)

        # 3. 余弦相似度
        norm_t = np.linalg.norm(emb_t)
        norm_s = np.linalg.norm(emb_s)
        if norm_t == 0.0 or norm_s == 0.0:
            return 0.0

        cos_sim = float(np.dot(emb_t, emb_s) / (norm_t * norm_s))

        # 裁剪并缩放到 [0, 0.5]
        cos_sim = max(0.0, min(1.0, cos_sim))
        return cos_sim * 0.5
    finally:
        if was_training:
            model.train()
