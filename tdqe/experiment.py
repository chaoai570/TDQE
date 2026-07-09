"""
分类器验证实验模块 (论文 Section 3)
=================================

按论文 Table 1 参数训练多标签文本分类器，进行以下实验:

    1. 训练基线分类器
    2. 删除低质量数据实验 (Section 3.3, 3.4)
    3. 删除高质量数据实验 (Section 3.3, 3.4)
    4. 消融实验: 仅鲁棒性 vs. 仅准确性 vs. 组合 TDQE (Section 3.5)

Table 1 参数:
    - 输出特征维度: 768
    - 输入最大长度: 512
    - 训练轮次: 10
    - 批大小: 8
    - 学习率: 0.01
    - 激活函数: Leaky-ReLU
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score

from .config import (
    MAX_INPUT_LENGTH,
    HIDDEN_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_CLASSES,
)


class TDQEClassifier(nn.Module):
    """
    多标签文本分类器。

    使用预训练编码器的均值池化嵌入，后接 Leaky-ReLU 激活的
    分类头，与 Table 1 参数一致。
    """

    def __init__(self, encoder, hidden_size: int = HIDDEN_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = encoder  # 预训练 transformer（如 GPT-2）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 对最后一层隐藏状态做均值池化
        last_hidden = outputs.hidden_states[-1]                          # (B, L, H)
        pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) # (B, H)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        logits = self.classifier(pooled)
        return logits


def _tokenize_batch(tokenizer, texts: List[str], device: str):
    """将文本列表分词，返回 (input_ids, attention_mask)。"""
    enc = tokenizer(
        texts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def train_classifier(
    model,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    device: str = "cpu",
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> Tuple[TDQEClassifier, float, float]:
    """
    按 Table 1 参数训练分类器，返回测试集上的准确率 (P) 和召回率 (R)。
    """
    clf = TDQEClassifier(model).to(device)
    optimizer = torch.optim.Adam(clf.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        clf.train()
        indices = np.random.permutation(len(train_texts))

        for i in range(0, len(train_texts), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_texts = [train_texts[j] for j in batch_idx]
            batch_labels = torch.tensor(
                [train_labels[j] for j in batch_idx], dtype=torch.long
            ).to(device)

            input_ids, attention_mask = _tokenize_batch(tokenizer, batch_texts, device)

            optimizer.zero_grad()
            logits = clf(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    return evaluate_classifier(clf, tokenizer, test_texts, test_labels, device)


def evaluate_classifier(
    clf: TDQEClassifier,
    tokenizer,
    test_texts: List[str],
    test_labels: List[int],
    device: str = "cpu",
) -> Tuple[float, float]:
    """评估分类器，返回测试集 (准确率, 召回率)。"""
    clf.eval()
    test_input_ids, test_attention_mask = _tokenize_batch(tokenizer, test_texts, device)

    eval_batch = 32
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), eval_batch):
            batch_ids = test_input_ids[i : i + eval_batch]
            batch_mask = test_attention_mask[i : i + eval_batch]
            logits = clf(batch_ids, batch_mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())

    p = precision_score(test_labels, all_preds, average="macro", zero_division=0)
    r = recall_score(test_labels, all_preds, average="macro", zero_division=0)
    return p, r


def run_data_removal_experiment(
    model,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    quality_scores: List[float],
    removal_fractions: List[float],
    remove_low: bool = True,
    device: str = "cpu",
) -> List[Tuple[float, float]]:
    """
    数据删除实验 (Section 3.3, 3.4)。

    按质量分数排序训练数据，删除一定比例（最低分或最高分），
    用剩余数据重新训练分类器，记录 (准确率, 召回率)。

    Parameters
    ----------
    remove_low : bool
        True  → 删除低质量数据  (Fig 2, Fig 4, Fig 6)
        False → 删除高质量数据  (Fig 3, Fig 5)

    Returns
    -------
    List[Tuple[float, float]]
        每个删除比例对应的 (准确率, 召回率)。
    """
    n = len(train_texts)
    sorted_idx = np.argsort(quality_scores)  # 升序: 低 → 高

    if not remove_low:
        sorted_idx = sorted_idx[::-1]  # 降序: 高 → 低

    results = []
    for frac in removal_fractions:
        n_remove = int(n * frac)
        keep_idx = sorted_idx[n_remove:] if n_remove < n else sorted_idx[:0]

        subset_texts = [train_texts[i] for i in keep_idx]
        subset_labels = [train_labels[i] for i in keep_idx]

        print(f"    删除比例 = {frac:.0%}: 剩余 {len(subset_texts)} 条样本")
        p, r = train_classifier(
            model, tokenizer,
            subset_texts, subset_labels,
            test_texts, test_labels,
            device=device,
        )
        results.append((p, r))
        print(f"      准确率 P = {p:.4f}, 召回率 R = {r:.4f}")

    return results


def run_ablation_experiment(
    model,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    robustness_scores: List[float],
    accuracy_scores: List[float],
    removal_fractions: List[float],
    device: str = "cpu",
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    消融实验 (Section 3.5, Figure 7)。

    对比三种评分策略:
        - 仅鲁棒性: 按 R(V(T)) 排序
        - 仅准确性: 按 M(T, S) 排序
        - 组合 (TDQE): 按 Q(T) = R + M 排序

    Returns
    -------
    (仅鲁棒性结果, 仅准确性结果, 组合结果)
    """
    q_scores = [r + a for r, a in zip(robustness_scores, accuracy_scores)]

    print("  --- 仅鲁棒性 ---")
    r_results = run_data_removal_experiment(
        model, tokenizer, train_texts, train_labels, test_texts, test_labels,
        robustness_scores, removal_fractions, remove_low=True, device=device,
    )

    print("  --- 仅准确性 ---")
    a_results = run_data_removal_experiment(
        model, tokenizer, train_texts, train_labels, test_texts, test_labels,
        accuracy_scores, removal_fractions, remove_low=True, device=device,
    )

    print("  --- 组合 (TDQE) ---")
    q_results = run_data_removal_experiment(
        model, tokenizer, train_texts, train_labels, test_texts, test_labels,
        q_scores, removal_fractions, remove_low=True, device=device,
    )

    return r_results, a_results, q_results
