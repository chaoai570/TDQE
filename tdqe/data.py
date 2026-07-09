"""
20 Newsgroups (20NG) 数据集加载模块。

按论文 Section 3.1:
    - 20 个类别，约 19,000 篇新闻文章
    - 平均长度: 266 词；最长: 10,334 词
    - 训练/测试划分: 8:2
"""

import os
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TRAIN_SPLIT, RANDOM_SEED, MAX_INPUT_LENGTH


# ── 20 Newsgroups 类别名称 ─────────────────────────────────
CATEGORIES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
    "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
    "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns",
    "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
]


def _extract_articles(file_path: str) -> List[str]:
    """
    从单个 20NG 文本文件中提取每篇文章。

    文章由 ``\\nNewsgroup:`` 分隔。有些 FAQ 或长文章内部会引用
    ``Newsgroup:`` 行，这些引用的片段很短（< 30 词），会被合并回
    前一篇完整文章，以保持 FAQ 类文章不被切碎。
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 按文章边界标记 "\nNewsgroup:" 切分
    parts = ("\n" + content).split("\nNewsgroup:")

    raw = []
    for part in parts:
        art = part.strip()
        if not art:
            continue
        if not art.startswith("Newsgroup:"):
            art = "Newsgroup:" + art
        raw.append(art)

    # 将过短的碎片合并回前一篇文章（它们是文章内部的引用）
    articles = []
    for art in raw:
        wc = len(art.split())
        if wc < 30 and articles:
            articles[-1] = articles[-1] + "\n" + art
        else:
            articles.append(art)

    # 最终过滤掉过短的文章
    return [a for a in articles if len(a) >= 100]


def load_20ng(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    加载完整 20 Newsgroups 数据集。

    Parameters
    ----------
    data_dir : str
        ``archive/`` 目录路径，目录下应有 20 个 ``.txt`` 文件。

    Returns
    -------
    texts : List[str]
        每篇文章的完整文本。
    labels : List[int]
        每篇文章的整数标签 (0–19)。
    categories : List[str]
        每篇文章的类别名称。
    """
    texts: List[str] = []
    labels: List[int] = []
    cat_names: List[str] = []

    for label_idx, cat in enumerate(CATEGORIES):
        file_path = os.path.join(data_dir, f"{cat}.txt")
        if not os.path.exists(file_path):
            print(f"  跳过缺失文件: {file_path}")
            continue

        articles = _extract_articles(file_path)
        for art in articles:
            texts.append(art)
            labels.append(label_idx)
            cat_names.append(cat)

    return texts, labels, cat_names


def split_dataset(
    texts: List[str],
    labels: List[int],
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    按 8:2 比例划分训练集和测试集（按论文 Section 3.1）。

    Returns
    -------
    (训练文本, 训练标签, 测试文本, 测试标签)
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(texts)
    indices = rng.permutation(n)
    n_train = int(n * TRAIN_SPLIT)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_texts, train_labels, test_texts, test_labels


class TDQEDataset(Dataset):
    """
    PyTorch Dataset，用于分类器训练 (Section 3.1)。
    将文本分词并打包为 (input_ids, attention_mask, label)。
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
