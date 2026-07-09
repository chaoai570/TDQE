"""
Dataset loader for the 20 Newsgroups (20NG) dataset.

Follows the paper's Section 3.1:
    - 20 categories, ~19 000 news articles
    - Average length: 266 words; max: 10 334 words
    - Train/test split: 8:2
"""

import os
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TRAIN_SPLIT, RANDOM_SEED, MAX_INPUT_LENGTH


# ── 20 Newsgroups category names ───────────────────────────────
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
    Extract individual news articles from a 20NG text file.

    Articles are separated by ``\\nNewsgroup:``. Some FAQ or long-form
    posts reference inline ``Newsgroup:`` lines which the splitter
    would fragment. We check lengths — split fragments shorter than
    30 words are merged back into the preceding article so FAQ posts
    (like ``alt.atheism`` FAQ) stay as one coherent unit.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split on article boundary marker
    parts = ("\n" + content).split("\nNewsgroup:")

    raw = []
    for part in parts:
        art = part.strip()
        if not art:
            continue
        if not art.startswith("Newsgroup:"):
            art = "Newsgroup:" + art
        raw.append(art)

    # Merge tiny fragments back — they are inline ``Newsgroup:``
    # references inside a longer article, not real article boundaries.
    articles = []
    for art in raw:
        wc = len(art.split())
        if wc < 30 and articles:
            # Merge into previous article
            articles[-1] = articles[-1] + "\n" + art
        else:
            articles.append(art)

    # Final length filter
    return [a for a in articles if len(a) >= 100]


def load_20ng(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load the full 20 Newsgroups dataset.

    Parameters
    ----------
    data_dir : str
        Path to the ``archive/`` directory containing the 20 ``.txt`` files.

    Returns
    -------
    texts : List[str]
        Full text content of each article.
    labels : List[int]
        Integer label (0–19) for each article.
    categories : List[str]
        Category name for each article.
    """
    texts: List[str] = []
    labels: List[int] = []
    cat_names: List[str] = []

    for label_idx, cat in enumerate(CATEGORIES):
        file_path = os.path.join(data_dir, f"{cat}.txt")
        if not os.path.exists(file_path):
            print(f"  ⚠  Skipping missing file: {file_path}")
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
    Split into train/test sets at 8:2 ratio (per Section 3.1).

    Returns
    -------
    (train_texts, train_labels, test_texts, test_labels)
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
    PyTorch Dataset wrapping tokenized texts and labels for classifier
    training (Section 3.1).
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
