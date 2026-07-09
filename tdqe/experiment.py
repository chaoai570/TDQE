"""
Classifier Validation Experiments (Section 3)
===============================================

Implements the paper's experimental pipeline:

    1. Train a multi-label text classifier with parameters from Table 1.
    2. Evaluate by deleting low-quality data (Section 3.3, 3.4).
    3. Evaluate by deleting high-quality data.
    4. Ablation study: robustness-only vs accuracy-only (Section 3.5).

Table 1 parameters:
    - Output feature dim: 768
    - Input max length: 512
    - Training epochs: 10
    - Batch size: 8
    - Learning rate: 0.01
    - Activation: Leaky-ReLU
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
    Simple multi-label text classifier.

    Uses mean-pooled embeddings from a pre-trained encoder followed
    by a classifier head with Leaky-ReLU activation, matching Table 1.
    """

    def __init__(self, encoder, hidden_size: int = HIDDEN_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = encoder  # pre-trained transformer (GPT-2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        # Use the transformer encoder to get hidden states
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Mean pool the last hidden state
        last_hidden = outputs.hidden_states[-1]                          # (B, L, H)
        pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) # (B, H)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        logits = self.classifier(pooled)
        return logits


def _tokenize_batch(tokenizer, texts: List[str], device: str):
    """Tokenize a list of texts and return (input_ids, attention_mask)."""
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
    Train a classifier using parameters from Table 1 and return
    test-set precision and recall (P, R).
    """
    # Build classifier
    clf = TDQEClassifier(model).to(device)
    optimizer = torch.optim.Adam(clf.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Tokenize test set once
    test_input_ids, test_attention_mask = _tokenize_batch(tokenizer, test_texts, device)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long).to(device)

    for epoch in range(num_epochs):
        clf.train()
        # Shuffle training data each epoch
        indices = np.random.permutation(len(train_texts))
        total_loss = 0.0

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
            total_loss += loss.item()

    # Final evaluation
    return evaluate_classifier(
        clf, tokenizer, test_texts, test_labels, device
    )


def evaluate_classifier(
    clf: TDQEClassifier,
    tokenizer,
    test_texts: List[str],
    test_labels: List[int],
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate classifier and return (precision, recall) on test set.
    """
    clf.eval()
    test_input_ids, test_attention_mask = _tokenize_batch(tokenizer, test_texts, device)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long).to(device)

    batch_size = 32
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_ids = test_input_ids[i : i + batch_size]
            batch_mask = test_attention_mask[i : i + batch_size]
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
    Run a data removal experiment (Sections 3.3, 3.4).

    Sorts training data by quality score, removes a fraction (either
    lowest or highest), trains a classifier on the remaining data,
    and records (precision, recall).

    Parameters
    ----------
    remove_low : bool
        True  → remove lowest-quality data  (Fig 2, Fig 4, Fig 6)
        False → remove highest-quality data (Fig 3, Fig 5)

    Returns
    -------
    List[Tuple[float, float]]
        (precision, recall) for each removal fraction.
    """
    n = len(train_texts)
    sorted_idx = np.argsort(quality_scores)  # ascending: low → high

    if not remove_low:
        sorted_idx = sorted_idx[::-1]  # descending: high → low

    results = []
    for frac in removal_fractions:
        n_remove = int(n * frac)
        keep_idx = sorted_idx[n_remove:] if n_remove < n else sorted_idx[:0]

        subset_texts = [train_texts[i] for i in keep_idx]
        subset_labels = [train_labels[i] for i in keep_idx]

        print(f"    Removal fraction={frac:.0%}: {len(subset_texts)} samples remaining")
        p, r = train_classifier(
            model, tokenizer,
            subset_texts, subset_labels,
            test_texts, test_labels,
            device=device,
        )
        results.append((p, r))
        print(f"      P={p:.4f}, R={r:.4f}")

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
    Run ablation experiment (Section 3.5, Figure 7).

    Compares three scoring strategies by removing low-quality data:
        - Robustness-only: sort by R(V(T))
        - Accuracy-only:   sort by M(T, S)
        - Combined:        sort by Q(T) = R + M

    Returns
    -------
    (robustness_results, accuracy_results, combined_results)
    """
    n = len(train_texts)
    q_scores = [r + a for r, a in zip(robustness_scores, accuracy_scores)]

    print("  --- Robustness-only ---")
    r_results = run_data_removal_experiment(
        model, tokenizer, train_texts, train_labels, test_texts, test_labels,
        robustness_scores, removal_fractions, remove_low=True, device=device,
    )

    print("  --- Accuracy-only ---")
    a_results = run_data_removal_experiment(
        model, tokenizer, train_texts, train_labels, test_texts, test_labels,
        accuracy_scores, removal_fractions, remove_low=True, device=device,
    )

    print("  --- Combined (TDQE) ---")
    q_results = run_data_removal_experiment(
        model, tokenizer, train_texts, train_labels, test_texts, test_labels,
        q_scores, removal_fractions, remove_low=True, device=device,
    )

    return r_results, a_results, q_results
