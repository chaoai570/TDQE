"""
Semantic Consistency Detection Module (Section 2.1)
=====================================================

Computes the **robustness** score R(V(T)) using Formula (1):

    R(V(T)) = σ( − 2 / (m(m−1)) * Σ_{i<j} d(v_i, v_j) )

where:
    - m   = number of stochastic forward passes (Dropout sub-networks)
    - v_i = embedding vector from the i-th stochastic sub-network
    - d(·,·) = Euclidean distance
    - σ(·)   = sigmoid, ensuring R ∈ [0, 0.5]
"""

import numpy as np
import torch

from .config import NUM_DROPOUT_PASSES


def _extract_embedding(model, tokenizer, text: str, device: str) -> np.ndarray:
    """
    Run a single forward pass through the model's encoder and extract
    the mean-pooled hidden-state embedding of the input text.

    The model is kept in **train** mode so Dropout is active, producing
    a unique stochastic sub-network on each call.
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
        # Mean-pool the last hidden state over the token dimension
        hidden = outputs.hidden_states[-1]          # (1, seq_len, hidden_size)
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
    Compute the robustness score R(V(T)) per Formula (1).

    Parameters
    ----------
    model : PreTrainedModel
        A text summarization / language model with Dropout layers
        (e.g. GPT-2, BART, T5). Must be in ``train()`` mode so
        Dropout is active.
    tokenizer : PreTrainedTokenizer
    text : str
        The input data sample T.
    m : int
        Number of stochastic forward passes (default 3, per paper).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    float
        R(V(T)) ∈ [0, 0.5].
    """
    if not text or not text.strip():
        return 0.0

    # Ensure dropout is active
    was_training = model.training
    model.train()

    try:
        embeddings = []
        for _ in range(m):
            emb = _extract_embedding(model, tokenizer, text, device)
            # L2-normalize so Euclidean distances are bounded in [0, 2]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)

        # Compute pairwise Euclidean distances
        distances = []
        for i in range(m):
            for j in range(i + 1, m):
                d = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(d)

        avg_distance = np.mean(distances) if distances else 0.0

        # Formula (1): R(V(T)) = σ( − avg_distance )
        # Note: paper defines σ on the scaled negative mean.
        # The coefficient 2/(m(m-1)) is absorbed by taking the mean
        # of pairwise distances, so the argument is simply ( − avg_distance ).
        score = 1.0 / (1.0 + np.exp(avg_distance))
        return float(score)
    finally:
        if not was_training:
            model.eval()
