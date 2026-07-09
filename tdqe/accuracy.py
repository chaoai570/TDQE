"""
Matching Degree Detection Module (Section 2.2)
===============================================

Computes the **accuracy** score M(T, S) by measuring the similarity
between a data sample T and its model-generated summary S.

The paper uses a "text similarity matching model" (Reference [16], BERT)
to compute M(T, S) ∈ [0, 0.5].

Implementation note:
    Since the paper requires a separate similarity matching model and
    an independent summarization model, but the user's environment
    provides a single GPT-2 checkpoint, we approximate the matching
    by:

    1. Generating a summary S from T using the language model.
    2. Computing the cosine similarity between the **mean-pooled
       hidden-state embeddings** of T and S from the same model
       (serving as the similarity matching function).
    3. Scaling the similarity to [0, 0.5].

    This preserves the paper's conceptual design: accuracy reflects
    how well the model "understands" T, as measured by the alignment
    between T and its summary.
"""

import numpy as np
import torch

from .config import MAX_INPUT_LENGTH, SUMMARY_MAX_NEW_TOKENS, SUMMARY_NUM_BEAMS


def _pool_embedding(model, tokenizer, text: str, device: str) -> np.ndarray:
    """Mean-pool the last hidden state of a text into a fixed-size vector."""
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
    """
    Generate a concise summary / continuation of the input text.
    Uses beam search as described in the paper.
    """
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

    # Decode only the newly generated tokens (skip the input prefix)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # The summary is the part beyond the input
    if len(full_text) > len(input_text):
        summary = full_text[len(input_text):].strip()
    else:
        summary = full_text.strip()

    return summary if summary else full_text[:200]


def accuracy_score(model, tokenizer, text: str, device: str = "cpu") -> float:
    """
    Compute the accuracy score M(T, S) per Section 2.2.

    1. Generate summary S from T.
    2. Compute cosine similarity between embeddings of T and S.
    3. Scale to [0, 0.5].

    Returns
    -------
    float
        M(T, S) ∈ [0, 0.5].
    """
    if not text or not text.strip():
        return 0.0

    was_training = model.training
    model.eval()

    try:
        # 1. Generate summary
        summary = _generate_summary(model, tokenizer, text, device)

        if not summary:
            return 0.0

        # 2. Compute embeddings of T and S in eval mode
        emb_t = _pool_embedding(model, tokenizer, text, device)
        emb_s = _pool_embedding(model, tokenizer, summary, device)

        # 3. Cosine similarity
        norm_t = np.linalg.norm(emb_t)
        norm_s = np.linalg.norm(emb_s)
        if norm_t == 0.0 or norm_s == 0.0:
            return 0.0

        cos_sim = float(np.dot(emb_t, emb_s) / (norm_t * norm_s))

        # Clamp and scale to [0, 0.5]
        cos_sim = max(0.0, min(1.0, cos_sim))
        return cos_sim * 0.5
    finally:
        if was_training:
            model.train()
