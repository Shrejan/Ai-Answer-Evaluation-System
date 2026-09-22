"""Token-to-word confidence from TrOCR generation scores (doc 08)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch

# Model-derived estimates — not calibrated probabilities until validated (doc 08).
CLIP_MIN = 0.0
CLIP_MAX = 1.0


def _token_probabilities(scores: Sequence[torch.Tensor]) -> List[float]:
    probs: List[float] = []
    for step_scores in scores:
        step = step_scores[0] if step_scores.dim() > 1 else step_scores
        prob = torch.softmax(step.float(), dim=-1).max().item()
        probs.append(prob)
    return probs


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(max(v, 1e-12)) for v in values)
    return math.exp(log_sum / len(values))


def compute_word_confidences(
    generated_sequence: torch.Tensor,
    token_scores: List[torch.Tensor],
    tokenizer,
    word_boundaries: List[Tuple[int, int]],
) -> List[float]:
    """Geometric mean of aligned token probabilities per word char span."""
    token_ids = generated_sequence.tolist()
    special_ids = set(tokenizer.all_special_ids)
    decoded_tokens: List[str] = []
    token_probs: List[float] = []
    step_probs = _token_probabilities(token_scores)

    step_idx = 0
    for token_id in token_ids:
        if token_id in special_ids:
            continue
        if step_idx >= len(step_probs):
            break
        decoded_tokens.append(tokenizer.decode([token_id]))
        token_probs.append(step_probs[step_idx])
        step_idx += 1

    char_spans: List[Tuple[int, int]] = []
    pos = 0
    for tok_text in decoded_tokens:
        char_spans.append((pos, pos + len(tok_text)))
        pos += len(tok_text)

    confidences: List[float] = []
    for start, end in word_boundaries:
        if end <= start:
            confidences.append(0.0)
            continue
        word_probs = [
            prob
            for (t_start, t_end), prob in zip(char_spans, token_probs)
            if t_start < end and t_end > start
        ]
        if not word_probs:
            confidences.append(0.5)
            continue
        value = _geometric_mean(word_probs)
        confidences.append(max(CLIP_MIN, min(CLIP_MAX, value)))
    return confidences


def clip_confidence(value: float) -> float:
    return max(CLIP_MIN, min(CLIP_MAX, value))
