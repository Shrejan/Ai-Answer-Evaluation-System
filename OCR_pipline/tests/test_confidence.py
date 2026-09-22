"""Phase 4 confidence tests."""

import torch

from ocr_pipeline.scoring.confidence import clip_confidence, compute_word_confidences


class FakeTokenizer:
    all_special_ids = [0, 1]

    def decode(self, ids):
        mapping = {2: "hel", 3: "lo", 4: "x"}
        return mapping.get(ids[0], "?")


def test_near_certain_crop():
    seq = torch.tensor([0, 2, 3, 1])
    scores = [
        torch.tensor([[-10.0, 10.0]]),
        torch.tensor([[-8.0, 8.0]]),
    ]
    confidences = compute_word_confidences(
        seq,
        scores,
        FakeTokenizer(),
        [(0, 3), (3, 5)],
    )
    assert confidences[0] > 0.9
    assert confidences[1] > 0.9


def test_one_uncertain_token_lowers_word_confidence():
    seq = torch.tensor([0, 2, 3, 1])
    scores = [
        torch.tensor([[-10.0, 10.0]]),
        torch.tensor([[-0.5, 0.5]]),
    ]
    confidences = compute_word_confidences(
        seq,
        scores,
        FakeTokenizer(),
        [(0, 3), (3, 5)],
    )
    assert confidences[1] < confidences[0]


def test_out_of_range_scores_clipped():
    assert clip_confidence(1.5) == 1.0
    assert clip_confidence(-0.2) == 0.0
