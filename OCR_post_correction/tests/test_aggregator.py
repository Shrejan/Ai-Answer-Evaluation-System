"""
tests/test_aggregator.py — Tests for candidate pool aggregator:
pool cap, original presence invariant, deduplication, deterministic ranking,
and property-based tests with random inputs.
"""

import random
import string
from stage3.schemas import Span, CandidatePool
from stage3.candidates.aggregator import build_candidate_pool


def test_aggregator_basic_and_cap():
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    sources_output = {
        "domain_dictionary": ["dental", "digital"],
        "symspell": ["dental", "denial", "detail", "dentist", "dentals", "capital"],
    }
    pool = build_candidate_pool(span, sources_output, max_pool_size=4)
    assert isinstance(pool, CandidatePool)
    assert len(pool.candidates) <= 4
    assert "denital" in pool.candidates
    assert "dental" in pool.candidates
    assert "denital" in pool.sources


def test_aggregator_preserves_original_when_ranked_last():
    span = Span(span_ids=[0], original_text="zzzzzz", clean_text="zzzzzz")
    # Provide 10 high-frequency valid English words that will heavily outrank "zzzzzz"
    sources_output = {
        "src1": ["the", "of", "and", "to", "in", "is", "it", "that", "for", "you"]
    }
    pool = build_candidate_pool(span, sources_output, max_pool_size=5)
    assert len(pool.candidates) == 5
    assert "zzzzzz" in pool.candidates, "Original MUST be preserved even when ranked last"


def test_aggregator_deduplication_and_determinism():
    span = Span(span_ids=[0], original_text="test", clean_text="test")
    sources_output = {
        "src1": ["apple", "banana", "apple"],
        "src2": ["banana", "cherry", "apple"],
    }
    pool1 = build_candidate_pool(span, sources_output, max_pool_size=6)
    pool2 = build_candidate_pool(span, sources_output, max_pool_size=6)

    # Check deduplication
    assert len(pool1.candidates) == len(set(pool1.candidates))
    # Check deterministic order
    assert pool1.candidates == pool2.candidates
    # Check source tracking
    assert set(pool1.sources["apple"]) == {"src1", "src2"}


def test_aggregator_property_original_always_in_pool():
    """Property test: for 50 random inputs and pool sizes, original ∈ pool always holds."""
    rng = random.Random(42)
    for _ in range(50):
        # Random original word
        orig_len = rng.randint(3, 10)
        orig = "".join(rng.choices(string.ascii_lowercase, k=orig_len))
        span = Span(span_ids=[0], original_text=orig, clean_text=orig)

        # Random sources output
        num_sources = rng.randint(1, 4)
        sources_out = {}
        for s in range(num_sources):
            num_cands = rng.randint(0, 8)
            cands = [
                "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10)))
                for _ in range(num_cands)
            ]
            sources_out[f"source_{s}"] = cands

        cap = rng.randint(2, 6)
        pool = build_candidate_pool(span, sources_out, max_pool_size=cap)

        assert len(pool.candidates) <= cap
        assert orig in pool.candidates, f"Failed invariant for orig '{orig}' with cap {cap}"
