"""
tests/test_candidate_sources.py — Comprehensive tests for Candidate Sources A–D.
Verifies empty inputs return [], worked examples produce expected candidate targets,
and measures/logs per-call latencies.
"""

import time
import pytest
from stage3.schemas import Span
from stage3.candidates.symspell_source import SymSpellSource
from stage3.candidates.confusion_source import ConfusionSource
from stage3.candidates.dictionary_source import DictionarySource
from stage3.candidates.phrase_retrieval_source import PhraseRetrievalSource


def test_empty_input_all_sources():
    empty_span = Span(span_ids=[0], original_text="", clean_text="")
    sources = [
        SymSpellSource(),
        ConfusionSource(),
        DictionarySource(),
        PhraseRetrievalSource(),
    ]
    for src in sources:
        start = time.perf_counter()
        cands = src.propose(empty_span)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert cands == [], f"{src.name} must return [] on empty input"
        print(f"[{src.name}] empty input latency: {elapsed_ms:.3f} ms")


def test_source_a_symspell():
    src = SymSpellSource()
    span = Span(span_ids=[0], original_text="transisttor", clean_text="transisttor")
    start = time.perf_counter()
    cands = src.propose(span)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[SymSpellSource] propose latency: {elapsed_ms:.3f} ms, candidates: {cands}")

    assert "transistor" in cands


def test_source_b_confusion():
    src = ConfusionSource()
    # Worked example: "pregnue" -> "pregnant"
    span = Span(span_ids=[0], original_text="pregnue", clean_text="pregnue")
    start = time.perf_counter()
    cands = src.propose(span)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[ConfusionSource] propose latency: {elapsed_ms:.3f} ms, candidates: {cands}")

    assert "pregnant" in cands


def test_source_c_dictionary():
    src = DictionarySource()
    # Worked example: "denital" -> "dental"
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    start = time.perf_counter()
    cands = src.propose(span)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[DictionarySource] propose latency: {elapsed_ms:.3f} ms, candidates: {cands}")

    assert "dental" in cands


def test_source_d_phrase_retrieval_split():
    src = PhraseRetrievalSource()
    # Worked example: "volontrar-y" -> "voluntary"
    span = Span(span_ids=[0], original_text="volontrar-y", clean_text="volontrary")
    start = time.perf_counter()
    cands = src.propose(span)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[PhraseRetrievalSource split] latency: {elapsed_ms:.3f} ms, candidates: {cands}")

    assert "voluntary" in cands


def test_source_d_phrase_retrieval_merge():
    src = PhraseRetrievalSource()
    # Worked example: "local lacking" -> "lactating"
    span = Span(
        span_ids=[0, 1],
        original_text="local lacking",
        clean_text="locallacking",
        is_phrase=True,
    )
    start = time.perf_counter()
    cands = src.propose(span)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[PhraseRetrievalSource merge] latency: {elapsed_ms:.3f} ms, candidates: {cands}")

    assert "lactating" in cands


def test_source_d_phrase_retrieval_prefix():
    src = PhraseRetrievalSource()
    # Worked example: "nubation" -> "incubation"
    span = Span(span_ids=[0], original_text="nubation", clean_text="nubation")
    cands = src.propose(span)
    assert "incubation" in cands


def test_source_d_phrase_retrieval_semantic():
    src = PhraseRetrievalSource()
    # Worked example: "attack blood pressure" -> "check blood pressure"
    span = Span(
        span_ids=[0, 1, 2],
        original_text="attack blood pressure",
        clean_text="attack blood pressure",
        prev_context="Doctors",
        next_context="and glucose levels",
        is_phrase=True,
    )
    cands = src.propose(span)
    assert "check blood pressure" in cands


def test_phonetic_domain_source_garbled_examples():
    from stage3.candidates.phonetic_source import PhoneticDomainSource
    src = PhoneticDomainSource()

    # "titzimmuneration" -> "immunization"
    s1 = Span(span_ids=[0], original_text="titzimmuneration", clean_text="titzimmuneration")
    c1 = src.propose(s1)
    assert "immunization" in c1

    # "makes jealous" -> "measles"
    s2 = Span(span_ids=[0, 1], original_text="makes jealous", clean_text="makes jealous", is_phrase=True)
    c2 = src.propose(s2)
    assert "measles" in c2

    # "pole" -> "polio"
    s3 = Span(span_ids=[0], original_text="pole", clean_text="pole")
    c3 = src.propose(s3)
    assert "polio" in c3

