"""
tests/test_resources.py — Tests for local dictionary loaders, SymSpell caching,
worked examples fixture, and sample page fixture.
"""

import json
from pathlib import Path
import yaml
from stage3.resources import (
    load_domain_terms,
    load_general_dictionary,
    load_phrase_corpus,
    load_confusion_matrix,
    get_symspell_engine,
)
from stage3.schemas import OCRInput


def test_load_domain_terms():
    terms = load_domain_terms()
    assert isinstance(terms, set)
    assert len(terms) > 50
    assert "transistor" in terms
    assert "amplifier" in terms
    assert "dental" in terms


def test_load_general_dictionary():
    dict_map = load_general_dictionary()
    assert isinstance(dict_map, dict)
    assert "the" in dict_map
    assert "health" in dict_map
    assert dict_map["the"] > dict_map["health"]


def test_load_phrase_corpus():
    phrases = load_phrase_corpus()
    assert isinstance(phrases, list)
    assert len(phrases) >= 5
    assert "text" in phrases[0]


def test_load_confusion_matrix():
    matrix = load_confusion_matrix()
    assert "substitutions" in matrix
    assert "i->e" in matrix["substitutions"]


from symspellpy import Verbosity

def test_symspell_engine():
    engine = get_symspell_engine()
    suggestions = engine.lookup("transisttor", Verbosity.CLOSEST, max_edit_distance=2)
    assert len(suggestions) > 0
    top_terms = [s.term for s in suggestions]
    assert "transistor" in top_terms


def test_worked_examples_fixture_exists():
    path = Path(__file__).resolve().parent / "fixtures" / "worked_examples.yaml"
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "examples" in data
    inputs = [ex["input"] for ex in data["examples"]]
    assert "transisttor" in inputs
    assert "denital" in inputs
    assert "pregnue" in inputs


def test_sample_page_fixture():
    path = Path(__file__).resolve().parent / "fixtures" / "sample_page.json"
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ocr_in = OCRInput(**payload)
    assert ocr_in.request_id == "page_001"
    assert len(ocr_in.words) == 19
