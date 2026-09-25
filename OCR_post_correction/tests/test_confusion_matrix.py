"""
tests/test_confusion_matrix.py — Tests for empirical character confusion matrix builder.
"""

from pathlib import Path
import json
from scripts.build_confusion_matrix import align_and_count, build_confusion_matrix


def test_align_and_count_known_pairs():
    pairs = [
        ("dental", "denital"),       # insertion 'i'
        ("transistor", "transisttor"), # insertion 't'
        ("terminal", "terninal"),     # substitution 'm->n'
        ("amplifier", "anplifier"),   # substitution 'm->n'
        ("resistor", "resestor"),     # substitution 'i->e'
    ]
    matrix = align_and_count(pairs, include_seed=False)

    assert "substitutions" in matrix
    assert "multi_char_patterns" in matrix
    assert "insertions" in matrix
    assert "probabilities" in matrix

    # 'm->n' occurred twice
    assert matrix["substitutions"].get("m->n") == 2
    # 'i->e' occurred once
    assert matrix["substitutions"].get("i->e") == 1
    # P(n | m) should be 1.0 since 'm' only substituted to 'n'
    assert matrix["probabilities"].get("m->n") == 1.0


def test_build_confusion_matrix_end_to_end(tmp_path):
    test_jsonl = tmp_path / "test_pairs.jsonl"
    with open(test_jsonl, "w", encoding="utf-8") as f:
        f.write('{"gt_text": "dental", "pred_text": "denital"}\n')
        f.write('{"gt_text": "resistor", "pred_text": "resestor"}\n')

    out_json = tmp_path / "char_confusion.json"
    result = build_confusion_matrix([test_jsonl], out_json, include_seed=True)

    assert out_json.exists()
    assert result["metadata"]["pair_count"] == 2
    assert "date" in result["metadata"]
    assert "source_file_hash" in result["metadata"]

    # Verify seed confusions are preserved
    assert result["substitutions"]["i->e"] >= 49
    assert result["substitutions"]["r->n"] >= 23
