"""
scripts/build_confusion_matrix.py — Builds character and multi-character confusion matrix
from Stage 2 alignment pairs (ground truth vs TrOCR predictions).
Normalizes counts to probabilities and exports to data/confusion_matrix/char_confusion.json.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from Levenshtein import opcodes


SEED_CONFUSIONS = {
    "i->e": 49,
    "i->a": 29,
    "e->o": 26,
    "r->n": 23,
    "o->a": 20,
    "a->o": 19,
    "c->o": 18,
    "i->o": 14,
}

SEED_MULTI_CHAR = {
    "t->tt": 12,
    "tt->t": 8,
    "m->rn": 15,
    "rn->m": 10,
}


def compute_file_hash(paths: List[Path]) -> str:
    hasher = hashlib.sha256()
    for p in sorted(paths):
        if p.exists():
            with open(p, "rb") as f:
                hasher.update(f.read())
    return hasher.hexdigest()[:16]


def align_and_count(
    pairs: List[Tuple[str, str]],
    include_seed: bool = True
) -> Dict[str, Any]:
    substitutions: Dict[str, int] = defaultdict(int)
    multi_char: Dict[str, int] = defaultdict(int)
    insertions: Dict[str, int] = defaultdict(int)
    deletions: Dict[str, int] = defaultdict(int)

    if include_seed:
        for k, v in SEED_CONFUSIONS.items():
            substitutions[k] += v
        for k, v in SEED_MULTI_CHAR.items():
            multi_char[k] += v

    for gt, pred in pairs:
        # Align using Levenshtein opcodes:
        # tag: 'replace', 'insert', 'delete', 'equal'
        codes = opcodes(gt, pred)
        for tag, i1, i2, j1, j2 in codes:
            gt_sub = gt[i1:i2]
            pred_sub = pred[j1:j2]

            if tag == "replace":
                if len(gt_sub) == 1 and len(pred_sub) == 1:
                    key = f"{gt_sub}->{pred_sub}"
                    substitutions[key] += 1
                else:
                    key = f"{gt_sub}->{pred_sub}"
                    multi_char[key] += 1
            elif tag == "insert":
                insertions[pred_sub] += 1
                if len(pred_sub) > 1:
                    multi_char[f"->{pred_sub}"] += 1
            elif tag == "delete":
                deletions[gt_sub] += 1
                if len(gt_sub) > 1:
                    multi_char[f"{gt_sub}->"] += 1

    # Normalize substitutions to probabilities per GT character: P(pred | gt)
    gt_totals: Dict[str, int] = defaultdict(int)
    for key, count in substitutions.items():
        gt_char = key.split("->")[0]
        gt_totals[gt_char] += count

    probabilities: Dict[str, float] = {}
    for key, count in substitutions.items():
        gt_char = key.split("->")[0]
        total = gt_totals[gt_char]
        probabilities[key] = round(count / total, 4) if total > 0 else 0.0

    return {
        "substitutions": dict(sorted(substitutions.items(), key=lambda x: x[1], reverse=True)),
        "multi_char_patterns": dict(sorted(multi_char.items(), key=lambda x: x[1], reverse=True)),
        "insertions": dict(sorted(insertions.items(), key=lambda x: x[1], reverse=True)),
        "deletions": dict(sorted(deletions.items(), key=lambda x: x[1], reverse=True)),
        "probabilities": dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)),
    }


def build_confusion_matrix(
    input_files: List[Path],
    output_path: Path,
    include_seed: bool = True
) -> Dict[str, Any]:
    pairs: List[Tuple[str, str]] = []
    for path in input_files:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                gt = data.get("gt_text") or data.get("ground_truth", "")
                pred = data.get("pred_text") or data.get("prediction", "")
                if gt and pred:
                    pairs.append((gt, pred))

    matrix = align_and_count(pairs, include_seed=include_seed)
    matrix["metadata"] = {
        "source_file_hash": compute_file_hash(input_files),
        "pair_count": len(pairs),
        "seed_included": include_seed,
        "date": datetime.now().isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build OCR empirical confusion matrix")
    parser.add_argument("--data-dir", type=str, default="data/stage2_pairs")
    parser.add_argument("--output", type=str, default="data/confusion_matrix/char_confusion.json")
    parser.add_argument("--no-seed", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    in_files = list(data_dir.glob("*.jsonl"))
    out_file = Path(args.output)

    result = build_confusion_matrix(in_files, out_file, include_seed=not args.no_seed)
    print(f"Confusion matrix written to {out_file}")
    print(f"Pairs processed: {result['metadata']['pair_count']}")
    print(f"Total substitution patterns: {len(result['substitutions'])}")
