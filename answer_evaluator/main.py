"""
Entry point.

Usage:
    python main.py
    python main.py --question input/question.txt --reference input/reference.txt --student input/student_answer.txt
"""

import argparse
import json
import sys
from pathlib import Path

import config
from evaluator import evaluate
from openrouter import generate_report, OpenRouterError


def read_text_file(path: str, label: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"Error: {label} file not found: {path}")
        sys.exit(1)

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        print(f"Error: {label} file is empty: {path}")
        sys.exit(1)

    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Fast descriptive-answer evaluator")
    parser.add_argument("--question", default=config.DEFAULT_QUESTION_PATH)
    parser.add_argument("--reference", default=config.DEFAULT_REFERENCE_PATH)
    parser.add_argument("--student", default=config.DEFAULT_STUDENT_PATH)
    parser.add_argument("--output", default=config.DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading embedding model...")
    # Model is loaded lazily on first use inside evaluate(), but we trigger
    # it here explicitly so the "Loading..." message and device print
    # happen before we start reading files/computing similarity.
    from embeddings import get_model
    get_model()

    print("Reading student answer...")
    question = read_text_file(args.question, "Question")
    reference_answer = read_text_file(args.reference, "Reference answer")
    student_answer = read_text_file(args.student, "Student answer")

    try:
        result = evaluate(question, reference_answer, student_answer)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print()
    print(f"Semantic similarity: {result['similarity']:.2f}")
    print(f"Concept coverage: {result['concept_coverage']:.2f}")
    print(f"Final score: {result['score']}/100")
    print()
    print("Generating evaluation report...")

    output = {
        "score": result["score"],
        "similarity": result["similarity"],
        "concept_coverage": result["concept_coverage"],
        "concepts": result["concepts"],
        "report": None,
    }

    try:
        report = generate_report(
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer,
            concepts_result=result["concepts"],
            final_score=result["score"],
        )
        output["report"] = report
    except OpenRouterError as exc:
        # The numerical evaluation must still succeed even if the LLM fails.
        output["llm_error"] = str(exc)
        print(f"Warning: LLM report generation failed ({exc}). "
              f"Numerical evaluation is still included below.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print("Evaluation completed.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
