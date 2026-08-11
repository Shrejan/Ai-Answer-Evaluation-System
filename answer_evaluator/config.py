"""
All tunable settings live here. Change values in this file (or the .env
file for secrets/model names) rather than hunting through the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Sentence Transformer (local embedding model)
# ---------------------------------------------------------------------------
# Change this to test other models, e.g. "all-mpnet-base-v2".
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Concept coverage thresholds (cosine similarity, 0-1)
# ---------------------------------------------------------------------------
COVERED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Final score formula weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SIMILARITY_WEIGHT = 0.55
CONCEPT_WEIGHT = 0.45

# ---------------------------------------------------------------------------
# OpenRouter (LLM used only to write the explanation, never the score)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# File paths (defaults; can be overridden with CLI args)
# ---------------------------------------------------------------------------
DEFAULT_QUESTION_PATH = "input/question.txt"
DEFAULT_REFERENCE_PATH = "input/reference.txt"
DEFAULT_STUDENT_PATH = "input/student_answer.txt"
DEFAULT_OUTPUT_PATH = "output/evaluation.json"
