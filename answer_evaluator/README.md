# Answer Evaluator

A small, fast, explainable module for grading handwritten descriptive
answers (after OCR turns them into `.txt` files).

```
OCR
 │
 ▼
student_answer.txt
 │
 ├──────────────► Sentence Transformer ──► Semantic Similarity ──► Numerical Score
 │
 └──────────────► OpenRouter LLM ──► Evaluation Report (Correct / Missing / Improvement)
```

The numerical score is always calculated by Python (Sentence-BERT +
cosine similarity). The LLM is used exactly once, only to explain the
result in plain language — it never sets or changes the score.

---

## 1. Project structure

```
answer_evaluator/
│
├── main.py          entry point / CLI
├── evaluator.py      scoring logic (the only place the score is computed)
├── embeddings.py      local Sentence-BERT model, splitting, cosine similarity
├── openrouter.py       single LLM call for the explanation report
├── prompts.py            prompt template
├── config.py               all tunable constants
│
├── input/
│   ├── question.txt
│   ├── reference.txt
│   └── student_answer.txt
│
├── output/
│   └── evaluation.json     (created after running)
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## 2. Install

```bash
cd answer_evaluator
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

The first run will download the Sentence-BERT model (`all-MiniLM-L6-v2`
by default, ~80MB) from Hugging Face and cache it locally. After that,
everything runs offline except the one OpenRouter call.

---

## 3. Run

Using the default files in `input/`:

```bash
python main.py
```

Or pointing at specific files:

```bash
python main.py \
    --question input/question.txt \
    --reference input/reference.txt \
    --student input/student_answer.txt
```

Example console output:

```
Loading embedding model...
Embedding device: cpu
Reading student answer...

Semantic similarity: 0.84
Concept coverage: 0.81
Final score: 82.4/100

Generating evaluation report...

Evaluation completed.
Output: output/evaluation.json
```

---

## 4. Example output (`output/evaluation.json`)

```json
{
  "score": 82.4,
  "similarity": 0.84,
  "concept_coverage": 0.81,
  "concepts": [
    {
      "concept": "A transformer transfers AC electrical energy from one circuit to another.",
      "similarity": 0.92,
      "status": "covered"
    },
    {
      "concept": "AC supplied to the primary winding creates changing magnetic flux.",
      "similarity": 0.89,
      "status": "covered"
    },
    {
      "concept": "The changing flux induces EMF in the secondary winding.",
      "similarity": 0.86,
      "status": "covered"
    },
    {
      "concept": "The voltage ratio depends on the turns ratio.",
      "similarity": 0.34,
      "status": "missing"
    }
  ],
  "report": {
    "correct_concepts": [
      "The student correctly explained electromagnetic induction and the changing magnetic flux."
    ],
    "missing_concepts": [
      "The student did not mention the relationship between voltage and turns ratio."
    ],
    "errors": [],
    "improvement": "Explain how the turns ratio between the primary and secondary windings determines the output voltage."
  }
}
```

If the LLM call fails for any reason (bad key, timeout, network issue,
invalid JSON), the numerical evaluation is still written out:

```json
{
  "score": 82.4,
  "similarity": 0.84,
  "concept_coverage": 0.81,
  "concepts": [ ... ],
  "report": null,
  "llm_error": "OpenRouter request failed: ..."
}
```

---

## 5. How the scoring formula works

**Step 1 — Split the reference answer into concepts.**
The reference answer is split into sentences (regex on `.`/`!`/`?`).
Each sentence is treated as one concept to check for.

**Step 2 — Split the student answer into sentences.**
Same splitting logic.

**Step 3 — Concept coverage.**
For every reference concept, compute its cosine similarity against
*every* student sentence and keep the maximum. That max similarity
determines a status:

| Similarity                        | Status  | Credit |
|-----------------------------------|---------|--------|
| `>= COVERED_THRESHOLD` (0.75)     | covered | 1.0    |
| `>= PARTIAL_THRESHOLD` (0.55)     | partial | 0.5    |
| below `PARTIAL_THRESHOLD`         | missing | 0.0    |

`concept_coverage` = average credit across all concepts (0–1).

**Step 4 — Overall similarity.**
Cosine similarity between the embedding of the full reference answer
and the full student answer (0–1).

**Step 5 — Final score.**

```python
final_score = (
    overall_similarity * SIMILARITY_WEIGHT   # 0.4
    + concept_coverage * CONCEPT_WEIGHT       # 0.6
) * 100
```

Concept coverage is weighted higher because it's a more direct measure
of "did the student cover the required points," while overall
similarity acts as a general sanity-check signal.

**Why not keyword matching?** Keyword overlap can't tell "changing
flux induces EMF" apart from "constant flux induces EMF" — same
keywords, opposite meaning. Semantic embeddings capture that
difference much better, and the LLM's error-detection step is there
specifically to flag conceptual mistakes like this one.

---

## 6. Changing the Sentence-BERT model

Edit `.env`:

```
EMBEDDING_MODEL=all-mpnet-base-v2
```

or edit the default directly in `config.py`:

```python
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
```

Any model supported by `sentence-transformers` works. Larger models
(e.g. `all-mpnet-base-v2`) are typically more accurate but slower;
`all-MiniLM-L6-v2` is the fast default.

---

## 7. Changing the OpenRouter model

Edit `.env`:

```
OPENROUTER_MODEL=anthropic/claude-3.5-haiku
```

Any model string supported by OpenRouter works — no code changes
needed.

---

## 8. Tuning thresholds and weights

All in `config.py`:

```python
COVERED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.55

SIMILARITY_WEIGHT = 0.4
CONCEPT_WEIGHT = 0.6
```

Adjust these after testing on your own dataset — there's nothing else
in the codebase that depends on their exact values.

---

## 9. GPU support

If an NVIDIA GPU with CUDA is available, it's used automatically:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The program prints which device it's using (`Embedding device: cuda`
or `Embedding device: cpu`) and works fine on CPU-only machines.

---

## 10. Important limitation

Sentence-BERT similarity is a **semantic similarity signal**, not a
perfect conceptual-correctness detector. It's good at catching
paraphrases and missing content, but it can occasionally miss subtle
factual errors that require real understanding (e.g. a single negated
word). That's exactly why the numerical score comes only from the
embedding pipeline, while explanation and light error-flagging come
from the LLM — keeping the system fast, cheap, and easy to reason
about, while acknowledging its limits.

Design priorities, in order: **accuracy, speed, simplicity, low API
cost, ease of modification.** No agents, no vector databases, no RAG,
no orchestration frameworks — just a few small Python files.
