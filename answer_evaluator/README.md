# Answer Evaluator

A friendly, fast, and explainable system for grading descriptive answers (hand‑written or typed) using OCR + semantic evaluation.

Imagine you upload a photo of a student's handwritten answer and, a few moments later, you get a clear score plus a short explanation of what they got right, what they missed, and how to improve. That's what this project does — and it's intentionally simple so teachers, students, and developers can understand and extend it.

Why this exists
- Teachers need fast, repeatable feedback for free‑text answers.
- Traditional keyword checks miss meaning — this uses semantic embeddings to judge concepts, not just words.
- An LLM explains the numeric result in plain language so feedback is useful and actionable.

Quick TL;DR
1. OCR converts an image of a written answer into text.
2. A Sentence‑BERT model compares that text to a reference answer and computes:
   - Concept coverage (did the student cover each reference idea?)
   - Overall semantic similarity
   - Final 0–100 score (computed deterministically from the above)
3. A single LLM call (via OpenRouter) converts the results into a short JSON report: correct concepts, missing concepts, errors, and improvement suggestions.

What makes it different
- The numeric score is always computed by code (no hallucinated grades).
- The LLM only writes the human‑friendly explanation — it cannot change the score.
- Splits the task into easy-to-reason-about pieces: OCR → embeddings → score → explanation.

Contents
```
answer_evaluator/
├── main.py            # CLI + server entrypoint
├── app.py             # FastAPI wrapper for evaluation service
├── evaluator.py       # Core scoring logic (the single source of truth)
├── embeddings.py      # Loads SentenceTransformer, splits text, does cosine similarity
├── openrouter.py      # One LLM call to produce the human explanation
├── prompts.py         # Prompt template used for the LLM
├── config.py          # All configurable thresholds & model names
├── input/             # Place question/reference/student text files here for CLI mode
└── output/            # evaluation.json is written here after a run
```

Quick start (5 minutes)
1. Install dependencies

```bash
cd answer_evaluator
pip install -r requirements.txt
cp .env.example .env
```

2. Edit .env and set your OpenRouter key (only required for the LLM explanation):

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

3. Run locally using the bundled example files:

```bash
python main.py --cli
```

Or run the server and call it from the client UI:

```bash
python main.py       # starts the evaluation server (FastAPI)
# OCR server is a separate component (see repo for OCR code) and typically runs on another laptop or process.
```

What you get
- output/evaluation.json containing:
  - score (0–100)
  - similarity (0–1)
  - concept_coverage (0–1)
  - concepts: per-reference-sentence similarity + status (covered/partial/missing)
  - report: LLM-produced JSON with correct/missing concepts and improvement tips (or null + llm_error if the LLM call failed)

Architecture (plain English)
- OCR (image → text): Convert image to text using a handwriting OCR model. The repository includes an OCR pipeline that uses TrOCR.
- Evaluator (text → score): Split the instructor's reference into sentences (each sentence = a "concept"). Split the student's text into sentences. Use Sentence‑BERT to embed all sentences, match each reference concept to the best student sentence, and mark it covered/partial/missing. Combine concept coverage and overall similarity into a final score.
- Explainer (score → human JSON): One carefully crafted prompt is sent to an LLM which returns a small JSON explaining the result.

Helpful tips
- First run downloads the Sentence‑BERT model (~80MB). Subsequent runs are fast.
- For small, fast setups, `all-MiniLM-L6-v2` works great. Switch to larger models in `.env` for better accuracy.
- If you have a GPU, the code will use it automatically for faster embeddings and OCR.

Security & privacy
- Keep your OpenRouter API key on the server only (evaluation machine). Do not expose it to the browser or clients.
- If you run across untrusted networks, put servers behind HTTPS and add simple API authentication.

Extending this project
- Replace the OCR model or improve pre-processing to handle messy handwriting.
- Tune thresholds and weights in config.py to match your grading style.
- Add caching for reference embeddings if you evaluate many students on the same question.
- Containerize (Docker) each component and orchestrate with docker‑compose for reproducible deployments.

Troubleshooting
- Empty reference/student errors: make sure input files contain text (app validates these fields).
- OpenRouter failures: check OPENROUTER_API_KEY and network connectivity.
- OCR produces nonsense: tune preprocessing (deskew, CLAHE) or try a different OCR model.

Contributing
All contributions are welcome. Open an issue to discuss changes, or send a pull request — aim for clear commits and small, testable changes.

License & credits
- Small educational project. See repository root for license information.

If anything above is unclear or you want the README to include diagrams, a short demo GIF, or step‑by‑step screenshots for non‑technical users, tell me and I can add them.