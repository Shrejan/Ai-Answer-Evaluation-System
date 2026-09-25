# Stage 3 — Post-OCR Retrieval-Constrained Correction

Post-OCR correction pipeline for handwritten descriptive student answers evaluated in an automated grading system.
`Kraken (Segmentation) → Fine-tuned TrOCR (Stage 2) → Stage 3 (this repo) → Stage 4 (Answer Evaluation)`

## Setup and Installation

### 1. Requirements
- Python 3.9+ (or Python 3.11)
- Ollama with `qwen3:1.7b` installed:
  ```bash
  ollama pull qwen3:1.7b
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Environment & Ollama Smoke Test
```bash
python scripts/smoke_ollama.py
```
Expected output:
- Cold latency: ~6s
- Warm latency: ~250ms
- Valid JSON schema compliance
- Zero `<think>` tokens
