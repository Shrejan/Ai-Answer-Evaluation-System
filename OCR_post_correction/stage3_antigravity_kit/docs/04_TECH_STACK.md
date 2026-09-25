# 04 — Tech Stack and Environment

## Runtime
| Need | Choice | Note |
|---|---|---|
| Language | Python 3.11 | venv |
| LLM runtime | **Ollama** + `qwen3:1.7b` | replaces `bitsandbytes` from SPEC §7.1/§14 (see D-001) |
| HTTP/Ollama client | `ollama` Python package (or `httpx`) | one client, hard timeout |
| Spell candidates | `symspellpy` | edit distance 2, prefix 7 |
| Fuzzy/edit distance | `rapidfuzz` | fast Levenshtein, phonetic via `jellyfish` |
| Phrase retrieval | `scikit-learn` TF-IDF (default); `sentence-transformers` optional | TF-IDF first for latency |
| Alignment | `python-Levenshtein` opcodes or `difflib` | confusion matrix |
| Schemas | `pydantic>=2`, `pyyaml` | |
| Metrics | `jiwer` (CER/WER), own code for others | |
| Tests/lint | `pytest`, `pytest-cov`, `ruff` | |
| Data | `pandas` (reports only) | |

`requirements.txt`
```
pydantic>=2
pyyaml
symspellpy
rapidfuzz
jellyfish
scikit-learn
python-Levenshtein
ollama
httpx
jiwer
pandas
numpy
pytest
pytest-cov
ruff
```

## Ollama setup (laptop, RTX 2050 4 GB)
```
ollama pull qwen3:1.7b
ollama serve                      # if not already running
curl http://localhost:11434/api/chat -d "{\"model\":\"qwen3:1.7b\",\"stream\":false,\"think\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with {\\\"ok\\\":true}\"}],\"format\":\"json\"}"
```
A 1.7B Q4 model is ~1.4 GB and fits in 4 GB VRAM. Keep TrOCR (Stage 2) and Stage 3 from occupying the GPU at the same time during benchmarking, or report both configurations.

## `config/model.yaml` (replaces SPEC §14 version)
```yaml
backend: ollama
host: "http://localhost:11434"
model: "qwen3:1.7b"
think: false
temperature: 0.0
seed: 42
num_predict: 40
num_ctx: 2048
keep_alive: "30m"
timeout_s: 10
```
Other configs (`thresholds.yaml`, `symspell.yaml`) as in SPEC §14; add `context_window_tokens: 6`, `max_pool_size: 6`, `conf_threshold: 0.85` (detector, to be tuned).
