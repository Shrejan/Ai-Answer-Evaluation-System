"""
scripts/smoke_ollama.py — Smoke test calling qwen3:1.7b on local Ollama.
Verifies JSON schema enforcement, think=false, latency (cold vs warm), and absence of <think> tags.
"""

import json
import time
import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """You are an OCR correction verifier.
Your task is to select the most likely original word or phrase from the supplied candidate list.

Rules:
1. You may select ONLY from the candidate list.
2. Never generate a new word.
3. Never modify any other part of the sentence.
4. Never paraphrase.
5. Never improve grammar.
6. Never add information.
7. If the original OCR text is already plausible, select the original.
8. When uncertain, select the original.
9. Return only valid JSON. No explanation, no markdown, no extra text."""

USER_PROMPT = """Context (before): specialized
OCR text: denital
Context (after): and eye clinics

Candidate list: ["dental", "denital"]

Return exactly:
{"selected_candidate": "...", "confidence": 0.0}"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_candidate": {
            "type": "string",
            "enum": ["dental", "denital"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        }
    },
    "required": ["selected_candidate", "confidence"]
}


def run_smoke_test(runs: int = 5):
    print(f"=== Starting Ollama smoke test on {MODEL} ({runs} runs) ===")
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        "format": SCHEMA,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 40,
            "num_ctx": 2048
        }
    }

    latencies = []
    has_think = False

    with httpx.Client(timeout=30.0) as client:
        for i in range(runs):
            start = time.perf_counter()
            resp = client.post(OLLAMA_URL, json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            if resp.status_code != 200:
                print(f"Run {i+1} FAILED with status {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            content = data.get("message", {}).get("content", "")
            
            # Check for <think> tags
            if "<think>" in content or "</think>" in content:
                has_think = True

            # Validate JSON
            try:
                parsed = json.loads(content)
                status = "VALID JSON"
            except Exception as e:
                parsed = None
                status = f"INVALID JSON: {e}"

            run_type = "COLD" if i == 0 else f"WARM #{i}"
            print(f"[{run_type}] {elapsed_ms:.1f}ms | {status} | raw: {content}")

    print("\n=== Summary ===")
    print(f"Cold latency: {latencies[0]:.1f} ms")
    if len(latencies) > 1:
        avg_warm = sum(latencies[1:]) / len(latencies[1:])
        print(f"Avg Warm latency ({runs-1} runs): {avg_warm:.1f} ms")
    print(f"<think> text detected: {'YES (ERROR)' if has_think else 'NO (PASSED)'}")
    assert not has_think, "Output must NOT contain <think> text!"


if __name__ == "__main__":
    run_smoke_test(5)
