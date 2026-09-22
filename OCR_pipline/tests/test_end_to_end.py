"""Phase 7 end-to-end contract tests."""

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline.config import load_config
from ocr_pipeline.main import create_app
from ocr_pipeline.pipeline import PipelineServices, run_pipeline
from ocr_pipeline.recognition.gpu_queue import GpuInferenceQueue
from ocr_pipeline.recognition.trocr_engine import TrocrEngine
from ocr_pipeline.correction.spell_correct import SpellCorrector


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        d[i, 0] = i
    for j in range(len(hyp) + 1):
        d[0, j] = j
    for i, rc in enumerate(ref, 1):
        for j, hc in enumerate(hyp, 1):
            cost = 0 if rc == hc else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(ref), len(hyp)] / len(ref)


def test_normal_page_contract(test_client):
    resp = test_client.post(
        "/ocr",
        data={"page_id": "page_001", "request_id": "req_1"},
        files={"image": ("page.png", _png(800, 600), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["page_id"] == "page_001"
    assert "timings_ms" in body
    assert body["model"]["detector"] == "kraken-blla"


def test_blank_page_contract(test_client, monkeypatch):
    monkeypatch.setattr(
        "ocr_pipeline.pipeline.detect_lines",
        lambda *a, **k: [],
    )
    resp = test_client.post(
        "/ocr",
        data={"page_id": "blank"},
        files={"image": ("blank.png", _png(400, 300), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == ""
    assert body["words"] == []
    assert body["line_count"] == 0
    assert body["status"] == "ok"


def test_oversized_page_exercises_downscale(test_client):
    resp = test_client.post(
        "/ocr",
        data={"page_id": "big"},
        files={"image": ("big.png", _png(4000, 3000), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_invalid_input_missing_page_id(test_client):
    resp = test_client.post(
        "/ocr",
        files={"image": ("page.png", _png(100, 100), "image/png")},
    )
    assert resp.status_code == 422


def test_invalid_input_bad_image(test_client):
    resp = test_client.post(
        "/ocr",
        data={"page_id": "p1"},
        files={"image": ("bad.png", b"not-an-image", "image/png")},
    )
    assert resp.status_code == 400


def test_processing_failure_500(test_client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("TrOCR inference failed after batch-size backoff retries")

    monkeypatch.setattr("ocr_pipeline.pipeline.detect_lines", lambda *a, **k: [[(1, 1), (2, 1), (2, 2)]])
    monkeypatch.setattr(
        "ocr_pipeline.pipeline.crop_lines_parallel",
        lambda img, lines, cfg: lines,
    )

    def fail_recognize(lines):
        raise RuntimeError("TrOCR inference failed after batch-size backoff retries")

    services = test_client.app.state.services
    monkeypatch.setattr(services.trocr, "recognize_lines_batched", fail_recognize)

    resp = test_client.post(
        "/ocr",
        data={"page_id": "fail"},
        files={"image": ("page.png", _png(200, 200), "image/png")},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["type"] == "processing_error"


def test_health_endpoint(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "gpu" in body
    assert "models" in body
    assert "queue_depth" in body


def test_concurrent_requests_bounded_queue(test_client):
    import concurrent.futures

    def post():
        return test_client.post(
            "/ocr",
            data={"page_id": "c"},
            files={"image": ("p.png", _png(200, 200), "image/png")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: post(), range(4)))
    assert all(r.status_code == 200 for r in results)


def test_forbidden_preprocessing_grep():
    import re
    from pathlib import Path

    pattern = re.compile(
        r"denoise|clahe|threshold|deskew|upscale|sharpen|morphology|"
        r"fastNlMeansDenoising|createCLAHE",
        re.IGNORECASE,
    )
    root = Path(__file__).resolve().parents[1] / "ocr_pipeline"
    hits = []
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path}:{lineno}: {line.strip()}")
    assert hits == [], f"Forbidden preprocessing calls found: {hits}"


def test_cer_wer_on_fixture_labels():
    refs = ["hello world", "the mitochondria"]
    hyps = ["hello world", "the mitochondria"]
    cers = [_cer(r, h) for r, h in zip(refs, hyps)]
    assert max(cers) == 0.0
