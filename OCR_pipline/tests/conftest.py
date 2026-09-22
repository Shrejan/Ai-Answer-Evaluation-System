"""Shared test fixtures."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline.config import load_config
from ocr_pipeline.correction.spell_correct import SpellCorrector
from ocr_pipeline.main import create_app
from ocr_pipeline.models import RecognitionStatus
from ocr_pipeline.pipeline import PipelineServices
from ocr_pipeline.recognition.gpu_queue import GpuInferenceQueue
from ocr_pipeline.recognition.trocr_engine import TrocrEngine


def _png_bytes(width: int, height: int, color=(255, 255, 255)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeTokenizer:
    all_special_ids = [0, 1]

    def decode(self, ids):
        return "hello"


@pytest.fixture
def blank_page_bytes():
    return _png_bytes(400, 300)


@pytest.fixture
def oversized_page_bytes():
    return _png_bytes(4000, 3000)


@pytest.fixture
def mock_services(monkeypatch):
    config = load_config()
    trocr = TrocrEngine(config.recognition)
    tokenizer = FakeTokenizer()
    trocr._bundle = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=tokenizer),
        model=MagicMock(),
        device=torch.device("cpu"),
        warmed_up=True,
    )
    queue = GpuInferenceQueue(config.queue)
    queue.start()
    spell = SpellCorrector(config.spell, protected_vocab=set())
    services = PipelineServices(config=config, trocr=trocr, gpu_queue=queue, spell=spell)

    def fake_detect(original, detection_config, segment_fn=None):
        h, w = original.shape[:2]
        if max(h, w) > 2048:
            return []
        return [[(20, 40), (300, 40), (300, 70), (20, 70)]]

    def fake_recognize(lines):
        for line in lines:
            if line.crop is None:
                continue
            line.raw_text = "hello world"
            line.recognized_text = "hello world"
            line.corrected_text = "hello world"
            line.recognition_status = RecognitionStatus.RECOGNIZED
            line.generation_output = SimpleNamespace(
                sequences=torch.tensor([[2, 3, 1]]),
                scores=[torch.tensor([[0.95, 0.05]])],
            )
        return lines

    monkeypatch.setattr("ocr_pipeline.pipeline.detect_lines", fake_detect)
    monkeypatch.setattr(trocr, "recognize_lines_batched", fake_recognize)
    yield services
    queue.stop()


@pytest.fixture
def test_client(mock_services):
    app = create_app(skip_model_load=True)
    with TestClient(app) as client:
        client.app.state.services = mock_services
        yield client
