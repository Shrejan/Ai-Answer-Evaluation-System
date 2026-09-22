"""Phase 3 recognition tests."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from ocr_pipeline.config import RecognitionConfig
from ocr_pipeline.models import LineRecord, RecognitionStatus, bounds_from_polygon
from ocr_pipeline.config import QueueConfig
from ocr_pipeline.recognition.gpu_queue import GpuInferenceQueue
from ocr_pipeline.recognition.trocr_engine import TrocrEngine


def _line(line_id: int, y: int) -> LineRecord:
    poly = [(10, y), (200, y), (200, y + 30), (10, y + 30)]
    bounds, vc, h = bounds_from_polygon(poly)
    crop = Image.new("RGB", (190, 30), (0, 0, 0))
    return LineRecord(
        line_id=line_id,
        polygon=poly,
        bounds=bounds,
        vertical_center=vc,
        estimated_height=h,
        crop=crop,
        recognition_status=RecognitionStatus.PENDING,
    )


def test_batch_success():
    engine = TrocrEngine(RecognitionConfig(batch_size=2))
    lines = [_line(0, 10), _line(1, 50)]

    mock_outputs = MagicMock()
    mock_outputs.sequences = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mock_outputs.scores = [torch.tensor([[0.9, 0.1], [0.8, 0.2]])]

    with patch.object(engine, "load") as load_mock, patch.object(
        engine, "_run_batch", return_value=mock_outputs
    ), patch.object(engine, "_decode_batch", return_value=["hello", "world"]):
        bundle = MagicMock()
        bundle.processor.tokenizer = MagicMock()
        load_mock.return_value = bundle
        engine._bundle = bundle
        result = engine.recognize_lines_batched(lines)

    assert result[0].raw_text == "hello"
    assert result[1].raw_text == "world"
    assert result[0].recognition_status == RecognitionStatus.RECOGNIZED


def test_oom_triggers_backoff():
    engine = TrocrEngine(RecognitionConfig(batch_size=4, min_batch_size=1))
    lines = [_line(0, 10), _line(1, 50), _line(2, 90)]

    call_sizes = []

    def fake_run(crops, batch_size, bundle):
        call_sizes.append(len(crops))
        if len(crops) > 1:
            raise torch.cuda.OutOfMemoryError("mock oom")
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1]])
        mock_outputs.scores = [torch.tensor([[0.9, 0.1]])]
        return mock_outputs

    with patch.object(engine, "load") as load_mock, patch.object(
        engine, "_run_batch", side_effect=fake_run
    ), patch.object(engine, "_decode_batch", return_value=["ok"]), patch(
        "torch.cuda.empty_cache"
    ):
        bundle = MagicMock()
        bundle.device.type = "cuda"
        load_mock.return_value = bundle
        engine._bundle = bundle
        engine.recognize_lines_batched(lines)

    assert 1 in call_sizes


def test_failed_crop_mid_batch_preserves_line_ids():
    engine = TrocrEngine(RecognitionConfig(batch_size=2))
    lines = [_line(0, 10), _line(1, 50)]
    lines[1].recognition_status = RecognitionStatus.DISCARDED_SMALL
    lines[1].crop = None

    mock_outputs = MagicMock()
    mock_outputs.sequences = torch.tensor([[1, 2]])
    mock_outputs.scores = [torch.tensor([[0.9, 0.1]])]

    with patch.object(engine, "load") as load_mock, patch.object(
        engine, "_run_batch", return_value=mock_outputs
    ), patch.object(engine, "_decode_batch", return_value=["only"]):
        bundle = MagicMock()
        load_mock.return_value = bundle
        engine._bundle = bundle
        result = engine.recognize_lines_batched(lines)

    assert result[0].line_id == 0
    assert result[1].line_id == 1
    assert result[0].raw_text == "only"
    assert result[1].recognition_status == RecognitionStatus.DISCARDED_SMALL


def test_gpu_queue_serializes_tasks():
    queue = GpuInferenceQueue(QueueConfig(max_depth=4, worker_count=1))
    queue.start()
    try:
        f1 = queue.submit(lambda x: x + 1, 1)
        f2 = queue.submit(lambda x: x + 2, 2)
        assert f1.result(timeout=5) == 2
        assert f2.result(timeout=5) == 4
    finally:
        queue.stop()
