"""TrOCR batched recognition with OOM backoff (doc 06)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocr_pipeline.config import RecognitionConfig
from ocr_pipeline.models import LineRecord, RecognitionStatus

logger = logging.getLogger(__name__)


def _slice_generation_output(outputs, index: int):
    """Extract single-sequence generation output from a batch result."""
    from types import SimpleNamespace

    seq = outputs.sequences[index : index + 1]
    return SimpleNamespace(sequences=seq, scores=outputs.scores)


@dataclass
class TrocrBundle:
    processor: TrOCRProcessor
    model: VisionEncoderDecoderModel
    device: torch.device
    warmed_up: bool = False


class TrocrEngine:
    def __init__(self, config: RecognitionConfig):
        self.config = config
        self._bundle: Optional[TrocrBundle] = None

    @property
    def bundle(self) -> TrocrBundle:
        if self._bundle is None:
            raise RuntimeError("TrOCR model not loaded")
        return self._bundle

    @property
    def warmed_up(self) -> bool:
        return self._bundle is not None and self._bundle.warmed_up

    def load(self) -> TrocrBundle:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading TrOCR on device: %s", device)
        processor = TrOCRProcessor.from_pretrained(self.config.model_id)
        model = VisionEncoderDecoderModel.from_pretrained(self.config.model_id)
        model.to(device)
        model.eval()
        if device.type == "cuda" and self.config.use_fp16_on_cuda:
            model.half()
        self._bundle = TrocrBundle(processor=processor, model=model, device=device)
        return self._bundle

    def warmup(self) -> None:
        bundle = self.bundle if self._bundle else self.load()
        dummy = Image.new("RGB", (384, 48), color=(255, 255, 255))
        self._run_batch([dummy], batch_size=1, bundle=bundle)
        bundle.warmed_up = True
        logger.info("TrOCR warm-up complete")

    def _run_batch(
        self,
        crops: List[Image.Image],
        batch_size: int,
        bundle: TrocrBundle,
    ):
        processor = bundle.processor
        model = bundle.model
        device = bundle.device

        pixel_values = processor(images=crops, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        if device.type == "cuda" and self.config.use_fp16_on_cuda:
            pixel_values = pixel_values.half()

        with torch.inference_mode():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=self.config.beam_width,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        return outputs

    def _decode_batch(self, outputs, bundle: TrocrBundle) -> List[str]:
        texts = bundle.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        return [t.strip() for t in texts]

    def recognize_lines_batched(
        self,
        ordered_lines: List[LineRecord],
        *,
        force_batch_size: Optional[int] = None,
    ) -> List[LineRecord]:
        bundle = self.bundle if self._bundle else self.load()
        pending = [
            line for line in ordered_lines if line.crop is not None and line.recognition_status == RecognitionStatus.PENDING
        ]
        if not pending:
            return ordered_lines

        batch_size = force_batch_size or self.config.batch_size
        idx = 0
        while idx < len(pending):
            chunk = pending[idx : idx + batch_size]
            crops = [line.crop for line in chunk]
            try:
                outputs = self._run_batch(crops, batch_size, bundle)
                texts = self._decode_batch(outputs, bundle)
                for i, line in enumerate(chunk):
                    text = texts[i]
                    line.raw_text = text
                    line.generation_output = _slice_generation_output(outputs, i)
                    if text:
                        line.recognized_text = text
                        line.recognition_status = RecognitionStatus.RECOGNIZED
                    else:
                        line.recognition_status = RecognitionStatus.RECOGNIZED_EMPTY
            except torch.cuda.OutOfMemoryError:
                if bundle.device.type == "cuda":
                    torch.cuda.empty_cache()
                if batch_size <= self.config.min_batch_size:
                    for line in chunk:
                        line.recognition_status = RecognitionStatus.DISCARDED_FAILED
                    raise RuntimeError(
                        "TrOCR inference failed after batch-size backoff retries"
                    ) from None
                batch_size = max(self.config.min_batch_size, batch_size // 2)
                logger.warning("CUDA OOM — retrying with batch_size=%d", batch_size)
                continue
            except Exception:
                logger.exception("TrOCR failed on batch")
                for line in chunk:
                    line.recognition_status = RecognitionStatus.DISCARDED_FAILED
            idx += len(chunk)
        return ordered_lines
