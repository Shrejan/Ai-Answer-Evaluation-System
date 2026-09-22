"""Application entry point — load models, warm up, mount routes (doc 06, 09)."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from ocr_pipeline.api.routes import router
from ocr_pipeline.config import load_config
from ocr_pipeline.correction.protected_vocab import load_protected_vocab
from ocr_pipeline.correction.spell_correct import SpellCorrector
from ocr_pipeline.detection.kraken_loader import load_kraken_model
from ocr_pipeline.pipeline import PipelineServices
from ocr_pipeline.recognition.gpu_queue import GpuInferenceQueue
from ocr_pipeline.recognition.trocr_engine import TrocrEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(*, skip_model_load: bool = False) -> FastAPI:
    app = FastAPI(title="Handwritten OCR Pipeline", version="0.1.0")
    app.include_router(router)

    @app.on_event("startup")
    def startup() -> None:
        if getattr(app.state, "services", None) is not None:
            return
        config = load_config()
        trocr = TrocrEngine(config.recognition)
        gpu_queue = GpuInferenceQueue(config.queue)
        protected = load_protected_vocab(config.spell.protected_vocab_path)
        spell = SpellCorrector(config.spell, protected_vocab=protected)

        kraken_model = None
        if not skip_model_load:
            kraken_model = load_kraken_model(config.detection.model_path)
            trocr.load()
            trocr.warmup()

        gpu_queue.start()
        app.state.services = PipelineServices(
            config=config,
            trocr=trocr,
            gpu_queue=gpu_queue,
            spell=spell,
            kraken_model=kraken_model,
        )
        logger.info("OCR pipeline startup complete (warmed_up=%s)", trocr.warmed_up)

    @app.on_event("shutdown")
    def shutdown() -> None:
        services = getattr(app.state, "services", None)
        if services:
            services.gpu_queue.stop()

    return app


app = create_app()
