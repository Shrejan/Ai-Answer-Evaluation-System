"""Bounded single-worker GPU inference queue (doc 06, 09)."""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Callable, Generic, Optional, TypeVar

from ocr_pipeline.config import QueueConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class GpuInferenceQueue(Generic[T, R]):
    """Serializes GPU work behind one consumer thread with a bounded queue."""

    def __init__(self, config: QueueConfig):
        self._config = config
        self._queue: queue.Queue = queue.Queue(maxsize=config.max_depth)
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="gpu-worker")
        self._started = False

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if not self._started:
            self._worker.start()
            self._started = True
            logger.info("GPU inference queue started (max_depth=%d)", self._config.max_depth)

    def stop(self) -> None:
        self._shutdown.set()
        self._queue.put(None)
        if self._started:
            self._worker.join(timeout=5)

    def submit(self, fn: Callable[[T], R], payload: T) -> Future:
        if not self._started:
            raise RuntimeError("GpuInferenceQueue not started")
        future: Future = Future()
        try:
            self._queue.put((fn, payload, future), block=True, timeout=30)
        except queue.Full as exc:
            future.set_exception(RuntimeError("OCR inference queue is full"))
            raise RuntimeError("OCR inference queue is full") from exc
        return future

    def _run(self) -> None:
        while not self._shutdown.is_set():
            item = self._queue.get()
            if item is None:
                break
            fn, payload, future = item
            try:
                result = fn(payload)
                future.set_result(result)
            except Exception as exc:
                logger.exception("GPU worker task failed")
                future.set_exception(exc)
            finally:
                self._queue.task_done()
