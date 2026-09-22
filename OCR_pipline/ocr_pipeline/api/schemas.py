"""Pydantic request/response models (doc 02_REQUEST_RESPONSE_CONTRACT)."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class OCRRequestMetadata(BaseModel):
    """JSON fields accompanying a multipart OCR upload."""

    request_id: Optional[str] = None
    page_id: Union[str, int]
    document_id: Optional[str] = None
    tenant_id: Optional[str] = None
    spell_correction: Optional[bool] = None


class WordRecord(BaseModel):
    id: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    line_id: Optional[int] = None
    bbox: Optional[List[int]] = None
    raw_text: Optional[str] = None
    corrected: Optional[bool] = None


class TimingsMs(BaseModel):
    decode: int = 0
    detection: int = 0
    cropping: int = 0
    recognition: int = 0
    spell_correction: int = 0
    total: int = 0


class ModelInfo(BaseModel):
    detector: str = "kraken-blla"
    recognizer: str = "microsoft/trocr-base-handwritten"
    spell_correction: str = "symspell"


class OCRSuccessResponse(BaseModel):
    request_id: str
    page_id: Union[str, int]
    text: str
    words: List[WordRecord]
    line_count: int
    recognized_line_count: int
    timings_ms: TimingsMs
    model: ModelInfo = Field(default_factory=ModelInfo)
    status: Literal["ok"] = "ok"


class ErrorDetail(BaseModel):
    type: Literal["invalid_input", "processing_error"]
    stage: Optional[str] = None
    message: str


class OCRErrorResponse(BaseModel):
    request_id: Optional[str] = None
    page_id: Optional[Union[str, int]] = None
    status: Literal["error"] = "error"
    error: ErrorDetail


class GpuHealthInfo(BaseModel):
    available: bool
    device: Optional[str] = None
    allocated_mb: Optional[float] = None
    reserved_mb: Optional[float] = None


class ModelsHealthInfo(BaseModel):
    detector: str = "kraken-blla"
    recognizer: str = "microsoft/trocr-base-handwritten"
    warmed_up: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"] = "ok"
    gpu: GpuHealthInfo
    models: ModelsHealthInfo
    queue_depth: int = 0
