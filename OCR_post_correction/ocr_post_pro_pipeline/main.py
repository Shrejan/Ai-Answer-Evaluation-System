"""
ocr_post_pro_pipeline/main.py — FastAPI application exposing Stage 3 Post-OCR Correction Pipeline.

Usage:
    uvicorn ocr_post_pro_pipeline.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from stage3.pipeline import correct_page, process_ocr_request
from stage3.correction_log import log_correction_event


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_post_pro_pipeline")

app = FastAPI(
    title="Stage 3 Post-OCR Correction Pipeline API",
    description="Local Qwen3-1.7B constrained verification and post-OCR error correction server.",
    version="1.0.0",
)


class WordItem(BaseModel):
    id: int
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OCRCorrectionRequest(BaseModel):
    request_id: Optional[str] = Field(default="page_001", description="Unique request identifier")
    student_id: Optional[str] = Field(default=None, description="Student ID")
    question_id: Optional[str] = Field(default=None, description="Question ID")
    text: Optional[str] = Field(default=None, description="Raw OCR input text")
    words: Optional[List[WordItem]] = Field(default_factory=list, description="Word level confidence array")


@app.get("/")
@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ocr_post_pro_pipeline",
        "version": "1.0.0",
    }


@app.post("/correct")
@app.post("/")
def correct_ocr(request: OCRCorrectionRequest) -> Dict[str, Any]:
    """
    Processes a Stage 2 OCR JSON request through the Stage 3 correction pipeline.
    Appends audit log and returns Stage 3 result payload.
    """
    try:
        req_dict = request.model_dump()
        result_obj = correct_page(req_dict)

        # Log correction event
        log_correction_event(result_obj)

        response_dict = process_ocr_request(req_dict)
        return response_dict
    except Exception as e:
        logger.error(f"Error processing OCR correction request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage 3 correction error: {str(e)}",
        )
