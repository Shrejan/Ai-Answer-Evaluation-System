"""HTTP endpoints (doc 02, 06)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ocr_pipeline.api.schemas import ErrorDetail, OCRErrorResponse, OCRRequestMetadata
from ocr_pipeline.diagnostics.health import build_health_response
from ocr_pipeline.pipeline import PipelineServices, run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_services(request: Request) -> PipelineServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "error": {
                    "type": "processing_error",
                    "stage": "startup",
                    "message": "OCR services not initialized",
                },
            },
        )
    return services


@router.get("/health")
def health(request: Request):
    services = _get_services(request)
    return build_health_response(
        services.config,
        warmed_up=services.trocr.warmed_up,
        queue_depth=services.gpu_queue.queue_depth,
    )


@router.post("/ocr")
async def ocr_page(
    request: Request,
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    page_id: Optional[str] = Form(None),
    request_id: Optional[str] = Form(None),
    spell_correction: Optional[bool] = Form(None),
):
    services = _get_services(request)

    meta: OCRRequestMetadata
    if metadata:
        try:
            meta = OCRRequestMetadata.model_validate(json.loads(metadata))
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=_invalid_input("metadata", str(exc)),
            ) from exc
    elif page_id is not None:
        meta = OCRRequestMetadata(page_id=page_id, request_id=request_id, spell_correction=spell_correction)
    else:
        raise HTTPException(
            status_code=422,
            detail=_invalid_input("page_id", "page_id is required"),
        )

    if not image.filename and not image.content_type:
        raise HTTPException(
            status_code=400,
            detail=_invalid_input("image", "image file is required"),
        )

    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=_invalid_input("image", f"failed to read upload: {exc}"),
        ) from exc

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail=_invalid_input("image", "empty image payload"),
        )

    req_id = meta.request_id or str(uuid.uuid4())

    try:
        result = run_pipeline(
            services,
            image_bytes,
            request_id=req_id,
            page_id=meta.page_id,
            spell_correction=meta.spell_correction,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=_invalid_input("image", str(exc)),
        ) from exc
    except RuntimeError as exc:
        logger.exception("Processing failure for request %s", req_id)
        body = OCRErrorResponse(
            request_id=req_id,
            page_id=meta.page_id,
            error=ErrorDetail(
                type="processing_error",
                stage="recognition",
                message=str(exc),
            ),
        )
        return JSONResponse(status_code=500, content=body.model_dump())
    except Exception as exc:
        logger.exception("Unexpected processing failure for request %s", req_id)
        body = OCRErrorResponse(
            request_id=req_id,
            page_id=meta.page_id,
            error=ErrorDetail(
                type="processing_error",
                stage="unknown",
                message=str(exc),
            ),
        )
        return JSONResponse(status_code=500, content=body.model_dump())


def _invalid_input(field: str, message: str) -> dict:
    return {
        "status": "error",
        "error": {
            "type": "invalid_input",
            "stage": field,
            "message": message,
        },
    }
