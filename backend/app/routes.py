"""
PixelPen Backend - routes.py
FastAPI router providing:
- GET  /health   : API and model diagnostic status
- POST /predict  : Predicts digit from base64 JSON or multipart/form-data upload
- POST /feedback : Collects user corrections & verified samples for retraining
"""

import os
import time
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, status
from fastapi.responses import JSONResponse

from schemas import (
    PredictBase64Request,
    PredictionResponse,
    HealthResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from preprocess import preprocess_digit, decode_image
from model_loader import model_loader

logger = logging.getLogger("pixelpen.routes")
router = APIRouter()

MAX_IMAGE_SIZE_BYTES = 3 * 1024 * 1024  # 3 MB limit

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns server status, model readiness, and hardware acceleration device.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded,
        model_version="1.0.0",
        device=model_loader.device,
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_digit(request: Request):
    """
    Primary prediction endpoint.
    Accepts either:
    1. JSON body with base64 encoded image string: {"image_base64": "...", "source": "canvas"|"photo"}
    2. Multipart file upload via form-data: file=<binary image>, source="photo"
    """
    overall_start = time.time()
    img_data = None
    input_source = "canvas"

    content_type = request.headers.get("content-type", "").lower()

    # Handle multipart file upload (camera or file picker)
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        file = form.get("file")
        if file is not None and hasattr(file, "read"):
            # Validate MIME type
            file_content_type = getattr(file, "content_type", "") or ""
            filename = getattr(file, "filename", "") or ""
            if not (file_content_type.startswith("image/") or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type '{file_content_type}'. Please upload a PNG, JPEG, or WebP image."
                )
            img_bytes = await file.read()
            if len(img_bytes) > MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image size exceeds 3 MB limit (size: {len(img_bytes) / 1024 / 1024:.2f} MB)."
                )
            img_data = img_bytes
            source = form.get("source")
            input_source = str(source) if source else "photo"
        elif "image_base64" in form:
            b64 = form.get("image_base64")
            if not b64 or len(str(b64).strip()) < 10:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty or missing image_base64 data."
                )
            img_data = str(b64)
            input_source = str(form.get("source", "canvas"))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must provide either JSON with 'image_base64' or a multipart form-data 'file'."
            )

    # Handle JSON base64 payload (canvas or base64 drag-and-drop)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must provide either JSON with 'image_base64' or a multipart form-data 'file'."
            )

        if not isinstance(body, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must provide either JSON with 'image_base64' or a multipart form-data 'file'."
            )

        b64 = body.get("image_base64")
        if not b64 or not isinstance(b64, str) or len(b64.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty or missing image_base64 data."
            )
        img_data = b64
        input_source = body.get("source") or "canvas"

    try:
        # Run dual-path preprocessing and extract pipeline stages
        batch_array, stages = preprocess_digit(img_data, source=input_source)
        is_valid_digit = bool(stages.pop("is_valid_digit", True))
        val_msg = str(stages.pop("validation_message", ""))

        # If preprocessing determined image is not a digit candidate (blank, too busy, solid block)
        if not is_valid_digit:
            total_latency = (time.time() - overall_start) * 1000.0
            return PredictionResponse(
                is_digit=False,
                digit=-1,
                confidence=0.0,
                probabilities=[0.1] * 10,
                message=val_msg or "No handwritten digit detected.",
                pipeline_stages=stages,
                latency_ms=round(total_latency, 2),
                source=input_source,
            )

        # Run CNN / topological inference
        digit, confidence, probabilities, model_latency = model_loader.predict(batch_array)
        total_latency = (time.time() - overall_start) * 1000.0

        is_recognized = (digit >= 0 and confidence >= 0.35)
        response_msg = None if is_recognized else "Input does not clearly resemble any digit 0–9."

        return PredictionResponse(
            is_digit=is_recognized,
            digit=digit if is_recognized else -1,
            confidence=round(confidence if is_recognized else 0.0, 4),
            probabilities=[round(p, 4) for p in probabilities],
            message=response_msg,
            pipeline_stages=stages,
            latency_ms=round(total_latency, 2),
            source=input_source,
        )

    except Exception as e:
        logger.exception("Error during digit preprocessing or prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing or inference error: {str(e)}"
        )

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Saves user feedback / ground truth corrections to storage for model retraining.
    """
    feedback_dir = os.path.join(os.path.dirname(__file__), "..", "data", "feedback")
    os.makedirs(feedback_dir, exist_ok=True)

    sample_id = f"feedback_{request.digit_actual}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    filename = f"{sample_id}.png"
    filepath = os.path.join(feedback_dir, filename)

    try:
        # Decode and save image
        rgb_img = decode_image(request.image_base64)
        from PIL import Image
        pil_img = Image.fromarray(rgb_img)
        pil_img.save(filepath)

        # Log entry in metadata TSV
        meta_path = os.path.join(feedback_dir, "feedback_log.tsv")
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(f"{sample_id}\t{request.digit_actual}\t{request.digit_predicted}\t{request.source}\t{time.time()}\n")

        return FeedbackResponse(
            success=True,
            message="Feedback saved successfully. Thank you for helping train PixelPen!",
            sample_id=sample_id,
        )
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return FeedbackResponse(
            success=False,
            message=f"Could not persist feedback: {str(e)}",
            sample_id="",
        )
