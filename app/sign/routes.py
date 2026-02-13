"""
Sign-to-text API routes.

Endpoints for video upload and sign language recognition.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import os
import tempfile

from .service import predict_video, predict_batch, load_model_instance

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Response Models ────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    predicted_word: str
    confidence: float
    top_k: List[Tuple[str, float]]


class SingleVideoResponse(BaseModel):
    success: bool
    result: PredictionResult = None
    error: str = None


class HealthResponse(BaseModel):
    service: str
    status: str
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/convert", response_model=SingleVideoResponse)
async def convert_sign_to_text(file: UploadFile = File(...)):
    """
    Upload a video and get sign language prediction.

    Parameters
    ----------
    file : UploadFile
        Video file (mp4, avi, etc.)

    Returns
    -------
    SingleVideoResponse
        Prediction result or error message
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Save uploaded file to temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"Processing uploaded file: {file.filename}")

        # Get prediction
        result = predict_video(tmp_path)

        # Clean up
        os.remove(tmp_path)

        if result is None:
            return SingleVideoResponse(
                success=False,
                error="Could not process video. Check video format and quality."
            )

        return SingleVideoResponse(
            success=True,
            result=PredictionResult(
                predicted_word=result['predicted_word'],
                confidence=result['confidence'],
                top_k=result['top_k']
            )
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        if 'tmp_path' in locals():
            try:
                os.remove(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns
    -------
    HealthResponse
        Service status and model availability
    """
    try:
        model = load_model_instance()
        model_loaded = model is not None
    except:
        model_loaded = False

    return HealthResponse(
        service="sign",
        status="ready" if model_loaded else "initializing",
        model_loaded=model_loaded
    )