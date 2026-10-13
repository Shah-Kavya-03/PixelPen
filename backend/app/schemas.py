"""
PixelPen Backend - schemas.py
Pydantic data schemas for request validation, structured responses, and pipeline tracking.
"""

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field

class PredictBase64Request(BaseModel):
    """
    Base64 image payload from canvas or file upload.
    Accepts data URLs (e.g. data:image/png;base64,...) or raw base64.
    """
    image_base64: str = Field(..., description="Base64 encoded image string")
    source: Literal["canvas", "photo"] = Field("canvas", description="Source of the digit image: 'canvas' or 'photo'")

class PipelineStages(BaseModel):
    """
    Intermediate visualization images encoded as base64 PNGs for the 'What the model sees' UI panel.
    """
    original: Optional[str] = Field(None, description="Original input image resized for preview")
    grayscale: Optional[str] = Field(None, description="Grayscale & denoised stage")
    thresholded: Optional[str] = Field(None, description="Binary threshold stage (adaptive/otsu)")
    cropped: Optional[str] = Field(None, description="Extracted bounding box of the digit")
    final_28x28: Optional[str] = Field(None, description="Centered 28x28 MNIST-normalized digit grid")

class PredictionResponse(BaseModel):
    """
    Full prediction result with softmax probabilities across 0-9 and pipeline images.
    """
    is_digit: bool = Field(True, description="Whether the image was recognized as a valid single digit")
    digit: int = Field(..., description="Top predicted digit (0-9) or -1 if not recognized as a digit")
    confidence: float = Field(..., description="Softmax confidence of top predicted digit (0.0 to 1.0)")
    probabilities: List[float] = Field(..., description="Array of 10 probabilities for digits 0 through 9")
    message: Optional[str] = Field(None, description="Diagnostic status or rejection message")
    pipeline_stages: Optional[Dict[str, str]] = Field(None, description="Base64 PNGs of each preprocessing stage")
    latency_ms: float = Field(..., description="Inference & preprocessing latency in milliseconds")
    source: str = Field("canvas", description="Source format processed")

class HealthResponse(BaseModel):
    """
    API health check response.
    """
    status: str = Field("ok", description="Server status")
    model_loaded: bool = Field(..., description="Whether the CNN model is loaded into memory")
    model_version: str = Field("1.0.0", description="PixelPen model and architecture version")
    device: str = Field("cpu", description="Execution device (CPU or GPU)")

class FeedbackRequest(BaseModel):
    """
    Feedback submission to store misclassified or verified user samples for retraining.
    """
    digit_predicted: int = Field(..., ge=-1, le=9)
    digit_actual: int = Field(..., ge=0, le=9)
    image_base64: str = Field(...)
    source: Literal["canvas", "photo"] = "canvas"

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    sample_id: Optional[str] = None
