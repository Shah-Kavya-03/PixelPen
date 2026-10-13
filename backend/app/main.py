"""
PixelPen Backend - main.py
FastAPI application entrypoint.
Configures CORS, lifespan model loading, logging, and error handling.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add current app directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes import router as api_router
from model_loader import model_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pixelpen.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan: Loads the model ONCE at application startup.
    Keeps model in memory across all requests.
    """
    logger.info("==========================================")
    logger.info("Initializing PixelPen AI Engine...")
    success = model_loader.load_model()
    if success:
        logger.info(f"Model successfully loaded from {model_loader.model_path}")
    else:
        logger.warning("Operating in fallback simulation mode until digit_cnn.keras is trained.")
    logger.info("==========================================")
    yield
    logger.info("PixelPen Backend shutting down...")

app = FastAPI(
    title="PixelPen API",
    description="Handwritten Digit Recognition AI Engine (CNN + Dual-Path Preprocessing)",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for local development and deployed frontend URLs
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler for uniform JSON responses
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )

# Include API routes
app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "app": "PixelPen: Handwritten Digit Recognition API",
        "tagline": "Draw it. Snap it. Let AI read it.",
        "status": "online",
        "model_loaded": model_loader.is_loaded,
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting PixelPen server at http://{host}:{port}")
    # Disable reload when running directly via python main.py to prevent Windows multiprocessing spawn recursion
    uvicorn.run("main:app", host=host, port=port, reload=False)

