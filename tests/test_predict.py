"""
PixelPen Tests - test_predict.py
Automated test suite verifying:
- Canvas & Photo preprocessing pipelines
- MNIST 28x28 centering & normalization
- FastAPI /health, /predict (JSON and Multipart), and /feedback endpoints
- Input validation & error boundaries
"""

import os
import sys
import io
import base64
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Ensure backend/app is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app")))

from main import app
from preprocess import preprocess_digit, decode_image, shared_final_mnist_stage
from model_loader import model_loader

client = TestClient(app)

def create_synthetic_canvas_image():
    """Generates a synthetic white-stroke-on-black-background digit '7' image."""
    img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw a 7
    draw.line([(80, 80), (220, 80)], fill=(255, 255, 255), width=24)
    draw.line([(220, 80), (120, 240)], fill=(255, 255, 255), width=24)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}", buffered.getvalue()

def create_synthetic_photo_image():
    """Generates a synthetic dark-ink-on-light-paper digit '3' image with simulated lighting."""
    img = Image.new("RGB", (300, 300), color=(240, 240, 245))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw a 3
    draw.arc([100, 70, 200, 160], start=270, end=90, fill=(30, 30, 40), width=18)
    draw.arc([100, 140, 200, 230], start=270, end=90, fill=(30, 30, 40), width=18)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}", buffered.getvalue()

# ================= 1. PREPROCESSING TESTS =================

def test_decode_image_base64():
    b64, _ = create_synthetic_canvas_image()
    arr = decode_image(b64)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (300, 300, 3)
    assert arr.dtype == np.uint8

def test_canvas_preprocessing_shape_and_range():
    b64, _ = create_synthetic_canvas_image()
    batch, stages = preprocess_digit(b64, source="canvas")
    
    # Check shape: batch of 1, 28x28, single channel
    assert batch.shape == (1, 28, 28, 1)
    # Check normalization: values in [0.0, 1.0]
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0
    # Must have detected stroke mass
    assert batch.sum() > 0.5

    # Check stage previews are generated
    assert "original" in stages
    assert "grayscale" in stages
    assert "thresholded" in stages
    assert "cropped" in stages
    assert "final_28x28" in stages

def test_photo_preprocessing_shape_and_range():
    b64, _ = create_synthetic_photo_image()
    batch, stages = preprocess_digit(b64, source="photo")
    
    assert batch.shape == (1, 28, 28, 1)
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0
    assert batch.sum() > 0.5
    assert stages["final_28x28"].startswith("data:image/png;base64,")

def test_shared_final_mnist_stage_centering():
    # Empty box returns all zeros
    empty = np.zeros((0, 0), dtype=np.uint8)
    res = shared_final_mnist_stage(empty)
    assert res.shape == (28, 28, 1)
    assert res.sum() == 0.0

# ================= 2. FASTAPI ENDPOINT TESTS =================

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "device" in data

def test_predict_canvas_json():
    b64, _ = create_synthetic_canvas_image()
    response = client.post("/predict", json={
        "image_base64": b64,
        "source": "canvas"
    })
    assert response.status_code == 200
    data = response.json()
    assert "digit" in data
    assert 0 <= data["digit"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 10
    assert "latency_ms" in data
    assert data["source"] == "canvas"

def test_predict_photo_multipart():
    _, raw_bytes = create_synthetic_photo_image()
    response = client.post(
        "/predict",
        data={"source": "photo"},
        files={"file": ("photo_sample.jpg", raw_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "digit" in data
    assert 0 <= data["digit"] <= 9
    assert len(data["probabilities"]) == 10
    assert data["source"] == "photo"

def test_predict_rejects_empty_payload():
    response = client.post("/predict", json={"image_base64": "", "source": "canvas"})
    assert response.status_code == 400

def test_predict_rejects_non_image_file():
    dummy_text = b"Not an image file"
    response = client.post(
        "/predict",
        data={"source": "photo"},
        files={"file": ("test.txt", dummy_text, "text/plain")}
    )
    assert response.status_code == 400

def test_feedback_endpoint():
    b64, _ = create_synthetic_canvas_image()
    response = client.post("/feedback", json={
        "digit_predicted": 7,
        "digit_actual": 7,
        "image_base64": b64,
        "source": "canvas"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "sample_id" in data

# ================= 3. DIGITS 0, 2, 3 RECOGNITION & NON-DIGIT REJECTION TESTS =================

def test_predict_recognizes_zero():
    img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([90, 60, 210, 240], outline=(255, 255, 255), width=24)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    resp = client.post("/predict", json={"image_base64": b64, "source": "canvas"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_digit"] is True
    assert data["digit"] == 0

def test_predict_recognizes_two():
    img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.arc([100, 60, 200, 160], start=180, end=0, fill=(255, 255, 255), width=22)
    draw.line([(200, 110), (100, 240)], fill=(255, 255, 255), width=22)
    draw.line([(90, 240), (210, 240)], fill=(255, 255, 255), width=22)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    resp = client.post("/predict", json={"image_base64": b64, "source": "canvas"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_digit"] is True
    assert data["digit"] == 2

def test_predict_recognizes_three():
    img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.arc([90, 60, 210, 150], start=220, end=90, fill=(255, 255, 255), width=22)
    draw.arc([90, 150, 210, 240], start=270, end=140, fill=(255, 255, 255), width=22)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    resp = client.post("/predict", json={"image_base64": b64, "source": "canvas"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_digit"] is True
    assert data["digit"] == 3

def test_predict_rejects_non_digit_blank_photo():
    img = Image.new("RGB", (300, 300), color=(245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    resp = client.post("/predict", json={"image_base64": b64, "source": "photo"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_digit"] is False
    assert data["digit"] == -1
    assert data["confidence"] == 0.0

def test_predict_rejects_non_digit_solid_block():
    img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 220, 220], fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    resp = client.post("/predict", json={"image_base64": b64, "source": "canvas"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_digit"] is False
    assert data["digit"] == -1
    assert data["confidence"] == 0.0

def test_predict_recognizes_seven_and_not_four():
    b64, _ = create_synthetic_canvas_image()
    resp = client.post("/predict", json={"image_base64": b64, "source": "canvas"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_digit"] is True
    assert data["digit"] == 7
    assert data["probabilities"][4] < data["probabilities"][7]

def test_feedback_with_negative_one_predicted():
    b64, _ = create_synthetic_canvas_image()
    resp = client.post("/feedback", json={
        "digit_predicted": -1,
        "digit_actual": 7,
        "image_base64": b64,
        "source": "canvas"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "sample_id" in data

