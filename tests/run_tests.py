"""
PixelPen Test Runner
Runs test suite using pure Python with detailed assertion reporting.
"""
import os
import sys
import io
import base64
import numpy as np
from PIL import Image

# Add backend/app to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app")))

from fastapi.testclient import TestClient
# pyrefly: ignore [missing-import]
from main import app
from preprocess import preprocess_digit, decode_image, shared_final_mnist_stage
from model_loader import model_loader

client = TestClient(app)

def create_synthetic_canvas_image(digit=7):
    img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    if digit == 7:
        draw.line([(80, 80), (220, 80)], fill=(255, 255, 255), width=24)
        draw.line([(220, 80), (120, 240)], fill=(255, 255, 255), width=24)
    elif digit == 0:
        draw.ellipse([90, 60, 210, 240], outline=(255, 255, 255), width=24)
    elif digit == 2:
        draw.arc([100, 60, 200, 160], start=180, end=0, fill=(255, 255, 255), width=22)
        draw.line([(200, 110), (100, 240)], fill=(255, 255, 255), width=22)
        draw.line([(90, 240), (210, 240)], fill=(255, 255, 255), width=22)
    elif digit == 3:
        draw.arc([90, 60, 210, 150], start=220, end=90, fill=(255, 255, 255), width=22)
        draw.arc([90, 150, 210, 240], start=270, end=140, fill=(255, 255, 255), width=22)


    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}", buf.getvalue()

def run_all_tests():
    passed = 0
    failed = 0
    test_names = []

    def check(name, condition, msg=""):
        nonlocal passed, failed
        test_names.append(name)
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} - {msg}")

    print("==================================================")
    print("Running PixelPen Comprehensive Test Suite")
    print("==================================================")

    # 1. Health check
    resp = client.get("/health")
    check("Health Endpoint 200 OK", resp.status_code == 200)
    data = resp.json()
    check("Health Status healthy", data.get("status") == "healthy")

    # 2. Canvas 7 Prediction
    b64, _ = create_synthetic_canvas_image(digit=7)
    resp = client.post("/predict", json={"image_base64": b64, "source": "canvas"})
    check("Predict 7 Status 200", resp.status_code == 200)
    data = resp.json()
    check("Predict 7 recognized as digit", data.get("is_digit") is True)
    check("Predict 7 predicted as 7 (NOT 4)", data.get("digit") == 7, f"Got: {data.get('digit')}")

    # 3. Canvas 0 Prediction
    b64_0, _ = create_synthetic_canvas_image(digit=0)
    resp = client.post("/predict", json={"image_base64": b64_0, "source": "canvas"})
    data = resp.json()
    check("Predict 0 predicted as 0", data.get("is_digit") is True and data.get("digit") == 0, f"Got: {data.get('digit')}")

    # 4. Canvas 2 Prediction
    b64_2, _ = create_synthetic_canvas_image(digit=2)
    resp = client.post("/predict", json={"image_base64": b64_2, "source": "canvas"})
    data = resp.json()
    check("Predict 2 predicted as 2", data.get("is_digit") is True and data.get("digit") == 2, f"Got: {data.get('digit')}")

    # 5. Canvas 3 Prediction
    b64_3, _ = create_synthetic_canvas_image(digit=3)
    resp = client.post("/predict", json={"image_base64": b64_3, "source": "canvas"})
    data = resp.json()
    check("Predict 3 predicted as 3", data.get("is_digit") is True and data.get("digit") == 3, f"Got: {data.get('digit')}")

    # 6. Blank non-digit photo rejection
    blank_img = Image.new("RGB", (300, 300), color=(245, 245, 245))
    buf = io.BytesIO()
    blank_img.save(buf, format="JPEG")
    b64_blank = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    resp = client.post("/predict", json={"image_base64": b64_blank, "source": "photo"})
    data = resp.json()
    check("Reject blank photo as non-digit", data.get("is_digit") is False and data.get("digit") == -1)

    # 7. Solid block non-digit rejection
    block_img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(block_img)
    draw.rectangle([80, 80, 220, 220], fill=(255, 255, 255))
    buf = io.BytesIO()
    block_img.save(buf, format="PNG")
    b64_block = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    resp = client.post("/predict", json={"image_base64": b64_block, "source": "canvas"})
    data = resp.json()
    check("Reject solid block as non-digit", data.get("is_digit") is False and data.get("digit") == -1)

    # 8. Feedback endpoint with digit_predicted: -1 (rejection feedback)
    resp = client.post("/feedback", json={
        "digit_predicted": -1,
        "digit_actual": 5,
        "image_base64": b64,
        "source": "canvas"
    })
    check("Feedback with predicted=-1 allowed", resp.status_code == 200, f"Status: {resp.status_code}")
    data = resp.json()
    check("Feedback response success is True", data.get("success") is True)

    # 9. Feedback endpoint with normal predicted digit
    resp = client.post("/feedback", json={
        "digit_predicted": 7,
        "digit_actual": 7,
        "image_base64": b64,
        "source": "canvas"
    })
    check("Feedback normal submission 200 OK", resp.status_code == 200)

    # 10. Pipeline stages verification
    batch, stages = preprocess_digit(b64, source="canvas")
    check("Pipeline stage original exists", "original" in stages)
    check("Pipeline stage grayscale exists", "grayscale" in stages)
    check("Pipeline stage thresholded exists", "thresholded" in stages)
    check("Pipeline stage cropped exists", "cropped" in stages)
    check("Pipeline stage final_28x28 exists", "final_28x28" in stages)

    print("==================================================")
    print(f"Results: {passed} PASSED, {failed} FAILED")
    print("==================================================")
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
