"""
Script to generate test samples in tests/sample_images/
Creates clean canvas drawings and simulated photo samples for automated testing.
"""

import os
import io
from PIL import Image, ImageDraw

def generate_samples():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Canvas drawing sample (Digit 7)
    canvas_img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas_img)
    draw.line([(70, 70), (230, 70)], fill=(255, 255, 255), width=26)
    draw.line([(230, 70), (120, 240)], fill=(255, 255, 255), width=26)
    canvas_path = os.path.join(out_dir, "sample_canvas_7.png")
    canvas_img.save(canvas_path)
    print(f"Generated: {canvas_path}")

    # 2. Simulated photo sample (Digit 3)
    photo_img = Image.new("RGB", (320, 320), color=(245, 245, 248))
    draw2 = ImageDraw.Draw(photo_img)
    draw2.arc([90, 60, 230, 170], start=270, end=90, fill=(25, 30, 45), width=20)
    draw2.arc([90, 150, 230, 260], start=270, end=90, fill=(25, 30, 45), width=20)
    photo_path = os.path.join(out_dir, "sample_photo_3.jpg")
    photo_img.save(photo_path, quality=92)
    print(f"Generated: {photo_path}")

if __name__ == "__main__":
    generate_samples()
