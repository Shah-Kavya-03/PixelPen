"""
PixelPen Backend - preprocess.py
Industrial-grade preprocessing engine for handwritten digits.
Implements two distinct paths feeding one shared final stage:
1. Canvas Path:
   - Direct grayscale conversion
   - Background polarity detection (white stroke on black vs black on white)
   - Stroke smoothing & cleaning
2. Photo Path (Phone camera snapshots / uploads):
   - EXIF orientation correction
   - Grayscale conversion
   - Gaussian blur denoising
   - Adaptive Gaussian thresholding (handles uneven shadows, vignetting, lighting gradients)
   - Background inversion (guarantees white digit on black background like MNIST)
   - Contour / foreground bounding box detection
3. Shared Final Stage (MNIST Standardization):
   - Aspect-ratio preserving square padding
   - High-quality downscaling to 20x20 pixels
   - Centering in standard 28x28 bounding frame (matching MNIST center-of-mass standard)
   - Normalization to float32 [0.0, 1.0] with shape (1, 28, 28, 1)

Robustness:
- Uses OpenCV (cv2) when installed.
- Gracefully falls back to Pillow + pure NumPy if cv2 is not yet installed in the environment.
- Generates base64 PNG previews for all 5 stages for the frontend 'What the model sees' panel.
"""

import io
import re
import base64
import logging
import numpy as np
from PIL import Image, ImageOps, ImageFilter

logger = logging.getLogger("pixelpen.preprocess")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False
    logger.info("OpenCV (cv2) not detected; using high-performance Pillow+NumPy fallback engine.")

def decode_image(image_input: bytes | str) -> np.ndarray:
    """
    Decodes an image from bytes or a base64 string (including data URI prefix).
    Handles EXIF orientation and converts to a standard RGB uint8 numpy array.
    """
    if isinstance(image_input, str):
        # Strip data URI prefix if present (e.g. data:image/png;base64,...)
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]
        image_bytes = base64.b64decode(image_input)
    else:
        image_bytes = image_input

    # Open with PIL to safely resolve EXIF orientation
    pil_img = Image.open(io.BytesIO(image_bytes))
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # Convert to RGB (dropping alpha channel if present with white background)
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        background.paste(pil_img, mask=pil_img.convert("RGBA").split()[3])
        pil_img = background.convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    return np.array(pil_img)

def encode_to_base64_png(image_array: np.ndarray) -> str:
    """
    Encodes a numpy array (grayscale or RGB) to a base64 PNG data URL string.
    """
    if image_array.dtype != np.uint8:
        # Scale float [0, 1] to uint8 [0, 255]
        image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)

    if HAS_CV2:
        success, buffer = cv2.imencode(".png", image_array)
        if success:
            b64_str = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"

    # Pure Pillow fallback
    pil_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def preprocess_canvas_path(img_rgb: np.ndarray):
    """
    Processes strokes drawn on the canvas.
    Canvas can be black chalkboard (white stroke) or white paper mode.
    """
    if HAS_CV2:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        corner_samples = [
            gray[0:10, 0:10],
            gray[0:10, w-10:w],
            gray[h-10:h, 0:10],
            gray[h-10:h, w-10:w]
        ]
        avg_corner_intensity = np.mean([np.mean(c) for c in corner_samples])
        if avg_corner_intensity > 127:
            binary = cv2.bitwise_not(gray)
        else:
            binary = gray.copy()

        blurred = cv2.GaussianBlur(binary, (3, 3), 0)
        # Threshold core stroke (exclude low-intensity blurred cyan glow beads)
        _, thresholded = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        # Dilate slightly so thin strokes match standard MNIST stroke width (2-3px at 28x28)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(thresholded, kernel, iterations=1)
        return gray, blurred, dilated
    else:
        pil_gray = Image.fromarray(img_rgb).convert("L")
        gray = np.array(pil_gray)
        h, w = gray.shape
        corners = [
            gray[0:10, 0:10],
            gray[0:10, w-10:w],
            gray[h-10:h, 0:10],
            gray[h-10:h, w-10:w]
        ]
        if np.mean([np.mean(c) for c in corners]) > 127:
            binary = 255 - gray
        else:
            binary = gray.copy()

        blurred_pil = Image.fromarray(binary).filter(ImageFilter.GaussianBlur(radius=1))
        blurred = np.array(blurred_pil)
        thresholded = np.where(blurred > 60, 255, 0).astype(np.uint8)
        dilated_pil = Image.fromarray(thresholded).filter(ImageFilter.MaxFilter(5))
        dilated = np.array(dilated_pil)
        return gray, blurred, dilated

def preprocess_photo_path(img_rgb: np.ndarray):
    """
    Processes real-world phone camera snapshots and photos.
    Handles desk shadows, textured paper, varying contrast, and pen widths.
    """
    if HAS_CV2:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Inverts so dark ink becomes white foreground
            blockSize=25,
            C=10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh_adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        return gray, denoised, cleaned
    else:
        pil_gray = Image.fromarray(img_rgb).convert("L")
        gray = np.array(pil_gray)
        denoised_pil = pil_gray.filter(ImageFilter.GaussianBlur(radius=2))
        denoised = np.array(denoised_pil)

        # Local adaptive threshold via local neighborhood box blur
        mean_local = np.array(denoised_pil.filter(ImageFilter.BoxBlur(radius=12)))
        thresh_adaptive = np.where(denoised < (mean_local.astype(np.int16) - 10), 255, 0).astype(np.uint8)

        # Clean noise
        pil_thresh = Image.fromarray(thresh_adaptive)
        cleaned_pil = pil_thresh.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        cleaned = np.array(cleaned_pil)
        return gray, denoised, cleaned

def extract_digit_bounding_box(binary_img: np.ndarray):
    """
    Finds the largest contour corresponding to the digit stroke.
    Crops the bounding box with padding.
    """
    # Zero out perimeter border (3px) to prevent adaptive threshold edge artifacts
    cleaned_bin = binary_img.copy()
    cleaned_bin[0:3, :] = 0
    cleaned_bin[-3:, :] = 0
    cleaned_bin[:, 0:3] = 0
    cleaned_bin[:, -3:] = 0

    h_orig, w_orig = cleaned_bin.shape

    if HAS_CV2:
        contours, _ = cv2.findContours(cleaned_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cy, cx = h_orig // 2, w_orig // 2
            size = min(h_orig, w_orig) // 2
            return cleaned_bin[max(0, cy - size):cy + size, max(0, cx - size):cx + size], None

        valid_contours = [
            c for c in contours 
            if cv2.contourArea(c) > 25 and (cv2.boundingRect(c)[2] < 0.95 * w_orig or cv2.boundingRect(c)[3] < 0.95 * h_orig)
        ]
        if not valid_contours:
            valid_contours = contours

        max_area = max(cv2.contourArea(c) for c in valid_contours)
        significant_contours = [c for c in valid_contours if cv2.contourArea(c) >= max_area * 0.15]

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for c in significant_contours:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        pad = 6
        x_min = max(0, int(x_min) - pad)
        y_min = max(0, int(y_min) - pad)
        x_max = min(w_orig, int(x_max) + pad)
        y_max = min(h_orig, int(y_max) + pad)

        cropped = cleaned_bin[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return cropped, bbox
    else:
        # Pure NumPy bounding box calculation with noise filtering
        cleaned_bin = binary_img.copy()
        cleaned_bin[0:3, :] = 0
        cleaned_bin[-3:, :] = 0
        cleaned_bin[:, 0:3] = 0
        cleaned_bin[:, -3:] = 0

        row_ink = np.sum(cleaned_bin > 30, axis=1)
        col_ink = np.sum(cleaned_bin > 30, axis=0)
        valid_rows = np.where(row_ink >= 3)[0]
        valid_cols = np.where(col_ink >= 3)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            cy, cx = h_orig // 2, w_orig // 2
            size = min(h_orig, w_orig) // 2
            return cleaned_bin[max(0, cy - size):cy + size, max(0, cx - size):cx + size], None

        y_min, y_max = int(valid_rows[0]), int(valid_rows[-1])
        x_min, x_max = int(valid_cols[0]), int(valid_cols[-1])

        pad = 6
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w_orig, x_max + pad + 1)
        y_max = min(h_orig, y_max + pad + 1)

        cropped = cleaned_bin[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return cropped, bbox

def validate_digit_image(img_rgb: np.ndarray, binary_img: np.ndarray, cropped_digit: np.ndarray, source: str = "canvas"):
    """
    Validates whether the preprocessed binary image and cropped region
    exhibit the characteristics of a single handwritten digit.
    Rejects:
    - Empty or nearly empty images
    - Highly cluttered/textured non-digit photos (scenery, faces, text documents)
    - Solid blocks or non-stroke geometry
    - Extreme aspect ratios or microscopic crops
    """
    h_orig, w_orig = binary_img.shape[:2]
    total_pixels = float(h_orig * w_orig)
    fg_pixels = float(np.count_nonzero(binary_img > 30))
    fg_ratio = fg_pixels / max(1.0, total_pixels)

    if fg_pixels < 25 or fg_ratio < 0.0015:
        return False, "No drawing or stroke detected. The image appears blank."

    if source.lower() == "photo":
        # Photos of single digits on paper should have clean paper background (>65%)
        # and moderate ink coverage (0.2% - 32%). Photos of rooms, nature, clutter exceed this.
        if fg_ratio > 0.32:
            return False, "No single handwritten digit detected. The photo is too busy, dark, or textured."

    ch, cw = cropped_digit.shape[:2]
    if ch < 8 or cw < 4:
        return False, "Detected stroke is too faint or small to be a digit."

    aspect_ratio = float(cw) / float(max(1, ch))
    if aspect_ratio > 5.0 or aspect_ratio < 0.08:
        return False, "Shape proportions do not resemble a handwritten digit."

    # Solidity / fill ratio inside the cropped box
    crop_fg = float(np.count_nonzero(cropped_digit > 30))
    crop_total = float(ch * cw)
    solidity = crop_fg / max(1.0, crop_total)
    if solidity > 0.85:
        return False, "Detected shape is a solid filled block, not a digit stroke."
    if solidity < 0.04:
        return False, "Detected stroke is too fragmented or sparse."

    return True, "Valid digit candidate"

def shared_final_mnist_stage(cropped_digit: np.ndarray) -> np.ndarray:
    """
    The MNIST Standardization Stage:
    1. Take the cropped bounding box.
    2. Resize to 20x20 preserving aspect ratio.
    3. Center inside 28x28 frame by bounding box (4-pixel border on 20x20 boundary).
    4. Normalize to float32 [0.0, 1.0].
    """
    h, w = cropped_digit.shape
    if h == 0 or w == 0:
        return np.zeros((28, 28, 1), dtype=np.float32)

    # Scale preserving aspect ratio into 20x20 box
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))

    if HAS_CV2:
        resized = cv2.resize(cropped_digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        pil_crp = Image.fromarray(cropped_digit)
        resized = np.array(pil_crp.resize((new_w, new_h), Image.Resampling.BILINEAR))

    # Center in 28x28 canvas by bounding box
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas28[top:top + new_h, left:left + new_w] = resized

    # Normalize to [0.0, 1.0] float32
    normalized = canvas28.astype(np.float32) / 255.0
    return np.expand_dims(normalized, -1)

def preprocess_digit(image_input: bytes | str, source: str = "canvas"):
    """
    Master preprocessing function.
    Dispatches to canvas or photo pipeline, executes shared MNIST final stage,
    validates candidate viability, and returns both model-ready array (1, 28, 28, 1)
    and base64 preview stages.
    """
    img_rgb = decode_image(image_input)

    # Dispatch to appropriate initial pipeline
    if source.lower() == "photo":
        gray, denoised, binary = preprocess_photo_path(img_rgb)
    else:
        gray, denoised, binary = preprocess_canvas_path(img_rgb)

    # Extract bounding box of digit
    cropped, bbox = extract_digit_bounding_box(binary)

    # Validate whether candidate is a valid digit
    is_valid, validation_msg = validate_digit_image(img_rgb, binary, cropped, source=source)

    # Draw bounding box on a visual copy for stage preview
    if HAS_CV2:
        bbox_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(bbox_preview, (x, y), (x + w, y + h), (0, 240, 255), 2)  # Neon cyan box
    else:
        bbox_preview = np.stack([binary, binary, binary], axis=-1)
        if bbox is not None:
            x, y, w, h = bbox
            y2 = min(bbox_preview.shape[0], y + h)
            x2 = min(bbox_preview.shape[1], x + w)
            bbox_preview[y:min(y+2, y2), x:x2] = [0, 240, 255]
            bbox_preview[max(0, y2-2):y2, x:x2] = [0, 240, 255]
            bbox_preview[y:y2, x:min(x+2, x2)] = [0, 240, 255]
            bbox_preview[y:y2, max(0, x2-2):x2] = [0, 240, 255]

    # Standardize to 28x28 MNIST format
    final_28x28_norm = shared_final_mnist_stage(cropped)
    # Shape for batch inference: (1, 28, 28, 1)
    batch_input = np.expand_dims(final_28x28_norm, 0)

    # Render 28x28 zoomed for inspection (scale up to 140x140 for display)
    if HAS_CV2:
        zoomed_28x28 = cv2.resize(
            (final_28x28_norm.squeeze() * 255.0).astype(np.uint8),
            (140, 140),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        pil_zoom = Image.fromarray((final_28x28_norm.squeeze() * 255.0).astype(np.uint8))
        zoomed_28x28 = np.array(pil_zoom.resize((140, 140), Image.Resampling.NEAREST))

    # Encode all stages into base64 PNGs for the 'What the model sees' UI panel
    stages = {
        "original": encode_to_base64_png(img_rgb),
        "grayscale": encode_to_base64_png(gray),
        "thresholded": encode_to_base64_png(binary),
        "cropped": encode_to_base64_png(bbox_preview),
        "final_28x28": encode_to_base64_png(zoomed_28x28),
        "is_valid_digit": is_valid,
        "validation_message": validation_msg,
    }

    return batch_input, stages
