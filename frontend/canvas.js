/**
 * PixelPen: canvas.js
 * High-performance pointer-event drawing canvas with undo stack, glowing stroke trail,
 * brush size adjustment, and debounced live prediction.
 */

class PixelPenCanvas {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });

    this.isDrawing = false;
    this.lastX = 0;
    this.lastY = 0;
    this.brushSize = options.brushSize || 24;
    this.undoStack = [];
    this.maxUndo = 15;
    this.hasDrawnAnything = false;

    // Callbacks
    this.onStrokeEnd = options.onStrokeEnd || null;
    this.onStrokeStart = options.onStrokeStart || null;

    this.initCanvasResolution();
    this.initListeners();
    this.saveState();
  }

  initCanvasResolution() {
    // Standardize internal canvas buffer to 300x300
    this.canvas.width = 300;
    this.canvas.height = 300;
    this.clear(false);
  }

  initListeners() {
    // Modern Pointer Events for Mouse, Touch & Stylus
    this.canvas.addEventListener("pointerdown", (e) => this.handlePointerDown(e));
    this.canvas.addEventListener("pointermove", (e) => this.handlePointerMove(e));
    this.canvas.addEventListener("pointerup", (e) => this.handlePointerUp(e));
    this.canvas.addEventListener("pointercancel", (e) => this.handlePointerUp(e));

    // Window resize handler to maintain aspect ratio
    window.addEventListener("resize", () => {
      // Internal coordinate system remains 300x300
    });
  }

  getPointerPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
      pressure: e.pressure && e.pressure > 0 ? e.pressure : 0.5
    };
  }

  handlePointerDown(e) {
    e.preventDefault();
    this.canvas.setPointerCapture(e.pointerId);
    this.isDrawing = true;
    const pos = this.getPointerPos(e);
    this.lastX = pos.x;
    this.lastY = pos.y;

    // Draw single dot on tap
    this.drawStroke(pos.x, pos.y, pos.x, pos.y, pos.pressure);
    this.hasDrawnAnything = true;

    if (this.onStrokeStart) {
      this.onStrokeStart();
    }
  }

  handlePointerMove(e) {
    if (!this.isDrawing) return;
    e.preventDefault();
    const pos = this.getPointerPos(e);
    this.drawStroke(this.lastX, this.lastY, pos.x, pos.y, pos.pressure);
    this.lastX = pos.x;
    this.lastY = pos.y;
  }

  handlePointerUp(e) {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    try {
      this.canvas.releasePointerCapture(e.pointerId);
    } catch (_) { }

    this.saveState();
    this.updateUndoButton();

    if (this.onStrokeEnd) {
      this.onStrokeEnd();
    }
  }

  drawStroke(x1, y1, x2, y2, pressure = 0.5) {
    this.ctx.save();
    // Chalkboard aesthetic: Thick white/cyan stroke on deep black
    this.ctx.strokeStyle = "#ffffff";
    this.ctx.fillStyle = "#ffffff";
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";

    // Dynamic width with slight pressure sensitivity (guarantees healthy stroke width for MNIST)
    const width = Math.max(16, this.brushSize * (0.9 + pressure * 0.2));
    this.ctx.lineWidth = width;

    // Outer subtle cyan glow
    this.ctx.shadowBlur = 4;
    this.ctx.shadowColor = "rgba(0, 240, 255, 0.5)";

    this.ctx.beginPath();
    this.ctx.moveTo(x1, y1);
    this.ctx.lineTo(x2, y2);
    this.ctx.stroke();

    // Secondary core pass for dense white ink
    this.ctx.shadowBlur = 0;
    this.ctx.beginPath();
    this.ctx.moveTo(x1, y1);
    this.ctx.lineTo(x2, y2);
    this.ctx.stroke();

    this.ctx.restore();
  }

  saveState() {
    if (this.undoStack.length >= this.maxUndo) {
      this.undoStack.shift();
    }
    const snapshot = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    this.undoStack.push(snapshot);
  }

  undo() {
    if (this.undoStack.length <= 1) return;
    // Discard current state
    this.undoStack.pop();
    const previous = this.undoStack[this.undoStack.length - 1];
    this.ctx.putImageData(previous, 0, 0);
    this.updateUndoButton();
    if (this.onStrokeEnd) {
      this.onStrokeEnd();
    }
  }

  clear(notify = true) {
    this.ctx.fillStyle = "#000000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.undoStack = [];
    this.saveState();
    this.hasDrawnAnything = false;
    this.updateUndoButton();
    if (notify && this.onStrokeEnd) {
      this.onStrokeEnd();
    }
  }

  updateUndoButton() {
    const undoBtn = document.getElementById("undoBtn");
    if (undoBtn) {
      undoBtn.disabled = this.undoStack.length <= 1;
    }
  }

  setBrushSize(size) {
    this.brushSize = parseInt(size, 10);
    const label = document.getElementById("brushSizeVal");
    if (label) label.textContent = `${this.brushSize}px`;
  }

  toBase64PNG() {
    return this.canvas.toDataURL("image/png");
  }

  isEmpty() {
    // Fast check: count non-black pixels
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
    let nonBlackCount = 0;
    for (let i = 0; i < imgData.length; i += 16) {
      if (imgData[i] > 20 || imgData[i + 1] > 20 || imgData[i + 2] > 20) {
        nonBlackCount++;
        if (nonBlackCount > 25) return false;
      }
    }
    return true;
  }

  loadSampleStroke(digit) {
    this.clear(false);
    this.ctx.save();
    this.ctx.strokeStyle = "#ffffff";
    this.ctx.fillStyle = "#ffffff";
    this.ctx.lineWidth = 24;
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.shadowBlur = 8;
    this.ctx.shadowColor = "rgba(0, 240, 255, 0.6)";

    const cx = 150;
    const cy = 150;

    // Draw stylized handwritten stroke vectors for test digits 0-9
    this.ctx.beginPath();
    switch (parseInt(digit, 10)) {
      case 0:
        this.ctx.ellipse(cx, cy, 50, 75, 0, 0, Math.PI * 2);
        break;
      case 1:
        this.ctx.moveTo(cx - 20, cy - 60);
        this.ctx.lineTo(cx + 5, cy - 75);
        this.ctx.lineTo(cx + 5, cy + 80);
        this.ctx.moveTo(cx - 30, cy + 80);
        this.ctx.lineTo(cx + 40, cy + 80);
        break;
      case 2:
        this.ctx.arc(cx - 5, cy - 35, 45, Math.PI, 0, false);
        this.ctx.lineTo(cx - 40, cy + 75);
        this.ctx.lineTo(cx + 50, cy + 75);
        break;
      case 3:
        this.ctx.arc(cx, cy - 35, 40, -Math.PI * 0.7, Math.PI * 0.5, false);
        this.ctx.arc(cx, cy + 35, 45, -Math.PI * 0.5, Math.PI * 0.8, false);
        break;
      case 4:
        this.ctx.moveTo(cx + 25, cy + 80);
        this.ctx.lineTo(cx + 25, cy - 70);
        this.ctx.lineTo(cx - 45, cy + 20);
        this.ctx.lineTo(cx + 50, cy + 20);
        break;
      case 5:
        this.ctx.moveTo(cx + 40, cy - 70);
        this.ctx.lineTo(cx - 30, cy - 70);
        this.ctx.lineTo(cx - 35, cy - 10);
        this.ctx.arc(cx, cy + 30, 48, -Math.PI * 0.6, Math.PI * 0.7, false);
        break;
      case 6:
        this.ctx.arc(cx, cy + 30, 48, 0, Math.PI * 2, false);
        this.ctx.moveTo(cx - 48, cy + 30);
        this.ctx.bezierCurveTo(cx - 45, cy - 40, cx - 10, cy - 75, cx + 25, cy - 75);
        break;
      case 7:
        this.ctx.moveTo(cx - 50, cy - 70);
        this.ctx.lineTo(cx + 45, cy - 70);
        this.ctx.lineTo(cx - 20, cy + 80);
        this.ctx.moveTo(cx - 15, cy);
        this.ctx.lineTo(cx + 15, cy);
        break;
      case 8:
        this.ctx.ellipse(cx, cy - 40, 38, 38, 0, 0, Math.PI * 2);
        this.ctx.ellipse(cx, cy + 40, 48, 48, 0, 0, Math.PI * 2);
        break;
      case 9:
        this.ctx.ellipse(cx, cy - 30, 45, 45, 0, 0, Math.PI * 2);
        this.ctx.moveTo(cx + 45, cy - 30);
        this.ctx.lineTo(cx + 45, cy + 20);
        this.ctx.bezierCurveTo(cx + 40, cy + 60, cx + 10, cy + 80, cx - 35, cy + 75);
        break;
      default:
        this.ctx.arc(cx, cy, 60, 0, Math.PI * 2);
    }
    this.ctx.stroke();
    this.ctx.restore();

    this.saveState();
    this.updateUndoButton();
    if (this.onStrokeEnd) {
      this.onStrokeEnd();
    }
  }
}

window.PixelPenCanvas = PixelPenCanvas;
