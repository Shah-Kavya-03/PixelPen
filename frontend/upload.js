/**
 * PixelPen: upload.js
 * Manages photo upload, mobile camera capture, drag-and-drop,
 * scanning-line animations, and instant sample test chips.
 */

class PixelPenUpload {
  constructor(options = {}) {
    this.dropzone = document.getElementById("dropzone");
    this.fileInput = document.getElementById("fileInput");
    this.cameraInput = document.getElementById("cameraInput");
    this.scanLaser = document.getElementById("scanLaser");
    this.previewWrap = document.getElementById("uploadPreviewWrap");
    this.previewImg = document.getElementById("uploadPreviewImg");
    this.promptWrap = document.getElementById("dropzonePrompt");
    this.removeBtn = document.getElementById("removeUploadBtn");
    this.sampleChips = document.getElementById("sampleChips");

    this.onImageSelected = options.onImageSelected || null;
    this.currentImageBase64 = null;
    this.currentFile = null;

    this.initListeners();
    this.initSampleChips();
  }

  initListeners() {
    if (!this.dropzone) return;

    // Drag & Drop events
    ["dragenter", "dragover"].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.add("drag-active");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.remove("drag-active");
      });
    });

    this.dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        this.handleFile(files[0]);
      }
    });

    // File input change
    if (this.fileInput) {
      this.fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFile(e.target.files[0]);
        }
      });
    }

    // Camera input change
    if (this.cameraInput) {
      this.cameraInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFile(e.target.files[0]);
        }
      });
    }

    // Remove upload
    if (this.removeBtn) {
      this.removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.clearUpload();
      });
    }
  }

  handleFile(file) {
    if (!file.type.startsWith("image/")) {
      const feedbackMsg = document.getElementById("feedbackMsg");
      if (feedbackMsg) {
        feedbackMsg.textContent = "Please upload an image file (PNG, JPEG, WebP).";
        feedbackMsg.classList.remove("hidden");
        feedbackMsg.style.color = "var(--neon-pink)";
        setTimeout(() => {
          feedbackMsg.classList.add("hidden");
          feedbackMsg.style.color = "";
        }, 4000);
      }
      return;
    }

    this.currentFile = file;
    const reader = new FileReader();

    reader.onload = (e) => {
      this.currentImageBase64 = e.target.result;
      this.showPreview(this.currentImageBase64);

      if (this.onImageSelected) {
        this.onImageSelected({
          file: this.currentFile,
          base64: this.currentImageBase64,
          source: "photo"
        });
      }
    };

    reader.readAsDataURL(file);
  }

  showPreview(src) {
    if (this.previewImg) {
      this.previewImg.src = src;
    }
    if (this.promptWrap) this.promptWrap.classList.add("hidden");
    if (this.previewWrap) this.previewWrap.classList.remove("hidden");
  }

  clearUpload() {
    this.currentImageBase64 = null;
    this.currentFile = null;
    if (this.fileInput) this.fileInput.value = "";
    if (this.cameraInput) this.cameraInput.value = "";
    if (this.promptWrap) this.promptWrap.classList.remove("hidden");
    if (this.previewWrap) this.previewWrap.classList.add("hidden");
  }

  startScanningAnimation() {
    if (this.scanLaser) {
      this.scanLaser.classList.remove("hidden");
    }
  }

  stopScanningAnimation() {
    if (this.scanLaser) {
      this.scanLaser.classList.add("hidden");
    }
  }

  initSampleChips() {
    if (!this.sampleChips) return;
    this.sampleChips.innerHTML = "";

    // Generate chips 0 through 9
    for (let i = 0; i <= 9; i++) {
      const btn = document.createElement("button");
      btn.className = "chip-btn";
      btn.textContent = i;
      btn.title = `Test sample digit ${i}`;

      btn.addEventListener("click", () => {
        this.generateSyntheticSamplePhoto(i);
      });

      this.sampleChips.appendChild(btn);
    }
  }

  generateSyntheticSamplePhoto(digit) {
    // Generates a simulated photo of paper with clean ink digit
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 240;
    offCanvas.height = 240;
    const ctx = offCanvas.getContext("2d");

    // Clean paper background with gentle lighting gradient
    const grad = ctx.createLinearGradient(0, 0, 240, 240);
    grad.addColorStop(0, "#f8fafc");
    grad.addColorStop(1, "#edf2f7");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 240, 240);

    // Draw dark blue/black ink digit
    ctx.strokeStyle = "#0f172a";
    ctx.fillStyle = "#0f172a";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const cx = 120;
    const cy = 120;

    ctx.beginPath();
    switch (digit) {
      case 0:
        ctx.ellipse(cx, cy, 34, 52, 0, 0, Math.PI * 2);
        break;
      case 1:
        ctx.moveTo(cx, cy - 55);
        ctx.lineTo(cx, cy + 55);
        ctx.moveTo(cx - 16, cy - 40);
        ctx.lineTo(cx, cy - 55);
        break;
      case 2:
        ctx.arc(cx, cy - 25, 28, Math.PI, 0, false);
        ctx.lineTo(cx - 28, cy + 54);
        ctx.lineTo(cx + 30, cy + 54);
        break;
      case 3:
        ctx.arc(cx, cy - 25, 26, -Math.PI * 0.75, Math.PI * 0.5, false);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy + 25, 28, -Math.PI * 0.5, Math.PI * 0.75, false);
        break;
      case 4:
        ctx.moveTo(cx - 16, cy - 50);
        ctx.lineTo(cx - 24, cy + 14);
        ctx.lineTo(cx + 28, cy + 14);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx + 12, cy - 46);
        ctx.lineTo(cx + 12, cy + 54);
        break;
      case 5:
        ctx.moveTo(cx + 25, cy - 50);
        ctx.lineTo(cx - 20, cy - 50);
        ctx.lineTo(cx - 22, cy - 4);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy + 24, 30, -Math.PI * 0.65, Math.PI * 0.75, false);
        break;
      case 6:
        ctx.arc(cx, cy + 22, 32, 0, Math.PI * 2, false);
        ctx.moveTo(cx - 32, cy + 22);
        ctx.bezierCurveTo(cx - 30, cy - 20, cx - 12, cy - 55, cx + 15, cy - 55);
        break;
      case 7:
        ctx.moveTo(cx - 32, cy - 52);
        ctx.lineTo(cx + 30, cy - 52);
        ctx.lineTo(cx - 10, cy + 55);
        break;
      case 8:
        ctx.ellipse(cx, cy - 26, 24, 24, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(cx, cy + 26, 30, 30, 0, 0, Math.PI * 2);
        break;
      case 9:
        ctx.ellipse(cx, cy - 22, 28, 28, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx + 28, cy - 22);
        ctx.lineTo(cx + 26, cy + 20);
        ctx.bezierCurveTo(cx + 24, cy + 45, cx + 5, cy + 56, cx - 20, cy + 52);
        break;
    }
    ctx.stroke();

    const dataUrl = offCanvas.toDataURL("image/jpeg", 0.92);
    this.currentImageBase64 = dataUrl;
    this.currentFile = null;
    this.showPreview(dataUrl);

    if (this.onImageSelected) {
      this.onImageSelected({
        file: null,
        base64: dataUrl,
        source: "photo"
      });
    }
  }
}

window.PixelPenUpload = PixelPenUpload;
