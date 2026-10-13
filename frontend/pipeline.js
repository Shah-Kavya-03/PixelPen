/**
 * PixelPen: pipeline.js
 * Controls the 'What the model sees' vision pipeline strip and interactive pixel inspection.
 */

class PixelPenPipeline {
  constructor() {
    this.stageOriginal = document.getElementById("stageOriginal");
    this.stageGray = document.getElementById("stageGray");
    this.stageThreshold = document.getElementById("stageThreshold");
    this.stageCropped = document.getElementById("stageCropped");
    this.stageFinal28 = document.getElementById("stageFinal28");
    this.finalPixelBox = document.getElementById("finalPixelBox");

    this.placeholderSvg = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='60' height='60'><rect width='60' height='60' fill='%23111827'/></svg>";

    this.initHoverInspection();
  }

  updateStages(stages) {
    if (!stages) return;

    if (stages.original && this.stageOriginal) {
      this.animateStageUpdate(this.stageOriginal, stages.original);
    }
    if (stages.grayscale && this.stageGray) {
      this.animateStageUpdate(this.stageGray, stages.grayscale);
    }
    if (stages.thresholded && this.stageThreshold) {
      this.animateStageUpdate(this.stageThreshold, stages.thresholded);
    }
    if (stages.cropped && this.stageCropped) {
      this.animateStageUpdate(this.stageCropped, stages.cropped);
    }
    if (stages.final_28x28 && this.stageFinal28) {
      this.animateStageUpdate(this.stageFinal28, stages.final_28x28);
    }
  }

  animateStageUpdate(imgElement, src) {
    imgElement.style.opacity = "0.3";
    imgElement.style.transform = "scale(0.92)";
    setTimeout(() => {
      imgElement.src = src;
      imgElement.style.opacity = "1";
      imgElement.style.transform = "scale(1)";
    }, 120);
  }

  resetStages() {
    [this.stageOriginal, this.stageGray, this.stageThreshold, this.stageCropped, this.stageFinal28].forEach((el) => {
      if (el) el.src = this.placeholderSvg;
    });
  }

  initHoverInspection() {
    if (!this.finalPixelBox) return;

    this.finalPixelBox.addEventListener("mouseenter", () => {
      this.finalPixelBox.style.transform = "scale(1.2)";
      this.finalPixelBox.style.zIndex = "10";
      this.finalPixelBox.style.boxShadow = "0 0 25px rgba(0, 240, 255, 0.7)";
    });

    this.finalPixelBox.addEventListener("mouseleave", () => {
      this.finalPixelBox.style.transform = "scale(1)";
      this.finalPixelBox.style.zIndex = "1";
      this.finalPixelBox.style.boxShadow = "none";
    });
  }
}

window.PixelPenPipeline = PixelPenPipeline;
