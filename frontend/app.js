/**
 * PixelPen: app.js
 * Master frontend controller:
 * - API communication (/predict, /health, /feedback)
 * - Result visualization (digit reveal, circular confidence gauge, probability bars)
 * - Web Audio API procedural sound synthesizer
 * - Dark/Light theme manager
 * - Prediction history strip
 * - Keyboard shortcuts & PWA installation
 */

class PixelPenApp {
  constructor() {
    this.apiUrl = localStorage.getItem("pixelpen_api_url") || this.detectDefaultApiUrl();
    this.isMuted = localStorage.getItem("pixelpen_muted") === "true";
    this.audioCtx = null;
    this.livePredictDebounce = null;
    this.lastPredictionData = null;
    this.history = JSON.parse(localStorage.getItem("pixelpen_history") || "[]");

    // UI Elements
    this.digitHero = document.getElementById("digitHero");
    this.confidenceVal = document.getElementById("confidenceVal");
    this.confidenceRingBar = document.getElementById("confidenceRingBar");
    this.latencyBadge = document.getElementById("latencyBadge");
    this.lowConfAlert = document.getElementById("lowConfAlert");
    this.probBarsGrid = document.getElementById("probBarsGrid");

    this.apiStatusDot = document.getElementById("apiStatusDot");
    this.apiStatusText = document.getElementById("apiStatusText");
    this.apiStatusBtn = document.getElementById("apiStatusBtn");

    this.soundToggleBtn = document.getElementById("soundToggleBtn");
    this.soundIcon = document.getElementById("soundIcon");
    this.themeToggleBtn = document.getElementById("themeToggleBtn");
    this.themeIcon = document.getElementById("themeIcon");

    this.pwaInstallBtn = document.getElementById("pwaInstallBtn");
    this.deferredPrompt = null;

    // Auto-Clear Timer elements
    this.autoClearProgress = document.getElementById("autoClearProgress");
    this.autoClearBar = document.getElementById("autoClearBar");
    this.autoClearText = document.getElementById("autoClearText");
    this.autoClearToggle = document.getElementById("autoClearToggle");
    this.autoClearInterval = null;

    // Sub-controllers
    this.pipeline = new PixelPenPipeline();
    this.initCanvas();
    this.initUpload();
    this.initGame();
    this.initProbBars();
    this.initTheme();
    this.initSound();
    this.initHistory();
    this.initListeners();
    this.initPWA();
    this.checkApiHealth();
  }

  detectDefaultApiUrl() {
    // If running on localhost:8000 (FastAPI), use origin; else default to http://localhost:8000
    if (window.location.port === "8000") {
      return window.location.origin;
    }
    return "http://localhost:8000";
  }

  // ================= CANVAS INITIALIZATION =================
  initCanvas() {
    this.canvasController = new PixelPenCanvas("digitCanvas", {
      brushSize: 22,
      onStrokeStart: () => {
        this.cancelAutoClearCountdown();
        this.playSound("brush");
      },
      onStrokeEnd: () => {
        const liveToggle = document.getElementById("livePredictToggle");
        if (liveToggle && liveToggle.checked && !this.canvasController.isEmpty()) {
          clearTimeout(this.livePredictDebounce);
          this.livePredictDebounce = setTimeout(() => {
            this.predictCanvas();
          }, 280);
        }
      }
    });

    const brushSlider = document.getElementById("brushSize");
    if (brushSlider) {
      brushSlider.addEventListener("input", (e) => {
        this.canvasController.setBrushSize(e.target.value);
      });
    }

    const clearBtn = document.getElementById("clearBtn");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        this.cancelAutoClearCountdown();
        this.canvasController.clear();
        this.resetResults();
        this.playSound("clear");
      });
    }

    const undoBtn = document.getElementById("undoBtn");
    if (undoBtn) {
      undoBtn.addEventListener("click", () => {
        this.cancelAutoClearCountdown();
        this.canvasController.undo();
        this.playSound("brush");
      });
    }

    const autoClearToggle = document.getElementById("autoClearToggle");
    if (autoClearToggle) {
      autoClearToggle.addEventListener("change", (e) => {
        if (!e.target.checked) {
          this.cancelAutoClearCountdown();
        }
      });
    }

    const predictBtn = document.getElementById("predictBtn");
    if (predictBtn) {
      predictBtn.addEventListener("click", () => {
        this.cancelAutoClearCountdown();
        this.predictCanvas();
      });
    }
  }

  // ================= UPLOAD INITIALIZATION =================
  initUpload() {
    this.uploadController = new PixelPenUpload({
      onImageSelected: (data) => {
        this.predictUpload(data);
      }
    });
  }

  // ================= GAME INITIALIZATION =================
  initGame() {
    this.gameController = new PixelPenGame({
      onChallengeSubmit: async (base64Img) => {
        const result = await this.sendPredictRequest({
          image_base64: base64Img,
          source: "canvas"
        });
        if (result) {
          this.displayPrediction(result, base64Img);
        }
        return result;
      },
      onPlaySound: (type) => this.playSound(type)
    });
  }

  // ================= PROBABILITY BARS SETUP =================
  initProbBars() {
    if (!this.probBarsGrid) return;
    this.probBarsGrid.innerHTML = "";

    for (let i = 0; i <= 9; i++) {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.id = `probRow${i}`;

      row.innerHTML = `
        <span class="prob-digit">${i}</span>
        <div class="prob-track">
          <div class="prob-fill" id="probFill${i}"></div>
        </div>
        <span class="prob-val" id="probVal${i}">0.0%</span>
      `;

      this.probBarsGrid.appendChild(row);
    }
  }

  // ================= GENERAL LISTENERS =================
  initListeners() {
    // Mode Navigation Tabs
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const mode = tab.dataset.tab;
        this.switchMode(mode);
      });
    });

    // API Modal Dialog
    const apiModal = document.getElementById("apiModal");
    if (this.apiStatusBtn && apiModal) {
      this.apiStatusBtn.addEventListener("click", () => {
        document.getElementById("apiUrlInput").value = this.apiUrl;
        apiModal.showModal();
      });
      document.getElementById("closeApiModalBtn").addEventListener("click", () => apiModal.close());
      document.getElementById("useLocalhostBtn").addEventListener("click", () => {
        document.getElementById("apiUrlInput").value = "http://localhost:8000";
      });
      document.getElementById("useLanBtn").addEventListener("click", () => {
        document.getElementById("apiUrlInput").value = `http://${window.location.hostname}:8000`;
      });
      document.getElementById("testApiBtn").addEventListener("click", () => this.testCustomApiUrl());
      document.getElementById("saveApiBtn").addEventListener("click", () => {
        const inputVal = document.getElementById("apiUrlInput").value.trim().replace(/\/$/, "");
        if (inputVal) {
          this.apiUrl = inputVal;
          localStorage.setItem("pixelpen_api_url", this.apiUrl);
          apiModal.close();
          this.checkApiHealth();
        }
      });
    }

    // Keyboard Shortcuts Dialog
    const shortcutsModal = document.getElementById("shortcutsModal");
    const shortcutsBtn = document.getElementById("shortcutsBtn");
    if (shortcutsBtn && shortcutsModal) {
      shortcutsBtn.addEventListener("click", () => shortcutsModal.showModal());
      document.getElementById("closeShortcutsModalBtn").addEventListener("click", () => shortcutsModal.close());
    }

    // Feedback Buttons
    const feedbackCorrectBtn = document.getElementById("feedbackCorrectBtn");
    const feedbackIncorrectBtn = document.getElementById("feedbackIncorrectBtn");
    const actualDigitSelect = document.getElementById("actualDigitSelect");

    if (feedbackCorrectBtn) {
      feedbackCorrectBtn.addEventListener("click", () => {
        if (!this.lastPredictionData) return;
        this.submitFeedback(this.lastPredictionData.digit, this.lastPredictionData.digit);
      });
    }

    if (feedbackIncorrectBtn && actualDigitSelect) {
      feedbackIncorrectBtn.addEventListener("click", () => {
        actualDigitSelect.classList.remove("hidden");
      });
      actualDigitSelect.addEventListener("change", (e) => {
        const actual = parseInt(e.target.value, 10);
        if (!isNaN(actual) && this.lastPredictionData) {
          this.submitFeedback(this.lastPredictionData.digit, actual);
          actualDigitSelect.classList.add("hidden");
        }
      });
    }

    // Keyboard Shortcuts Handling
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

      if (e.key === "Enter") {
        e.preventDefault();
        this.cancelAutoClearCountdown();
        this.predictCanvas();
      } else if (e.key.toLowerCase() === "c") {
        this.cancelAutoClearCountdown();
        this.canvasController.clear();
        this.resetResults();
        this.playSound("clear");
      } else if (e.key.toLowerCase() === "u") {
        this.cancelAutoClearCountdown();
        this.canvasController.undo();
        this.playSound("brush");
      } else if (e.key.toLowerCase() === "m") {
        this.toggleSound();
      } else if (e.key.toLowerCase() === "t") {
        this.toggleTheme();
      } else if (e.key === "1") {
        this.switchMode("canvas");
      } else if (e.key === "2") {
        this.switchMode("upload");
      } else if (e.key === "3") {
        this.switchMode("challenge");
      }
    });
  }

  switchMode(mode) {
    this.cancelAutoClearCountdown();
    document.querySelectorAll(".tab-btn").forEach((t) => {
      const isActive = t.dataset.tab === mode;
      t.classList.toggle("active", isActive);
      t.setAttribute("aria-selected", isActive);
    });

    document.querySelectorAll(".panel-mode").forEach((p) => {
      p.classList.remove("active");
    });

    const activePanel = document.getElementById(`mode${mode.charAt(0).toUpperCase() + mode.slice(1)}`);
    if (activePanel) {
      activePanel.classList.add("active");
    }
  }

  // ================= API PREDICTION CALLS =================
  async predictCanvas() {
    if (this.canvasController.isEmpty()) {
      return;
    }
    const base64Img = this.canvasController.toBase64PNG();
    const result = await this.sendPredictRequest({
      image_base64: base64Img,
      source: "canvas"
    });
    if (result) {
      this.displayPrediction(result, base64Img);
    }
  }

  async predictUpload(data) {
    this.uploadController.startScanningAnimation();
    let result = null;

    if (data.file) {
      const formData = new FormData();
      formData.append("file", data.file);
      formData.append("source", "photo");
      result = await this.sendPredictMultipart(formData);
    } else if (data.base64) {
      result = await this.sendPredictRequest({
        image_base64: data.base64,
        source: "photo"
      });
    }

    this.uploadController.stopScanningAnimation();

    if (result) {
      this.displayPrediction(result, data.base64);
    }
  }

  async sendPredictRequest(payload) {
    try {
      const response = await fetch(`${this.apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${response.status}`);
      }

      return await response.json();
    } catch (err) {
      console.error("[PixelPen API Error]:", err);
      this.handleApiError(err.message);
      return null;
    }
  }

  async sendPredictMultipart(formData) {
    try {
      const response = await fetch(`${this.apiUrl}/predict`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${response.status}`);
      }

      return await response.json();
    } catch (err) {
      console.error("[PixelPen API Error]:", err);
      this.handleApiError(err.message);
      return null;
    }
  }

  handleApiError(msg) {
    if (this.digitHero) {
      this.digitHero.textContent = "!";
      this.digitHero.style.color = "var(--neon-pink)";
    }
    if (this.latencyBadge) {
      this.latencyBadge.textContent = "Error";
    }
    this.playSound("error");

    const isNetworkError = !msg || msg.includes("Failed to fetch") || msg.includes("NetworkError");
    const feedbackMsg = document.getElementById("feedbackMsg");
    if (feedbackMsg) {
      feedbackMsg.textContent = isNetworkError
        ? `Could not reach PixelPen Backend API at ${this.apiUrl}. Ensure server is running.`
        : `API Error: ${msg}`;
      feedbackMsg.classList.remove("hidden");
      feedbackMsg.style.color = "var(--neon-pink)";
      setTimeout(() => {
        feedbackMsg.classList.add("hidden");
        feedbackMsg.style.color = "";
      }, 5000);
    }
  }

  // ================= RESULTS DISPLAY =================
  displayPrediction(data, inputThumb) {
    this.lastPredictionData = { ...data, inputThumb };
    const isDigit = data.is_digit !== false && data.digit !== undefined && data.digit !== null && Number(data.digit) >= 0;

    // Update Big Digit Reveal with bounce animation
    if (this.digitHero) {
      this.digitHero.textContent = isDigit ? data.digit : "?";
      this.digitHero.style.color = isDigit ? "var(--neon-cyan)" : "var(--neon-pink)";
      this.digitHero.style.transform = "scale(1.2)";
      setTimeout(() => {
        this.digitHero.style.transform = "scale(1)";
      }, 180);
    }

    // Update Confidence Gauge Meter
    const pct = isDigit ? Math.round(data.confidence * 100) : 0;
    if (this.confidenceVal) {
      this.confidenceVal.textContent = `${pct}%`;
    }
    if (this.confidenceRingBar) {
      const maxOffset = 314.16; // 2 * PI * 50
      const offset = maxOffset - (pct / 100) * maxOffset;
      this.confidenceRingBar.style.strokeDashoffset = offset;
    }

    // Latency
    if (this.latencyBadge) {
      this.latencyBadge.textContent = `⚡ ${data.latency_ms.toFixed(0)} ms`;
    }

    // Low confidence alert or non-digit rejection message
    if (this.lowConfAlert) {
      if (!isDigit) {
        this.lowConfAlert.classList.remove("hidden");
        const alertText = this.lowConfAlert.querySelector(".alert-text");
        if (alertText) {
          alertText.textContent = data.message || "No clear handwritten digit detected. Please upload or draw a single digit (0–9) on a clean background.";
        }
      } else {
        this.lowConfAlert.classList.toggle("hidden", data.confidence >= 0.6);
        const alertText = this.lowConfAlert.querySelector(".alert-text");
        if (alertText && data.confidence < 0.6) {
          alertText.textContent = "Confidence is low (<60%). Try centering the digit or redrawing with a thicker stroke.";
        }
      }
    }

    // Softmax probabilities bars
    if (data.probabilities && data.probabilities.length === 10) {
      for (let i = 0; i <= 9; i++) {
        const prob = isDigit ? data.probabilities[i] : 0.0;
        const row = document.getElementById(`probRow${i}`);
        const fill = document.getElementById(`probFill${i}`);
        const val = document.getElementById(`probVal${i}`);

        if (row && fill && val) {
          const barPct = (prob * 100).toFixed(1);
          fill.style.width = `${barPct}%`;
          val.textContent = `${barPct}%`;

          const isTop = isDigit && (i === data.digit);
          row.classList.toggle("top-guess", isTop);
        }
      }
    }

    // Update Vision Pipeline Strip
    if (data.pipeline_stages) {
      this.pipeline.updateStages(data.pipeline_stages);
    }

    // Reset feedback message
    const feedbackMsg = document.getElementById("feedbackMsg");
    if (feedbackMsg) feedbackMsg.classList.add("hidden");

    // Add to history strip only if valid digit
    if (isDigit) {
      this.addToHistory(data.digit, pct, inputThumb);
      this.playSound("predict");

      // Trigger Idle Timer Auto-Clear countdown if in canvas mode
      const isCanvasMode = document.getElementById("modeCanvas")?.classList.contains("active");
      if (isCanvasMode) {
        this.startAutoClearCountdown(2000);
      }
    } else {
      this.cancelAutoClearCountdown();
      this.playSound("clear");
    }
  }

  resetResults() {
    if (this.digitHero) this.digitHero.textContent = "?";
    if (this.confidenceVal) this.confidenceVal.textContent = "0%";
    if (this.confidenceRingBar) this.confidenceRingBar.style.strokeDashoffset = 314.16;
    if (this.latencyBadge) this.latencyBadge.textContent = "⚡ -- ms";
    if (this.lowConfAlert) this.lowConfAlert.classList.add("hidden");

    for (let i = 0; i <= 9; i++) {
      const row = document.getElementById(`probRow${i}`);
      const fill = document.getElementById(`probFill${i}`);
      const val = document.getElementById(`probVal${i}`);
      if (row) row.classList.remove("top-guess");
      if (fill) fill.style.width = "0%";
      if (val) val.textContent = "0.0%";
    }
    this.pipeline.resetStages();
  }

  // ================= AUTO-CLEAR COUNTDOWN LOGIC =================
  startAutoClearCountdown(delayMs = 2000) {
    const toggle = document.getElementById("autoClearToggle") || this.autoClearToggle;
    if (!toggle || !toggle.checked) return;
    this.cancelAutoClearCountdown();

    const progressEl = document.getElementById("autoClearProgress") || this.autoClearProgress;
    const barEl = document.getElementById("autoClearBar") || this.autoClearBar;
    const textEl = document.getElementById("autoClearText") || this.autoClearText;

    if (!progressEl || !barEl || !textEl) return;

    progressEl.classList.remove("hidden");
    const startTime = Date.now();

    const updateCountdown = () => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, delayMs - elapsed);
      const progressPct = Math.max(0, (remaining / delayMs) * 100);

      barEl.style.width = `${progressPct}%`;
      textEl.textContent = `Auto-clearing in ${(remaining / 1000).toFixed(1)}s`;

      if (remaining <= 0) {
        this.cancelAutoClearCountdown();
        if (this.canvasController && !this.canvasController.isEmpty()) {
          this.canvasController.clear();
          this.playSound("clear");
        }
      }
    };

    updateCountdown();
    this.autoClearInterval = setInterval(updateCountdown, 40);
  }

  cancelAutoClearCountdown() {
    if (this.autoClearInterval) {
      clearInterval(this.autoClearInterval);
      this.autoClearInterval = null;
    }
    const progressEl = document.getElementById("autoClearProgress") || this.autoClearProgress;
    const barEl = document.getElementById("autoClearBar") || this.autoClearBar;
    if (progressEl) {
      progressEl.classList.add("hidden");
    }
    if (barEl) {
      barEl.style.width = "100%";
    }
  }

  // ================= FEEDBACK SUBMISSION =================
  async submitFeedback(predicted, actual) {
    if (!this.lastPredictionData || !this.lastPredictionData.inputThumb) return;

    try {
      const response = await fetch(`${this.apiUrl}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit_predicted: predicted,
          digit_actual: actual,
          image_base64: this.lastPredictionData.inputThumb,
          source: this.lastPredictionData.source || "canvas"
        })
      });

      const res = await response.json();
      const feedbackMsg = document.getElementById("feedbackMsg");
      if (feedbackMsg) {
        feedbackMsg.textContent = res.message || "Feedback saved for retraining!";
        feedbackMsg.classList.remove("hidden");
      }
      this.playSound("fanfare");
    } catch (e) {
      console.warn("Feedback save failed:", e);
    }
  }

  // ================= HEALTH MONITORING =================
  async checkApiHealth() {
    if (this.apiStatusDot) this.apiStatusDot.className = "status-dot status-loading";
    if (this.apiStatusText) this.apiStatusText.textContent = "Connecting...";

    try {
      const resp = await fetch(`${this.apiUrl}/health`, { method: "GET" });
      if (resp.ok) {
        const data = await resp.json();
        if (this.apiStatusDot) this.apiStatusDot.className = "status-dot status-online";
        if (this.apiStatusText) this.apiStatusText.textContent = data.model_loaded ? "Model Ready" : "Simulation Mode";
      } else {
        throw new Error("Health check returned error");
      }
    } catch (_) {
      if (this.apiStatusDot) this.apiStatusDot.className = "status-dot status-offline";
      if (this.apiStatusText) this.apiStatusText.textContent = "Offline";
    }
  }

  async testCustomApiUrl() {
    const input = document.getElementById("apiUrlInput").value.trim().replace(/\/$/, "");
    const statusEl = document.getElementById("modalPingStatus");
    statusEl.textContent = "Testing connection...";
    try {
      const resp = await fetch(`${input}/health`);
      if (resp.ok) {
        const data = await resp.json();
        statusEl.textContent = `✅ Connected! Model: ${data.model_loaded ? "Loaded" : "Simulation"} (${data.device})`;
        statusEl.style.color = "var(--neon-emerald)";
      } else {
        statusEl.textContent = `❌ Server responded with code ${resp.status}`;
        statusEl.style.color = "var(--neon-pink)";
      }
    } catch (e) {
      statusEl.textContent = `❌ Connection failed: ${e.message}`;
      statusEl.style.color = "var(--neon-pink)";
    }
  }

  // ================= WEB AUDIO SYNTHESIZER =================
  initSound() {
    this.updateSoundIcon();
    if (this.soundToggleBtn) {
      this.soundToggleBtn.addEventListener("click", () => this.toggleSound());
    }
  }

  toggleSound() {
    this.isMuted = !this.isMuted;
    localStorage.setItem("pixelpen_muted", this.isMuted.toString());
    this.updateSoundIcon();
  }

  updateSoundIcon() {
    if (this.soundIcon) {
      this.soundIcon.textContent = this.isMuted ? "🔇" : "🔊";
    }
  }

  playSound(type) {
    if (this.isMuted) return;

    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      if (!this.audioCtx) {
        this.audioCtx = new AudioContext();
      }
      if (this.audioCtx.state === "suspended") {
        this.audioCtx.resume();
      }

      const now = this.audioCtx.currentTime;

      if (type === "brush") {
        // Soft white noise brush whoosh
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(320, now);
        osc.frequency.exponentialRampToValueAtTime(180, now + 0.08);
        gain.gain.setValueAtTime(0.04, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.08);

      } else if (type === "predict") {
        // High-tech futuristic crystal chime
        const freqs = [587.33, 880, 1174.66]; // D5, A5, D6
        freqs.forEach((f, i) => {
          const osc = this.audioCtx.createOscillator();
          const gain = this.audioCtx.createGain();
          osc.type = "sine";
          osc.frequency.setValueAtTime(f, now + i * 0.04);
          gain.gain.setValueAtTime(0.08, now + i * 0.04);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35 + i * 0.04);
          osc.connect(gain);
          gain.connect(this.audioCtx.destination);
          osc.start(now + i * 0.04);
          osc.stop(now + 0.35 + i * 0.04);
        });

      } else if (type === "clear") {
        // Downward sweep whoosh
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(440, now);
        osc.frequency.exponentialRampToValueAtTime(80, now + 0.15);
        gain.gain.setValueAtTime(0.09, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.15);

      } else if (type === "fanfare") {
        // Victory triad chord
        [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => {
          const osc = this.audioCtx.createOscillator();
          const gain = this.audioCtx.createGain();
          osc.type = "triangle";
          osc.frequency.setValueAtTime(f, now + i * 0.08);
          gain.gain.setValueAtTime(0.12, now + i * 0.08);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
          osc.connect(gain);
          gain.connect(this.audioCtx.destination);
          osc.start(now + i * 0.08);
          osc.stop(now + 0.5);
        });

      } else if (type === "error") {
        // Error buzzer
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(160, now);
        osc.frequency.setValueAtTime(120, now + 0.12);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.25);
      }
    } catch (_) { }
  }

  // ================= THEME MANAGER =================
  initTheme() {
    const savedTheme = localStorage.getItem("pixelpen_theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);
    this.updateThemeIcon(savedTheme);

    if (this.themeToggleBtn) {
      this.themeToggleBtn.addEventListener("click", () => this.toggleTheme());
    }
  }

  toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("pixelpen_theme", next);
    this.updateThemeIcon(next);
  }

  updateThemeIcon(theme) {
    if (this.themeIcon) {
      this.themeIcon.textContent = theme === "dark" ? "🌙" : "☀️";
    }
  }

  // ================= HISTORY STRIP =================
  initHistory() {
    this.renderHistory();
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    if (clearHistoryBtn) {
      clearHistoryBtn.addEventListener("click", () => {
        this.history = [];
        localStorage.removeItem("pixelpen_history");
        this.renderHistory();
      });
    }
  }

  addToHistory(digit, confidencePct, thumb) {
    const item = {
      digit,
      confidencePct,
      thumb: thumb || "",
      timestamp: Date.now()
    };
    this.history.unshift(item);
    if (this.history.length > 20) this.history.pop();
    localStorage.setItem("pixelpen_history", JSON.stringify(this.history));
    this.renderHistory();
  }

  renderHistory() {
    const historyStrip = document.getElementById("historyStrip");
    if (!historyStrip) return;

    if (this.history.length === 0) {
      historyStrip.innerHTML = '<div class="history-empty">Draw or upload a digit to build your history log.</div>';
      return;
    }

    historyStrip.innerHTML = "";
    this.history.forEach((h) => {
      const card = document.createElement("div");
      card.className = "history-item";
      card.innerHTML = `
        ${h.thumb ? `<img src="${h.thumb}" class="history-thumb" alt="digit">` : ''}
        <span class="history-digit">${h.digit}</span>
        <span class="history-conf">${h.confidencePct}%</span>
      `;
      historyStrip.appendChild(card);
    });
  }

  // ================= PWA INSTALLATION =================
  initPWA() {
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      this.deferredPrompt = e;
      if (this.pwaInstallBtn) {
        this.pwaInstallBtn.classList.remove("hidden");
        this.pwaInstallBtn.addEventListener("click", () => {
          this.deferredPrompt.prompt();
          this.deferredPrompt.userChoice.then(() => {
            this.deferredPrompt = null;
            this.pwaInstallBtn.classList.add("hidden");
          });
        });
      }
    });
  }
}

// Instantiate PixelPen Master Controller when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.pixelPenApp = new PixelPenApp();
});
