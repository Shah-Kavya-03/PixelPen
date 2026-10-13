/**
 * PixelPen: game.js
 * 'Digit Challenge' arcade mini-game.
 * Challenges players to quickly draw specified digits, scoring based on CNN confidence & speed.
 * Completely eliminates intrusive browser alert() dialogs, rendering rich feedback
 * in the main Results Panel and inline animated challenge status banners.
 */

class PixelPenGame {
  constructor(options = {}) {
    this.targetDigitEl = document.getElementById("targetDigit");
    this.timerEl = document.getElementById("challengeTimer");
    this.streakEl = document.getElementById("challengeStreak");
    this.highScoreEl = document.getElementById("challengeHighScore");
    this.startBtn = document.getElementById("startChallengeBtn");
    this.clearBtn = document.getElementById("clearChallengeBtn");
    this.submitBtn = document.getElementById("submitChallengeBtn");

    // Inline Feedback Banner elements
    this.banner = document.getElementById("challengeBanner");
    this.bannerIcon = document.getElementById("challengeBannerIcon");
    this.bannerTitle = document.getElementById("challengeBannerTitle");
    this.bannerDesc = document.getElementById("challengeBannerDesc");

    // Results Card Challenge Badge
    this.resultBadge = document.getElementById("challengeResultBadge");
    this.resultBadgeIcon = document.getElementById("challengeBadgeIcon");
    this.resultBadgeText = document.getElementById("challengeBadgeText");

    this.onChallengeSubmit = options.onChallengeSubmit || null;
    this.onPlaySound = options.onPlaySound || null;

    this.targetDigit = null;
    this.timer = 10;
    this.timerInterval = null;
    this.streak = 0;
    this.highScore = parseInt(localStorage.getItem("pixelpen_high_score") || "0", 10);
    this.isActive = false;

    this.initCanvas();
    this.initListeners();
    this.updateStatsDisplay();
  }

  initCanvas() {
    this.challengeCanvas = new PixelPenCanvas("challengeCanvas", {
      brushSize: 22,
      onStrokeEnd: () => {
        if (this.isActive && this.submitBtn) {
          this.submitBtn.disabled = this.challengeCanvas.isEmpty();
        }
      }
    });
  }

  initListeners() {
    if (this.startBtn) {
      this.startBtn.addEventListener("click", () => this.startNewRound());
    }
    if (this.clearBtn) {
      this.clearBtn.addEventListener("click", () => {
        this.challengeCanvas.clear();
        if (this.submitBtn) this.submitBtn.disabled = true;
      });
    }
    if (this.submitBtn) {
      this.submitBtn.addEventListener("click", () => this.evaluateAttempt());
    }
  }

  updateStatsDisplay() {
    if (this.streakEl) this.streakEl.textContent = this.streak;
    if (this.highScoreEl) this.highScoreEl.textContent = this.highScore;
  }

  showBanner(typeClass, icon, title, desc) {
    if (!this.banner) return;
    this.banner.className = `challenge-banner ${typeClass}`;
    if (this.bannerIcon) this.bannerIcon.textContent = icon;
    if (this.bannerTitle) this.bannerTitle.textContent = title;
    if (this.bannerDesc) this.bannerDesc.textContent = desc;
    this.banner.classList.remove("hidden");
  }

  hideBanner() {
    if (this.banner) this.banner.classList.add("hidden");
  }

  showResultBadge(typeClass, icon, text) {
    if (!this.resultBadge) return;
    this.resultBadge.className = `challenge-result-badge ${typeClass}`;
    if (this.resultBadgeIcon) this.resultBadgeIcon.textContent = icon;
    if (this.resultBadgeText) this.resultBadgeText.textContent = text;
    this.resultBadge.classList.remove("hidden");
  }

  hideResultBadge() {
    if (this.resultBadge) this.resultBadge.classList.add("hidden");
  }

  startNewRound() {
    clearInterval(this.timerInterval);
    this.isActive = true;
    this.hideBanner();
    this.hideResultBadge();

    // Pick random target digit 0-9 different from last
    let nextDigit;
    do {
      nextDigit = Math.floor(Math.random() * 10);
    } while (nextDigit === this.targetDigit);
    this.targetDigit = nextDigit;

    if (this.targetDigitEl) {
      this.targetDigitEl.textContent = this.targetDigit;
      this.targetDigitEl.style.transform = "scale(1.3)";
      setTimeout(() => {
        this.targetDigitEl.style.transform = "scale(1)";
      }, 200);
    }

    this.challengeCanvas.clear();
    if (this.startBtn) this.startBtn.textContent = "Restart";
    if (this.submitBtn) this.submitBtn.disabled = true;

    // Reset 10s countdown
    this.timer = 10;
    this.updateTimerDisplay();

    this.timerInterval = setInterval(() => {
      this.timer--;
      this.updateTimerDisplay();
      if (this.timer <= 0) {
        this.handleTimeUp();
      }
    }, 1000);
  }

  updateTimerDisplay() {
    if (this.timerEl) {
      this.timerEl.textContent = `${this.timer}s`;
      this.timerEl.style.color = this.timer <= 3 ? "var(--neon-pink)" : "var(--text-primary)";
    }
  }

  handleTimeUp() {
    clearInterval(this.timerInterval);
    this.isActive = false;
    if (this.submitBtn) this.submitBtn.disabled = true;

    // Streak broken
    this.streak = 0;
    this.updateStatsDisplay();

    if (this.onPlaySound) this.onPlaySound("error");

    // Display result in challenge banner and results card without alert popup
    this.showBanner(
      "banner-timeout",
      "⏰",
      "Time's Up!",
      `Target digit was ${this.targetDigit}. Hit 'Restart' or 'Start Challenge' to try again!`
    );
    this.showResultBadge(
      "badge-mismatch",
      "⏰",
      `TIME'S UP! Target was ${this.targetDigit}`
    );
  }

  async evaluateAttempt() {
    if (!this.isActive) return;
    clearInterval(this.timerInterval);
    this.isActive = false;

    const base64Img = this.challengeCanvas.toBase64PNG();

    if (this.onChallengeSubmit) {
      const result = await this.onChallengeSubmit(base64Img);
      if (result) {
        const isDigit = result.is_digit !== false && result.digit !== undefined && result.digit !== null && Number(result.digit) >= 0;
        const isMatch = isDigit && (Number(result.digit) === Number(this.targetDigit));
        const confidence = result.confidence || 0.0;

        if (isMatch) {
          // Success! Target digit was recognized
          this.streak++;
          const roundPoints = Math.round(confidence * 100 + this.timer * 15);
          if (this.streak > this.highScore) {
            this.highScore = this.streak;
            localStorage.setItem("pixelpen_high_score", this.highScore.toString());
          }
          this.updateStatsDisplay();

          if (this.onPlaySound) this.onPlaySound("fanfare");

          // Display in inline banner and results card (NO alert box)
          this.showBanner(
            "banner-success",
            "🎯",
            `TARGET HIT! Digit ${result.digit} (${(confidence * 100).toFixed(0)}%)`,
            `+${roundPoints} pts • Current Streak: ${this.streak}`
          );
          this.showResultBadge(
            "badge-success",
            "🎯",
            `TARGET HIT! Digit ${result.digit} matches target!`
          );

          // Transition to next round after 1.8s
          setTimeout(() => {
            if (!this.isActive) {
              this.startNewRound();
            }
          }, 1800);

        } else if (!isDigit) {
          // Input was not recognized as a digit stroke
          this.streak = 0;
          this.updateStatsDisplay();
          if (this.onPlaySound) this.onPlaySound("error");

          this.showBanner(
            "banner-mismatch",
            "⚠️",
            "No Digit Detected",
            `AI could not detect a clean digit stroke. Target is ${this.targetDigit}. Streak reset.`
          );
          this.showResultBadge(
            "badge-mismatch",
            "⚠️",
            `UNCERTAIN: No clear digit detected (Target: ${this.targetDigit})`
          );

        } else {
          // AI saw a different digit
          this.streak = 0;
          this.updateStatsDisplay();
          if (this.onPlaySound) this.onPlaySound("error");

          // Display mismatch in banner and results card (NO alert box)
          this.showBanner(
            "banner-mismatch",
            "❌",
            `Mismatch: AI saw ${result.digit} instead of ${this.targetDigit}`,
            `Confidence: ${(confidence * 100).toFixed(0)}% • Streak reset to 0. Draw again!`
          );
          this.showResultBadge(
            "badge-mismatch",
            "❌",
            `MISMATCH: Target was ${this.targetDigit}, AI detected ${result.digit}`
          );
        }
      }
    }
  }
}

window.PixelPenGame = PixelPenGame;
