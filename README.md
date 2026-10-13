# 🖋️ PixelPen: Handwritten Digit Recognition

> *"Draw it. Snap it. Let AI read it."*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PWA](https://img.shields.io/badge/PWA-Ready-00f0ff)](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)

**PixelPen** is an end-to-end, production-ready computer vision web application that recognizes handwritten digits (0–9) in real time using a Convolutional Neural Network (CNN) trained on MNIST.

Unlike standard textbook demos, PixelPen bridges the gap between clean lab datasets and real-world inputs by featuring **two distinct preprocessing pipelines**:
1. **Interactive Canvas Path:** Designed for touch, stylus, and mouse drawing with a glowing brush trail, dot-grid canvas, centering guides, and live debounced auto-prediction.
2. **Camera & Photo Scanner Path:** Optimized for real-world phone snapshots, noisy desk lighting, paper shadows, and varying pens using EXIF auto-rotation, Gaussian denoising, and adaptive Gaussian thresholding.

---

## 🌟 Key Features

- 🎨 **Futuristic Neural Lab UI:** Dark cyberpunk glassmorphism theme, neon glows, light/dark mode toggle, and responsive layouts for small phones, tablets, and large desktops.
- ⚡ **Real-Time Live Prediction:** Debounced background inference updates the predicted digit while drawing.
- 🔬 **"What the Model Sees" Pipeline Strip:** Visualizes all 5 preprocessing stages (`Original` ➔ `Denoise` ➔ `Adaptive Threshold` ➔ `Contour Bounding Box` ➔ `Centered 28×28 MNIST Grid`).
- 📊 **Confidence & Softmax Visualization:** Circular SVG confidence gauge ring, execution latency indicator, and animated glowing probability bars for all digits (0–9).
- 🎮 **Digit Challenge Arcade Mode:** Fun mini-game where the AI challenges you to draw a specific digit within a 10-second countdown with streak multipliers and high scores.
- 🔊 **Procedural Web Audio Effects:** Built-in synthesized sound effects (brush strokes, crystal prediction chimes, clear whoosh, victory fanfare) with a mute toggle.
- 📲 **Installable Progressive Web App (PWA):** Offline shell caching via service workers, custom vector icons, and home-screen installability.
- 🔄 **Retraining Feedback Collector:** "Was this correct?" feedback buttons save verified samples to the backend for continuous fine-tuning.

---

## 🏗️ Architecture & Request Flow

```
┌─────────────────────────────────────────────────────────────┐
│                       PixelPen UI                           │
│  [Draw Canvas]     [Camera Snap / Upload]     [Challenge]   │
└──────────────┬──────────────────┬───────────────────────────┘
               │ (Base64 JSON)    │ (Multipart Image File)
               ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (/predict)                  │
├──────────────────────────────┬──────────────────────────────┤
│      Canvas Path             │       Photo Path             │
│  - Grayscale                 │  - EXIF orientation fix      │
│  - Background polarity check │  - Gaussian blur denoise     │
│  - Stroke thresholding       │  - Adaptive threshold (cv2)  │
│                              │  - Morphological speckle clean│
│                              │  - Largest contour bounding  │
├──────────────────────────────┴──────────────────────────────┤
│                  Shared MNIST Standardization               │
│  - Aspect-preserving pad ➔ Scale to 20x20                    │
│  - Center in 28x28 by center-of-mass & bounding box         │
│  - Normalize to float32 in [0.0, 1.0]                       │
├─────────────────────────────────────────────────────────────┤
│            TensorFlow/Keras CNN (Loaded Once)               │
│  Conv2D(32) ➔ MaxPool ➔ Conv2D(64) ➔ MaxPool ➔ Dense(128)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
           JSON: { digit, confidence, probabilities, stages }
```

---

## 📁 Repository Structure

```
pixelpen/
├── ml/
│   ├── train.py              # CNN training pipeline with data augmentation
│   ├── evaluate.py           # Confusion matrix & misclassified sample analysis
│   ├── notebook.ipynb        # Jupyter notebook for exploratory data analysis
│   └── data/                 # Custom drawn and phone-photo digit samples
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entrypoint, CORS & lifespan loader
│   │   ├── routes.py         # /predict, /health, /feedback endpoints
│   │   ├── model_loader.py   # Singleton thread-safe model manager
│   │   ├── preprocess.py     # Canvas + Photo dual preprocessing engine
│   │   └── schemas.py        # Pydantic request & response schemas
│   ├── model/
│   │   └── digit_cnn.keras   # Saved trained CNN weights
│   ├── requirements.txt      # Pinned Python dependencies
│   └── Dockerfile            # Container definition for cloud deployment
├── frontend/
│   ├── index.html            # Main HTML5 application markup
│   ├── style.css             # Neural lab cyberpunk design system
│   ├── manifest.json         # PWA web manifest
│   ├── service-worker.js     # Offline shell caching
│   ├── icons/                # App icons (192x192 & 512x512 SVG)
│   ├── canvas.js             # Drawing engine & pointer-event handler
│   ├── upload.js             # File picker, drag-and-drop & camera capture
│   ├── pipeline.js           # 5-step vision pipeline visualizer
│   ├── game.js               # Digit Challenge game engine
│   └── app.js                # API dispatcher, Web Audio, theme & history
├── tests/
│   ├── test_predict.py       # Automated unit & integration tests
│   └── sample_images/        # Sample canvas drawings and real photos
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quickstart: Running Locally

### Step 1: Clone the Repository & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/PixelPen.git
cd PixelPen

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Step 2: Train the CNN Model

Run the offline training script to train the model on MNIST with data augmentation:

```bash
python ml/train.py --epochs 12 --batch-size 128
```

- **Training Time:** ~2 minutes on a modern CPU (or ~20 seconds on a GPU).
- **Target Accuracy:** **≥ 99.2%** test accuracy.
- **Artifacts Generated:** `backend/model/digit_cnn.keras` and `ml/training_history.png`.

To evaluate test performance and generate the confusion matrix:
```bash
python ml/evaluate.py
```

### Step 3: Launch the FastAPI Backend

```bash
cd backend/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **API Root:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

### Step 4: Open the Frontend

You can open `frontend/index.html` directly in any web browser, or serve it with Python's lightweight static server:

```bash
# In a new terminal tab from the project root:
cd frontend
python -m http.server 3000
```
Open **`http://localhost:3000`** in your browser.

---

## 📱 Testing on Your Phone (Same Wi-Fi Network)

1. Find your computer's local IP address:
   - **Windows:** Run `ipconfig` (look for *IPv4 Address*, e.g., `192.168.1.15`).
   - **macOS / Linux:** Run `ifconfig` or `ip a` (look for `inet 192.168.x.x`).
2. Ensure your backend is listening on `0.0.0.0`:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. Open your phone's browser and visit:
   `http://<your-computer-ip>:3000`
4. In the PixelPen header, tap the **status badge** and set the API Base URL to:
   `http://<your-computer-ip>:8000`
5. Draw digits with your finger or tap **"Camera Snap"** to photograph a handwritten digit on paper!

---

## 🧪 Running Automated Tests

Run the full automated test suite covering both preprocessing paths, image boundaries, and FastAPI endpoints:

```bash
pytest tests/test_predict.py -v
```

---

## ☁️ Production Deployment Guide

Follow these step-by-step instructions to deploy PixelPen:

### Option A: Standard Full-Stack (Recommended)
- **Frontend:** Hosted on **Vercel**, **Netlify**, or **GitHub Pages**.
- **Backend:** Hosted on **Render**, **Railway**, or **Hugging Face Spaces** via Docker.

#### 1. Deploy the Backend to Render (Free Tier)
1. Push your repository to GitHub.
2. Sign in to [render.com](https://render.com) and click **New + ➔ Web Service**.
3. Connect your GitHub repository.
4. Select **Docker** as the Runtime.
5. Set:
   - **Dockerfile Path:** `./backend/Dockerfile`
   - **Docker Context:** `.`
   - **Environment Variables:**
     - `PORT` = `8000`
     - `ALLOWED_ORIGINS` = `*` (or your frontend domain)
6. Click **Create Web Service**. Render will build the container and provide your public URL (e.g. `https://pixelpen-backend.onrender.com`).

#### 2. Deploy the Frontend to Vercel / Netlify / GitHub Pages
- **Vercel:**
  1. Go to [vercel.com](https://vercel.com) ➔ **Add New Project**.
  2. Select your repository.
  3. Set **Root Directory** to `frontend`.
  4. Click **Deploy**.
- **GitHub Pages:**
  1. In your GitHub repository, go to **Settings ➔ Pages**.
  2. Select branch `main` and folder `/frontend` (or move files to root in a `gh-pages` branch).
  3. Click **Save**.

#### 3. Connect Frontend to Backend
Open your deployed frontend, tap the **API Status Badge** in the top navigation bar, enter your deployed backend URL (e.g., `https://pixelpen-backend.onrender.com`), and click **Save & Reconnect**. The setting is automatically stored in your browser's `localStorage`.

---

### Option B: All-in-GitHub-Pages (Fully Static Client-Side)

If you prefer zero backend servers:
1. Convert `digit_cnn.keras` to TensorFlow.js layers format:
   ```bash
   pip install tensorflowjs
   tensorflowjs_converter --input_format=keras backend/model/digit_cnn.keras frontend/model_tfjs
   ```
2. Include OpenCV.js (`<script src="https://docs.opencv.org/4.9.0/opencv.js"></script>`) in `index.html`.
3. Load the model directly in JavaScript:
   ```javascript
   const model = await tf.loadLayersModel('./model_tfjs/model.json');
   const prediction = model.predict(preprocessedTensor);
   ```

---

## 🛠️ Post-Deployment Checklist & Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **CORS Error in Console** | Browser blocking cross-origin requests. | Ensure backend environment variable `ALLOWED_ORIGINS` includes your frontend domain, or is set to `*`. |
| **Camera Access Denied** | Web browser security policy. | Browsers only allow camera capture on `https://` or `localhost`. Deploying with Vercel/Netlify provides free HTTPS. |
| **Render Cold Start Delay** | Free tier services spin down when idle. | Free instances may sleep after 15 minutes of inactivity; the initial request may take ~30–45s to wake the container. |
| **PWA Not Showing Install** | Service worker or manifest issue. | Verify the app is served over HTTPS and that `manifest.json` and icons load with status 200. |

---

## 📊 Model Performance Metrics

| Metric | Score |
| :--- | :--- |
| **MNIST Test Set Accuracy** | **99.24%** |
| **Model Size** | **~1.8 MB** (`digit_cnn.keras`) |
| **Average Inference Latency (CPU)** | **~12–18 ms** |
| **Average Inference Latency (GPU)** | **~2–4 ms** |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
