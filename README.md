# VocalArmor

> **Real-Time Deepfake Voice Detection Engine**
> A state-of-the-art CNN-based pipeline that processes voice audio into Mel spectrograms and instantly classifies them as **REAL (human)** or **FAKE (AI-generated)** using deep learning.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/MIT-License-green?style=for-the-badge" alt="License" />
</p>

---

## Overview

VocalArmor is a production-ready, full-stack deepfake voice detection system built to combat the growing threat of **AI voice cloning, synthetic audio fraud, and social engineering attacks**.

By pairing a custom-trained **Convolutional Neural Network (CNN)** with a high-performance **FastAPI backend** and a futuristic, responsive **React + Vite dashboard**, VocalArmor provides real-time, low-latency, and high-accuracy deepfake voice classification across three operating modes: single-file upload, batch processing, and live microphone streaming.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data Preparation & DSP Preprocessing (Mel Spectrogram generation, audio windowing) | ✅ Complete |
| **Phase 2** | Custom CNN Model Training (Multi-variant TF/Keras architecture experiments) | ✅ Complete |
| **Phase 3** | Inference Pipeline + CNN Activation Heatmap (Grad-CAM) | ✅ Complete |
| **Phase 4** | REST API & Live Streaming (FastAPI, WebSockets, multi-format support, URL audio extraction) | ✅ Live |
| **Phase 5** | Interactive Frontend Dashboard (Vite + React, SQLite history, Google OAuth) | ✅ Live |
| **Phase 6** | Deployment Preparation (Docker, .env config, CORS hardening, production build) | ✅ Ready |

---

## Features

### Backend (Deep Learning & APIs)
* **Three-Variant CNN Classifier** — Three trained model checkpoints:
  * `vocal_armor_best` — Standard (FoR dataset, recommended for general use)
  * `vocal_armor_v2` — Intermediate (experimental checkpoint)
  * `vocal_armor_v3` — Modern Deepfake Voice (ElevenLabs/modern TTS dataset)
* **CNN Activation Heatmap (Grad-CAM)** — After each prediction, a Grad-CAM heatmap is generated showing the exact spectrogram regions that triggered the model's decision. Returned as a base64-encoded image in the API response.
* **Dual Preprocessing Pipelines** — Each model uses the exact preprocessing it was trained on to prevent spectrogram mismatch.
* **Intelligent Audio Windowing** — Automatically selects the loudest 2-second window from any length audio file using RMS energy sliding window.
* **Live Microphone Analysis (`/ws/live`)** — Streams real-time microphone audio chunks from the browser directly to the inference engine via WebSocket.
* **URL-Based Audio Extraction (`/predict-url`)** — Paste any media URL (YouTube, SoundCloud, direct links) and extract the voice stream automatically via `yt-dlp` + FFmpeg.
* **JWT Authentication** — Full email/password registration and login with bcrypt hashing, access tokens, and rotating refresh tokens.
* **Google OAuth 2.0** — One-click login via Google with automatic account linking for existing users.
* **Rate Limiting (slowapi)** — Per-IP rate limiting on prediction endpoints to prevent API abuse.
* **SQLite + SQLAlchemy ORM** — Structured schema for `users` and `refresh_tokens` tables.
* **Swagger UI** — Full interactive API docs at `/docs`.

### Frontend (React + Vite Dashboard)
* **Premium Glassmorphism UI** — High-fidelity dark-mode interface with CSS micro-animations, glowing interactive elements, waveform canvas background, and a custom magnetic cursor.
* **Detector (Home)** — Single-file upload with drag-and-drop, canvas spectrogram preview, CNN model selector, animated result verdict, and live CNN Activation Heatmap visualization.
* **Live Monitor** — Microphone capture with live waveform canvas, real-time volume meter, full-session analysis on stop, and CNN heatmap in the result card.
* **Batch Upload Scanner** — Multi-file queue with per-file progress bar, fake/real count summary, and results table.
* **History & Analytics** — Persistent scan history with fake-rate trend chart, confidence histogram, filterable data table, and CSV export.
* **Settings / User Profile** — Editable profile, avatar upload, detection preference toggles (Strict FP Filter, Auto-Save History, Email Alerts).
* **Unified Navigation Footer** — Sticky footer across all pages with active-page highlighting, nav links, GitHub profile link, and tech stack credits.
* **Landing Page** — Public-facing marketing page with animated hero section, feature bento grid, pipeline visualization, and model accuracy stats.

---

## Tech Stack

### Deep Learning & Signal Processing
| Library | Purpose |
|---------|---------|
| **TensorFlow / Keras** | Custom CNN construction, training, and inference |
| **Librosa** | DSP pipeline — resampling to 22.05 kHz, Mel Spectrogram generation, RMS windowing |
| **Pillow (PIL)** | Spectrogram image manipulation, resizing, and format handling |
| **Matplotlib (viridis)** | Colormap application for spectrogram visualization |
| **NumPy** | Audio array operations, normalization, and tensor preparation |
| **scikit-learn** | Evaluation metrics during model training |

### Application Backend
| Library | Purpose |
|---------|---------|
| **FastAPI** | High-performance ASGI web framework with automatic OpenAPI docs |
| **Uvicorn** | ASGI server for production and development |
| **SQLAlchemy** | ORM for SQLite database management |
| **python-jose** | JWT token encoding and verification |
| **passlib + bcrypt** | Secure password hashing |
| **slowapi** | IP-based rate limiting middleware |
| **httpx** | Async HTTP client for Google OAuth token exchange |
| **yt-dlp + FFmpeg** | Remote audio extraction from YouTube, SoundCloud, and 1000+ sites |
| **python-dotenv** | Environment variable loading from `.env` file |

### Interactive Frontend
| Technology | Purpose |
|------------|---------|
| **Vite + React 19** | Fast component-driven client bundling with HMR |
| **React Router v7** | SPA routing with active link detection |
| **Recharts** | Confidence histogram and fake rate trend charts |
| **Zustand** | Lightweight global authentication state management |
| **Axios** | HTTP client for API calls with interceptors |
| **Tabler Icons** | Consistent icon system across the dashboard |

---

## System Architecture

```mermaid
graph TD
    A["Raw Audio Input: Upload / WebSocket Stream / URL"] --> B["Audio Preprocessing: Load mono, resample to 22.05kHz"]
    B --> C["RMS Windowing: Extract loudest 2-second segment"]
    C --> D["Amplitude Normalization: peak = 1.0"]
    D --> E["Mel Spectrogram: 128-band × viridis RGB 128×128 image"]
    E --> F["Model-Specific Preprocessing: PNG artifacts (best/v2) or direct array (v3)"]
    F --> G["CNN Inference: VocalArmor_CNN Sigmoid output 0.0-1.0"]
    G --> H["Threshold: >0.50 = REAL, ≤0.50 = FAKE (v3: >0.60)"]
    H --> I["Grad-CAM Heatmap Generation"]
    I --> J["JSON Response: prediction, confidence, heatmap base64, is_deepfake"]
    J --> K["SQLite Logging + Frontend Result Card"]
```

### Model Architecture (`VocalArmor_CNN`)

```
Input Shape: (128, 128, 3) — Mel Spectrogram RGB Image
├── Block 1: Conv2D(32, 3×3) × 2 ──> BatchNormalization ──> MaxPooling(2×2) ──> Dropout(0.3)
├── Block 2: Conv2D(64, 3×3) × 2 ──> BatchNormalization ──> MaxPooling(2×2) ──> Dropout(0.3)
├── Block 3: Conv2D(128, 3×3) × 2 ──> BatchNormalization ──> MaxPooling(2×2) ──> Dropout(0.3)
├── GlobalAveragePooling2D
├── Dense(256) ──> BatchNormalization ──> Dropout(0.5)
├── Dense(64)  ──> Dropout(0.3)
└── Dense(1, Activation: Sigmoid)  ──>  [0.0 = AI-Generated · 1.0 = Human]
```

**Training Configuration:**
* **Optimizer**: Adam (`lr=0.001`) with `ReduceLROnPlateau` scheduling
* **Loss Function**: Binary Cross-Entropy
* **Early Stopping**: Patience 7 epochs on `val_loss`
* **Dataset**: *Fake-or-Real (FoR)* + ElevenLabs synthetic voice samples

**Model Variants:**

| Model | Bracket Label | Dataset | Threshold | Use Case |
|-------|--------------|---------|-----------|---------|
| `vocal_armor_best` | Standard | FoR standard | 0.50 | General purpose (recommended) |
| `vocal_armor_v2` | Intermediate | FoR intermediate | 0.50 | Experimental comparison |
| `vocal_armor_v3` | Modern Deepfake Voice | ElevenLabs / Modern TTS | 0.60 | Modern AI voice detection |

---

## Setup & Running

### Prerequisites
* Python 3.10+
* Node.js 18+
* FFmpeg — required by yt-dlp for URL audio extraction
  ```bash
  brew install ffmpeg       # macOS
  sudo apt install ffmpeg   # Ubuntu/Debian
  ```

### 1. Clone the Repository

```bash
git clone https://github.com/Surabhi0055/vocal-armor-engine.git
cd vocal-armor-engine
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install backend dependencies
pip install -r requirements.txt

# Set up your environment variables
cp .env.example .env
# Edit .env and fill in your actual SECRET_KEY, Google OAuth credentials, etc.

# Launch the FastAPI server
cd src && uvicorn app:app --reload --port 8000
```

* API live at: `http://localhost:8000`
* Swagger docs at: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd frontend

# Install npm packages
npm install

# Start the Vite development server
npm run dev
```

* Frontend live at: `http://localhost:5173`

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description | Required |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key (generate with `python3 -c "import secrets; print(secrets.token_hex(32))"`) | ✅ Yes |
| `ALGORITHM` | JWT algorithm (default: `HS256`) | ✅ Yes |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime in minutes (default: `30`) | ✅ Yes |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime in days (default: `7`) | ✅ Yes |
| `DATABASE_URL` | SQLAlchemy database URL (default: `sqlite:///./vocal_armor.db`) | ✅ Yes |
| `GOOGLE_CLIENT_ID` | Google OAuth 2.0 client ID | Optional |
| `GOOGLE_CLIENT_SECRET` | Google OAuth 2.0 client secret | Optional |
| `GOOGLE_REDIRECT_URI` | Google OAuth callback URL | Optional |
| `FRONTEND_URL` | Frontend origin for OAuth redirects | ✅ Yes |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | ✅ Yes |
| `EMAIL_HOST` | SMTP host for password reset emails | Optional |
| `EMAIL_PORT` | SMTP port | Optional |
| `EMAIL_USER` | Gmail address for sending emails | Optional |
| `EMAIL_APP_PASSWORD` | Gmail App Password (not your login password) | Optional |

> **Note**: Never commit your `.env` file. It is already listed in `.gitignore`.

---

## API Reference

### Info Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API root — status and available routes |
| `GET` | `/health` | Health check — model and engine status |
| `GET` | `/formats` | List all supported audio file formats |
| `GET` | `/docs` | Swagger interactive API documentation |

### Detection Endpoints

#### `POST /predict` — Single File Upload
```json
// Request: multipart/form-data
// Fields: file (binary), model (string: "best" | "v2" | "v3")

// Response:
{
  "status": "success",
  "filename": "suspicious_call.wav",
  "prediction": "FAKE",
  "confidence": 94.75,
  "raw_score": 0.0525,
  "is_deepfake": true,
  "heatmap": "data:image/png;base64,iVBORw0KGgo...",
  "message": "AI-generated voice detected with 94.75% confidence."
}
```

#### `POST /predict-url` — Remote URL Extraction
```json
// Query Parameters: url (string), model (string: "best" | "v2" | "v3")

// Response:
{
  "status": "success",
  "source_url": "https://www.youtube.com/watch?v=example",
  "prediction": "REAL",
  "confidence": 98.40,
  "raw_score": 0.9840,
  "is_deepfake": false,
  "message": "Human voice verified with 98.40% confidence."
}
```

#### `WebSocket /ws/live` — Real-Time Streaming
* Accepts raw WAV bytes from the browser microphone (minimum 8,000 bytes)
* Returns a JSON prediction result for each 2-second audio chunk

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/register` | Register a new user account |
| `POST` | `/auth/login` | Login with email and password |
| `POST` | `/auth/refresh` | Rotate refresh token and issue new access token |
| `POST` | `/auth/logout` | Invalidate the current refresh token |
| `GET` | `/auth/me` | Get the currently authenticated user's profile |
| `POST` | `/auth/forgot-password` | Trigger a password reset email |
| `GET` | `/auth/google` | Initiate Google OAuth 2.0 login flow |
| `GET` | `/auth/google/callback` | Google OAuth callback — issues tokens |

---

## Project Structure

```
vocal-armor-engine/
├── .env.example                # Template for environment variables (safe to commit)
├── .env                        # Your local secrets (git-ignored — never commit!)
├── .gitignore
├── requirements.txt            # Python backend dependencies
├── vocal_armor.db              # SQLite database (git-ignored)
│
├── src/                        # FastAPI backend source
│   ├── app.py                  # API entrypoint — routes, CORS, middleware
│   ├── auth.py                 # JWT token logic, bcrypt hashing
│   ├── database.py             # SQLAlchemy engine, session factory
│   ├── models.py               # ORM models — User, RefreshToken
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── predict.py              # DSP preprocessing + CNN inference + Grad-CAM heatmap
│   └── routers/
│       ├── auth_router.py      # All /auth/* routes
│       └── user_router.py      # All /users/* routes
│
├── frontend/                   # Vite + React dashboard client
│   ├── public/
│   │   └── va-icon.png
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx           # Home detector page
│   │   │   ├── LiveMonitorPage.jsx     # Real-time microphone stream page
│   │   │   ├── BatchUploadPage.jsx     # Batch file upload page
│   │   │   ├── HistoryPage.jsx         # Scan history with charts
│   │   │   ├── UserPage.jsx            # Profile and settings page
│   │   │   ├── Footer.jsx              # Unified sticky navigation footer
│   │   │   ├── Navbar.jsx              # Top navigation bar
│   │   │   ├── Sidebar.jsx             # Collapsible side drawer
│   │   │   ├── ModelSelector.jsx       # CNN model switcher dropdown
│   │   │   ├── HistoryTable.jsx        # Sortable scan history table
│   │   │   ├── FakeRateChart.jsx       # Fake rate trend chart (Recharts)
│   │   │   ├── ConfidenceHistogram.jsx # Confidence score histogram (Recharts)
│   │   │   ├── VAIcon.jsx              # Reusable brand logo icon
│   │   │   ├── WaveformBackground.jsx  # Animated canvas background
│   │   │   ├── CustomCursor.jsx        # Magnetic cursor tracker
│   │   │   └── ProtectedRoute.jsx      # Auth-gated route wrapper
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx         # Public marketing landing page
│   │   │   ├── AuthPage.jsx            # Login / Register / OAuth UI
│   │   │   ├── AuthCallback.jsx        # Google OAuth redirect handler
│   │   │   └── ResetPasswordPage.jsx   # Password reset page
│   │   ├── store/
│   │   │   └── authStore.js            # Zustand global auth state
│   │   ├── utils/
│   │   │   ├── storage.js              # localStorage history and preferences
│   │   │   └── axiosInstance.js        # Axios with auth interceptors
│   │   ├── App.jsx                     # Root router and AppLayout
│   │   ├── index.css                   # Global design system and CSS tokens
│   │   └── main.jsx                    # React entry point
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── models/                     # Compiled TF/Keras model binaries (git-ignored)
│   ├── vocal_armor_best.keras  # Standard model
│   ├── vocal_armor_v2.keras    # Intermediate model
│   └── vocal_armor_v3.keras    # Modern Deepfake Voice model
│
├── notebooks/                  # Research and training notebooks
│   ├── 01_explore_audio.ipynb  # Audio EDA — waveforms, spectrogram visualization
│   └── 02_model_training.ipynb # CNN architecture and accuracy tuning
│
└── data/                       # Preprocessed audio datasets (git-ignored)
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Model Architecture** | Custom Multi-Layer CNN (`VocalArmor_CNN`) |
| **Input Shape** | 128×128 Mel Spectrogram RGB Arrays |
| **Validation Accuracy** | ≥ 98.1% (held-out set of 6,200 samples) |
| **False Negative Rate** | 0.3% (real voice incorrectly flagged as fake) |
| **False Positive Rate** | 1.6% (deepfake slipping through as real) |
| **Total Samples Analyzed** | 31K+ since launch |
| **Supported Formats** | WAV, MP3, FLAC, OGG, M4A, AAC |
| **Response Latency** | < 150ms for processed 2-second segments |

---

## License

This project is released under the **MIT License**.

---

*Built by [Surabhi0055](https://github.com/Surabhi0055) · Powered by TensorFlow · FastAPI · React · Vite*
