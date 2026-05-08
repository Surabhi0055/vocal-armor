# Vocal-Armor

> Real-time AI deepfake voice detection system

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![Status](https://img.shields.io/badge/Status-API%20Live-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Vocal-Armor is a CNN-based deepfake audio detection system that converts voice recordings into Mel spectrograms and classifies them as real or AI-generated.

Built to combat the growing threat of AI voice cloning and audio fraud.

## Status

* ✅ Phase 1: Data Preparation & Preprocessing — Complete
* ✅ Phase 2: CNN Model Training — Complete
* ✅ Phase 3: Inference Pipeline — Complete
* ✅ Phase 4: REST API (FastAPI) — Live
* 🔄 Phase 5: Frontend / Deployment — Next

## Tech Stack

* **Python 3.12+**
* **TensorFlow / Keras** — CNN model training & inference
* **Librosa** — audio loading, resampling, mel spectrogram generation
* **Pillow** — spectrogram image processing
* **FastAPI + Uvicorn** — REST API backend
* **Matplotlib** — spectrogram visualization
* **Scikit-learn** — evaluation metrics (confusion matrix, ROC, classification report)

## Dataset

* Fake-or-Real (FoR) Dataset — 2-second pre-trimmed clips
* Splits: Training / Validation / Testing

## Approach

1. Load audio files and convert to mono, 22050 Hz
2. Extract the loudest 2-second window per file
3. Generate 128×128 Mel Spectrogram images (viridis colormap)
4. Train CNN classifier on spectrogram images
5. Serve predictions via REST API
6. Test on real-world downloaded AI voice samples

## Model Architecture

Custom CNN (`VocalArmor_CNN`) built with TensorFlow/Keras:

```
Input (128×128×3)
├── Block 1 — Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.3)
├── Block 2 — Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.3)
├── Block 3 — Conv2D(128) × 2 → BatchNorm → MaxPool → Dropout(0.3)
├── GlobalAveragePooling2D
├── Dense(256) → BatchNorm → Dropout(0.5)
├── Dense(64) → Dropout(0.3)
└── Dense(1, sigmoid)  →  0.0 = FAKE · 1.0 = REAL
```

**Training config:** Adam (lr=0.001) · Binary Crossentropy · EarlyStopping (patience=7) · ReduceLROnPlateau

## API

Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint     | Description                         |
| ------ | ------------ | ----------------------------------- |
| GET    | `/`        | Health check                        |
| POST   | `/predict` | Upload audio file → get prediction |

### Example Response

```json
{
  "status": "success",
  "filename": "voice_sample.wav",
  "prediction": "FAKE",
  "confidence": 91.43,
  "raw_score": 0.0857
}
```

**Supported formats:** `.wav` · `.mp3` · `.flac` · `.ogg` · `.m4a` · `.aac`

**Max file size:** 50 MB

## Project Structure

```
vocal-armor-engine/
├── data/                   # Dataset folder
├── notebooks/
│   ├── 01_explore_audio.ipynb      # EDA & spectrogram visualization
│   └── 02_model_training.ipynb     # CNN training & evaluation
├── models/
│   └── vocal_armor_best.keras      # Saved best model
├── results/                # Training plots, confusion matrix, logs
├── app.py                  # FastAPI server
├── predict.py              # Inference pipeline
└── README.md
```

## Results

| Metric          | Value                         |
| --------------- | ----------------------------- |
| Model           | VocalArmor_CNN                |
| Input           | 128×128 Mel Spectrogram      |
| Optimizer       | Adam                          |
| Training target | val_accuracy > 85%            |
| Supported audio | WAV, MP3, FLAC, OGG, M4A, AAC |
