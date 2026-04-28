# Vocal-Armor

> Real-time AI deepfake voice detection system

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Status](https://img.shields.io/badge/Status-Data%20Prep%20Complete-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Vocal-Armor is a CNN-based deepfake audio detection
system that converts voice recordings into spectrograms
and classifies them as real or AI-generated.

Built to combat the growing threat of AI voice cloning
and audio fraud.

## Status

 Phase 1: Data Preparation & Preprocessing — Complete
 Phase 2: CNN Model Training — Next

## Tech Stack

- Python 3.12+
- Librosa (audio processing)
- Joblib & Pillow (parallel processing & fast image generation)
- TensorFlow / PyTorch (CNN model)
- Matplotlib (spectrogram visualization)

## Dataset

- Fake-or-Real (FoR) Dataset
- ASVspoof 2021
- In-The-Wild Audio Deepfake Dataset

## Approach

1. Load audio files
2. Convert to Mel Spectrogram
3. Train CNN classifier
4. Classify — Real or Fake
5. Test on real world In-The-Wild audio

## Project Structure

vocal-armor-engine/
├── data/           # Dataset folder
├── notebooks/      # Experiments and EDA
├── models/         # Trained model files
├── src/            # Source code
├── results/        # Accuracy reports
└── README.md

## Results

Data preprocessing complete. Audio files successfully converted to 224x224 Mel Spectrograms.
Model training starting soon.
