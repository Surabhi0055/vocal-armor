# Vocal-Armor 

> Real-time AI deepfake voice detection system

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
Vocal-Armor is a CNN-based deepfake audio detection 
system that converts voice recordings into spectrograms 
and classifies them as real or AI-generated.

Built to combat the growing threat of AI voice cloning 
and audio fraud.

## Status
🚧 Active Development — Started April 2026

## Tech Stack
- Python 3.10+
- Librosa (audio processing)
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
Training in progress — results coming soon
