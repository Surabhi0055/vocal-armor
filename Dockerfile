# ── Base Image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System Dependencies ────────────────────────────────────────────────────────
# ffmpeg  → required by yt-dlp for audio extraction from URLs
# libsndfile1 → required by soundfile / librosa for audio file reading
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ────────────────────────────────────────────────
# Copy requirements first (Docker cache layer — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Source ────────────────────────────────────────────────────
COPY src/ ./src/
COPY models/ ./models/

# ── Create uploads directory ───────────────────────────────────────────────────
RUN mkdir -p uploads/avatars

# ── Expose Port ────────────────────────────────────────────────────────────────
# Hugging Face Spaces requires port 7860
EXPOSE 7860

# ── Start FastAPI Server ───────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "src"]
