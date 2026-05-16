from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
from pathlib import Path
from predict import load_vocal_armor, predict_voice, ALLOWED_EXTENSIONS

app = FastAPI(
    title="Vocal-Armor API",
    description=(
        "Real-time deepfake voice detection API. "
        "Upload any audio file and the engine will classify it as "
        "REAL (human) or FAKE (AI-generated)."
    ),
    version="1.2",
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = load_vocal_armor()

#  Endpoints 

@app.get("/", tags=["Info"])
def read_root():
    return {
        "status":  "online",
        "message": "Vocal-Armor API is live and ready for inference.",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health", tags=["Info"])
def health_check():
    return {
        "status":            "healthy",
        "model":             "VocalArmor_CNN",
        "model_version":     "3.0",
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
        "max_file_size_mb":  50,
        "engine_loaded":     engine is not None,
    }


@app.get("/formats", tags=["Info"])
def get_supported_formats():
    """Returns the list of audio formats accepted by the /predict endpoint."""
    return {
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
        "note": (
            "Upload any of these audio formats to POST /predict "
            "for deepfake detection."
        ),
    }


@app.post("/predict", tags=["Detection"])
async def predict_audio(file: UploadFile = File(...)):

    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # 2. Read content and enforce 50 MB size limit
    MAX_BYTES = 50 * 1024 * 1024
    contents  = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 50 MB.",
        )

    # 3. Write to a uniquely-named temp file, run inference, clean up
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        result = predict_voice(temp_path, engine)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    # 4. Return rich response
    is_fake = result["prediction"] == "FAKE"
    return JSONResponse(content={
        "status":      "success",
        "filename":    file.filename,
        "prediction":  result["prediction"],
        "confidence":  result["confidence"],
        "raw_score":   result["raw_score"],
        "is_deepfake": is_fake,
        "message": (
            f"AI-generated voice detected with {result['confidence']}% confidence."
            if is_fake else
            f"Human voice verified with {result['confidence']}% confidence."
        ),
    })

@app.post("/predict-url", tags=["Detection"])
async def predict_from_url(url: str):
    import yt_dlp
    import re

    # Basic URL validation
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL.")

    temp_path = None
    try:
        # yt-dlp handles YouTube, SoundCloud, direct links, and 1000+ sites
        with tempfile.TemporaryDirectory() as tmp_dir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'{tmp_dir}/audio.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'quiet': True,
                'no_warnings': True,
                'max_filesize': 50 * 1024 * 1024,  # 50MB limit
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find the downloaded file
            files = list(Path(tmp_dir).glob('*.mp3'))
            if not files:
                raise Exception("Could not download audio from URL.")

            temp_path = str(files[0])
            result = predict_voice(temp_path, engine)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process URL: {str(e)}"
        )

    is_fake = result["prediction"] == "FAKE"
    return JSONResponse(content={
        "status":      "success",
        "source_url":  url,
        "prediction":  result["prediction"],
        "confidence":  result["confidence"],
        "raw_score":   result["raw_score"],
        "is_deepfake": is_fake,
        "message": (
            f"AI-generated voice detected with {result['confidence']}% confidence."
            if is_fake else
            f"Human voice verified with {result['confidence']}% confidence."
        ),
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)