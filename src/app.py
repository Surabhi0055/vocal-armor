from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, Form, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from database import create_tables
from routers.auth_router import router as auth_router
from routers.user_router import router as user_router
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
# rate limiter to track users by their IP address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# handle users who break the rules
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable sessions (for Google OAuth callback states)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "change-me"))

# CORS 
frontend_urls = os.getenv("FRONTEND_URLS", "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://127.0.0.1:5173,http://127.0.0.1:5174")
allowed_origins = [url.strip() for url in frontend_urls.split(",") if url.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(auth_router)
app.include_router(user_router)

# Mount static files for uploads
os.makedirs("uploads/avatars", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Create tables on startup
@app.on_event("startup")
def startup():
    create_tables()
    print("Database tables created")

engines = load_vocal_armor()

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
        "engine_loaded":     len(engines) > 0,
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
@limiter.limit("100/minute")  
async def predict_audio(request: Request, file: UploadFile = File(...), model: str = Form("best")):

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

        engine_to_use = engines.get(model, engines.get("best"))
        result = predict_voice(temp_path, engine_to_use, model)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        import traceback
        traceback.print_exc()
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
        "heatmap":     result.get("heatmap"),
        "message": (
            f"AI-generated voice detected with {result['confidence']}% confidence."
            if is_fake else
            f"Human voice verified with {result['confidence']}% confidence."
        ),
    })

@app.post("/predict-url", tags=["Detection"])
@limiter.limit("5/minute") 
async def predict_from_url(request: Request,url: str, model: str = "best"):
    
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
            engine_to_use = engines.get(model, engines.get("best"))
            result = predict_voice(temp_path, engine_to_use, model)

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
        "heatmap":     result.get("heatmap"),
        "message": (
            f"AI-generated voice detected with {result['confidence']}% confidence."
            if is_fake else
            f"Human voice verified with {result['confidence']}% confidence."
        ),
    })


@app.websocket("/ws/live")
async def live_inference(websocket: WebSocket):
    await websocket.accept()
    print("Live monitor client connected")
    
    try:
        while True:
            # Receive WAV bytes from browser
            data = await websocket.receive_bytes()
            
            print(f"Received audio chunk: {len(data)} bytes")
            
            # Reject suspiciously small payloads (silence/empty)
            if len(data) < 8000:
                print(f"Skipping — too small ({len(data)} bytes), likely silence")
                await websocket.send_json({
                    "error": "Audio chunk too small or silent, skipping"
                })
                continue
            
            temp_path = None
            try:
                # Save as .wav file
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(data)
                    temp_path = tmp.name
                
                print(f"Running inference on: {temp_path}")
                result = predict_voice(temp_path, engines['best'], "best")
                is_fake = result["prediction"] == "FAKE"
                
                response = {
                    "prediction":  result["prediction"],
                    "confidence":  result["confidence"],
                    "raw_score":   result["raw_score"],
                    "is_deepfake": is_fake,
                    "message": (
                        f"AI-generated voice detected with "
                        f"{result['confidence']}% confidence."
                        if is_fake else
                        f"Human voice verified with "
                        f"{result['confidence']}% confidence."
                    ),
                }
                print(f"Sending result: {response}")
                await websocket.send_json(response)
                
            except Exception as e:
                print(f"Inference error: {e}")
                await websocket.send_json({"error": str(e)})
                
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    except Exception as e:
        print(f"Live monitor client disconnected: {e}")
        

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)