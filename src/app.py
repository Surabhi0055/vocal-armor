from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from pathlib import Path
from predict import load_vocal_armor, predict_voice, ALLOWED_EXTENSIONS

app = FastAPI(
    title="Vocal-Armor API",
    description="Backend API for real-time deepfake voice detection.",
    version="1.1"
)

# Load the neural network into memory
engine = load_vocal_armor()

@app.get("/")
def read_root():
    return {
        "status":  "success",
        "message": "Welcome to Vocal-Armor API! The engine is ready.",
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
    }


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):

    #  Validate file extension 
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        )

    #  Enforce a file-size limit (50 MB) 
    MAX_BYTES = 50 * 1024 * 1024
    contents  = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is 50 MB."
        )

    #  Write to a uniquely-named temp file
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False
        ) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        # Run inference 
        result = predict_voice(temp_path, engine)

    except ValueError as ve:
        # Raised by validate_audio_file for bad extensions (double-check)
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Catch corrupted audio, librosa errors, TF errors, etc.
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        # clean up the temp file 
        if temp_file and os.path.exists(temp_path):
            os.remove(temp_path)
        elif 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

    return JSONResponse(content={
        "status":     "success",
        "filename":   file.filename,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "raw_score":  result["raw_score"],  
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)