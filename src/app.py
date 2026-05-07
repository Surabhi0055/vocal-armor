from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import os
import tempfile
from predict import load_vocal_armor, predict_voice

app = FastAPI(
    title="Vocal-Armor API",
    description="Backend API for real-time deepfake voice detection.",
    version="1.0"
)

# Load the neural network into memory exactly once when the server starts
engine = load_vocal_armor()

@app.get("/")
def read_root():
    return {"status": "success", "message": "Welcome to Vocal-Armor API! The engine is ready."}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Send the saved file to our Neural Network!
    result = predict_voice(temp_audio_path, engine)
    
    # Clean up the temporary file
    os.remove(temp_audio_path)
    
    return {
        "status": "success",
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)