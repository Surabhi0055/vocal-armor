from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import os
import tempfile

app = FastAPI(
    title="Vocal-Armor API",
    description="Backend API for real-time deepfake voice detection.",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"status": "success", "message": "Welcome to Vocal-Armor API! The engine is ready."}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "status": "success",
        "filename": file.filename,
        "message": "Audio file received and saved successfully!"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)