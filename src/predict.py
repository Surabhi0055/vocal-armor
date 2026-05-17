import os
import numpy as np
from pathlib import Path

# Disable Metal GPU — Apple Metal uses float16 which causes activation collapse
# and produces wrong predictions. CPU float32 gives correct results.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import librosa
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use a path relative to this file, not the working directory
model_path = Path(__file__).parent.parent / 'models' / 'vocal_armor_best.keras'

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

def load_vocal_armor():
    models_dir = Path(__file__).parent.parent / 'models'
    loaded_models = {}
    
    for name, filename in [('best', 'vocal_armor_best.keras'), ('v2', 'vocal_armor_v2.keras'), ('v3', 'vocal_armor_v3.keras')]:
        path = models_dir / filename
        if path.exists():
            print(f"Loading {name} model from {path}...")
            loaded_models[name] = tf.keras.models.load_model(path)
            
    if not loaded_models:
        raise FileNotFoundError("Could not find any models in the models directory!")
        
    print(f"Successfully loaded models: {list(loaded_models.keys())}")
    return loaded_models


def preprocess_audio(audio_path, model_name="best"):

    import matplotlib.cm as cm
    from PIL import Image

    print(f"Analyzing audio features for: {audio_path}")

    SR_TARGET = 22050
    DURATION  = 2.0
    N_MELS    = 128
    IMG_SIZE  = (128, 128)

    #  1. Load audio 
    y, sr = librosa.load(audio_path, sr=SR_TARGET, mono=True)

    #  2. Find the loudest 2-second window 
    # If the file is already ~2 seconds (pre-processed), just pad — don't re-window.
    # Re-windowing pre-trimmed files shifts the start index and distorts the input.
    expected_samples = int(SR_TARGET * DURATION)
    if len(y) <= expected_samples:
        # File is 2 s or shorter — just pad it
        y = np.pad(y, (0, max(0, expected_samples - len(y))))
    else:
        # Slide a 2-second window and pick the one with highest RMS energy
        hop = SR_TARGET // 10          # 0.1-second hops
        best_start = 0
        best_rms   = -1.0
        for start in range(0, len(y) - expected_samples + 1, hop):
            window = y[start : start + expected_samples]
            rms    = float(np.sqrt(np.mean(window ** 2)))
            if rms > best_rms:
                best_rms   = rms
                best_start = start
        y = y[best_start : best_start + expected_samples]

    #  3. Normalise amplitude 
  
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    #  4. Compute mel spectrogram 
    mel    = librosa.feature.melspectrogram(y=y, sr=SR_TARGET, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    #  5. Convert to RGB image 
    norm_mel    = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    color_mapped = cm.viridis(norm_mel)
    uint8_img   = (color_mapped[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(uint8_img)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.resize(IMG_SIZE, resample=Image.LANCZOS)
    
    # CRITICAL FIX: Dataset Mismatch!
    # The original dataset (used for 'best' and 'v2') was created by saving plots as PNGs.
    # The ElevenLabs dataset (used for 'v3') was created directly using pure numpy arrays.
    # We must apply the exact preprocessing that the specific model was trained on!
    
    if model_name == "v3":
        # Directly convert PIL image to array and normalize (EXACTLY like notebook for v3)
        img_array = np.array(img) / 255.0
        final = np.expand_dims(img_array, axis=0)
    else:
        # Apply PNG compression artifacts (required for 'best' and 'v2')
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name
            img.save(temp_path)
        try:
            loaded_img = tf.keras.utils.load_img(temp_path, target_size=IMG_SIZE)
            img_array  = tf.keras.utils.img_to_array(loaded_img) / 255.0
            final      = np.expand_dims(img_array, axis=0)
        finally:
            os.remove(temp_path)

    return final


def validate_audio_file(audio_path: str):
    """Raise ValueError if the file extension is not a supported audio format."""
    ext = Path(audio_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def predict_voice(audio_path: str, model, model_name: str = "best") -> dict:

    print(f"\nAnalyzing voice sample: {Path(audio_path).name}")

    validate_audio_file(audio_path)

    img_array  = preprocess_audio(audio_path, model_name)

    print("Running neural network...")
    prediction = model.predict(img_array, verbose=0)
    score      = float(prediction[0, 0])

    THRESHOLD = 0.50
    if model_name == "v3":
        THRESHOLD = 0.60  # Slight bump for v3

    print(f"Raw model score: {score:.4f}  (>{THRESHOLD} = REAL, ≤{THRESHOLD} = FAKE)")

    if score > THRESHOLD:
        confidence = 50 + ((score - THRESHOLD) / (1.0 - THRESHOLD)) * 50 if THRESHOLD != 0.5 else score * 100
        label      = "REAL"
        print(f"RESULT: REAL HUMAN VOICE (Confidence: {confidence:.2f}%)")
    else:
        confidence = 50 + ((THRESHOLD - score) / THRESHOLD) * 50 if THRESHOLD != 0.5 else (1.0 - score) * 100
        label      = "FAKE"
        print(f"RESULT: AI DEEPFAKE DETECTED (Confidence: {confidence:.2f}%)")

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "raw_score":  round(score, 4),
    }


if __name__ == "__main__":
    engines = load_vocal_armor()
    test_file = "../data/for-2seconds/testing/fake/file1001.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"
    if Path(test_file).exists():      
        predict_voice(test_file, engines['best'], "best")
    else:
        print(f"\nCould not find {test_file}. Please update the path!")