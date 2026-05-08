import os
import numpy as np
from pathlib import Path
import tensorflow as tf
import librosa
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = Path(__file__).parent.parent / 'models' / 'vocal_armor_best.keras'

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

def load_vocal_armor():
    print(f"Loading Vocal-Armor Engine from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model at {model_path}!")

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    model.summary()
    return model


def preprocess_audio(audio_path):

    import matplotlib.cm as cm
    from PIL import Image

    print(f"Analyzing audio features for: {audio_path}")

    SR_TARGET = 22050
    DURATION  = 2.0
    N_MELS    = 128
    IMG_SIZE  = (128, 128)

    #  Load audio 
    y, sr = librosa.load(audio_path, sr=SR_TARGET, mono=True)

    #  Find the loudest 2-second window 
    expected_samples = int(SR_TARGET * DURATION)

    if len(y) <= expected_samples:
        y = np.pad(y, (0, max(0, expected_samples - len(y))))
    else:
        # Slide a 2-second window and pick the one with highest RMS energy
        hop = SR_TARGET // 10         
        best_start = 0
        best_rms   = -1.0
        for start in range(0, len(y) - expected_samples + 1, hop):
            window = y[start : start + expected_samples]
            rms    = float(np.sqrt(np.mean(window ** 2)))
            if rms > best_rms:
                best_rms   = rms
                best_start = start
        y = y[best_start : best_start + expected_samples]

    # Normalise amplitude 
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Compute mel spectrogram 
    mel    = librosa.feature.melspectrogram(y=y, sr=SR_TARGET, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Convert to RGB image 
    norm_mel    = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    color_mapped = cm.viridis(norm_mel)
    uint8_img   = (color_mapped[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(uint8_img)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.resize(IMG_SIZE, resample=Image.LANCZOS)   # direct to 128×128

    # Save to a secure temp file, load back as array 
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


def predict_voice(audio_path: str, model) -> dict:

    print(f"\nAnalyzing voice sample: {Path(audio_path).name}")
    validate_audio_file(audio_path)
    img_array  = preprocess_audio(audio_path)

    print("Running neural network...")
    prediction = model.predict(img_array, verbose=0)
    score      = float(prediction[0, 0])

    print(f"Raw model score: {score:.4f}  (>0.5 = REAL, ≤0.5 = FAKE)")

    if score > 0.5:
        confidence = score * 100
        label      = "REAL"
        print(f"RESULT: REAL HUMAN VOICE (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1.0 - score) * 100
        label      = "FAKE"
        print(f"RESULT: AI DEEPFAKE DETECTED (Confidence: {confidence:.2f}%)")

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "raw_score":  round(score, 4),
    }


if __name__ == "__main__":
    engine = load_vocal_armor()
    test_file = "../data/for-2seconds/testing/fake/file1001.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"
    if Path(test_file).exists():         
        predict_voice(test_file, engine)
    else:
        print(f"\nCould not find {test_file}. Please update the path!")