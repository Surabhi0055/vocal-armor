import os
import numpy as np
from pathlib import Path
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_path = Path('../models/vocal_armor_best.keras')
def load_vocal_armor ():
    print(f"Loading Vocal-Armor Engine from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model at {model_path}!")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    model.summary()
    return model

def preprocess_audio (audio_path):
    print(f"Analyzing audio features for: {audio_path}")
    import matplotlib.cm as cm
    from PIL import Image
    
    # Load audio 
    duration = 2.0
    sr_target = 22050
    y, sr = librosa.load(audio_path, duration=duration, sr=sr_target)
    
    expected_length = int(sr_target * duration)
    if len(y) < expected_length:
        y = np.pad(y, (0, expected_length - len(y)))

    # Compute Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Direct Image Conversion
    norm_mel = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    color_mapped = cm.viridis(norm_mel)
    uint8_img = (color_mapped[:, :, :3] * 255).astype(np.uint8)

    # Save using PIL matching the generator step
    temp = tempfile.mktemp(suffix='.png')
    img = Image.fromarray(uint8_img)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.resize((224, 224), resample=Image.LANCZOS)
    img.save(temp)

    # Load and preprocess exactly like ImageDataGenerator
    loaded_img = tf.keras.utils.load_img(temp, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(loaded_img)
    img_array = img_array / 255.0

    final = np.expand_dims(img_array, axis=0)
    os.remove(temp)
    return final

def predict_voice(audio_path, model):
    print(f"\nAnalyzing voice sample: {Path(audio_path).name}")
    img_array = preprocess_audio(audio_path)
    print("Running neural network...")
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0,0]
    if score > 0.5:
        confidence = score * 100
        print(f"RESULT: REAL HUMAN VOICE (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1.0 - score) * 100
        print(f"RESULT: AI DEEPFAKE DETECTED (Confidence: {confidence:.2f}%)")


if __name__ == "__main__":
    engine = load_vocal_armor()
    test_file = "../data/for-2seconds/testing/fake/file1001.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"
    if Path(test_file).exists :
        predict_voice(test_file, engine)
    else:
        print(f"\nCould not find {test_file}. Please update the path!")

