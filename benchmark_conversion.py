import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def audio_to_spectrogram_image(audio_path, save_path, duration=2, n_mels=128, sr_target=22050):
    try:
        y, sr = librosa.load(audio_path, duration=duration, sr=sr_target)
        expected_length = int(sr_target * duration)
        if len(y) < expected_length:
            y = np.pad(y, (0, expected_length - len(y)))
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        plt.figure(figsize=(2, 2))
        plt.axes([0, 0, 1, 1])
        librosa.display.specshow(mel_db, sr=sr, cmap="viridis")
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Find a sample file
data_root = Path("/Users/surabhi/Desktop/vocal-armor-engine/data/for-2seconds")
sample_file = next(data_root.rglob("*.wav"), None)

if sample_file:
    print(f"Testing with: {sample_file}")
    start = time.time()
    success = audio_to_spectrogram_image(sample_file, "test_output.png")
    end = time.time()
    if success:
        print(f"Time taken: {end - start:.4f} seconds")
    else:
        print("Function failed")
else:
    print("No sample file found")
