import time
import librosa
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path

def audio_to_spectrogram_optimized(audio_path, save_path, duration=2, n_mels=128, sr_target=22050):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, duration=duration, sr=sr_target)
        expected_length = int(sr_target * duration)
        if len(y) < expected_length:
            y = np.pad(y, (0, expected_length - len(y)))
        
        # Compute Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to 0-1 range for image saving
        # We want to match the "viridis" look or just save as grayscale
        # Viridis mapping:
        norm_mel = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        
        # Use viridis colormap
        color_mapped = cm.viridis(norm_mel) # Returns RGBA
        uint8_img = (color_mapped[:, :, :3] * 255).astype(np.uint8) # Keep RGB
        
        # Resize to 200x200 to match the user's intent (plt.figure(figsize=(2,2)) + dpi=100)
        img = Image.fromarray(uint8_img)
        img = img.transpose(Image.FLIP_TOP_BOTTOM) # specshow plots with origin='lower'
        img = img.resize((200, 200), resample=Image.LANCZOS)
        
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

data_root = Path("/Users/surabhi/Desktop/vocal-armor-engine/data/for-2seconds")
sample_file = next(data_root.rglob("*.wav"), None)

if sample_file:
    print(f"Testing optimized with: {sample_file}")
    start = time.time()
    success = audio_to_spectrogram_optimized(sample_file, "test_output_opt.png")
    end = time.time()
    if success:
        print(f"Optimized Time taken: {end - start:.4f} seconds")
    else:
        print("Optimized Function failed")
