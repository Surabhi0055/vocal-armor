import os
import numpy as np
from pathlib import Path
import tensorflow as tf

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

if __name__ == "__main__":
    engine = load_vocal_armor()
