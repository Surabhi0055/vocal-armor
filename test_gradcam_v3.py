import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import sys
sys.path.append('src')
from predict import predict_voice, load_vocal_armor

engines = load_vocal_armor()
result = predict_voice('data/for-2seconds/testing/fake/file1001.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav', engines['v3'], 'v3')
print('Heatmap present v3:', bool(result.get('heatmap')))
