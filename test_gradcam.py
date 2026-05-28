import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/vocal_armor_best.keras')
img = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
last_conv = 'conv2d_11'

with tf.GradientTape() as tape:
    tape.watch(img)
    x = img
    last_out = None
    for l in model.layers:
        x = l(x)
        if l.name == last_conv:
            last_out = x
    preds = x
    class_channel = preds[:, 0]

grads = tape.gradient(class_channel, last_out)
print('Grads none?', grads is None)
