# Program 6: Develop a dnoisy autoencoder to reconstruct clean images from noisy input data

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# 1. Data Prep & Noise
(x, _), (xt, _) = tf.keras.datasets.mnist.load_data()
x = x.reshape(-1, 28, 28, 1) / 255.0
xt = xt.reshape(-1, 28, 28, 1) / 255.0

# Create Noisy Data
xn = np.clip(x + 0.3 * np.random.normal(0, 1, x.shape), 0, 1)
xtn = np.clip(xt + 0.3 * np.random.normal(0, 1, xt.shape), 0, 1)

# 2. Autoencoder Model (Encoder -> Decoder)
model = Sequential([
    # Encoder
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D(2, padding='same'),
    Conv2D(32, 3, activation='relu', padding='same'),
    MaxPooling2D(2, padding='same'),
    
    # Decoder
    Conv2D(32, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(32, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(1, 3, activation='sigmoid', padding='same')
])

# 3. Train
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(xn, x, epochs=5, batch_size=256, validation_data=(xtn, xt))

# 4. Predict & Plot (Row 1: Original, Row 2: Noisy, Row 3: Result)
preds = model.predict(xtn)
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(3, 10, i + 1); plt.imshow(xt[i], cmap='gray'); plt.axis('off')
    plt.subplot(3, 10, i + 11); plt.imshow(xtn[i], cmap='gray'); plt.axis('off')
    plt.subplot(3, 10, i + 21); plt.imshow(preds[i], cmap='gray'); plt.axis('off')
plt.show()