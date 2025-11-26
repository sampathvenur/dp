# Program 6: Develop a dnoisy autoencoder to reconstruct clean images from noisy input data

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# 1. Load & Reshape
(x, _), (xt, _) = mnist.load_data()
x = x.reshape(-1, 28, 28, 1) / 255.0
xt = xt.reshape(-1, 28, 28, 1) / 255.0

# 2. Add Noise (randn is shorter)
xn = np.clip(x + 0.5 * np.random.randn(*x.shape), 0, 1)
xtn = np.clip(xt + 0.5 * np.random.randn(*xt.shape), 0, 1)

# 3. Model (Encode -> Decode)
model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D(2, padding='same'),
    UpSampling2D(2),
    Conv2D(1, 3, activation='sigmoid', padding='same')
])

model.compile('adam', 'binary_crossentropy')
model.fit(xn, x, epochs=5, batch_size=256, validation_data=(xtn, xt))

# 4. Plot Results
p = model.predict(xtn[:10])
# Stack images: Original (top), Noisy (middle), Predicted (bottom)
res = np.vstack([np.hstack(xt[:10]), np.hstack(xtn[:10]), np.hstack(p[:10])])
plt.imshow(res.squeeze(), cmap='gray'); plt.axis('off'); plt.show()