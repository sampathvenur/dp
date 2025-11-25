# Program 10: Implement a simple generative adversarial network (GAN) on the mnist dataset to generate new handwritten digits.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# 1. Data (Normalize to [-1, 1])
(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.reshape(-1, 28, 28, 1) - 127.5) / 127.5

# 2. Generator (Noise 100 -> Image 28x28)
gen = Sequential([
    Dense(7*7*128, input_dim=100), Reshape((7, 7, 128)),
    Conv2DTranspose(64, 5, strides=2, padding='same', activation='relu'),
    Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh')
])

# 3. Discriminator (Image 28x28 -> Real/Fake)
disc = Sequential([
    Conv2D(64, 5, strides=2, padding='same', input_shape=(28, 28, 1)), LeakyReLU(0.2),
    Dropout(0.3),
    Conv2D(128, 5, strides=2, padding='same'), LeakyReLU(0.2),
    Flatten(), Dense(1, activation='sigmoid')
])
disc.compile(optimizer='adam', loss='binary_crossentropy')

# 4. GAN (Connect Gen to Disc)
disc.trainable = False
gan = Sequential([gen, disc])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# 5. Training Loop
batch = 128
for i in range(3001):
    # Train Discriminator (Real + Fake)
    noise = np.random.normal(0, 1, (batch, 100))
    fake = gen.predict(noise, verbose=0)
    real = x[np.random.randint(0, x.shape[0], batch)]
    
    X = np.concatenate([real, fake])
    y = np.concatenate([np.ones((batch, 1)), np.zeros((batch, 1))])
    d_loss = disc.train_on_batch(X, y)

    # Train Generator (Trick Disc into thinking Fake is Real)
    g_loss = gan.train_on_batch(noise, np.ones((batch, 1)))

    if i % 500 == 0:
        print(f"Step {i} | D Loss: {d_loss:.3f} | G Loss: {g_loss:.3f}")
        plt.imshow(fake[0, :, :, 0], cmap='gray'); plt.axis('off'); plt.show()