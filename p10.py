# Program 10: Implement a simple generative adversarial network (GAN) on the mnist dataset to generate new handwritten digits.

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# 1. Data
(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.reshape(-1, 28, 28, 1) - 127.5) / 127.5

# 2. Models
g = Sequential([
    Dense(7*7*128, input_dim=100), Reshape((7, 7, 128)),
    Conv2DTranspose(64, 5, 2, 'same', activation='relu'),
    Conv2DTranspose(1, 5, 2, 'same', activation='tanh')
])

d = Sequential([
    Conv2D(64, 5, 2, 'same', input_shape=(28, 28, 1)), LeakyReLU(0.2),
    Conv2D(128, 5, 2, 'same'), LeakyReLU(0.2),
    Flatten(), Dense(1, activation='sigmoid')
])
d.compile('adam', 'binary_crossentropy')

d.trainable = False
gan = Sequential([g, d])
gan.compile('adam', 'binary_crossentropy')

# 3. Training
for i in range(3001):
    z = np.random.randn(128, 100)
    fake = g.predict(z, verbose=0)
    real = x[np.random.randint(0, 60000, 128)]
    
    # Train D (Real then Fake)
    d_loss = d.train_on_batch(real, np.ones((128, 1))) + d.train_on_batch(fake, np.zeros((128, 1)))
    
    # Train G
    g_loss = gan.train_on_batch(z, np.ones((128, 1)))

    if i % 500 == 0:
        print(f"Step {i} D:{d_loss/2:.2f} G:{g_loss:.2f}")
        plt.imshow(fake[0,:,:,0], cmap='gray'); plt.show()