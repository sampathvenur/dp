# Program 2: Implement regularization techniques in deep learning models using parameters norm penalties, dataset augmentation & noise robustness for improved generalization.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load & Preprocess (Reshape & Normalize)
(X, y), _ = tf.keras.datasets.mnist.load_data()
X = X.reshape(-1, 28, 28, 1) / 255.0

# 2. Add Noise (Noise Robustness)
X = np.clip(X + np.random.normal(0, 0.1, X.shape), 0, 1)

# 3. Augmentation (Dataset Augmentation)
dg = ImageDataGenerator(rotation_range=10, validation_split=0.2)

# 4. Model (L2 Penalty & Dropout for Regularization)
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 5. Compile & Train (Using sparse_categorical_crossentropy saves label conversion)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train with Generator & Early Stopping
model.fit(dg.flow(X, y, subset='training'), 
          validation_data=dg.flow(X, y, subset='validation'), 
          epochs=50, 
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])