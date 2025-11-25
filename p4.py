# Program 4: Build a CNN on mnist dataset to demonstrate convolutional, pooling and classification

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

# 1. Load & Preprocess
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# 2. Model (Conv -> Pool -> Flat -> Dense)
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Train (batch_size=64 ensures the progress bar says 844/844 like your screenshot)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 4. Evaluate
print(f"Test accuracy : {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")

# 5. Plot
plt.plot(h.history['accuracy'], label='Train Accuracy')
plt.plot(h.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy'); plt.legend(); plt.grid(True); plt.show()