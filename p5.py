# Program 5: Implement a LSTM based RNN for text classification to demonstrate sequence modelling, unfolding, computational graphs and handling long term dependencies.

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 1. Load & Pad Data
(X, y), (Xt, yt) = imdb.load_data(num_words=20000)
X = pad_sequences(X, maxlen=200, padding='post')
Xt = pad_sequences(Xt, maxlen=200, padding='post')

# 2. Model (Embedding -> LSTM -> Dropout -> Dense)
model = Sequential([
    Embedding(20000, 64),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 3. Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(X, y, epochs=5, batch_size=64, validation_split=0.2)

# 4. Evaluate
print(f"Test Accuracy : {model.evaluate(Xt, yt, verbose=0)[1]:.4f}")

# 5. Plot (Loop generates both Accuracy and Loss graphs)
for k in ['accuracy', 'loss']:
    plt.plot(h.history[k], label='Train')
    plt.plot(h.history['val_'+k], label='Validation')
    plt.xlabel('Epochs'); plt.ylabel(k.capitalize()) 
    plt.title(f'{k.capitalize()} over Epoch'); plt.legend(); plt.show()