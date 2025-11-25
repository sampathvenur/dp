# Program 1: solving XOR problem using multiplayer perceptron.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data Preparation
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# 2. Define Model (input_dim=2 replaces the Input layer)
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 3. Compile & Train (increased epochs to ensure 100% accuracy)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=2000, verbose=0)

# 4. Evaluate & Output Accuracy
print(f"Accuracy : {model.evaluate(X, y, verbose=0)[1]*100:.2f}%")

# 5. Predict & Format Output
print("\nPredictions: ")
p = (model.predict(X) > 0.5).astype(int)

for i in range(4):
    print(f"Input : {X[i]} - Predicted Output : {p[i][0]} - True Output : {y[i]}")