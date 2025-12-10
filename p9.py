import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Data (Random Binary)
X = np.array([[1,1,0,0,0,0], [0,0,0,1,1,0], [1,1,0,0,0,0], [0,0,1,1,1,0]])
W = np.random.normal(scale=0.01, size=(6, 2))  # Weights
bv, bh = np.zeros(6), np.zeros(2)              # Biases
losses = []

def sigmoid(x): return 1 / (1 + np.exp(-x))

# 2. Train RBM (5 Epochs)
for epoch in range(5):
    # Positive phase
    h_prob = sigmoid(np.dot(X, W) + bh)
    
    # Reconstruction (Negative phase)
    v_recon = sigmoid(np.dot(h_prob, W.T) + bv)
    
    # Update weights (Contrastive Divergence)
    W += 0.1 * (np.dot(X.T, h_prob) - np.dot(v_recon.T, sigmoid(np.dot(v_recon, W) + bh)))
    
    # Calculate Loss (MSE) and Print
    loss = np.mean((X - v_recon) ** 2)
    losses.append(loss)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 3. Plot Output
plt.plot(losses, marker='o')
plt.title("RBM Loss")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()