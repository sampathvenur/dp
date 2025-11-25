# Program 9: Implement a restricted boltzmann machine (RBM) for learning binary data representations.

import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Data (Load, Reshape, Binarize in one go)
(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.reshape(-1, 784) / 255.0 > 0.5).astype('float32')
ds = tf.data.Dataset.from_tensor_slices(x).batch(64)

# 2. RBM Class
class RBM(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([784, 128], stddev=0.01))
        self.hb = tf.Variable(tf.zeros([128]))
        self.vb = tf.Variable(tf.zeros([784]))

    def train(self, v):
        # Forward Pass (Positive Phase)
        ph0 = tf.nn.sigmoid(v @ self.W + self.hb)
        h0 = tf.cast(tf.random.uniform(ph0.shape) < ph0, tf.float32) # Bernoulli Sample
        
        # Backward Pass (Reconstruction / Negative Phase)
        pv1 = tf.nn.sigmoid(h0 @ tf.transpose(self.W) + self.vb)
        v1 = tf.cast(tf.random.uniform(pv1.shape) < pv1, tf.float32)
        ph1 = tf.nn.sigmoid(v1 @ self.W + self.hb)
        
        # Update Weights (Contrastive Divergence)
        self.W.assign_add(0.05 * (tf.transpose(v) @ ph0 - tf.transpose(v1) @ ph1) / 64)
        self.vb.assign_add(0.05 * tf.reduce_mean(v - v1, axis=0))
        self.hb.assign_add(0.05 * tf.reduce_mean(ph0 - ph1, axis=0))
        
        return tf.reduce_mean(tf.square(v - v1)) # MSE Loss

# 3. Train Loop
rbm = RBM()
losses = []
for epoch in range(5):
    loss = [rbm.train(batch) for batch in ds] # Train whole dataset
    avg = sum(loss) / len(loss)
    losses.append(avg)
    print(f"Epoch {epoch+1}, Loss: {avg:.4f}")

# 4. Plot
plt.plot(range(1, 6), losses, 'o-')
plt.title("RBM Reconstruction Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.grid(True); plt.show()