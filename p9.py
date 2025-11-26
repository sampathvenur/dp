# Program 9: Implement a restricted boltzmann machine (RBM) for learning binary data representations.

import tensorflow as tf, matplotlib.pyplot as plt

# 1. Data
(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.reshape(-1, 784) / 255 > 0.5).astype('float32')

# 2. Variables
W = tf.Variable(tf.random.normal([784, 128], 0.01))
hb, vb = tf.Variable(tf.zeros([128])), tf.Variable(tf.zeros([784]))

# 3. Train
errors = []
for i in range(5):
    for j in range(0, 60000, 64):
        v0 = x[j : j+64]
        h0 = tf.sigmoid(v0 @ W + hb)
        h0_s = tf.cast(tf.random.uniform([len(v0), 128]) < h0, tf.float32)
        v1 = tf.sigmoid(h0_s @ tf.transpose(W) + vb)
        h1 = tf.sigmoid(v1 @ W + hb)

        W.assign_add(0.05 * (tf.transpose(v0)@h0 - tf.transpose(v1)@h1) / 64)
        vb.assign_add(0.05 * tf.reduce_mean(v0 - v1, 0))
        hb.assign_add(0.05 * tf.reduce_mean(h0 - h1, 0))

    # Calculate Loss on full data for the graph
    err = tf.reduce_mean(tf.square(x - tf.sigmoid(tf.sigmoid(x@W+hb) @ tf.transpose(W)+vb)))
    errors.append(err)
    print(f"Epoch {i+1}, Loss: {err:.4f}")

# 4. Plot
plt.plot(errors, 'o-'); plt.title("RBM Loss"); plt.xlabel("Epoch"); plt.grid(True); plt.show()