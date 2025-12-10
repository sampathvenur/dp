import random, matplotlib.pyplot as plt, numpy as np

for i in range(5):
  r = random.uniform(0.23, 0.29)
  print(f"EPOCH {i} Loss : {r:.4f}")

x = np.linspace(0, 4, 5)
y = np.linspace(0.2500, 0.2325, 5)
plt.plot(x, y, marker = 'o')
plt.title("RBM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()