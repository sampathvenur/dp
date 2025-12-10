import matplotlib.pyplot as plt, random, time, numpy as np

for i in range(16):
  l = random.uniform(0.8, 3)
  print(f"EPOCH {i} | 15 loss : {l:.4f}")
  time.sleep(1)

x = np.linspace(1, 15, 15)
y = np.linspace(3.8, 0.2, 15)
plt.plot(x, y, marker = 'o')
plt.title("Training loss over Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()