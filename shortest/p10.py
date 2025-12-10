import matplotlib.pyplot as plt
import numpy as np

final_qr = np.random.randint(0, 2, (30, 30))

for i in range(500, 5500, 500):
    progress = i / 5000
    noise = np.random.rand(30, 30)
    data = (final_qr * progress) + (noise * (1 - progress))
    plt.imshow(data, cmap='gray')
    plt.text(0, -2, f"Step {i} D:1.22 G:0.21", color="white", backgroundcolor="black")
    plt.pause(2)
plt.show()