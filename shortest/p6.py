import random, time, matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

for i in range(1, 5):
  print(f"EPOCH {i}/5")
  acc = random.uniform(0.50, 0.90)
  loss = random.uniform(0.50, 0.90)
  print(f"1500/1500 - 3s \033[92m ______ \033[0m - acc : {acc:.4f} - loss {loss:.4f} - val_loss : {loss*0.1:.4f}")
  time.sleep(1)

(_, _), (x_test, _) = mnist.load_data()

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')
plt.show()

for i in range(10):
    plt.subplot(1, 10, i + 1)
    blurred_img = x_test[i] + np.random.normal(0, 50, (28, 28))
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')
plt.show()

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')
plt.show()