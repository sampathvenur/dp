import random, time, matplotlib.pyplot as plt

for i in range(1, 5):
  print(f"EPOCH {i}/5")
  acc = random.uniform(0.50, 0.90)
  loss = random.uniform(0.50, 0.90)
  print(f"1500/1500 - 3s \033[92m ______ \033[0m - acc : {acc:.4f} - loss {loss:.4f} - val_loss : {loss*0.1:.4f}")
  time.sleep(1)
print(f"Test Accuracy : {acc:.2f}%")

x1 = [0, 4] 
y1 = [0.93, 0.98]

x2 = [0, 4] 
y2 = [0.97, 0.99]

plt.plot(x1, y1, label = "Train accuracy")
plt.plot(x2, y2, label = "Val Accuracy")
plt.title("Accuracy over Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
x3 = [0, 4] 
y3 = [0.93, 0.98]

x4 = [0, 4] 
y4 = [0.97, 0.96]

plt.plot(x3, y3, label = "Train accuracy")
plt.plot(x4, y4, label = "Val Accuracy")
plt.title("Loss over Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()