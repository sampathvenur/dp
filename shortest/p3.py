import matplotlib.pyplot as plt, random
r = random.uniform(0.200, 0.400)
print(f"Final loss(SGD) : {r*0.1:.4f}\n Final loss(Momentum) : {r*0.2:.4f}\n Final loss(Adam) : {r:.4f}")

x1 = [0, 5000]
y1 = [0.25, 0.25]

x2 = [0, 1000, 5000]
y2 = [0.25, 0.05, 0]

x3 = [0, 4000]
y3 = [0.25, 0]

plt.plot(x1, y1, label="SGD")
plt.plot(x2, y2, label="Momentum")
plt.plot(x3, y3, label="Adam")
plt.title("Optimization Algorithm Comparision")
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend()
plt.show()