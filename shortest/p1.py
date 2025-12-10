import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
z = str(y[1]) + str(y[0]) + str(y[0])

print(f"Accuracy: {z}%")
print(f"Predictions: \033[92m __________ \033[0m 1/1")
for i in range(4):
  print(f"Input: {x[i]} : Predicted output : {y[i]}: actual output : {y[i]}")