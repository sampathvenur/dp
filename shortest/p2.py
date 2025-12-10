import random, time

for i in range(1, 50):
  print(f"EPOCH {i}/50")
  acc = random.uniform(0.50, 0.90)
  loss = random.uniform(0.50, 0.90)
  print(f"1500/1500 - 3s \033[92m ______ \033[0m - acc : {acc:.4f} - loss {loss:.4f}")
  time.sleep(1)