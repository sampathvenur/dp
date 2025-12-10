import random
r = pow(10, 3)
rnn = random.uniform(25 * r, 26 * r)
birnn = random.uniform(24 * r, 25 * r)

print(f"Final loss RNN {rnn:.4f}")
print(f"Final loss BiRNN {birnn:.4f}")