# Program 8: Implement an encoder - decoder architecture with LSTM for sequence to sequence learning

import torch, torch.nn as nn, matplotlib.pyplot as plt

# 1. Data
x = torch.randint(0, 50, (1000, 10))

# 2. Model
class S2S(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(50, 32)
        self.e = nn.LSTM(32, 64, batch_first=True) # Encoder
        self.d = nn.LSTM(32, 64, batch_first=True) # Decoder
        self.f = nn.Linear(64, 50)
        
    def forward(self, s, t):
        _, h = self.e(self.E(s))       # Encoder
        y, _ = self.d(self.E(t), h)    # Decoder
        return self.f(y)

# 3. Train
m = S2S()
opt = torch.optim.Adam(m.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
hist = []

for i in range(15):
    opt.zero_grad()
    # view(-1, 50) flattens data for loss function
    loss = loss_fn(m(x, x).view(-1, 50), x.view(-1))
    loss.backward(); opt.step(); hist.append(loss.item())
    print(f"EPOCH {i+1} | 15, Loss : {loss.item():.4f}")

# 4. Plot
plt.plot(range(1, 16), hist, 'o-')
plt.title("Training loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True); plt.show()