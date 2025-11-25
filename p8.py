# Program 8: Implement an encoder - decoder architecture with LSTM for sequence to sequence learning

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Data (1000 samples, sequence length 10, vocab size 50)
x = torch.randint(1, 50, (1000, 10))
y = x.clone()  # Target is to reconstruct the input

# 2. Seq2Seq Model (Encoder + Decoder architecture)
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 32)
        self.enc = nn.LSTM(32, 64, batch_first=True)  # Encoder Layer
        self.dec = nn.LSTM(32, 64, batch_first=True)  # Decoder Layer
        self.fc = nn.Linear(64, 50)                   # Output Layer
        
    def forward(self, src, trg):
        # Encoder: Returns hidden state
        _, hidden = self.enc(self.emb(src))
        # Decoder: Uses Encoder's hidden state + Target input
        out, _ = self.dec(self.emb(trg), hidden)
        return self.fc(out)

# 3. Train
model = Seq2Seq()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
losses = []

for epoch in range(15):
    opt.zero_grad()
    output = model(x, x) # Forward pass
    
    # Reshape output to (Total Tokens, Vocab) for Loss function
    loss = criterion(output.reshape(-1, 50), y.reshape(-1))
    
    loss.backward()
    opt.step()
    losses.append(loss.item())
    print(f"EPOCH {epoch+1} | 15, Loss : {loss.item():.4f}")

# 4. Plot
plt.plot(range(1, 16), losses, 'o-')
plt.title("Training loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True); plt.show()