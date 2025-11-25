# Program 7: Construct a Bidirectional RNN and compare it's performance with a standard RNN on sequence prediction tasks.

import torch
import torch.nn as nn

# 1. Data (100 samples, sequence length 9)
# We treat input as numbers 0 to 899. Target is just Input + 1.
x = torch.arange(900).float().view(100, 9, 1)
y = x + 1

# 2. Universal Model (Handles both RNN and BiRNN)
class Net(nn.Module):
    def __init__(self, bi):
        super().__init__()
        # If bidirectional, hidden size doubles, so we adjust linear input
        self.rnn = nn.RNN(1, 32, batch_first=True, bidirectional=bi)
        self.fc = nn.Linear(64 if bi else 32, 1)
    
    def forward(self, x):
        # rnn returns (output, hidden). We only need output.
        return self.fc(self.rnn(x)[0])

# 3. Train Function
def train(bi_flag):
    model = Net(bi_flag)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()

# 4. Compare & Print
print(f"Final loss RNN {train(False):.4f}")
print(f"Final loss BiRNN {train(True):.4f}")