# Program 3: Implement and compare different optimization algorithms (SGO momentum, adam) for training a sample neural network with a toy dataset.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Data
X = torch.Tensor([[0,0], [0,1], [1,0], [1,1]])
y = torch.Tensor([[0], [1], [1], [0]])

# 2. Training Loop for each Optimizer
for opt_name in ['SGD', 'Momentum', 'Adam']:
    # Define Model (Reset for each loop)
    model = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid())
    criterion = nn.MSELoss()
    
    # Select Optimizer
    if opt_name == 'Adam': 
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
    elif opt_name == 'Momentum': 
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    else: 
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Train
    losses = []
    for _ in range(5000):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Plot & Print
    plt.plot(losses, label=opt_name)
    print(f"Final loss ({opt_name}) : {losses[-1]:.3f}")

# 3. Show Graph
plt.legend()
plt.title('Optimization Algorithm Comparison')
plt.show()