import torch
import argparse
import os

N, D_in, D_out = 64, 1024, 16

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Linear(D_in, D_out)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

loss_fn = torch.nn.MSELoss()

for t in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()