import numpy as np
import torch


batch_size = 100

linear = torch.nn.Linear(2, 1)

opt = torch.optim.SGD(linear.parameters(), lr=1e-3)

idx = 0
while True:
    idx += 1
    opt.zero_grad()

    x1 = np.random.random(size=batch_size)
    x2 = np.random.random(size=batch_size)

    #define operation here
    y = x1 - x2

    x = np.column_stack((x1, x2))

    x_t = torch.Tensor(x)
    y_t = torch.Tensor(y)

    #training
    out = linear(x_t)

    loss = torch.nn.functional.mse_loss(out.squeeze(-1), y_t)
    loss.backward()
    opt.step()
    
    if idx % 100 == 0:
        print(loss)


    if loss < 0.00001:
        print(loss)
        print(linear.weight, linear.bias)
        break