import numpy as np
import torch.nn as nn
import torch

def p(x):
    return torch.FloatTensor(np.array(x, copy=False))

class Netw(nn.Module):
    def __init__(self, device="cpu", preprocessor=p):
        super().__init__()
        self.prep = preprocessor
        self.lin = nn.Sequential(
            nn.Linear(2,1),
        )
        self.device = device

    def __call__(self, x):
        return self.lin(x)

device = "cpu"

net = Netw(device)
print(net)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

class RegressionTask:
    def __init__(self, slope, bias, noise_beta):
        assert isinstance(noise_beta, float)
        self.bnoise = noise_beta
        self.slope = slope
        self.bias = bias
        
    def get(self, batch_size, x_lim, x_count):
        noise = (np.random.random(size=(batch_size, x_count)) - 0.5) * self.bnoise
        xs = np.random.randint(-x_lim, x_lim, size=(batch_size,x_count))
        target_v  = xs * self.slope + self.bias + noise

        return xs.astype(np.float32), target_v.astype(np.float32)



'''slope = 0.356
bias = 345.
noise = 14.
batch = 100
xrange = 1000
n_sample = 1

reg = RegressionTask(slope, bias, noise)

x, y = reg.get(batch, xrange, n_sample)

scaling_factor = xrange * slope + bias

for x in range(10000):

    x, y = reg.get(batch, xrange, n_sample)
    x /= scaling_factor
    y /= scaling_factor

    optimizer.zero_grad()
    x = p(x).to(device)
    y = p(y).to(device)
    pred = net(x)
    losses = (y - pred)**2
    loss = losses.mean()
    loss.backward()
    optimizer.step()
    print(loss)
'''


class SubstrationTask:
    def __init__(self):
        self.buffer = None
    def get(self, batch_size):
        input_v = np.random.random(size=(batch_size,2))

        target_v = input_v[:, 0] + input_v[:, 1]
        return input_v, target_v
    
    def create_buffer(self, size):
        self.buffer = self.get(batch_size=size)
    
    def sample(self, size):
        
        idx = np.random.randint(0, len(self.buffer[0]) - 1, size)
        return self.buffer[0][idx], self.buffer[1][idx]



device = "cpu"

env = SubstrationTask()
env.create_buffer(1000)
print(env.sample(10))

lin = nn.Linear(2, 1)
optimizer = torch.optim.SGD(lin.parameters(), lr=1e-3)
lin.to(device)
idx = 0


for x in range(10000):
    optimizer.zero_grad()
    x, y = env.sample(4)
    x = p(x).to(device)
    y = p(y).to(device)
    pred = lin(x)
    losses = (y - pred)**2
    loss = losses.mean()
    loss.backward()
    optimizer.step()
    print(loss)


print([x for x in lin.parameters()])