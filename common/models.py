from torch import nn
import torch
import numpy as np
import math
import torch.nn.functional as F
class DenseDQN(nn.Module):
    def __init__(self, input ,HIDDEN, output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, output)
        )

    def forward(self, x):
        return self.net(x)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.flat = nn.Flatten(1, 3)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.flat(self.conv(x))
        #print(conv_out.shape)
        return self.fc(conv_out)


class DuelDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(1,3)
        )

        conv_out = self._get_conv_out(input_shape)
        self.value_net = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.adv_net = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        convx = self.conv(x)
        values = self.value_net(convx)
        adv = self.adv_net(convx)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        adv_v = adv-adv_mean
        return values + adv_v


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer('epsilon_weight', z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)


class NoisyFactorizedLinear(nn.Linear):
 def __init__(self, in_features, out_features,sigma_zero=0.4, bias=True):
    super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
    sigma_init = sigma_zero / math.sqrt(in_features)
    w = torch.full((out_features, in_features), sigma_init)
    self.sigma_weight = nn.Parameter(w)
    z1 = torch.zeros(1, in_features)
    self.register_buffer("epsilon_input", z1)
    z2 = torch.zeros(out_features, 1)
    self.register_buffer("epsilon_output", z2)
    if bias:
        w = torch.full((out_features,), sigma_init)
        self.sigma_bias = nn.Parameter(w)
        
 def forward(self, input):
    self.epsilon_input.normal_()
    self.epsilon_output.normal_()
    func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
    eps_in = func(self.epsilon_input.data)
    eps_out = func(self.epsilon_output.data)
    bias = self.bias
    if bias is not None:
        bias = bias + self.sigma_bias * eps_out.t()
    noise_v = torch.mul(eps_in, eps_out)
    v = self.weight + self.sigma_weight * noise_v
    return F.linear(input, v, bias)


class NoisyDuelDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(1,-1)
        )

        conv_out = self._get_conv_out(input_shape)
        self.value_net = nn.Sequential(
            NoisyLinear(conv_out, 256),
            nn.ReLU(),
            NoisyLinear(256, 1)
        )
        self.adv_net = nn.Sequential(
            NoisyLinear(conv_out, 256),
            nn.ReLU(),
            NoisyLinear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        convx = self.conv(x)
        values = self.value_net(convx)
        adv = self.adv_net(convx)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        adv_v = adv-adv_mean
        return values + adv_v


class A2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(1,-1)
        )

        conv_shape = self._get_conv_out(input_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.policy = nn.Sequential(
            nn.Linear(conv_shape, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        act_v = self.policy(conv_out)
        value = self.value(conv_out)
        return act_v, value

        




