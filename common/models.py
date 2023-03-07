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
        convx = self.conv(x)
        convx = convx.view(convx.shape[0], -1)
        return self.fc(convx)


class DualDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.ReLU(),
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
        convx = convx.view(convx.shape[0], -1)
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


#WARNING: FOR NEWER NETWORKS, MAKE SURE THAT ALL COMPONENTS OF NETWORK CAN BE CAUGHT IN network_reset METHOD IF YOU WANT MULTIPROCESSING

class NoisyDualDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        #input_shape is shape of observation as (filter, height, width) DO NOT CALL AS BATCH
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        conv_out = self._get_conv_out(input_shape)
        
        self.value_net = nn.Sequential(
            NoisyLinear(conv_out, 256),
            nn.LeakyReLU(),
            NoisyLinear(256, 1)
        )
        self.adv_net = nn.Sequential(
            NoisyLinear(conv_out, 256),
            nn.LeakyReLU(),
            NoisyLinear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        convx = self.conv(x)
        convx = convx.view(convx.shape[0], -1)
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
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        conv_shape = self._get_conv_out(input_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_shape, 516),
            nn.LeakyReLU(),
            nn.Linear(516, 1)
        )
        self.policy = nn.Sequential(
            nn.Linear(conv_shape, 516),
            nn.LeakyReLU(),
            nn.Linear(516, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        convx = conv_out.view(conv_out.shape[0], -1)
        act_v = self.policy(convx)
        value = self.value(convx)
        return act_v, value

class LinearA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.policy = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    
    def forward(self, x):
        x = self.lin(x)
        act_v = self.policy(x)
        value = self.value(x)
        return act_v, value


class BiasedFilter(nn.Module):
    def __init__(self, feature_size, alpha=0.16):
        super().__init__()
        self.alpha = alpha
        self.feature_size = feature_size
        b = nn.Parameter(torch.empty(size=feature_size))
        self.register_parameter('bias', b)
        self.register_buffer("b_noise", torch.zeros_like(b))
    
        self.reset_parameters()

    def forward(self, x):
        self.b_noise.uniform_()
        return x + self.bias * self.b_noise 

    def reset_parameters(self):
        nn.init.uniform_(self.bias, a=0., b=self.alpha)


class SinglyConnected(nn.Module):
    def __init__(self, feature_size, alpha=0.16, bias=True, bias_ratio=1/20):
        super().__init__()
        self.b_percent = bias_ratio
        self.alpha = alpha
        w = nn.Parameter(torch.empty(size=feature_size, dtype=torch.float32))
        self.register_parameter('weight', w)
        if bias:
            b = nn.Parameter(torch.empty(size=feature_size, dtype=torch.float32))
            self.register_parameter('bias', b)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        if self.bias is None:
            return x * self.weight
        return x * self.weight+ self.bias

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=1.0-self.alpha, b=1.0+self.alpha)

        if self.bias is not None:
            nn.init.uniform_(self.bias, a=-self.alpha * self.b_percent, b=self.alpha * self.b_percent)
    

def network_reset(layer):
    #useful for disapearing parameters in multiprocessing cases where a cuda network is shared across processes
    #example: net.apply(network_reset)
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
            layer.reset_parameters()


class deepprint(nn.Module):
    def __init__(self, name, func="rainbow", treshold=1000000, n_skip=1):
        super().__init__()
        self.name = name
        if func =="rainbow":
            self.func = self.rainbow
        self.tres = treshold
        self.id = 0
        self.n_skip = n_skip

    def forward(self, x):
        self.id +=1
        with torch.no_grad():
            if x.mean() > self.tres or x.mean() < -self.tres:
                print(self.name, self.func(x))
                raise MemoryError("(deepprint) Weight explosion caught!")
            elif self.id % self.n_skip ==0:
                print(self.name, self.func(x))
        return x

    @staticmethod
    def zeros(obs):
        a = obs.any()!=0
        return a
    @staticmethod
    def rainbow(x):
        return [x.max().item(), x.mean().item(), x.min().item()]