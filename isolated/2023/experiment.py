import torch.nn as nn 
import torch
from PR2L import agent
import numpy as np
import torch.nn.functional as F

class BiasedFilter(nn.Module):
    def __init__(self, feature_size, alpha=0.0016):
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

#no use found for this layer
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
    
class FilterAgent(agent.Agent):
    def __init__(self,filter, net, device="cpu", Selector= agent.ProbabilitySelector(), preprocessing=agent.numpytoFloatTensor_preprossesing, inconn=None):
        super().__init__()
        assert isinstance(Selector, agent.ActionSelector)
        if inconn is not None:
            self.inconn = inconn
        self.selector = Selector
        self.net = net
        self.filter = filter
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x):
        x1 = self.preprocessing(x)
        x = self.filter(x1.to(self.device))
        if self.inconn is not None:
            img = np.concatenate((x.cpu().numpy()[0], x1.cpu().numpy()[0]), axis=2)
            self.inconn.send(img)
        act_v = self.net(x)[0]
        act_v = F.softmax(act_v, dim=1)
        actions = self.selector(act_v.cpu().numpy())
        noise = self.filter.b_noise.cpu().numpy()
        return actions, [[noise]] * actions.shape[0]
    

