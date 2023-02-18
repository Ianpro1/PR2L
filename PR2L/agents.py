import torch
import numpy as np

class ActionSelector:
    def __call__(self, x):
        raise NotImplementedError

class ArgmaxSelector(ActionSelector):

    def __init__(self):
        super().__init__()

    def __call__(self,x):
        return x.argmax(dim=1)

class Agent:
    def __call__(self):
        raise NotImplementedError

def numpytotensor_preprossesing(x):
    return torch.tensor(np.array(x))

class BasicAgent(Agent):
    def __init__(self, net, device="cpu", Selector= ArgmaxSelector(), preprocessing=numpytotensor_preprossesing):
        super().__init__()
        assert isinstance(Selector, ActionSelector)
        self.selector = Selector
        self.net = net
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x)
        x.to(self.device)
        values = self.net(x)
        actions = self.selector(values)
        return actions.cpu().numpy()