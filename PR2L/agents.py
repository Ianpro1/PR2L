import torch
import numpy as np
import copy

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

def numpytofloattensor_preprossesing(x):
    return torch.FloatTensor(np.array(x))

class BasicAgent(Agent):
    def __init__(self, net, device, Selector= ArgmaxSelector(), preprocessing=numpytotensor_preprossesing):
        super().__init__()
        assert isinstance(Selector, ActionSelector)
        self.selector = Selector
        self.net = net
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x)
        
        values = self.net(x.to(self.device))
        actions = self.selector(values)
        return actions.cpu().numpy()


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)