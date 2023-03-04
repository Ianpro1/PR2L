import torch
import torch.nn.functional as F
import numpy as np
import copy

class ActionSelector:
    def __call__(self, x):
        raise NotImplementedError

class ArgmaxSelector(ActionSelector):
    def __init__(self):
        super().__init__()

    def __call__(self,x):
        return np.argmax(x,axis=1)

class ProbabilitySelector(ActionSelector):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        actions = []
        for probs in x:
            actions.append(np.random.choice(len(probs), p=probs))
        return np.array(actions)


class EpsilonGreedySelector(ActionSelector):
    def __init__(self, epsilon, selector=None):
        super().__init__()
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxSelector()

    def __call__(self, x):
        batch_size, n_actions = x.shape
        actions = self.selector(x)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_action = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_action
        return actions


def numpytotensor_preprossesing(x):
    return torch.tensor(np.array(x, copy=False))

def numpytoFloatTensor_preprossesing(x):
    return torch.FloatTensor(np.array(x, copy=False))

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


class Agent:
    def __call__(self):
        raise NotImplementedError
    

class BasicAgent(Agent):
    def __init__(self, net, device="cpu", Selector= ArgmaxSelector(), preprocessing=numpytoFloatTensor_preprossesing):
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
        actions = self.selector(values.cpu().numpy())
        return actions

class PolicyAgent(Agent):
    def __init__(self, net, device="cpu", Selector= ProbabilitySelector(), preprocessing=numpytoFloatTensor_preprossesing):
        super().__init__()
        assert isinstance(Selector, ActionSelector)
        self.selector = Selector
        self.net = net
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x)
        act_v, _ = self.net(x.to(self.device))
        act_v = F.softmax(act_v, dim=1)
        actions = self.selector(act_v.cpu().numpy())
        return actions
