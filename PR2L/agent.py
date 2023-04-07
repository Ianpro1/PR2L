#agent includes reinforcement learning agents as well as simple action selectors
import torch
import torch.nn.functional as F
import numpy as np
import copy

class ActionSelector:
    """
    Action Selector class required by some agent classes
    """
    def __call__(self, x):
        raise NotImplementedError

class ArgmaxSelector(ActionSelector):
    """Argmax action sampling"""
    def __init__(self):
        super().__init__()

    def __call__(self,x):
        return np.argmax(x,axis=1)

class ProbabilitySelector(ActionSelector):
    """Action sampling that uses the inputs as probabilies"""
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        actions = []
        for probs in x:
            actions.append(np.random.choice(len(probs), p=probs))
        return np.array(actions)


class EpsilonGreedySelector(ActionSelector):
    """Epsilon Greedy sampling of actions"""
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


def preprocessing(x):
    return torch.tensor(np.array(x, copy=False))

def float32_preprocessing(x):
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
    """Agent class required for all Experience processing classes"""
    def __call__(self):
        raise NotImplementedError
    
    def initial_state(self):
        return None


class BasicAgent(Agent):
    """An agent that samples the action (single output returned by network) using ArgmaxSelector()"""
    def __init__(self, net, device="cpu", Selector= ArgmaxSelector(), preprocessing=float32_preprocessing):
        super().__init__()
        assert isinstance(Selector, ActionSelector)
        self.selector = Selector
        self.net = net
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x, internal_states):
        x = self.preprocessing(x)
        values = self.net(x.to(self.device))
        actions = self.selector(values.data.cpu().numpy())
        return actions, internal_states


class PolicyAgent(Agent):
    """An agent that samples the action (from first argument returned by network) using ProbabilitySelector()"""
    def __init__(self, net, device="cpu", Selector= ProbabilitySelector(), preprocessing=float32_preprocessing):
        super().__init__()
        assert isinstance(Selector, ActionSelector)
        self.selector = Selector
        self.net = net
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x, internal_states):
        x = self.preprocessing(x)
        act_v = self.net(x.to(self.device))[0]
        act_v = F.softmax(act_v, dim=1)
        actions = self.selector(act_v.data.cpu().numpy())
        return actions, internal_states


class ContinuousNormalAgent(Agent):
    """An agent that receives 3 matrices: mean, variance and other (usually value) 
    and samples the actions using Normal/Gaussian distribution"""
    def __init__(self, net, device="cpu", preprocessor=float32_preprocessing, clipping=True):
        super().__init__()
        self.net = net
        self.device = device
        self.preprocessor = preprocessor
        self.clipping = clipping

    @torch.no_grad()
    def __call__(self, states, internal_states):
        states_v = self.preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        if self.clipping:
            actions = np.clip(actions, -1, 1)
        return actions, internal_states
    
