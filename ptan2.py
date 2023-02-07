import torch
from torch import nn 
import numpy as np
import copy
import collections
import gym
import ptan
import cv2

#target networks

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


#Agents

class BaseAgent:

    def __call__(self, obs):

        assert isinstance(obs, list)
        raise NotImplementedError

import common 

class DQNAgent(BaseAgent):

    def __init__(self, net, selector, device="cpu", preprocessor=common.ndarray_preprocessor(common.VariableTensor_preprocessor())): #need changes
        self.net = net
        self.selector = selector
        self.device = device
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, obs):
        
        if self.preprocessor is not None:
            obs = self.preprocessor(obs)
            if torch.is_tensor(obs):
                obs = obs.to(self.device)

        q_v = self.net(obs)
        q = q_v.data.cpu().numpy()
        actions = self.selector(q)
        return actions
    
#Selection

class ActionSelector:

    def __call__(self, scores):

        raise NotImplementedError

    
class ArgmaxActionSelector(ActionSelector):

    def __call__(self, scores):
        #consider using torch.argmax
        assert isinstance(scores, np.ndarray)

        return np.argmax(scores, axis = 1)

class EpsilonGreedyActionSelector(ActionSelector):

    def __init__(self, selector=ArgmaxActionSelector(), epsilon=0.05):
        self.epsilon = epsilon
        self.selector = selector
    
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions

Experience = collections.namedtuple('Experience', ['state', 'action', 'reward', 'done'])


def FirstOnlyResetSelector(obs):
    return obs[0]

#Experience

class ExperienceSource:

# works as n-step experience source for single or multiple environments

    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False, ResetSelector=FirstOnlyResetSelector):
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert isinstance(vectorized,bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]

        self.ResetSelector = ResetSelector
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.vectorized = vectorized
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):

        states, histories, cur_rewards, cur_steps = [], [], [], []
        env_lens = []


        for env in self.pool:
            obs = env.reset()

            if self.ResetSelector is not None:
                obs = self.ResetSelector(obs)

            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)

            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(collections.deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)


        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []

            for idx, state in enumerate(states): #expect states to be [obs, obs, ...]
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()
                else:
                    states_input.append(state)
                    states_indices.append(idx)

            if states_input:
                states_actions = self.agent(states_input) #expect assert error (need ndarray)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action

            grouped_actions = _group_list(actions, env_lens)
            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done_n, _ = env.step(action_n) 
                else:
                    next_state, r, is_done, _ = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                for ofs, (actions, next_state, r, is_done) in enumerate(zip(action_n,next_state_n, r_n, is_done_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    cur_rewards[idx] += r
                    cur_steps[idx] +=1
                    
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        #agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1
                
    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res




def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))

class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)




class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)