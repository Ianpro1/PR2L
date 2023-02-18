import common.extentions as ex
import common.models as models
import torch
import torch.nn as nn
import gym
import numpy as np
from collections import namedtuple, deque

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
        

Experience = namedtuple("Experience", ("state", "action", "reward", "done"))


#NEED TO VERIFY THAT ALL ACTIONS ARE NOT SHUFFLED THROUGH EXPERIENCES
class ExperienceSource:
    def __init__(self, env, agent):
        assert isinstance(agent, Agent)
        assert isinstance(env, (gym.Env, list, tuple))
        if isinstance(env, (list, tuple)):
            self.env = env
            env_len = len(env)
        else: 
            self.env = [env]
            env_len = 1
        self.agent = agent
        self.env_len = env_len        
        self.tot_reward = [0.]*env_len
        self.tot_rewards = []
        self.tot_step = [0.]*env_len
        self.tot_steps = []

    def __iter__(self):
        #initialization setup
        states, rewards, dones, nextstates = deque(maxlen=self.env_len), deque(maxlen=self.env_len), deque(maxlen=self.env_len), deque(maxlen=self.env_len)
        
        for env in self.env:
            obs, _ = env.reset()
            states.append(obs)
            nextstates.append(None)

        while True:
            
            actions = self.agent(states)
            
            for i, env in enumerate(self.env):
                obs, reward, done, info, _ = env.step(actions[i])
                rewards.append(reward)
                dones.append(done)
                nextstates.append(obs)
                self.sum_rewards_steps(reward, done, i)

            exp = Experience(states, actions, rewards, dones)
            yield exp
            
            for env in self.env:
                if dones.popleft():
                    obs, _ = env.reset()
                    states.append(obs)
                    nextstates.popleft()
                else:
                    states.append(nextstates.popleft())

    def sum_rewards_steps(self, reward, done, env_id):
        #keeps track of rewards and steps
        self.tot_step[env_id] += 1
        self.tot_reward[env_id] += reward
        if done:
            self.tot_rewards.append(self.tot_reward[env_id])
            self.tot_reward[env_id] = 0
            self.tot_steps.append(self.tot_step[env_id])
            self.tot_step[env_id] = 0

    def pop_rewards_steps(self):
        res = list(zip(self.tot_rewards, self.tot_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


def unpack_exp(exp, env_len): #might be suboptimal
    states = [[]*env_len]
    actions = [[]*env_len]
    rewards = [[]*env_len]
    dones = [[]*env_len]
    for i in range(env_len):
        states[i].append(exp.state.popleft())
        actions[i].append(exp.action.popleft())
        rewards[i].append(exp.reward.popleft())
        dones[i].append(exp.done.popleft())
    return states, actions, rewards, dones


class FirstLastExperienceSource(ExperienceSource):
    def __init__(self, env, agent, n_steps=2):
        super().__init__(env, agent, n_steps)

    def __iter__(self):
        
        #goup individual experiences
        for exp in super().__iter__():
            #states, actions, rewards, dones = unpack_exp(exp, self.env_len) -> suboptimal

            #before popping the deque into the remainders deque, verify the content!
            pass


            




import common.playground as pg

device = "cpu"
'''env = gym.make("Breakout-v4")

class wrapobs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, obs):
        return np.moveaxis(obs, -1, 0).astype(np.float32)

env = wrapobs(env)'''

def wrapper(x):
    x = pg.VaryObservation()(x)
    return pg.ScaleRGBimage()(x)

env1 = pg.DummyEnv((1), 1.,None, wrapper, pg.EpisodeLength(3))
env2 = pg.DummyEnv((1), 2.,None, wrapper, pg.EpisodeLength(4))
env3 = pg.DummyEnv((1), 3.,None, wrapper, pg.EpisodeLength(2))

env = [env1, env2, env3]

net = models.DenseDQN(1,10,4)

agent = BasicAgent(net)

exp_source = ExperienceSource(env, agent)

#import ptan
#import common.atari_wrappers as aw
#agent2 = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector())
#exp_source2 = ptan.experience.ExperienceSource(env, agent2, 3)

for i, x in enumerate(exp_source):
    print(x, '\n')
    if i >100:
        break



