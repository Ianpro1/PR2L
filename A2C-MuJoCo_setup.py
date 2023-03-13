#MuJoCo env is used through gymnasium instead of the outdated openai gym
#tutorial: https://mujoco.readthedocs.io/en/latest/programming/#getting-started


#here is an implementation of the A2C method

import gymnasium as gym
from PR2L import rendering, agent, experience, training
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import math

ENV_ID = "InvertedPendulum-v4"
HIDDEN = 128
GAMMA = 0.99
N_STEPS = 2
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4
TEST_ITERS = 100
NUM_ENVS = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

class ContinuousA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HIDDEN),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(HIDDEN, act_size),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(HIDDEN, act_size),
            nn.Softplus()
        )

        self.value = nn.Linear(HIDDEN, 1)

    
    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)

class ContinuousA2CAgent(agent.Agent):
    def __init__(self, net, device="cpu", preprocessor=agent.float32_preprocessing):
        super().__init__()
        self.net = net
        self.device = device
        self.preprocessor = preprocessor


    @torch.no_grad()
    def __call__(self, states):
        states_v = self.preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions

def test_net(net, env, count=10, device="cpu", preprocessor=agent.float32_preprocessing):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = preprocessor([obs])
            obs_v = obs_v.to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count

def calc_logprob(mu_v, var_v, actions_v):
    p1 = -((mu_v - actions_v)**2) / (2*var_v.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2*math.pi *var_v))
    return p1 + p2

class LengthWrapper(gym.Wrapper):
    def __init__(self, env, episode_length):
        super().__init__(env)
        self.ep_len = episode_length
        self.step_count = 0

    def reset(self):
        obs = self.env.reset()
        self.step_count = 0
        return obs
    
    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        self.step_count += 1
        if self.ep_len <= self.step_count:
            done = True
        
        return obs, rew, done, trunc, info

def make_env(ENV_NAME, queue=None, episode_length=1000):
    
    if queue is not None:
        print("created new render env...")
        env = gym.make(ENV_NAME, render_mode='rgb_array')
        env =  rendering.ReplayWrapper(env, queue)
        if episode_length != -1:
            env = LengthWrapper(env, episode_length)
    else:
        env = gym.make(ENV_NAME)
        if episode_length != -1:
            env = LengthWrapper(env, episode_length)
    return env

def build_envs(ENV_NAME, number=1, queue=None, episode_length=1000):
    envs = []
    if queue is not None:
        envs = [make_env(ENV_NAME, None, episode_length) for _ in range(number-1)]
        envs.append(make_env(ENV_NAME, queue, episode_length))
    else:
        envs = [make_env(ENV_NAME, None, episode_length) for _ in range(number)]
    return envs

if __name__ == "__main__":
#problem is that environments seem to reset themselves without the env.step() call
    replayqueue = mp.Queue(maxsize=1)

    envs = build_envs(ENV_ID, NUM_ENVS, replayqueue, -1)
    test_env = make_env(ENV_ID)
    action_shape = envs[0].action_space._shape[0]
    observation_shape = envs[0].observation_space._shape[0]
    
    #this type of display shouldn't be used on single environments
    display = mp.Process(target=rendering.init_replay, args=(replayqueue, 10))
    display.start()
    
    net = ContinuousA2C(observation_shape, action_shape)
    net.to(device)

    _agent = ContinuousA2CAgent(net, device)
    exp_source = experience.ExperienceSource(envs, _agent, track_rewards=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    batch = []
    idx = 0
    while True:
        idx += 1
        #print(idx)
        for i, exp in enumerate(exp_source):
            batch.append(exp)
            if i >= BATCH_SIZE:
                break
        
        states, actions, refs_q_v = training.unpack_batch_A2C(batch, net, GAMMA, N_STEPS, device, value_index_pos=2)
        batch.clear()

        optimizer.zero_grad()

        mu_v, var_v, values = net(states)
        value_loss = F.mse_loss(values.squeeze(-1), refs_q_v)

        adv_v = refs_q_v.unsqueeze(-1) - values
        log_prob_v = calc_logprob(mu_v, var_v, actions)
        policy_loss = -(adv_v * log_prob_v).mean()
        
        ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
        entropy_loss = ENTROPY_BETA * ent_v.mean()

        loss = policy_loss + entropy_loss + value_loss

        loss.backward()
        optimizer.step()

        if idx % TEST_ITERS == 0:
            rewards, steps = test_net(net, test_env, device=device)
            print('steps %d, rewards %.2f)' % (steps, rewards))
            if rewards > 100000:
                break
    
    replayqueue.put(None)
    display.join()

        

