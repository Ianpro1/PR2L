import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import numpy as np
from common import models, extentions
import gym
import math
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from PR2L import common_wrappers, agents, experience, utilities
import multiprocessing as mp
import common.Rendering as rendering
from collections import deque
import torch.utils.tensorboard as tensorboard
ENV_NAME = "CartPole-v1"
ENTROPY_BETA = 0.01
N_STEPS = 4
EPSILON_START = 1.0
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.001
CLIP_GRAD = 0.1
BETA_POLICY = 1.0
device = "cuda"
NUM_ENV = 50

env = []
for e in range(NUM_ENV):
    env.append(e)
preprocessor = agents.numpytoFloatTensor_preprossesing
net = models.LinearA2C((4,), 2).to(device)
agent = agents.PolicyAgent(net, device)
exp_source = experience.ExperienceSourceV2(env, agent, N_STEPS, GAMMA)
buffer = []
for idx, exp in enumerate(exp_source):
    for rewards, steps in exp_source.pop_rewards_steps():
        print('idx %d, rewards %.3f, steps %.3f')
    
    buffer.append(exp)

    if len(buffer) < BATCH_SIZE:
        continue

    states, actions, rewards, last_states, not_dones = utilities.unpack_batch(buffer)
    buffer.clear()
    
    states = preprocessor(states).to(device)
    rewards = preprocessor(rewards).to(device)
    #actions = torch.LongTensor(np.array(actions))

    if len(last_states) < 1:
        tgt_q_v = rewards
    else:
        next_r = torch.zeros_like(states)
        next_r[not_dones] = net(last_states)[1]
        tgt_q_v = rewards + next_r * math.pow(GAMMA, N_STEPS)

    adv, baseline = net(states)

    value_loss = F.mse_loss(baseline.squeeze(-1), tgt_q_v)

    G_v = tgt_q_v - baseline.detach()
    log_probs = F.log_softmax(adv, dim=1)
    log_prob_a = log_probs[range(BATCH_SIZE), actions]
    policy = -BETA_POLICY * log_prob_a * G_v
    policy_loss = policy.mean()

    probs = F.softmax(adv, dim=1)
    ent = (probs * log_probs).sum(dim=1).mean()
    entropy_loss = -ENTROPY_BETA * ent

    loss = entropy_loss + policy_loss + value_loss

