import gymnasium as gym
import PR2L as p2
from PR2L.agent import float32_preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PR2L.rendering import displayScreen
import multiprocessing as mp

class A2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_shape, 516),
            nn.LeakyReLU(),
            nn.Linear(516, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        self.policy = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_actions)
        )

    def get_logits(self, x):
        out = self.base(x)
        act_v = self.policy(out)
        return act_v
    
    def get_values(self, x):
        out = self.base(x)
        value = self.value(out)
        return value
    
    def forward(self, x):
        out = self.base(x)
        act_v = self.policy(out)
        value = self.value(out)
        return act_v, value

def swap(img):
    return img.transpose(1,0,2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAE_LAMBDA = 0.95
GAMMA = 0.99
REPLAY_SIZE = 2000
LEARNING_RATE = 1e-4
PPO_EPSILON = 0.2
PPO_BATCH_SIZE = 50
PPO_EPOCHES = 10
GRAD_CLIP = 0.2
ENV_TAG = "CartPole-v1"

class render_wrapper:
    def __init__(self, env, inconn):
        self.env = env
        self.inconn = inconn
    
    def step(self, action):
        obs_tuple = self.env.step(action)
        self.inconn.send(self.env.render())
        return obs_tuple

    def reset(self):
        obs_tuple = self.env.reset()
        self.inconn.send(self.env.render())
        return obs_tuple

if __name__ == "__main__":
    inconn, outconn = mp.Pipe()
    env = gym.make(ENV_TAG)
    net = A2C(env.observation_space.shape[0], env.action_space.n).to(device)
    agent = p2.agent.PolicyAgent(net, device)
    render_agent = p2.agent.PolicyAgent(net, device, Selector=p2.agent.ArgmaxSelector())
    render_env = render_wrapper(gym.make(ENV_TAG, render_mode="rgb_array"), inconn)
    opt = torch.optim.Adam(net.parameters(), LEARNING_RATE)

    idx = 0
    
    exp_source = p2.experience.SingleExperienceSource(env, agent)
    render_source = iter(p2.experience.EpisodeSource(render_env, render_agent))
    smooth_reward = []
    smooth_steps = []
    batch = []
    
    #rendering
    screen = mp.Process(target=displayScreen, args=(outconn,1,swap))
    screen.start()
    next(render_source)
    for exp in exp_source:
        batch.append(exp)
        if (len(batch) < REPLAY_SIZE + 1):
            continue
        #this approach doesn't seem to work
        batch, old_adv_v, refs_q_v = p2.training.calc_gae(batch, net.get_values, GAMMA, GAE_LAMBDA, device)
        if (idx % 10 == 0):
            print(idx)
        idx += 1

        states = [exp.state for exp in batch]
        actions = [exp.action for exp in batch]

        with torch.no_grad():
            states = float32_preprocessing(states).to(device)
            actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
            
            logits = net.get_logits(states)
            old_log_prob_a = F.log_softmax(logits, dim=1)
            old_log_prob_a = old_log_prob_a[range(len(old_log_prob_a)), actions]


        for epoch in range(PPO_EPOCHES):
            for batch_ofs in range(0, REPLAY_SIZE, PPO_BATCH_SIZE):
                batch_end = batch_ofs + PPO_BATCH_SIZE
                states_v = states[batch_ofs:batch_end]
                actions_v = actions[batch_ofs:batch_end]
                adv_v = old_adv_v[batch_ofs:batch_end]
                refs_v = refs_q_v[batch_ofs:batch_end]
                old_logprob_v = old_log_prob_a[batch_ofs:batch_end]

                opt.zero_grad()
                new_values = net.get_values(states_v)
                value_loss = F.mse_loss(new_values.squeeze(-1), refs_v)
                value_loss.backward()
                opt.step()

                opt.zero_grad()
                new_logits = net.get_logits(states_v)
                new_logprob_v = F.log_softmax(new_logits, dim=1)
                new_logprob_v = new_logprob_v[range(len(new_logprob_v)), actions_v]
                ratio = torch.exp(new_logprob_v - old_logprob_v)
                
                surr_obj = ratio * adv_v

                clipped_surr_obj = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * adv_v

                policy_loss = -torch.min(clipped_surr_obj, surr_obj).mean()
                
                policy_loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                opt.step()

        for reward, steps in exp_source.pop_rewards_steps():
            smooth_reward.append(reward)
            smooth_steps.append(steps)
            print('reward: %.3f, steps: %d' % (reward, steps))
        
        if len(smooth_reward) >= 10:
            print('mean_reward: %.3f, mean_steps: %d' % (sum(smooth_reward) / len(smooth_reward), sum(smooth_steps)/len(smooth_steps)))
            smooth_reward.clear()
            smooth_steps.clear()
            next(render_source)

        batch.clear()

