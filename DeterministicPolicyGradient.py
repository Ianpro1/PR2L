#off policy
import torch
import torch.nn as nn
import numpy as np
from PR2L.agent import Agent, float32_preprocessing, preprocessing
import mjct
from PR2L import experience, utilities, agent
from torch import optim
import torch.nn.functional as F
from collections import deque
import time 

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, HIDDEN1=400, HIDDEN2=300):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, act_size),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)
    
class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size, HIDDEN1=400, HIDDEN2=300):
        super(DDPGCritic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, HIDDEN1),
            nn.ReLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(HIDDEN1 + act_size, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class AgentDDPG(Agent):
    def __init__(self, net, device="cpu", preprocessing=float32_preprocessing, \
        ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        super().__init__()
        self.net = net
        self.device = device
        self.preprocessing = preprocessing
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def getState(self):
        return None

    @torch.no_grad()
    def __call__(self, obs, agent_states):
        state_v = float32_preprocessing(obs).to(self.device)
        mu_v = self.net(state_v)
        actions = mu_v.data.cpu().numpy()
        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)
                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states


device = "cpu" if torch.cuda.is_available() else "cpu"
ENV_NUM = 30
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
N_STEPS = 4
GAMMA = 0.99
APIRATE = 30
ENV_ID = "TosserCPP"
EPSILON_IDX = 50000

if __name__ == "__main__":

    envs = [mjct.make(ENV_ID, apirate=APIRATE) for _ in range(ENV_NUM)]
    envs.append(mjct.make(ENV_ID, render= "autogl", apirate=APIRATE))
    inp_size = 10
    out_size = 2

    act_net = DDPGActor(10, 2, 200, 50).to(device)
    crt_net = DDPGCritic(10, 2, 200, 50).to(device)
    
    tgt_act = agent.TargetNet(act_net)
    tgt_crt = agent.TargetNet(crt_net)

    print(crt_net)
    print(act_net)

    modelmanager = utilities.ModelBackupManager(ENV_ID, "DDPG_001", [act_net, crt_net])
    modelmanager.load()

    #make sure to enable OU-noise when training
    _agent = AgentDDPG(act_net, device, ou_enabled=True, ou_epsilon=1.0)
    exp_source = experience.ScarsedExperienceSource(100, envs, _agent, GAMMA = GAMMA, n_steps=N_STEPS)

    #loop to test the network
    if True:
        test_agent = AgentDDPG(act_net, device, ou_enabled=True, ou_epsilon=0.01)
        test_exp_source = experience.ExperienceSource(envs, test_agent, GAMMA = GAMMA, n_steps=N_STEPS)
        for exp in test_exp_source:
            continue

    buffer = experience.SimpleReplayBuffer(exp_source, REPLAY_BUFFER_SIZE)

    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    idx = 0
    score = deque(maxlen=100)
    running = True
    while running:
        idx += 1
        buffer.populate(1)

        _agent.ou_epsilon = max(1.0 - idx/EPSILON_IDX, 0.02)

        if idx % 100000 == 0:
            modelmanager.save()

        if len(buffer) < BATCH_SIZE * 2:
            continue
        
        batch = buffer.sample(BATCH_SIZE)

        states, actions, rewards, next_states, not_dones = utilities.unpack_batch(batch)
        states = float32_preprocessing(states).to(device)
        actions = preprocessing(actions).to(device)
        rewards = float32_preprocessing(rewards).to(device)

        with torch.no_grad():
            if next_states:
                next_q_v = torch.zeros_like(rewards)
                next_states = float32_preprocessing(next_states).to(device)
                next_act_v = tgt_act.target_model(next_states)
                next_crt_v = tgt_crt.target_model(next_states, next_act_v)
                next_q_v[not_dones] = next_crt_v.squeeze(-1)
                refs_q_v = rewards + next_q_v * GAMMA**N_STEPS

            else:
                refs_q_v = rewards
        
        crt_opt.zero_grad()
        q_v = crt_net(states, actions)
        critic_loss = F.mse_loss(q_v.squeeze(-1), refs_q_v)
        critic_loss.backward()
        crt_opt.step()

        act_opt.zero_grad()
        cur_action_v = act_net(states)
        actor_loss = -crt_net(states, cur_action_v)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        act_opt.step()

        tgt_act.alpha_sync(alpha=1-1e-3)
        tgt_crt.alpha_sync(alpha=1-1e-3)

        for reward, step in exp_source.pop_rewards_steps():
            score.append(reward)
            m_score = np.array(score).mean()
            print("idx %d, reward %.3f, steps %d, total_score %.3f" % (idx, reward, step, m_score))
            
            if m_score > 5.0:
                modelmanager.save()
                running = False




"""
A larger batch tend to be much more stable and perform much better than a smaller batch.
recommended batch size is 500. It achieves an average score of > 5.00 over 100 episodes!
"""