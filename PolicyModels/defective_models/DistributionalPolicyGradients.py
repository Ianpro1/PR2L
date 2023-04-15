#D4PG is a version of DDPG that is supposedly better however,
#this script seems to be slower at this particular task or might not function properly


import torch
import torch.nn as nn
from PR2L import experience, utilities, agent
from PR2L.agent import float32_preprocessing, Agent, preprocessing
import mjct
import numpy as np
import torch.nn.functional as F
from torch import optim
from collections import deque


class D4PGActor(nn.Module):
    def __init__(self, input_size, output_size, HIDDEN=400, HIDDEN2=300):
        super().__init__()
        self.act_net = nn.Sequential(
            nn.Linear(input_size, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.act_net(x)


class D4PGCritic(nn.Module):
    def __init__(self, input_size, output_size, v_max, v_min, n_atoms=51, HIDDEN=400, HIDDEN2=300):
        super().__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(input_size, HIDDEN),
            nn.ReLU()
        )
        self.v_net = nn.Sequential(
            nn.Linear(HIDDEN+output_size, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("support_atoms", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.v_net(torch.cat([obs, a], dim=1))
    
    def get_expected_v(self, distr):
        w = F.softmax(distr, dim=1) * self.support_atoms
        res = w.sum(dim=1)
        return res.unsqueeze(dim=-1)
    

class AgentD4PG(Agent):
    def __init__(self, act_net, device="cpu", epsilon=0.3, clipping=True):
        super().__init__()
        self.act_net = act_net
        self.device = device
        self.epsilon = epsilon
        self.clipping = clipping
    
    def __call__(self, obs, internal_states):
        states = float32_preprocessing(obs).to(self.device)
        mu_v = self.act_net(states)
        actions = mu_v.data.cpu().numpy()
        actions += self.epsilon * np.random.normal(size=actions.shape) #adding normal noise
        if self.clipping:
            actions = np.clip(actions, -1, 1)
        return actions, internal_states


def distr_projection(next_v_distr, rewards, not_dones, gamma, device="cpu"):
    #handle episodes that were not terminated
    if not_dones.any():
        #next_v_distr is based on next_states which does not have the same shape as rewards!
        next_distr = next_v_distr.data.cpu().numpy()
        rewards = rewards.data.cpu().numpy()
        batch_size = len(next_distr) #length of next_distr for batch which doesn't match rewards!
        proj_distr = np.zeros(shape=(batch_size,N_ATOMS), dtype=np.float32)
        for atom in range(N_ATOMS):
            #project the atom from reward while respecting range
            proj_atom = np.minimum(V_MAX, np.maximum(V_MIN, rewards[not_dones] + (V_MIN + atom * DELTA)*gamma))
            #get index
            proj_id = (proj_atom - V_MIN) / DELTA
            lower = np.floor(proj_id).astype(np.int64)
            upper = np.ceil(proj_id).astype(np.int64)
            #handle rare event when atom falls directly on a boundary
            same = (lower == upper)
            proj_distr[same, lower[same]] += next_distr[same, atom]
            not_same = (lower != upper)
            proj_distr[not_same, lower[not_same]] += next_distr[not_same, atom] * (upper - proj_id)[not_same]
            proj_distr[not_same, upper[not_same]] += next_distr[not_same, atom] * (proj_id - lower)[not_same]
    
    #handle episodes that were terminated
    dones = (not_dones == False)

    if dones.any():
        resized_proj_distr = np.zeros(shape=(len(rewards), N_ATOMS), dtype=np.float32)
        resized_proj_distr[not_dones] = proj_distr
        #this will handle the situation when the episode is over,
        #in this case the projected distribution will be a single or double strand representing the rewards
        proj_atom = np.minimum(V_MAX, np.maximum(V_MIN, rewards[dones]))
        proj_id = (proj_atom-V_MIN) / DELTA
        lower = np.floor(proj_id).astype(np.int64)
        upper = np.ceil(proj_id).astype(np.int64)
        same = (lower == upper)
        same_dones = dones.copy()
        same_dones[dones] = same
        if same_dones.any():
            resized_proj_distr[same_dones, lower[same]] = 1.0
        not_same = (lower != upper)
        not_same_dones = dones.copy()
        not_same_dones[dones] = not_same
        if not_same_dones.any():
            resized_proj_distr[not_same_dones, lower[not_same]] = (upper - proj_id)[not_same]
            resized_proj_distr[not_same_dones, upper[not_same]] = (lower - proj_id)[not_same]
        #overwrite proj_distr
        proj_distr = resized_proj_distr
    return torch.FloatTensor(proj_distr).to(device)
         

device = "cuda" if torch.cuda.is_available() else "cpu"
ENV_NUM = 50
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
BATCH_SIZE = 200
N_STEPS = 4
GAMMA = 0.99
APIRATE = 30
ENV_ID = "TosserCPP"
EPSILON_IDX = 50000
V_MAX = 10
V_MIN = -10
N_ATOMS = 51
DELTA = (V_MAX - V_MIN) / (N_ATOMS - 1)


if __name__ == "__main__":
    envs = [mjct.make(ENV_ID, apirate=APIRATE) for _ in range(ENV_NUM)]
    envs.append(mjct.make(ENV_ID, render= "autogl", apirate=APIRATE))
    inp_size = 10
    out_size = 2

    act_net = D4PGActor(inp_size, out_size).to(device)
    crt_net = D4PGCritic(inp_size, out_size, V_MAX, V_MIN).to(device)
    
    tgt_act = agent.TargetNet(act_net)
    tgt_crt = agent.TargetNet(crt_net)

    print(crt_net)
    print(act_net)

    modelmanager = utilities.ModelBackupManager(ENV_ID, "D4PG_001", [act_net, crt_net])
    #modelmanager.load()

    _agent = AgentD4PG(act_net, device, 0.3)
    exp_source = experience.ExperienceSource(envs, _agent, GAMMA = GAMMA, n_steps=N_STEPS)

    buffer = experience.SimpleReplayBuffer(exp_source, REPLAY_BUFFER_SIZE)

    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    idx = 0
    score = deque(maxlen=100)
    running = True
    while running:
        idx += 1
        buffer.populate(1)

        _agent.epsilon = max(0.3 - idx/EPSILON_IDX, 0.02)

        if idx % 100000 == 0:
            modelmanager.save()

        if len(buffer) < BATCH_SIZE * 2:
            continue
        
        batch = buffer.sample(BATCH_SIZE)

        states, actions, rewards, next_states, not_dones = utilities.unpack_batch(batch)
        states = float32_preprocessing(states).to(device)
        actions = preprocessing(actions).to(device)
        rewards = float32_preprocessing(rewards).to(device)
        not_dones = np.array(not_dones, copy=False)
        crt_opt.zero_grad()

        #get distribution for v
        crt_v_distr = crt_net(states, actions)
        #get target distribution for last_states if any
        if next_states:
            next_states = float32_preprocessing(next_states).to(device)
            next_actions = tgt_act.target_model(next_states)
            next_v_distr = tgt_crt.target_model(next_states, next_actions)
            next_v_distr = F.softmax(next_v_distr, dim=1)
        #project target distribution for last_states
        proj_next_v_distr = distr_projection(next_v_distr, rewards, not_dones, gamma=GAMMA**N_STEPS, device=device)
        #get log_prob of distribution values * projected distr. 
        logprob_distr_v = -F.log_softmax(crt_v_distr, dim=1) * proj_next_v_distr
        #critic loss
        critic_loss = logprob_distr_v.sum(dim=1).mean()
        critic_loss.backward()
        crt_opt.step()

        act_opt.zero_grad()

        #get expected value
        act_v = act_net(states)
        v_distr = crt_net(states, act_v)
        mu_v = crt_net.get_expected_v(v_distr)
        #actor loss
        actor_loss = -mu_v.mean()
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