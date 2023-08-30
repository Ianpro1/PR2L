#the script may not sample experiences properly (still works)
#Also, using a single environment is better and more efficient (the implementation here is technically wrong)
#PPO_Linear is a much better example
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PR2L.agent as ag
from PR2L.agent import float32_preprocessing
import mjct
from collections import namedtuple
import math
HIDDEN = 64

class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.mu = nn.Sequential(
        nn.Linear(obs_size, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, act_size),
        nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))


    def forward(self, x):
        return self.mu(x)
    
class Critic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.value = nn.Sequential(
        nn.Linear(obs_size, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
        )
    def forward(self, x):
        return self.value(x)

class A2C(ag.Agent):
    def __init__(self, net, device="cpu"):
        super().__init__()
        self.net = net
        self.device = device

    def __call__(self, states, internal_states):
        logstd = self.net.logstd.data.cpu().numpy()
        noise = np.random.normal(loc=0.0, scale=1.0, size=logstd.shape)

        states_v = float32_preprocessing(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        actions = mu + np.exp(logstd) * noise
        actions = np.clip(actions, -1, 1)
        return actions, internal_states
    

device = "cpu" if torch.cuda.is_available() else "cpu"
ENV_NUM = 20
TRAJECTORY_SIZE = 102 #this is equal 2040 experiences (20 * 102)

N_STEPS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
APIRATE = 30
ENV_ID = "TosserCPP"
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4
PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 60
TEST_ITERS = 1000

#trajectory and states must be ordered
def calc_adv_ref(trajectory, net_crt, states, device='cpu'):
    values = net_crt(states)
    values = values.squeeze().data.cpu().numpy()
    last_gae = 0.0
    result_adv = []
    result_ref = []
    
    for val, next_val, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)
    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)

def calc_logprob(mu_v, logstd_v, actions_v):
    #NOTE: this function doesn't calculate the probability (else it would contradict probability theory) but the probability density.
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done'))

class expSource:
    def __init__(self, envs, agent):
        self.envs = envs
        self.agent = agent

    def __iter__(self):
        states = []
        
        for e in self.envs:
            state, _ = e.reset()
            states.append(state)

        while (1):
            exp_list = []

            actions, _ = self.agent(states, None)

            for i, e in enumerate(self.envs):
                obs, reward, done, _, _ = e.step(actions[i])

                exp_list.append(Experience(states[i], actions[i], reward, done))

                if (done):
                    obs, _ = e.reset()

                states[i] = obs

            yield exp_list

class playSource:
    def __init__(self, env, agent):
        assert isinstance(agent, ag.Agent)
        self.agent = agent
        self.env = env

    def __iter__(self):
        
        while (1):
            obs, _ = self.env.reset()
            internal_state = self.agent.initial_state()
            while(1):
                actions, internal_state = self.agent(obs, internal_state)
                obs, rew, done, term, info = self.env.step(actions)
                if done:
                    break
                yield None


def transpose_list(lst):
    return list(map(list, zip(*lst)))

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]   
           
if __name__ == "__main__":
    envs = [mjct.make(ENV_ID, apirate=APIRATE) for _ in range(ENV_NUM)]
    render_env = mjct.make(ENV_ID, render= "autogl", apirate=APIRATE)
    obs_size = 10
    act_size = 2
    
    act_net = Actor(obs_size, act_size).to(device)
    crt_net = Critic(obs_size).to(device)
    agent = A2C(act_net, device)
    exp_source = iter(expSource(envs, agent))
    render_source = iter(playSource(render_env, agent))
    
    trajectory = []
    opt_crt = torch.optim.Adam(crt_net.parameters(), LEARNING_RATE_CRITIC)
    opt_act = torch.optim.Adam(act_net.parameters(), LEARNING_RATE_ACTOR)

    idx = 0
    while (1):
        trajectory.append(next(exp_source))

        if len(trajectory) < (TRAJECTORY_SIZE):
            continue
        
        trajectory = transpose_list(trajectory)

        adv_v = []
        ref_v = []
        act_v = []

        final_trajectory = []
        
        for subtrajectory in trajectory:
            traj_states = [t.state for t in subtrajectory]
            traj_actions = [t.action for t in subtrajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)
            #repeated calls of calc_adv_ref is ressource-wasteful (normally you'd want to pass all states at once into the network)
            traj_adv_v, traj_ref_v = calc_adv_ref(subtrajectory, crt_net, traj_states_v, device=device)
            adv_v.append(traj_adv_v)
            ref_v.append(traj_ref_v)
            act_v.append(traj_actions_v[:-1])
            final_trajectory.append(traj_states_v[:-1])

        traj_adv_v = torch.cat(adv_v)
        traj_ref_v = torch.cat(ref_v)
        traj_actions_v = torch.cat(act_v)   
        
        final_trajectory = torch.cat(final_trajectory)
    
        mu_v = act_net(final_trajectory)
        old_logprob_v = calc_logprob(mu_v, act_net.logstd, traj_actions_v).detach()
        traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
        traj_adv_v /= torch.std(traj_adv_v)

        #old_logprob_v = old_logprob_v[:-1].detach()
        print(idx)
        idx +=1
        for epoch in range(PPO_EPOCHES):
            for batch_ofs in range(0, len(final_trajectory), PPO_BATCH_SIZE):
                if (batch_ofs < 100):
                    next(render_source)
                batch_l = batch_ofs + PPO_BATCH_SIZE
                states_v = final_trajectory[batch_ofs:batch_l]
                actions_v = traj_actions_v[batch_ofs:batch_l]
                batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                batch_adv_v = batch_adv_v.unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                batch_old_logprob_v = old_logprob_v[batch_ofs:batch_l]

                opt_crt.zero_grad()
                value_v = crt_net(states_v)
                loss_value_v = F.mse_loss(
                value_v.squeeze(-1), batch_ref_v)
                loss_value_v.backward()
                opt_crt.step()
                opt_act.zero_grad()
                mu_v = act_net(states_v)
                logprob_pi_v = calc_logprob(mu_v, act_net.logstd, actions_v)
                
                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                c_ratio_v = torch.clamp(ratio_v,1.0 - PPO_EPS,1.0 + PPO_EPS)
                clipped_surr_v = batch_adv_v * c_ratio_v
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward()
                opt_act.step()
        

