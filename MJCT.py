import mjct
from common import models
from PR2L import agent, experience, training, rendering, utilities
import torch
import math
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import time

ENV_ID = "Tosser"
HIDDEN = 128
GAMMA = 0.99
N_STEPS = 2
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4
TEST_ITERS = 10000
NUM_ENVS = 30
SAVE_ITER = 100000
APIRATE = 30
TIMESTEPS = 0.005
device = "cuda" if torch.cuda.is_available() else "cpu"

def test_net(net, env, count=3, device="cpu", preprocessor=agent.float32_preprocessing):
    rewards = 0.0
    steps = 0
    time.sleep(1)
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

def test_in_parallel(net, device, preprocessor=agent.float32_preprocessing):
    #need to enable render!
    env = mjct.make("TosserCPP", True, TIMESTEPS, APIRATE, False)
    while True:
        rewards = 0.0
        steps = 0
        obs, _ = env.reset()
        while True:
            
            obs_v = preprocessor([obs])
            obs_v = obs_v.to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _, _ = env.step(action)
            env.render()
            rewards += reward
            steps += 1
            if done:
                break
        print( "reward %.3f, steps %d" % (rewards, steps))

if __name__ == "__main__":
#problem is that environments seem to reset themselves without the env.step() call

    envs = []
    if False:
        e = mjct.make("TosserCPP", True, 0.02, 50, True)
        envs.append(e)

    for _ in range(NUM_ENVS):
        envs.append(mjct.make(ENV_ID, False, TIMESTEPS, APIRATE, False)) 
    
    action_size = 2
    observation_size = 10
    net = models.ContinuousA2C(observation_size, action_size, HIDDEN)
    net.to(device)
    net.share_memory()

    #parallel testing 
    testing = mp.Process(target=test_in_parallel, args=(net, device))
    testing.start()

    _agent = agent.ContinuousNormalAgent(net, device)
    exp_source = iter(experience.ExperienceSource(envs, _agent, track_rewards=False))
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    batch = []
    idx = 0
    modelwriter = utilities.ModelBackup(ENV_ID, "001", net, True)

    net.apply(models.network_reset)

    while True:
        idx += 1
        if idx % 100 == 0:
            print(idx)
        while (len(batch) < BATCH_SIZE):
            batch.append(next(exp_source))
            
        
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

        if idx % SAVE_ITER == 0:
            modelwriter.save()
