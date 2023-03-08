import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from PR2L import agent, utilities, rendering, experience
import torch.multiprocessing as mp
import numpy as np
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from collections import namedtuple, deque
from common import common_wrappers, models, extentions
from common.performance import FPScounter
import os

GAMMA = 0.99
LEARNING_RATE = 0.001
BETA_ENTROPY = 0.01

N_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 8

GRAD_BATCH = 48
TRAIN_BATCH = 2
ENV_NAME = "PongNoFrameskip-v4"
solved = 20
preprocessor = agent.numpytoFloatTensor_preprossesing

class SingleChannelWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array([observation])
    
def make_env(ENV_ID, inconn=None, render=False):
    
    if render:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = utilities.render_env(env)
    else:
        env = gym.make(ENV_ID)
        if inconn is not None:
                env = rendering.SendimgWrapper(env, inconn, frame_skip=12)
    env = AtariPreprocessing(env)
    env = common_wrappers.RGBtoFLOAT(env)
    env = common_wrappers.BetaSumBufferWrapper(env, 3, 0.4)
    env = SingleChannelWrapper(env)
    return env


def grads_func(proc_name, net, device, train_queue, inconn=None):
    if inconn is None:
        envs = [make_env(ENV_NAME) for _ in range(NUM_ENVS)]
    else:
        envs = [make_env(ENV_NAME) for _ in range(NUM_ENVS-1)]
        env = make_env(ENV_NAME, inconn)
        envs.append(env)

    _agent = agent.PolicyAgent(net, device)
    exp_source = experience.ExperienceSource(envs, _agent, n_steps=N_STEPS, GAMMA=GAMMA)
    batch = []
    frame_idx = 0
    writer =SummaryWriter(comment=proc_name)
    mean_rewards = deque(maxlen=20)
    mean_r = 0
    for exp in exp_source:
        frame_idx += 1
        for reward, steps in exp_source.pop_rewards_steps():
            mean_rewards.append(reward)
            mean_r = np.array(mean_rewards).mean()
            print("mean_rewards %.2f, rewards %.2f, steps %d" %(mean_r, reward, steps) )

        if mean_r >= solved:
            break

        batch.append(exp)

        if len(batch) < GRAD_BATCH:
                continue
        
        states, actions, rewards, last_states, not_dones = utilities.unpack_batch(batch)
        batch.clear()

        states = preprocessor(states).to(device)
        rewards = preprocessor(rewards).to(device)
        actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
        if last_states:
            last_states = preprocessor(last_states).to(device)
            tgt_q = torch.zeros_like(rewards)
            with torch.no_grad():
                next_q_v = net(last_states)[1]

            tgt_q[not_dones] = next_q_v.squeeze(-1)
            refs_q_v = rewards + tgt_q * GAMMA**N_STEPS
        else:
            refs_q_v = rewards

        #training steps
        net.zero_grad()
        
        logits, values = net(states)
        values = values.squeeze(-1)
        value_loss = F.mse_loss(values, refs_q_v)
        
        adv_v = refs_q_v - values.detach()
        log_probs = F.log_softmax(logits, dim=1)
        
        log_prob_a = log_probs[range(GRAD_BATCH), actions]
        
        policy_loss = -(log_prob_a*adv_v).mean()

        probs = F.softmax(logits, dim=1)
        entropy_loss = (probs*log_probs).sum(dim=1).mean()
        entropy_loss = BETA_ENTROPY * entropy_loss

        loss =value_loss + entropy_loss + policy_loss
        loss.backward()

        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        
        grads = [
                param.grad.data.cpu() if param.grad is not None else None for param in net.parameters()
        ]
        train_queue.put(grads)
    train_queue.put(None) 



if __name__ == "__main__":
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = "1"
    device = "cuda"

    net = models.A2C((1,84,84), 4).to(device)
    net.share_memory()

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []

    if True:
        inconn, outconn = mp.Pipe()
        display = mp.Process(target=rendering.init_display, args=(outconn, (420, 320)))
        display.start()
    else:
        inconn=None


    for i, proc_idx in enumerate(range(PROCESSES_COUNT)):
        proc_name = "-a3c-grad_pong_" + str(proc_idx)
        if i == 0:
            p_args = (proc_name, net, device, train_queue, inconn)
        else:
            p_args = (proc_name, net, device, train_queue)
        data_proc = mp.Process(target=grads_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)

    net.apply(models.network_reset)
    
    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break
            step_idx += 1
            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad
            
            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(device)
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()
                    grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
            display.join()
