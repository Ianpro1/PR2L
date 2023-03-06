import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from PR2L import agent, utilities, rendering, common_wrappers, experience
import torch.multiprocessing as mp
import numpy as np
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from collections import namedtuple, deque
from common import models, extentions
from common.performance import FPScounter

parameters = {
"ENV_NAME":"BreakoutNoFrameskip-v4",
"BETA_ENTROPY": 0.01,
"BETA_POLICY": 1.0,
"N_STEPS": 4,
"GAMMA":0.99,
"LEARNING_RATE":0.001,
"CLIP_GRAD":0.5,
"PROCESS_COUNT": 4,
"eps": 1e-3,
"MINIBATCH_SIZE":36,
"NUM_ENVS": 12,
"solved":400
}

GAMMA = parameters.get("GAMMA", 0.99)
N_STEPS = parameters.get("N_STEPS", 4)
CLIP_GRAD = parameters["CLIP_GRAD"]
BETA_POLICY = parameters["BETA_POLICY"]
BETA_ENTROPY = parameters["BETA_ENTROPY"]
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def make_envlist(number, ENV_ID, inconn=None):
    env = []
    env.append(make_env(ENV_ID, inconn))
    for _ in range(number-1):
        env.append(make_env(ENV_ID))
    return env


EpisodeEnd = namedtuple("EpisodeEnd", ("rewards", "steps"))
import time
def play_env(env_number, net, queue, minibatch_size, device="cpu", inconn=None):

    env = make_envlist(env_number, parameters["ENV_NAME"], inconn)
    _agent = agent.PolicyAgent(net, device)
    exp_source = experience.ExperienceSourceV2(env, _agent, n_steps=parameters["N_STEPS"], GAMMA=parameters["GAMMA"])

    minibatch = []
    fpscount = FPScounter(10000)
    
    for exp in exp_source:
        fpscount.step()
        
        minibatch.append(exp)

        if len(minibatch) >= minibatch_size:
            #random_batch = random.sample(minibatch, minibatch_size)
            queue.put(minibatch.copy())
            minibatch.clear() 
        
        for rewards, steps in exp_source.pop_rewards_steps():
            end = EpisodeEnd(rewards, steps)
            queue.put(end)
 

class BatchGenerator:
    def __init__(self, exp_queue, PROCESS_COUNT):
        self.queue = exp_queue
        self.PROCESS_COUNT = PROCESS_COUNT
        self.rewards = []
        self.steps = []

    def __iter__(self):
        batch = []
        idx = 0
        while True:
            exp = self.queue.get()         
            if isinstance(exp, EpisodeEnd):
                self.rewards.append(exp.rewards)
                self.steps.append(exp.steps)
                continue
            idx +=1
            batch.extend(exp)

            if idx % self.PROCESS_COUNT == 0:
                yield batch
                batch.clear()

    def pop_rewards_steps(self):
        res = list(zip(self.rewards, self.steps))
        if res:
            self.rewards.clear()
            self.steps.clear() 
        return res


if __name__ == "__main__":

    mp.set_start_method("spawn")
    exp_queue = mp.Queue(maxsize=parameters["PROCESS_COUNT"])

    if False:
        inconn, outconn = mp.Pipe()
        display = mp.Process(target=rendering.init_display, args=(outconn, (420, 320)))
        display.start()
    else:
        inconn = None

    net = models.A2C((1,84,84), 4).to(device)
    net.share_memory()

    processes = []
    p1 = mp.Process(target=play_env, args=(parameters["NUM_ENVS"], net, exp_queue, parameters["MINIBATCH_SIZE"], device, inconn))
    processes.append(p1)
    for _ in range(parameters["PROCESS_COUNT"]-1):
        p = mp.Process(target=play_env, args=(parameters["NUM_ENVS"], net, exp_queue, parameters["MINIBATCH_SIZE"], device))
        processes.append(p)

    for p in processes:
        p.start()

    buffer = BatchGenerator(exp_queue, parameters["PROCESS_COUNT"])

    mean_rewards = deque(maxlen=20)
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters["LEARNING_RATE"], eps=parameters["eps"])
    
    print(net)
    net.apply(models.network_reset)


    net.load_state_dict(torch.load("model_saves/BreakoutNoFrameskip-v4/model_002/state_dicts/2023-03-06/save-00-40.pt"))
    writer = SummaryWriter()
    render_agent = agent.PolicyAgent(net, device)
    render_env = make_env(parameters["ENV_NAME"], render=True)
    backup = utilities.ModelBackup(parameters["ENV_NAME"], "002", net, render_env=render_env, agent=render_agent)
    
    solved = parameters["solved"]
    mean_r = 0
    
    for idx, batch in enumerate(buffer):
        if idx % 10000 == 0:
            backup.save(parameters)
            backup.mkrender(fps=140.0, frametreshold=5000)

        for rewards, steps in buffer.pop_rewards_steps():
            mean_rewards.append(rewards)
            mean_r = np.asarray(mean_rewards).mean()
            print("idx %d, mean_rewards %.2f, cur_rewards %.2f, steps %d" % (idx, mean_r, rewards, steps))

        if mean_r > solved:
            backup.save(parameters)
            backup.mkrender(fps=140.0, frametreshold=5000)
        
        batch_len = len(batch)
        
        states, actions, rewards, last_states, not_dones = utilities.unpack_batch(batch)
        
        states = preprocessor(states).to(device)
        rewards = preprocessor(rewards).to(device)
        actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
        if last_states:
            last_states = preprocessor(last_states).to(device)
            tgt_q = torch.zeros_like(rewards)
            with torch.no_grad():
                next_q_v = net(last_states)[1]

            tgt_q[not_dones] = next_q_v.squeeze(-1)
            refs_q_v = rewards + tgt_q * (GAMMA**N_STEPS)
        else:
            refs_q_v = rewards

        #training steps
        optimizer.zero_grad()

        logits, values = net(states)
        values = values.squeeze(-1)
        value_loss = F.mse_loss(values, refs_q_v)
        
        adv_v = refs_q_v - values.detach()
        log_probs = F.log_softmax(logits, dim=1)     
        log_prob_a = log_probs[range(batch_len), actions]
        policy_loss = -(log_prob_a * adv_v).mean()
        
        probs = F.softmax(logits, dim=1)
        entropy_loss = (probs * log_probs).sum(dim=1).mean()
        entropy_loss = BETA_ENTROPY * entropy_loss

        #policy_loss.backward(retain_graph=True)
        
        '''grad_mean, grad_max = extentions.calc_grad(net)
        writer.add_scalar("mean_grad", grad_mean, idx)
        writer.add_scalar("max_grad", grad_max, idx)'''
    
        loss = value_loss + policy_loss + entropy_loss
        loss.backward()

        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        optimizer.step()

        '''writer.add_scalar("entropy", entropy_loss, idx)
        writer.add_scalar("policy_loss", policy_loss, idx)
        writer.add_scalar("value_loss", value_loss, idx)'''

        if idx % 100 == 0:
            print("value_loss", value_loss.item())
        