#This A3C model is used to test noise and gradient flow

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from PR2L import agent, utilities, rendering, common_wrappers, experience
import torch.multiprocessing as mp
import numpy as np
import torch.nn as nn
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from collections import namedtuple, deque
from common import models, extentions
from common.performance import FPScounter
import time

#doesn't work with adv_v (seems like the avg adv_v is moving too fast for this to work)
#TODO try to use with value estimation, try to place noise in gradient instead of on input
class BiasedFilter(nn.Module):
    def __init__(self, feature_size, alpha=0.0016):
        super().__init__()
        self.alpha = alpha
        self.feature_size = feature_size
        b = nn.Parameter(torch.empty(size=feature_size))
        self.register_parameter('bias', b)
        self.register_buffer("b_noise", torch.zeros_like(b))
    
        self.reset_parameters()

    def forward(self, x):
        self.b_noise.uniform_()
        return x + self.bias * self.b_noise 

    def reset_parameters(self):
        nn.init.uniform_(self.bias, a=0., b=self.alpha)

#no use found for this layer
class SinglyConnected(nn.Module):
    def __init__(self, feature_size, alpha=0.16, bias=True, bias_ratio=1/20):
        super().__init__()
        self.b_percent = bias_ratio
        self.alpha = alpha
        w = nn.Parameter(torch.empty(size=feature_size, dtype=torch.float32))
        self.register_parameter('weight', w)
        if bias:
            b = nn.Parameter(torch.empty(size=feature_size, dtype=torch.float32))
            self.register_parameter('bias', b)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        if self.bias is None:
            return x * self.weight
        return x * self.weight+ self.bias

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=1.0-self.alpha, b=1.0+self.alpha)

        if self.bias is not None:
            nn.init.uniform_(self.bias, a=-self.alpha * self.b_percent, b=self.alpha * self.b_percent)
    
class FilterAgent(agent.Agent):
    def __init__(self,filter, net, device="cpu", Selector= agent.ProbabilitySelector(), preprocessing=agent.numpytoFloatTensor_preprossesing, inconn=None):
        super().__init__()
        assert isinstance(Selector, agent.ActionSelector)
        if inconn is not None:
            self.inconn = inconn
        self.selector = Selector
        self.net = net
        self.filter = filter
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x):
        x1 = self.preprocessing(x)
        x = self.filter(x1.to(self.device))
        if self.inconn is not None:
            img = np.concatenate((x.cpu().numpy()[0], x1.cpu().numpy()[0]), axis=2)
            self.inconn.send(img)
        act_v = self.net(x)[0]
        act_v = F.softmax(act_v, dim=1)
        actions = self.selector(act_v.cpu().numpy())
        noise = self.filter.b_noise.cpu().numpy()
        return actions, [[noise]] * actions.shape[0]


parameters = {
"ENV_NAME":"BreakoutNoFrameskip-v4",
"BETA_ENTROPY": 0.01,
"BETA_POLICY": 1.0,
"N_STEPS": 4,
"GAMMA":0.99,
"LEARNING_RATE":0.001,
"CLIP_GRAD":0.5,
"PROCESS_COUNT": 1,
"eps": 1e-3,
"MINIBATCH_SIZE":32,
"NUM_ENVS": 8,
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

def play_env(env_number,filter, net, queue, minibatch_size, device="cpu", inconn=None):

    env = make_envlist(env_number, parameters["ENV_NAME"])

    _agent = FilterAgent(filter, net, device, inconn=inconn)
    exp_source = experience.MemorizedExperienceSource(env, _agent, n_steps=parameters["N_STEPS"], GAMMA=parameters["GAMMA"])

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

    if True:
        inconn, outconn = mp.Pipe()
        display = mp.Process(target=rendering.init_display, args=(outconn, (336, 672), rendering.ChannelFirstPreprocessing))
        display.start()
    else:
        inconn = None

    net = models.A2C((1,84,84), 4).to(device)
    net.share_memory()
    netfilter = BiasedFilter((84, 84)).to(device)
    netfilter.share_memory()

    processes = []

    p1 = mp.Process(target=play_env, args=(parameters["NUM_ENVS"], netfilter, net, exp_queue, parameters["MINIBATCH_SIZE"], device, inconn))
    processes.append(p1)

    for _ in range(parameters["PROCESS_COUNT"]-1):
        p = mp.Process(target=play_env, args=(parameters["NUM_ENVS"], netfilter, net, exp_queue, parameters["MINIBATCH_SIZE"], device))
        processes.append(p)

    for p in processes:
        p.start()

    buffer = BatchGenerator(exp_queue, parameters["PROCESS_COUNT"])

    mean_rewards = deque(maxlen=20)
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters["LEARNING_RATE"], eps=parameters["eps"])
    filter_optimizer = torch.optim.Adam(netfilter.parameters(), lr=0.01)
    print(net)

    net.apply(models.network_reset)
    netfilter.reset_parameters()

    net.load_state_dict(torch.load("model_saves/BreakoutNoFrameskip-v4/model_002/state_dicts/2023-03-06/save-14-30.pt"))
    writer = SummaryWriter()
    backup = utilities.ModelBackup(parameters["ENV_NAME"], "003", net)
    solved = parameters["solved"]
    mean_r = 0
    
    for idx, batch in enumerate(buffer):
        #if idx % 30000 == 0:
            #backup.save(parameters)

        for rewards, steps in buffer.pop_rewards_steps():
            mean_rewards.append(rewards)
            mean_r = np.asarray(mean_rewards).mean()
            print("idx %d, mean_rewards %.2f, cur_rewards %.2f, steps %d" % (idx, mean_r, rewards, steps))

        if mean_r > solved:
            backup.save(parameters)
        
        batch_len = len(batch)
        
        initstates, actions, rewards, last_states, not_dones, memories = utilities.unpack_memorizedbatch(batch)

        initstates = preprocessor(initstates).to(device)
        memories = preprocessor(memories).to(device)

        


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
        filter_optimizer.zero_grad()
        states = initstates + netfilter.bias * memories

        logits, values = net(states.detach())
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
        print(adv_v.mean())
        filter_loss = ((states - initstates)**2).mean(dim=(1,2,3))
        filter_loss = -(filter_loss * adv_v).mean()
        filter_loss.backward()

        loss = value_loss + policy_loss + entropy_loss
        loss.backward()

        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

        filter_optimizer.step()
        optimizer.step()

        '''writer.add_scalar("entropy", entropy_loss, idx)
        writer.add_scalar("policy_loss", policy_loss, idx)
        writer.add_scalar("value_loss", value_loss, idx)'''

        if idx % 100 == 0:
            print("value_loss", value_loss.item())