import torch
import torch.utils.tensorboard as tensorboard
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from PR2L import agent, utilities, rendering, common_wrappers, experience
import torch.multiprocessing as mp
import numpy as np
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from collections import namedtuple, deque
from common import models, Rendering
from common.performance import FPScounter

parameters = {
"ENV_NAME":"PongNoFrameskip-v4",
"BETA_ENTROPY": 0.01,
"N_STEPS": 4,
"BATCH_SIZE":128,
"GAMMA":0.99,
"LEARNING_RATE":0.005,
"CLIP_GRAD":0.1,
"device":"cuda",
"PROCESS_COUNT": 3,
"eps": 1e-3,
"MINIBATCH_SIZE":32,
"NUM_ENVS": 8
}

GAMMA = parameters.get("GAMMA", 0.99)
N_STEPS = parameters.get("N_STEPS", 4)
CLIP_GRAD = parameters["CLIP_GRAD"]
BETA_ENTROPY = parameters["BETA_ENTROPY"]
device = parameters["device"]

class SingleChannelWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array([observation])
    
def make_env(ENV_ID, inconn=None, render=False):
    env = gym.make(ENV_ID)
    if render:
        env = utilities.render_env(env)
    else:
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
            
            if exp:                
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

    inconn, outconn = mp.Pipe()
    display = mp.Process(target=rendering.init_display, args=(outconn, (420, 320)))
    display.start()
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
    preprocessor = agent.numpytoFloatTensor_preprossesing
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters["LEARNING_RATE"], eps=parameters["eps"])
    writer = tensorboard.SummaryWriter()


    net.apply(models.network_reset)
    for idx, batch in enumerate(buffer):
        for rewards, steps in buffer.pop_rewards_steps():
            mean_rewards.append(rewards)
            mean_r = np.asarray(mean_rewards).mean()
            print("mean_rewards %.2f, rewards %.2f, steps %d" % (mean_r, rewards, steps))

        batch_len = len(batch)
        states, actions, rewards, last_states, not_dones = utilities.unpack_batch(batch)
        
        states = preprocessor(states).to(device)
        rewards = preprocessor(rewards).to(device)
        actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
        last_states = preprocessor(last_states).to(device)

        with torch.no_grad():
            if len(last_states) < 1:
                tgt_q_v = torch.zeros_like(rewards)
                #get last_states_v (missing dones)
                #receiving empty tensor, error is due to not_dones being all false
                last_vals_v = net(last_states)[1]

                tgt_q_v[not_dones] = last_vals_v.squeeze(-1)

                #bellman equation
                ref_q_v = rewards + tgt_q_v * (GAMMA**N_STEPS)
            else:
                ref_q_v = rewards

        optimizer.zero_grad()

        logits_v, values = net(states)
        
        #mse for value network
        loss_value_v = F.mse_loss(values.squeeze(-1), ref_q_v)
        #policy gradient
        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = ref_q_v - values.detach()
        log_p_a = log_prob_v[range(batch_len), actions]
        log_prob_act_v = adv_v * log_p_a
        loss_policy_v = -log_prob_act_v.mean() 

        #entropy (goal: maximizing the entropy)
        prob_v = F.softmax(logits_v, dim=1)
                
        ent = (prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -BETA_ENTROPY * ent 

        #code to keep track of maximum gradient (careful, if backpropagation is done here then the loss must be changed accordingly)
        writer.add_scalar("entropy", entropy_loss_v, idx)
        writer.add_scalar("policy_loss", loss_policy_v, idx)
        writer.add_scalar("value_loss", loss_value_v, idx)

        loss_v = 0.1* loss_policy_v + loss_value_v
        loss_v.backward()

        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        
        optimizer.step()
