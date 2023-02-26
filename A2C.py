import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import numpy as np
from common import models, extentions
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from PR2L import common_wrappers, agents, experience, utilities
import multiprocessing as mp
import common.Rendering as rendering
from collections import deque

ENV_NAME = "BreakoutNoFrameskip-v4"
ENTROPY_BETA = 0.01
N_STEPS = 4
EPSILON_START = 1.0
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.001
CLIP_GRAD = 0.1

device = "cpu"

class HighLevelSendimg:
    def __init__(self, inconn, frame_skip=2):
        self.inconn = inconn
        self.frame_skip = frame_skip
        self.count = 0
    def __call__(self, img):
        self.count +=1
        if self.count % self.frame_skip ==0:
            self.count = 0
            img = (img.transpose(1,2,0) * 255. * 1.5).astype(np.uint8)
            self.inconn.send(img)
        return img

class SingleChannelWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array([observation])

def make_env(ENV_ID, inconn=None):
    env = gym.make(ENV_ID)
    env = AtariPreprocessing(env)
    env = common_wrappers.RGBtoFLOAT(env)
    env = common_wrappers.BetaSumBufferWrapper(env, 5, 0.3)
    env = SingleChannelWrapper(env)
    if inconn is not None:
            env = common_wrappers.LiveRenderWrapper(env, HighLevelSendimg(inconn, frame_skip=8))
    return env



if __name__ == "__main__":

    mp.set_start_method("spawn")

    inconn, outconn = mp.Pipe()

    env = make_env(ENV_NAME, inconn)

    p1 = mp.Process(target=rendering.init_display, args=(outconn, 336, 336, (84, 84))) #args=(outconn, 320, 420, (210, 160))
    p1.start()
    net = models.A2C((1, 84, 84), 4).to(device)
    print(net)
    selector = agents.ProbabilitySelector()
    selector = agents.EpsilonGreedySelector(EPSILON_START, selector)
    agent = agents.PolicyAgent(net, device, selector)
    exp_source = experience.ExperienceSource(env, agent, N_STEPS)
    preprocessor = agents.numpytoFloatTensor_preprossesing
    obs,_ = env.reset()

    

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    backer = extentions.ModelBackup(ENV_NAME, net)

    batch = []

    rewardbuffer = deque(maxlen=20)
    solved = 0.
    for idx, exp in enumerate(exp_source):

        

        batch.append(exp)

        for rewards, steps in exp_source.pop_rewards_steps():
            rewardbuffer.append(rewards)
            solved = np.asarray(rewardbuffer).mean()

            print("steps %d, mean_rewards %.2f, cur_reward %.2f" % (steps, solved, rewards))

            if steps > 30000:
                backer.save(parameters={"A2C":True})

        if solved > 400:
            print("Solved!")
            backer.save()

        if len(batch) < BATCH_SIZE:
            continue

        states, actions, rewards, last_states, not_dones = utilities.unpack_batch(batch)

        states = preprocessor(states).to(device)
        rewards = preprocessor(rewards).to(device)
        actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
        last_states = preprocessor(last_states).to(device)
        tgt_q_v = torch.zeros_like(rewards)

        with torch.no_grad():
            #get last_states_v (missing dones)
            last_vals_v = net(last_states)[1]

            tgt_q_v[not_dones] = last_vals_v.squeeze(-1)

            #bellman equation
            ref_q_v = rewards + tgt_q_v * (GAMMA**N_STEPS)

        batch.clear()
        optimizer.zero_grad()
        logits_v, values = net(states)

        #mse for value network
        loss_value_v = F.mse_loss(values.squeeze(-1), ref_q_v)

        #policy gradient
        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = ref_q_v - values.detach()
        log_p_a = log_prob_v[range(BATCH_SIZE), actions]
        log_prob_act_v = log_p_a * adv_v
        loss_policy_v = -log_prob_act_v.mean() #negative to minimize the policy

        #entropy (goal: maximizing the entropy)
        prob_v = F.softmax(logits_v, dim=1)
        ent = (prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = ENTROPY_BETA * ent

        #code to keep track of maximum gradient (careful, if backpropagation is done here then the loss must be changed accordingly)
        '''
        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate([
        p.grad.data.cpu().numpy().flatten()
        for p in net.parameters()
        if p.grad is not None
        ])'''

        loss_v = entropy_loss_v + loss_policy_v + loss_value_v
        loss_v.backward()

        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

        optimizer.step()

        print("loss: %.3f" % loss_v.item())

















    
    


