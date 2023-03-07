import torch.nn as nn
import torch
import math
import gym
import numpy as np 
from PR2L import utilities, rendering, common_wrappers
import PR2L.agent as agnt
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import torch.multiprocessing as mp

class SinglyConnected(nn.Module):
    def __init__(self, feature_size, bias=True, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.feature_size = feature_size
        w = nn.Parameter(torch.empty(size=feature_size))
        self.register_parameter('weight', w)
        if bias:
            b = nn.Parameter(torch.empty(size=feature_size))
            self.register_parameter('bias', b)
        else:
            self.register_parameter('bias', None)
            self.bias = None
        self.reset_parameters()

    def forward(self, x):
        if self.bias is None:
            return x * self.weight
        return x * self.weight + self.bias
    
    def reset_parameters(self):
        nn.init.uniform_(self.weight, 1.-self.alpha, 1.+self.alpha)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -self.alpha, self.alpha)



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


import time

if __name__ == "__main__":
    net = SinglyConnected((84,84), True)
    env = make_env("BreakoutNoFrameskip-v4")
    inconn, outconn = mp.Pipe()

    prep = agnt.numpytoFloatTensor_preprossesing
    p_args = (outconn, (336, 672), rendering.ChannelFirstPreprocessing)
    
    display = mp.Process(target=rendering.init_display, args=p_args)
    display.start()

    
    done = True
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    time.sleep(3)
    while True:
        
        if done:
            obs, _ = env.reset()
        
        obs, _, done, _, _ = env.step(np.random.choice(4))
        
        obs = prep(obs)     

        out = net(obs)

        loss = nn.functional.mse_loss(out, obs)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        

        img = np.concatenate((obs.numpy(), out.detach().numpy()), axis=2)
        
        inconn.send(img)

        

        
        

    
