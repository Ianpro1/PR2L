import torch.multiprocessing as mp
from common.models import NoisyDuelDQN
import time
from common.extentions import ModelBackup
import torch.nn as nn
import torch
import ptan


net = NoisyDuelDQN((3,84,84), 4)
import pandas as pd
import numpy as np
import common.extentions as E
import time
import torch 
import gym
import common.models as models
env = gym.make("Breakout-v4")
net = models.NoisyDuelDQN((3,84,84), 4)



class Agent:
    def __init__(self, net,  device, preprocessing=E.ndarray_preprocessor(E.FloatTensor_preprocessor())):
        self.net = net
        self.device = device
        self.preprocessing = preprocessing
    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x)
        return self.net(x)

t1 = time.time()

frame = np.random.randint(0, 255, (1920, 1080, 3))

import timeit

obj = np.random.randint(0, 255, (4000, 4000, 3))
#
def prepro(x):
    return  torch.FloatTensor(np.array(x, copy=False))

def test_code():
    E.ndarray_preprocessor(E.FloatTensor_preprocessor(obj))

stmt = "test_code"
setup = "from __main__ import test_code"

print(
    timeit.timeit(stmt, setup, number=1000000000)
)


#improvements
"""
transpose -> moveaxis

"""