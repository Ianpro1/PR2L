import torch.multiprocessing as mp
from common.models import NoisyDuelDQN
import time
from common.extentions import ModelBackup
import torch.nn as nn
import torch
import ptan
'''def train(model):
    #model.to("cuda")
    agent = ptan.agent.DQNAgent(model, action_selector=ptan.actions.ArgmaxActionSelector())

    for x in range(10):

        time.sleep(1)
        if x==5:
            print(model.state_dict())
    # Construct data_loader, optimizer, etc.
    
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
def get_def(x):
    print(x)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    num_processes = 4
    model = NoisyDuelDQN((3,84,84), 4)
    model.apply(get_def)
    model.to("cuda")    # placing model in gpu prior to parallelization deletes all weights
    model.share_memory()
    p = mp.Process(target=train, args=(model,))
    p.start()
    model.apply(weight_reset) # does not work well
    #might need to reset all parameters
    p.join()

'''


net = NoisyDuelDQN((3,84,84), 4)

def completeReset(x):
    if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d):
        print(x)
        x.reset_parameters()


net.apply(completeReset)
print(net.state_dict())















