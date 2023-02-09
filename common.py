import torch
from torch import nn 
import numpy as np
import collections
import gym
import ptan
import cv2
import os
import datetime
#basic networks

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.flat = nn.Flatten(1, 3)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.flat(self.conv(x))
        #print(conv_out.shape)
        return self.fc(conv_out)


#preprocessing

class preprocessor:
    # process data from environment to any compatible type 
    # Note: inherit class should be used like wrappers
    def __init__(self, pre_p=None):
        if pre_p is None:
            self.pre_p = lambda x: x
        else:
            self.pre_p = pre_p
    
    def __call__(self, input):
        return self.pre_p(input)


class ndarray_preprocessor(preprocessor):
    def __call__(self, states):
        
        states = np.array(states, copy=False)
        return super().__call__(states)

class VariableTensor_preprocessor(preprocessor):
    def __call__(self, states):
        assert isinstance(states, np.ndarray)
        states = torch.tensor(states)
        return super().__call__(states)

class FloatTensor_preprocessor(preprocessor):
    def __call__(self, states):
        assert isinstance(states, np.ndarray)
        states = torch.FloatTensor(states)
        return super().__call__(states)


#batch handeling

def unpack_batch(batch:list[ptan.experience.ExperienceFirstLast]): #List was undefined
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state, copy=False)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
        np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=np.uint8), \
        np.array(last_states, copy=False)

#model save
class ModelBackup:
    def __init__(self, root, net, notify=True):
        self.dateroot = str(datetime.datetime.now().date())
        self.date = datetime.datetime.now().strftime("(%H-%M)")
        
        self.path = os.path.join(root, self.dateroot)
        if os.path.isdir(self.path) ==False:
            os.makedirs(self.path)

        self.net = net
        self.root = root
        self.notify=notify
        self.id = 0
            
    def save(self, parameters=None):
        name = "modelsave%d_"%self.id + self.date + ".pt"
        torch.save(self.net.state_dict(), os.path.join(self.path, name))
        if parameters:
            assert isinstance(parameters, dict)
            with open(os.path.join(self.path, "parameters%d_"%self.id + self.date + ".txt"), 'w') as f:
                f.write(str(parameters))

        if self.notify:
            print("created " + name)
        self.id += 1

#rendering
def playandsave_episode(render_source, env):
    assert type(render_source) == ptan.experience.ExperienceSource
    print("Warning: Possible Reduced Frames, Make sure to use special wrapper for frame-skip")
    for i, x in enumerate(render_source):
        print(i)
        if x[0][3] or i>2000:
           print("done")
           break

def create_video(frames, output_filename):
    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for MP4 video
    video = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))
    for frame in frames:
        frame = (frame * 255).astype(np.uint8)  # scale values from 0-1 to 0-255
        video.write(frame)
    video.release()

'''class Renderer:
    def __init__(self, env):
        self.env = env

    def sample(self.count):
        ''' 


    
    