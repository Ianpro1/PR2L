import torch
import numpy as np
import ptan
import cv2
import os
import datetime


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
        #print(states)
        states = torch.FloatTensor(states)
        return super().__call__(states)


#batch handeling
def unpack_batch(batch, obs_shape): # return states, actions, calculated tgt_q_v = r + tgt_net(last_state)*GAMMA
    states = []
    rewards = []
    actions = []
    last_states = []
    dones = []
    for exp in batch: # make it in 2 for loops for if statement
        states.append(exp.state)
        rewards.append(exp.reward)
        actions.append(exp.action)
        
        if exp.last_state is not None:
            dones.append(False)
            last_states.append(exp.last_state) 
        else:
            dones.append(True)
            last_states.append(np.empty(shape=obs_shape)) #might be suboptimal
   
    return states, actions, rewards, last_states, dones
    
def calc_grad(net):
    grad_max = 0
    grad_count = 0
    grad_means = 0
    for p in net.parameters():
        grad_max = max(grad_max, p.grad.abs().max().item())
        grad_means += (p.grad ** 2).mean().sqrt().item()
        grad_count += 1

    return grad_means/grad_count, grad_max
    
def meanmax_weight(net):
    mean_w = []
    max_w = []
    for x in net.parameters():
        max_w.append(x.max().item())
        mean_w.append(x.mean().item())
    return mean_w, max_w
    
#model save
class ModelBackup:
    def __init__(self, root, net, notify=True, Temp_disable=False):
        self.dateroot = str(datetime.datetime.now().date())
        self.date = datetime.datetime.now().strftime("(%H-%M)")
        
        self.path = os.path.join(root, self.dateroot)
        if os.path.isdir(self.path) ==False:
            os.makedirs(self.path)
        self.Temp_disable = Temp_disable
        self.net = net
        self.root = root
        self.notify=notify
        self.id = 0
            
    def save(self, parameters=None):
        if self.Temp_disable:
            return None
        name = "modelsave%d_"%self.id + self.date + ".pt"
        torch.save(self.net.state_dict(), os.path.join(self.path, name))
        if parameters:
            assert isinstance(parameters, dict)
            with open(os.path.join(self.path, "parameters%d_"%self.id + self.date + ".txt"), 'w') as f:
                f.write(str(parameters))

        if self.notify:
            print("created " + name)
        self.id += 1
        self.date = self.date = datetime.datetime.now().strftime("(%H-%M)")

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


    
    