import torch
import datetime
import os
from gym import Wrapper
import cv2
import numpy as np


def unpack_batch(batch):
    #Note: the unpack function returns a reduce size last_state list (does not include terminated states)
    states = []
    rewards = []
    actions = []
    last_states = []
    not_dones = []
    for exp in batch:
        states.append(exp.state)
        rewards.append(exp.reward)
        actions.append(exp.action)
        
        if exp.next is not None:
            not_dones.append(True)
            last_states.append(exp.next) 
        else:
            not_dones.append(False)   
    return states, actions, rewards, last_states, not_dones


#backup
class render_env(Wrapper):
    #simple wrapper that keeps a rgb_array_list from last env reset
    def __init__(self, env):
        super().__init__(env)
        self.rframes = []

    def reset(self):
        self.rframes.clear()
        obs, info = self.env.reset()
        self.rframes.append(self.env.render())
        return obs, info
    
    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        self.rframes.append(self.env.render())
        return obs, rew, done, trunc, info


class ModelBackup:
    def __init__(self, ENV_NAME, iid, net, notify=True, render_env=None, agent=None, folder="model_saves", prefix="model_"):
        assert isinstance(iid, str)
        if render_env is None or agent is None and notify:
            self.disable_mkrender = True
            print("Warning, argument missing for ModelBackup.mkrender and has been disabled")
        else:
            self.disable_mkrender = False
        self.agent = agent
        self.render_env = render_env
        self.net = net
        self.notify = notify
        self.path = os.path.join(folder, ENV_NAME, prefix+iid)
        if os.path.isdir(self.path) == False:
            os.makedirs(self.path)

    def save(self, parameters=None):
        assert isinstance(parameters, (dict, type(None)))
        date = str(datetime.datetime.now().date())
        time = datetime.datetime.now().strftime("-%H-%M")

        if parameters is not None:
            temproot = os.path.join(self.path, "parameters", date)

            if os.path.isdir(temproot) == False:
                os.makedirs(temproot)

            with open(os.path.join(temproot, "save" + time +".txt"), 'w') as f:
                f.write(str(parameters))
        
        temproot = os.path.join(self.path, "state_dicts", date)
        if os.path.isdir(temproot) == False:
            os.makedirs(temproot)

        location = os.path.join(temproot, "save" + time +".pt")
        torch.save(self.net.state_dict(), location)
        
        if self.notify:
            print("created " + location)
    
    def mkrender(self, fps=60.0, frametreshold=2000, colors=cv2.COLOR_RGB2BGR):
        if self.disable_mkrender:
            return 0
        else:
            obs, info = self.render_env.reset()
            obs = [obs]
            action = self.agent(obs)[0]
            frameid=0
            while True:
                frameid += 1
                if self.notify:
                    print(frameid)
                obs, rew, done, trunc, info = self.render_env.step(action)
                obs = [obs]
                action = self.agent(obs)[0]
                
                if done or frameid > frametreshold:
                    break
            
            date = str(datetime.datetime.now().date())
            time = datetime.datetime.now().strftime("-%H-%M")
            temproot = os.path.join(self.path, "renders", date)
            if os.path.isdir(temproot) == False:
                os.makedirs(temproot)
            frames = self.render_env.rframes
            height, width, channels = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            location = os.path.join(temproot, "save" + time +".mp4")
            video = cv2.VideoWriter(location, fourcc, fps, (width, height))
            for frame in frames:
                frame = cv2.cvtColor(frame, colors)
                video.write(frame)
            video.release()
            return 1