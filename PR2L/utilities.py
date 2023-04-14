#utilities includes basic extentions for other libraries (including PR2L)
#listed libraries: Pytorch, gym
import torch
import datetime
import os
from gym import Wrapper
import cv2
import numpy as np

def unpack_batch(batch):
    """
    A class used to unpack a batch of experiences of type experience.Experience

    Returns the following: states, actions, rewards, next_states, not_dones

    NOTE: next_states batch size is smaller than states when there are terminations. This is because not_dones should be used to
     mask a torch.zero_like tensor and replace the values with next_states.
    #This function assumes len(next_states) < 1 is handled properly during training"""
    states = []
    rewards = []
    actions = []
    next_states = []
    not_dones = []
    for exp in batch:
        states.append(exp.state)
        rewards.append(exp.reward)
        actions.append(exp.action)
        
        if exp.next is not None:
            not_dones.append(True)
            next_states.append(exp.next) 
        else:
            not_dones.append(False)   
    return states, actions, rewards, next_states, not_dones

def unpack_memorizedbatch(batch):
    """
    A class used to unpack a batch of experiences of type experience.MemorizedExperience

    NOTE: next_states batch size is smaller than states when there are terminations. This is because not_dones should be used to
     mask a torch.zero_like tensor and replace the values with next_states.
    #This function assumes len(next_states) < 1 is handled properly during training"""
    states = []
    rewards = []
    actions = []
    next_states = []
    not_dones = []
    memories = []
    for exp in batch:
        states.append(exp.state)
        rewards.append(exp.reward)
        actions.append(exp.action)
        memories.append(exp.memory)
        if exp.next is not None:
            not_dones.append(True)
            next_states.append(exp.next) 
        else:
            not_dones.append(False)   
    return states, actions, rewards, next_states, not_dones, memories

#backup
class render_env(Wrapper):
    """
    Simple wrapper class that keeps an rgb array list from last environment reset() inside the attribute: rframes.
    This wrapper is mostly used along the ModelBackup class
    
    """
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

    """
    Class that creates torch.state_dict() of model passed as argument.
    It stores the saves inside a directory called model_saves.
    The class can also create videos of the agent if both an agent and a render environment are passed as arguments
    (gym.Env wrapped inside utilities.render_env). 
    """

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
        

class ModelBackupManager:
    """"
    Variant of ModelBackup: allows backup creation and (Not yet) loading of multiple networks at once.
    """
    def __init__(self,ENV_ID,iid, net, directory="model_saves", notify=True, error=True):
        assert isinstance(iid, str)
        assert isinstance(ENV_ID, str)
        if isinstance(net, (list,tuple)) == False:
            self.net = [net]
        else:
            self.net = net
        self.error = error
        self.path = os.path.join(directory, ENV_ID, iid)
        self.notify = notify

    #TODO add parameter backup
    def save(self, parameters=None):
        save_dir = os.path.join(self.path, "state_dicts", str(datetime.datetime.now().date()))
        
        for network in self.net:
            if os.path.isdir(save_dir) == False:
                os.makedirs(save_dir)
            modelname = str(type(network).__name__) + datetime.datetime.now().strftime("-%H-%M") + ".pt"
            temp_path = os.path.join(save_dir, modelname)
            torch.save(network.state_dict(), temp_path)
            if self.notify:
                print("(ModelBackupManager) created: " + temp_path)

    def load(self):
        # Find the latest dates folder within self.path/state_dicts
        sd_path = os.path.join(self.path, "state_dicts")
        dates_folders = [f for f in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, f)) and f.startswith('20')]
        if not dates_folders:
            if self.error == False:
                return None
            raise FileNotFoundError(f"No matching directory found in {sd_path}")
        latest_date_folder = max(dates_folders)

        # Find the files that were created the latest with matching net name
        net_files = {}
        previous = ''
        previous_name = ''
        expected_len = len(self.net)
        for net_class in self.net:
            net_name = str(type(net_class).__name__)
            matching_files = [f for f in os.listdir(os.path.join(sd_path, latest_date_folder)) if f.startswith(net_name) and f.endswith('.pt')]
            if not matching_files:
                if self.error == False:
                    return None
                raise FileNotFoundError(f"No matching file found for {net_name} in {sd_path}/{latest_date_folder}")
            
            latest_save = ''
            latest = ''
            for f in matching_files:
                timestamp = f[-8:-3]
                if timestamp > latest:
                    latest = timestamp
                    latest_save = f

            if latest != previous and previous != '':
                if self.error == False:
                    return None
                raise ValueError(f"Timestamp for {latest_save} does not match with: {previous_save}")
            
            previous_save = latest_save
            previous = latest
            net_files[net_name] = os.path.join(sd_path, latest_date_folder, latest_save)           

        if len(net_files) != expected_len:
            if self.error == False:
                    return None
            raise ValueError(f"Expected to find {expected_len} files but found {len(net_files)}")
        
        if self.notify:
            print("(ModelBackupManager) Loading the following models: ", net_files)
        for net in self.net:
            net.load_state_dict(torch.load(net_files[type(net).__name__]))






