import multiprocessing as mp
import numpy as np
import pygame
import torch
import pandas as pd
import torch.nn as nn

class grads_manager:
    #should make a simpler function to return distribution values
    @torch.no_grad()
    def __init__(self, net, func="rainbow", path="output.csv", scaler="minmax"):
        assert isinstance(func, str)
        self.net = net
        self.column = None
        self.path = path
        self.scaler = scaler
        if func == 'mean':
            self.func = self.layer_mean
        elif func == 'min':
            self.func = self.layer_min
        elif func == 'max':
            self.func = self.layer_max
        elif func == 'rainbow':
            self.func = self.layer_rainbow
            self.column = ["max", "mean", "min"]
        else:
            raise ValueError(func + " does not exist!")
    
    def write_csv(self):
        names = []
        grads = []
        for name, param in self.net.named_parameters():
            names.append(name)
            grads.append(self.func(param.grad))

        if self.scaler == "minmax":
            max_grads = max(grads)
            min_grads = min(grads)
            grads = self.minmax(grads)
            grads = np.append(grads, (max_grads, min_grads))
            names.extend(["max_grads", "min_grads"])

        if self.column is not None:
            frame = pd.DataFrame(grads, index=names, columns=self.column)
        else:
            frame = pd.DataFrame(grads, index=names)

        self.frame = frame
        frame.to_csv(self.path)

    def get(self):
        names = []
        grads = []
        for name, param in self.net.named_parameters():
            names.append(name)
            grads.append(self.func(param.grad))  
        if self.scaler == "minmax":
            grads = self.minmax(grads)
        return names, grads
    
    def minmax(self, x):
        maxx = max(x)
        minx = min(x)
        rangex = maxx - minx
        scaled = (x-minx)/rangex
        return scaled

    @staticmethod
    def layer_mean(layer):
        return layer.cpu().mean().numpy()
    @staticmethod
    def layer_max(layer):
        return layer.cpu().max().numpy()
    @staticmethod
    def layer_min(layer):
        return layer.cpu().min().numpy()
    @staticmethod
    def layer_rainbow(layer):
        max = layer.cpu().max().numpy()
        mean = layer.cpu().mean().numpy()
        min = layer.cpu().min().numpy()
        return [max, mean, min]

class params_toDataFrame:
    #create an output file or returns a pandas DataFrame on demand which contains labeled parameters of the network
    #arguments: mean, max, min, rainbow-> return all
    @torch.no_grad()
    def __init__(self, net, func="rainbow", path="output.csv"):
        assert isinstance(func, str)
        names = []
        params = []
        column = None
        if func == 'mean':
            func = self.layer_mean
        elif func == 'min':
            func = self.layer_min
        elif func == 'max':
            func = self.layer_max
        elif func == 'rainbow':
            func = self.layer_rainbow
            column = ["max", "mean", "min"]
        else:
            raise ValueError(func + " does not exist!")
        
        for name, param in net.named_parameters():
            names.append(name)
            params.append(func(param))

        if column is not None:
            frame = pd.DataFrame(params, index=names, columns=column)
        else:
            frame = pd.DataFrame(params, index=names)
        self.frame = frame
        frame.to_csv(path)

    @staticmethod
    def layer_mean(layer):
        return layer.cpu().mean().numpy()
    @staticmethod
    def layer_max(layer):
        return layer.cpu().max().numpy()
    @staticmethod
    def layer_min(layer):
        return layer.cpu().min().numpy()
    @staticmethod
    def layer_rainbow(layer):
        max = layer.cpu().max().numpy()
        mean = layer.cpu().mean().numpy()
        min = layer.cpu().min().numpy()
        return [max, mean, min]
    
    def get(self):
        return self.frame



def init_display(conn, width, height, frame_shape):
    #creates a pygame instance of rgb_array upon receive from a Pipe() (used for live rendering of agent)
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    while True:
        
        frame = conn.recv()
        
        if frame is None:
            break
        
        channel_id = np.argmin(frame.shape)
        
        channel_num = np.min(frame.shape)

        frame = np.moveaxis(frame, channel_id, 2)
        screen.fill((255, 255, 255))
        frame = np.array(frame, dtype=np.float32).transpose(1,0,2)
        frame = np.repeat(frame, height // frame_shape[0], axis=0)
        frame = np.repeat(frame, width // frame_shape[1], axis=1)

        
        if channel_num < 3:
            frame = frame[:, :, :1]
            frame = np.repeat(frame, 3, axis=2)
        elif channel_num > 3:
            frame = frame[:, :, :3]
        
        frame = (frame).astype(np.uint8)
        
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return



def init_bar(data_shape, height, bar_width, conn, pool=1):
    #creates a pygame instance of a bar chart of user-defined pixel length (this can be used for live rendering)
    if len(data_shape) != 1:
        raise MemoryError
    pygame.init()
    dlen = data_shape[-1]
    screen = pygame.display.set_mode((dlen * bar_width, height))
    while True:
        
        data = conn.recv()
        assert isinstance(data, (np.ndarray))
        if data is None:
            break
        screen.fill((255, 255, 255))

        frame = np.full(shape=(dlen, height), fill_value=255, dtype=np.uint8)
        dmin = data.min()
        dmax = data.max()
        drange = dmax - dmin
        if drange !=0:
            scaled_v = (data - dmin )/ drange
            scaled_v = np.floor(scaled_v * height)
            scaled_v = scaled_v.astype(np.uint8)
        else:
            scaled_v = data
        
        for i, x in enumerate(scaled_v):
            frame[i][:x] = 0
        frame = np.repeat(frame, bar_width, 0)
        frame = (frame).astype(np.uint8)
        frame = np.array([frame] * 3).transpose(1,2,0)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
               


