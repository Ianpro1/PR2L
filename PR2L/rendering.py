#rendering contains classes for live-rendering of the agent and tools for neural network analysis (not yet implemented)
import pygame
import warnings
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        raise ImportError("Failed to import both gymnasium and gym")

import matplotlib.pyplot as plt
import multiprocessing as mp

class RenderWrapper(gym.Wrapper):
    """
    This render wrapper will send the render() output of the environment through the mp.Pipe() given as argument.
    """
    def __init__(self, env, inconn, frame_skip=4):
        super().__init__(env)
        self.inconn = inconn
        self.f_skip = frame_skip
        self.count = 0
    
    def step(self, action):
        obs = self.env.step(action)
        self.count += 1
        if self.count % self.f_skip == 0:
            frame = self.env.render()
            self.inconn.send(frame)
        return obs

    def reset(self):
        obs = self.env.reset()
        self.count = 0
        if self.f_skip == 1:
            frame = self.env.render()
            self.inconn.send(frame)
        return obs

class SendimgWrapper(gym.Wrapper):
    """
    This render wrapper will send the raw observation of the environment through the mp.Pipe() given as argument.
    """
    #assumes the input is an rgb_array
    def __init__(self, env, inconn, frame_skip=4):
        #parameters-> env: gym.Env, inconn: in-connection of Pipe, frame_skip: sending observations every nth frame
        super().__init__(env)
        self.inconn = inconn
        self.frame_skip = frame_skip
        self.count = 0

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        self.count +=1
        if self.count % self.frame_skip == 0:
            self.inconn.send(obs)
        return obs, rew, done, trunc, info
    
    def reset(self):
        obs, info = self.env.reset()
        self.count=0
        self.inconn.send(obs)
        return obs, info
    

def ChannelFirstPreprocessing(img):
    """
    Moves the first channel of an image to the back
    """
    img = (img.transpose(2,1,0) * 255.).astype(np.uint8)
    return img


def HandleFrame(frame, screen, screen_size, preprocessing=None):
    if preprocessing is not None:
        frame = preprocessing(frame)
    else:
        frame = frame.transpose(1,0,2)

    screen.fill((255, 255, 255))

    frame = np.repeat(frame, screen_size[0] // frame.shape[1], axis=1)
    frame = np.repeat(frame, screen_size[1] // frame.shape[0], axis=0)

    #Assumes channel_index is the smallest of the observation
    channel_num = np.min(frame.shape)

    #Then, reduces or increases num of channels to 3 
    if channel_num < 3:
        frame = frame[:, :, :1]
        frame = np.repeat(frame, 3, axis=2)
    elif channel_num > 3:
        frame = frame[:, :, :3]
    
    pygame.surfarray.blit_array(screen, frame)
    pygame.display.update()


def init_display(outconn, screen_size=None, preprocessing=None):

    warnings.warn("init_display() is deprecated. Use displayScreen() instead.")
    """
    (DEPRECATED): displayScreen is prefered over init_display


    #default img format is rgb_array as Height, Width and Channels 
    #parameters-> outconn: receiving end of Pipeconnection (mp.Pipe())

    This class creates a pygame window instance which uses the outputs of mp.Pipe() as frames for the rendering.
    """
    assert isinstance(screen_size, (list, tuple, type(None)))

    if screen_size is None:
        frame = outconn.recv()
        screen_size = frame.shape[:-1]

    pygame.init()
    screen = pygame.display.set_mode((screen_size[1], screen_size[0]))
    while True:
        
        frame = outconn.recv()         

        if frame is None:
            break

        HandleFrame(frame, screen, screen_size, preprocessing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
    pygame.quit()
    return


class ReplayWrapper(gym.Wrapper):
    """
    This wrapper will build an episode of frame using render() and, once the environment terminated, will send it through the mp.Pipe()
    """
    def __init__(self, env, queue, copy=False):
        super().__init__(env)
        self.queue = queue
        self.framebuffer = []
        self.fullqueue = False
        self.copy_ = copy
    
    def reset(self):
        if self.queue.full() == False and len(self.framebuffer) > 0:
            if self.copy_:
                self.queue.put(self.framebuffer.copy())
            else:
                self.queue.put(self.framebuffer)
        obs, info = self.env.reset()
        self.framebuffer.clear()
        self.framebuffer.append(self.env.render())
        return obs, info
    
    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        self.framebuffer.append(self.env.render())
       
        return obs, rew, done, trunc, info


def init_replay(queue, delay, screen_size=None, preprocessing=None):
    """
    #default img format is rgb_array as Height, Width and Channels 
    #parameters-> outconn: receiving end of Pipeconnection (mp.Pipe())

    This class creates a pygame window instance which uses the outputs of mp.Pipe() as frames for the rendering.
    
    NOTE: The difference between  init_display is that this class renders full episodes as opposed to individual frames received
    from the mp.Pipe(). Hence, the episodes can be replayed using a custom delay (FPS) without too much bottleneck. This class should
    be used with many environments in parralled and series, else the multiprocess bottleneck will be more apparent.
    """
    assert isinstance(screen_size, (list, tuple, type(None)))
    
    if screen_size is None:
        #cannot pass
        while True:
            replay = queue.get(block=True)
            if len(replay) <1:
                "keeps going..."
                continue
            screen_size = replay[0].shape[:-1]
            break
        
    
    pygame.init()
    screen = pygame.display.set_mode((screen_size[1], screen_size[0]))
    frame=None
    while True:
        
        replay = queue.get(block=True)        
        if replay is None:
                break 
        
        for frame in replay: 
            HandleFrame(frame, screen, screen_size, preprocessing)
            pygame.time.delay(delay)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        for _ in range(3):
            if frame is not None:
                frame = frame * 0.8
                HandleFrame(frame, screen, screen_size, preprocessing)
                pygame.time.delay(delay)

        
    pygame.quit()
    return


def barprint(array : np.ndarray, delaysec=None):
    assert isinstance(array, np.ndarray)
    if array.ndim == 1:
        plt.title(str(len(array)))
        plt.bar(range(len(array)), array)
    elif array.ndim == 2:
        plt.title('(' + str(array.shape[0]) + ', ' + str(array.shape[1]) + ')')
        for arr in array:
            plt.bar(range(len(arr)), arr)
            if delaysec is not None:
                plt.pause(delaysec)
    else:
        print("(barprint) cannot graph 3d bar diagram...")
    plt.show()


class pltprint:
    def __init__(self, graph=plt.bar, delay=0.001, xlim=None, ylim=(0,1), **kwargs):
        inconn, outconn = mp.Pipe()
        alims = [xlim, ylim]
        self.thread = mp.Process(target=self.plt_thread, args=(outconn, graph, delay, alims), kwargs=kwargs)
        self.thread.start()
        self.inconn = inconn
        self.buffer = []

    def drawavg(self, x, buffer_len = 10):
        if buffer_len == 1:
            self.inconn.send(x)
        else:
            self.buffer.append(x)
            if len(self.buffer) >= buffer_len:
                temp_buffer = np.array(self.buffer, copy=False)
                self.inconn.send(temp_buffer.mean(0))
                self.buffer.clear()

    def drawbuffer(self, x, buffer_len=10):
        self.buffer.append(x)
        if len(self.buffer) >= buffer_len:
            self.inconn.send(np.array(self.buffer, copy=False))
            self.buffer.clear()

    def plt_thread(self, outconn, graph, delay, axis_lim : list, **kwargs):
        x = outconn.recv()
        assert isinstance(x, np.ndarray)
        ndim = x.ndim
        xsize = range(x.shape[-1])
        if ndim ==1:
            while True:
                if axis_lim[0] is not None:
                    plt.xlim(axis_lim[0])
                plt.ylim(axis_lim[1])
                graph(xsize, x, **kwargs)
                plt.pause(delay)
                plt.draw()
                plt.clf()
                x = outconn.recv()
                if x is None:
                    break
        elif ndim == 2:
            while True:
                if axis_lim[0] is not None:
                    plt.xlim(axis_lim[0])
                plt.ylim(axis_lim[1])
                for arr in x:
                    graph(xsize, arr, **kwargs)
                    plt.pause(delay)
                plt.draw()
                plt.clf()
                x = outconn.recv()
                if x is None:
                    break
        else:
            print("(pltprint) Cannot plot {ndim}d array!")


import cv2

def displayScreen(outconn, upscale_factor : int = 1, preprocessing=None):
    """
    pygame screen that is used for rendering rgb_arrays.
    parameters: 
    outconn -> receiving end of mp.Pipe() connection. It expects np.ndarray of integer values (min: 0, max : 255)
    
    upscale_factor -> small arrays might not function properly with pygame surfaces. This parameter will do a cv2.resize with linear interpolation.
    You can disable upscale_factor by passing 1.
    
    preprocessing -> A user-defined function for any extra preprocessing.

    """
    pygame.init()
    
    img = outconn.recv()
    assert img is not None
    
    if preprocessing is not None:
        img = preprocessing(img)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3
 
    #handle channel position
    channel_argmin = np.argmin(img.shape)
    order = [id % 3 for id in range(channel_argmin+1, 3 + channel_argmin + 1)]
    img = img.transpose((order))

    #handle channel number

    channel_size = np.min(img.shape)
    assert channel_size == 1 or channel_size == 3

    if (channel_size == 1):
        img = np.repeat(img, 3, 2)

    #upscale img (if required)
    if (upscale_factor != 1):
        img = cv2.resize(img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
    
    shape = img.shape
    screen = pygame.display.set_mode(shape[:-1])

    while(1):
        screen.fill((255, 255, 255))
        pygame.surfarray.blit_array(screen, img)
        pygame.display.update()

        #get new img
        img = outconn.recv()
        if img is None:
            break
        #handle processing
        if preprocessing is not None:
            img = preprocessing(img)
        img = img.transpose((order))
        if (channel_size == 1):
            img = np.repeat(img, 3, 2)
        if (upscale_factor != 1):
            img = cv2.resize(img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
    pygame.quit()