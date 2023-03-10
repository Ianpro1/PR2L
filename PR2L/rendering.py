#rendering contains classes for live-rendering of the agent and tools for neural network analysis (not yet implemented)
import pygame
import numpy as np
from gymnasium import Wrapper

class RenderWrapper(Wrapper):
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

class SendimgWrapper(Wrapper):
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
    img = (img.transpose(2,1,0) * 255.).astype(np.uint8)
    return img


def HandleFrame(frame,screen, screen_size, preprocessing=None):
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
    #default img format is rgb_array as Height, Width and Channels 
    #parameters-> outconn: receiving end of Pipeconnection 
    #creates a pygame instance of rgb_array upon receive from a Pipe() (used for live rendering of agent)
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


class ReplayWrapper(Wrapper):
    def __init__(self, env, queue):
        super().__init__(env)
        self.queue = queue
        self.framebuffer = []
        self.fullqueue = False
    
    def reset(self):
        print("reset")
        if self.queue.full() == False and len(self.framebuffer) > 0:
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


