import pygame
import numpy as np
from gym import Wrapper



class SendimgWrapper(Wrapper):
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


def init_display(outconn, screen_size, preprocessing=None):
    #default img format is rgb_array as Height, Width and Channels 
    #parameters-> outconn: receiving end of Pipeconnection 
    #creates a pygame instance of rgb_array upon receive from a Pipe() (used for live rendering of agent)
    assert isinstance(screen_size, (list, tuple))

    pygame.init()
    screen = pygame.display.set_mode((screen_size[1], screen_size[0]))
    while True:
        
        frame = outconn.recv()

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

