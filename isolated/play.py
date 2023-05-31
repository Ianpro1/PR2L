import mjct
import numpy as np
import PR2L.playground as p
from PR2L import rendering      
import gym
#env = mjct.make("TosserCPP", render="autogl", apirate=60)
class normalcallback:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def __call__(self, x):
        x = int(x[0])
        _, reward, done, _, _ = self.env.step(x)
        if done:
            print(reward)
            self.env.reset()
    
    def reload(self):
        self.env.reset()


import multiprocessing as mp
if __name__ == "__main__":

    inconn, outconn = mp.Pipe()


    display = mp.Process(target=rendering.init_display, args=(outconn,))
    env = gym.make("MsPacmanNoFrameskip-v4", render_mode="rgb_array")
    env = rendering.RenderWrapper(env, inconn, frame_skip=1)
    display.start()

    cb = normalcallback(env)
    ctrl = p.BasicController((1,), cb, cb.reload, ' ')

    inp = ['w', 'a', 's', 'd']
    out = [
        [1, [0]],
        [2, [0]],
        [3, [0]],
        [0, [0]]
    ]
    ctrl.loadController(inp, out)
    #ctrl.makeController()
    ctrl.play(20)
