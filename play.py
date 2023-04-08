import mjct
import numpy as np
import PR2L.playground as p         

env = mjct.make("TosserCPP", render="autogl", apirate=360)

class normalcallback:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def __call__(self, x):
        _, reward, done, _, _ = self.env.step(x)
        if done:
            print(reward)
            self.env.reset()
    
    def reload(self):
        self.env.reset()

cb = normalcallback(env)
ctrl = p.BasicController((2,), cb, cb.reload, ' ')

inp = ['w', 'a', 's', 'd']
out = [
    [1., [0]],
    [1., [1]],
    [-1., [0]],
    [-1., [1]]
]
ctrl.loadController(inp, out)
#ctrl.makeController()
ctrl.play(1)
