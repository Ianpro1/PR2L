import mjct
import numpy as np
import PR2L.playground as p         

env = mjct.make("TosserCPP", True, 0.002, 1000)

class custom_callback:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def __call__(self, x):
        x = np.array(x, copy=True)
        _, _, done, _, _ = self.env.step(x)
        self.env.render()
        if done:
            self.env.reset()
    def reload(self):
        self.env.reset()


cb = custom_callback(env)
ctrl = p.BasicController((2,), cb, cb.reload)

inp = ['w', 'a', 's', 'd']
out = [
    [1., [0]],
    [1., [1]],
    [-1., [0]],
    [-1., [1]]
]
ctrl.loadController(inp, out)
ctrl.play(0)
