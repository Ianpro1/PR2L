import mjct
import numpy as np 
from PR2L.playground import BasicController, CustomControllerCallback



env = mjct.make("TosserCPP", render="autogl", timestep=0.002, apirate=60)

cb = CustomControllerCallback(env)
ctrl = BasicController((2,), cb, cb.reload)

inp = ['w', 'a', 's', 'd']

out = [
[-1,[0]],
[1, [0]],
[-1,[1]],
[1, [1]],
]

ctrl.loadController(inp, out)
ctrl.play(1)