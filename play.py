import mjct
import numpy as np
import PR2L.playground as p         

env = mjct.make("TosserCPP", render="glwindow", apirate=360)

cb = p.CustomControllerCallback(env)
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
