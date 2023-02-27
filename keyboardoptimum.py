# default keboard layout

#1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
#1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
#1, 2, 3, 4, 5, 6, 7, 8, 9, 10

import numpy as np


layout_grid = np.zeros(shape=(3,12))
layout_grid[1, 11:] = -1
layout_grid[2, 10:] = -1
print(layout_grid)
default_pos = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 6),(1, 7),(1, 8),(1, 9)]

def reset_placement(placements):
    for i,x in enumerate(placements):
        layout_grid[x] = i + 1
reset_placement(default_pos)
print(layout_grid)