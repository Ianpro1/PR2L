import matplotlib.pyplot as plt
import numpy as np
from collections import deque



for i in range(1000):
    x = np.random.randint(0, 1000+i*1000)
    y = np.random.randint(0, 1000+i*1000)
    plt.scatter(x, y)
    plt.pause(0.000001)

plt.show()