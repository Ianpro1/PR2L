from collections import deque
import timeit
import numpy as np
d = deque(maxlen=5)
l = []
nparr = np.ones(shape=(10,10))
print(nparr.size)

dtype = np.dtype([('arr', np.float32, (84, 84, 3))])
arr = np.empty(5, dtype=dtype)
l = [None]*5

def test1():
    idd = 0
    for x in range(10000):
        idd += 1
        obj = np.empty(shape=(84,84,3), dtype=np.float32)
        arr['arr'][idd] = obj
        if idd == 4:
            idd = 0

def test2():
    idd = 0
    for x in range(10000):
        idd += 1
        obj = np.empty(shape=(84,84,3), dtype=np.float32)
        l[idd] = obj
        if idd == 4:
            idd = 0


print(timeit.timeit(stmt=test1, number=1))
print(timeit.timeit(stmt=test2, number=1))