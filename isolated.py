from collections import deque
import numpy as np
import timeit


for x in range(100):
    print(x)
    if x % 60 >= 1:
        break



'''def test_code1():
    l = list([])

    for x in range(100):
        l.append(x)
    


def test_code2():
    t = tuple([])
    for x in range(100):
        t.append(x)
    
    


number = 100000000
speed = timeit.timeit(stmt="test_code1", setup="from __main__ import test_code1", number=number)
speed2 = timeit.timeit(stmt="test_code2", setup="from __main__ import test_code2", number=number)

print("test1->",speed)
print("test2->",speed2)'''


