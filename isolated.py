from collections import deque
import numpy as np
import timeit




'''def test_code1():
    list=[]
    for x in range(5):
        list.append(x)
    id = 0
    for x in range(5):
        id +=x

    
    


def test_code2():
    d = deque(maxlen=5)
    for x in range(5):
        d.append(x)
    id = 0
    for x in range(5):   
        id+=d.popleft()
        
    
    


number = 10000000
speed = timeit.timeit(stmt="test_code1", setup="from __main__ import test_code1", number=number)
speed2 = timeit.timeit(stmt="test_code2", setup="from __main__ import test_code2", number=number)

print("test1->",speed)
print("test2->",speed2)
'''

def decay_rewards(rewards, gamma):
    decayed = 0.0
    for i in range(len(rewards)):
        r = rewards.pop()
        decayed *= gamma
        decayed += r
        if i == (len(rewards)):
            rewards.appendleft(decayed)
        else:
            rewards.appendleft(r)
    return rewards


rewards = deque([1., 0., 1.])
rewards.append(2.)
print(rewards)
print(decay_rewards(rewards, 0.90))