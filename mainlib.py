from collections import deque
import timeit
import numpy as np

def decay_oldest_reward(rewards, gamma):
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

def decay_oldest_rewardv2(rewards, gamma):
        for i in range(1, len(rewards)):
             rewards[0] += rewards[i] * gamma**i
        return rewards

def decay_oldest_rewardv3(rewards, gamma):
        tot = 0.0
        for r in reversed(rewards):
             tot = r + tot*gamma
        rewards[0] = tot
        
def all1():
    my_deque = deque(np.ones_like(range(10)))
    return decay_oldest_reward(my_deque, 0.99)

def all2():
    my_deque = deque(np.ones_like(range(10)))
    return decay_oldest_rewardv2(my_deque, 0.99)

def all3():
    my_deque = deque(np.ones_like(range(10)))
    return decay_oldest_rewardv3(my_deque, 0.99)

def alllist():
     my_list = list(np.ones_like(range(10)))
     return decay_oldest_rewardv3(my_list, 0.99)


print(all1(), all2(), all3())

print(timeit.timeit(all1, number=100000))
print(timeit.timeit(all2, number=100000))
print(timeit.timeit(all3, number=100000))
print(timeit.timeit(alllist, number=100000))

'''print(
    timeit.timeit(d, number=1000),
    timeit.timeit(d2, number=1000),
    timeit.timeit(l, number=1000)
)'''


