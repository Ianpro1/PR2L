
gamma = 0.99


def decay_all_rewards(rewards):
        prev = 0.0
        res = []
        for r in reversed(rewards):
            r += prev
            res.append(r)
            prev = r * gamma
        return res

def decay_oldest_reward(rewards):
    tot = 0.0
    for r in reversed(rewards):
            tot = r + tot* gamma
    rewards[0] = tot

