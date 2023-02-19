from PR2L.experience import ExperienceSource, EpisodeReplayBuffer
from PR2L.agents import BasicAgent
import common.models as models
import common.playground as pg

device = "cpu"

def wrapper(x):
    x = pg.VaryObservation()(x)
    return pg.ScaleRGBimage()(x)

env1 = pg.DummyEnv((1), 1.,None, wrapper, pg.EpisodeLength(8))
env2 = pg.DummyEnv((1), 2.,None, wrapper, pg.EpisodeLength(100))
env3 = pg.DummyEnv((1), 3.,None, wrapper, pg.EpisodeLength(2))

env = [env1, env2, env3]

net = models.DenseDQN(1,10,4)

agent = BasicAgent(net)

exp_source = ExperienceSource(env, agent, 4)
buffer = EpisodeReplayBuffer(exp_source, 10000)

for x in range(1):
    buffer.populate(3)
