from common import playground, models, extentions
import numpy as np
import torch
from PR2L import experience, agents, utilities
import os

net = models.DenseDQN(4, 100, 4)

backup = extentions.ModelBackup("TESTENV", "001", net)
parameters = {"test":True}

backup.save(parameters)