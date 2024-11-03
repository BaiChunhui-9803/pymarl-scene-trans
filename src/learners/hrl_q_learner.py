import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
import pandas as pd
import numpy as np


class QLearner:
    def __init__(self, controller, scheme, logger, args):
        self.args = args
        self.controller = controller
        self.logger = logger

        # self.qtable = pd.DataFrame(columns=self.actions, dtype=np.float64)

        # self.params = list(controller.parameters())

        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1