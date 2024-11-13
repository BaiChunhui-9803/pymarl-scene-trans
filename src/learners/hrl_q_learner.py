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

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        batch_list = [batch.data.transition_data for batch in batch.buffer_pool]

        train_batch = []
        for batch in batch_list:
            n_episodes = len(batch["filled"])
            for k in range(n_episodes):
                t_max_step = len(batch["filled"][k])
                for t in range(t_max_step - 1):
                    upper_state = batch["upper_state"][k][t]
                    upper_action = batch["upper_action"][k][t]
                    global_reward = batch["global_reward"][k][t]
                    upper_next_state = batch["upper_state"][k][t+1]

                    lower_id = batch["lower_id"][k][t]
                    lower_state = batch["lower_state"][k][t]
                    lower_action = batch["lower_action"][k][t]
                    short_reward = batch["short_reward"][k][t]
                    lower_next_state = batch["lower_state"][k][t+1]
                    if t == t_max_step - 2:
                        upper_next_state = 'terminal'
                        lower_next_state = 'terminal'

                    self.controller.agent.cluster_qtable.learn(upper_state, upper_action, global_reward, upper_next_state)
                    self.controller.agent.combat_qtable_dict[lower_id].learn(lower_state, lower_action, short_reward, lower_next_state)

                    # train_batch.append([state, action, reward, next_state])


        # TODO - Log the stats
        # if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("grad_norm", grad_norm, t_env)
            # self.log_stats_t = t_env
