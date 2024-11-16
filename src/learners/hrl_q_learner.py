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

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, agent=None):
        if self.args.runner == "parallel_cbs":
            batch_list = batch.buffer_pool.values()
            train_batch = []
            for batch in batch_list:
                t_max_step = len(batch["filled"])
                for t in range(t_max_step - 1):
                    upper_state = batch["upper_state"][t]
                    upper_action = batch["upper_action"][t]
                    global_reward = batch["global_reward"][t]
                    upper_next_state = batch["upper_state"][t + 1]

                    lower_id = batch["lower_id"][t]
                    lower_state = batch["lower_state"][t]
                    lower_action = batch["lower_action"][t]
                    short_reward = batch["global_reward"][t]
                    lower_next_state = batch["lower_state"][t + 1]
                    if t == t_max_step - 2:
                        upper_next_state = 'terminal'
                        lower_next_state = 'terminal'

                    self.controller.agent.cluster_qtable.learn(upper_state, upper_action, global_reward,
                                                               upper_next_state)
                    self.controller.agent.combat_qtable_dict[lower_id].learn(lower_state, lower_action, short_reward,
                                                                             lower_next_state)
        elif self.args.runner == "episode_cbs":
            assert len(batch["filled"]) == 1
            t_max_step = len(batch["filled"][0])
            for t in range(t_max_step - 1):
                upper_state = batch["upper_state"][0][t]
                upper_action = batch["upper_action"][0][t]
                global_reward = batch["global_reward"][0][t]
                upper_next_state = batch["upper_state"][0][t + 1]
                if t == t_max_step - 2:
                    upper_next_state = 'terminal'
                self.controller.agent.cluster_qtable.learn(upper_state, upper_action, global_reward,
                                                           upper_next_state)

                lower_id = batch["lower_id"][0][t]
                lower_state = batch["lower_state"][0][t]
                lower_action = batch["lower_action"][0][t]
                short_reward = 0 if t == 0 else batch["global_reward"][0][t-1]
                if agent.previous_combat_action[lower_id] is not None:
                    self.controller.agent.combat_qtable_dict[lower_id].learn(
                        agent.previous_combat_state[lower_id],
                        agent.previous_combat_action[lower_id],
                        short_reward,
                        'terminal' if t == t_max_step - 1 else lower_state)

                agent.previous_sub_tag = lower_id
                agent.previous_combat_state[lower_id] = lower_state
                agent.previous_combat_action[lower_id] = lower_action

                # short_reward = batch["short_reward"][0][t]
                # lower_next_state = batch["lower_state"][0][t + 1]

                    # lower_next_state = 'terminal'

                # self.controller.agent.combat_qtable_dict[lower_id].learn(lower_state, lower_action,
                #                                                          short_reward, lower_next_state)

                    # train_batch.append([state, action, reward, next_state])




        # TODO - Log the stats
        # if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("grad_norm", grad_norm, t_env)
            # self.log_stats_t = t_env

    def cuda(self):
        self.controller.cuda()
