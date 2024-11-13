from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.custom_episode_buffer import CustomEpisodeCBSBatch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, controller):
        self.new_batch = partial(CustomEpisodeCBSBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.controller = controller

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        # self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                # binich - self.env.get_state() -> self.env.get_im_state()
                # using influence map hashing state
                "upper_state": self.env.get_im_state(),
                # binich - Preserved the original state.
                "original_state": self.env.get_original_state(),
                "avail_actions": self.env.get_avail_actions(),
                # "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, multi_action_data = self.controller.select_actions(self.env, self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            global_reward, short_reward, terminated, env_info = self.env.step(actions, self.args)
            episode_return += global_reward

            post_transition_data = {
                "upper_action": multi_action_data["upper_action"],
                "lower_id": multi_action_data["lower_id"],
                "lower_state": multi_action_data["lower_state"],
                "lower_action": multi_action_data["lower_action"],
                "global_reward": global_reward,
                "short_reward": short_reward,
                "terminated": terminated != env_info.get("episode_limit", False),
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "upper_state": self.env.get_im_state(),
            "original_state": self.env.get_original_state(),
            "avail_actions": self.env.get_avail_actions(),
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, multi_action_data = self.controller.select_actions(self.env, self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        last_transition_data = {
            "upper_action": multi_action_data["upper_action"],
            "lower_id": multi_action_data["lower_id"],
            "lower_state": multi_action_data["lower_state"],
            "lower_action": multi_action_data["lower_action"],
        }
        self.batch.update(last_transition_data, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        plt.close()
        matplotlib.pyplot.figure().clear()
        matplotlib.pyplot.close()

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.controller, "get_epsilon"):
                self.logger.log_stat("epsilon", self.controller.get_epsilon(self.t_env), self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
