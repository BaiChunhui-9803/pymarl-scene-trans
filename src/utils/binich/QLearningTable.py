import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, avail_actions, learning_rate=0.1, reward_decay=0.9):
        self.actions = avail_actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state, e_greedy=1):
        self.check_state_exist(state)
        if np.random.uniform() > e_greedy:
            state_action = self.q_table.loc[state, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(list(self.actions.keys()))
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = pd.concat([self.q_table, pd.Series([0] * len(self.actions),
                                                              index=self.q_table.columns,
                                                              name=state).to_frame().T])