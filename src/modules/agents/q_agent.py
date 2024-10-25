import pandas as pd
import numpy as np


class QAgent:
    def __init__(self, avail_actions, args, learning_rate=0.1, reward_decay=0.9):
        super(QAgent, self).__init__()
        self.args = args
        self.avail_actions = avail_actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.cluster_qtable = pd.DataFrame(columns=self.avail_actions["avail_cluster_strengths"], dtype=np.float64)
        self.combat_qtable_dict = {}



    def get_cluster_qtable(self):
        return self.cluster_qtable

    def get_combat_qtable_dict(self):
        return self.combat_qtable_dict




