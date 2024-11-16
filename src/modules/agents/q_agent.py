import pandas as pd
import numpy as np
from src.utils.binich.QLearningTable import QLearningTable


class QAgent:
    def __init__(self, avail_actions, args, learning_rate=0.1, reward_decay=0.9):
        super(QAgent, self).__init__()
        self.args = args
        self.avail_actions = avail_actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.cluster_qtable = QLearningTable(self.avail_actions["avail_cluster_strengths"], learning_rate, reward_decay)
        # self.cluster_qtable = pd.DataFrame(columns=self.avail_actions["avail_cluster_strengths"], dtype=np.float64)
        self.combat_qtable_dict = {}
        self.sub_clusters_qtable_tag = None
        self.previous_sub_tag = None

        self.previous_combat_state = {}
        self.previous_combat_action = {}





    def get_cluster_qtable(self):
        return self.cluster_qtable

    def get_combat_qtable_dict(self):
        return self.combat_qtable_dict

    def check_sub_table_exist(self, combat_table_tag):
        if combat_table_tag in self.combat_qtable_dict:
            return True
        else:
            return False

    def update_combat_qtable_dict(self, cluster_list):
        combat_table_tag = (cluster_list[0], cluster_list[1])
        if not self.check_sub_table_exist(combat_table_tag):
            self.combat_qtable_dict.update({combat_table_tag: QLearningTable(self.avail_actions["avail_scripts"])})
            self.previous_combat_state.update({combat_table_tag: None})
            self.previous_combat_action.update({combat_table_tag: None})
        return combat_table_tag

    def cuda(self):
        pass




