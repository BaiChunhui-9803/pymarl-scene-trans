from smac.env import StarCraft2Env

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

from src.utils.binich.influence_map import InfluenceMap
from src.utils.binich.cluster import Cluster, distance

import numpy as np
from absl import logging

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

attributes = [
    "owner",
    "tag",
    "unit_type",
    "health",
    "health_max",
    "pos"
]

scripts = {
    "action_ATK_nearest": 1,
    "action_ATK_clu_nearest": 2,
    "action_ATK_nearest_weakest": 3,
    "action_ATK_clu_nearest_weakest": 4,
    "action_ATK_threatening": 5,
    "action_DEF_clu_nearest": 6,
    "action_MIX_gather": 7,
    "action_MIX_lure": 8,
    "action_MIX_lure_2": 9,
}


class CustomStarCraft2Env(StarCraft2Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.window_size = (860, 600)
        self.sorted_agents = None
        self.sorted_enemies = None
        self.featured_agents = None
        self.featured_enemies = None
        self.window_size = (1280, 960)

        self.im = InfluenceMap(self.n_agents)
        self.cluster = Cluster(self.n_agents)

    def reset(self):
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        return self.get_obs(), self.get_state()

    # binich - custom method for getting state using influence map hashing
    def get_im_state(self):
        self.im.update(self.agents, self.enemies)
        im_state = self.im.get_im_hash()
        return im_state

    def get_original_state(self):
        self.im.update(self.agents, self.enemies)
        original_state = {'featured_agents': self.im.featured_agents, 'featured_enemies': self.im.featured_enemies}
        return original_state

    def update(self, agents, enemies):
        self.sorted_agents = [{'tag': agent.tag, 'x': agent.pos.x, 'y': agent.pos.y,
                              'health': agent.health, 'health_max': agent.health_max}
                              for agent in self.agents.values()]
        self.sorted_enemies = [{'tag': enemy.tag, 'x': enemy.pos.x, 'y': enemy.pos.y,
                               'health': enemy.health, 'health_max': enemy.health_max}
                               for enemy in self.enemies.values()]
        self.featured_agents = sorted([(item['tag'], item['x'], item['y'], item['health'], item['health_max'])
                                       for item in self.sorted_agents], key=lambda x: x[0])
        self.featured_enemies = sorted([(item['tag'], item['x'], item['y'], item['health'], item['health_max'])
                                        for item in self.sorted_enemies], key=lambda x: x[0])

    def get_avail_actions(self):
        avail_actions = {
            "avail_cluster_strengths": self.cluster.cluster_strengths,
            "avail_scripts": scripts
        }
        # self.cluster.update(self.agents, self.enemies)
        # cluster = self.cluster.kmeans(3)
        return avail_actions

    def get_nearest_enemy(self, mp, enemies):
        min_dis = 99.
        min_tag = -1
        for enemy in enemies:
            dis = distance((enemy[1], enemy[2]), mp)
            if dis < min_dis:
                min_dis = dis
                min_tag = enemy[0]
        return min_tag

    def get_local_enemy(self, local_agents, enemies):
        local_enemies = []
        for enemy in enemies:
            sum_distance = 0
            count = 0
            for unit in local_agents:
                sum_distance += distance((enemy[1], enemy[2]), (unit[1], unit[2]))
                if distance((enemy[1], enemy[2]), (unit[1], unit[2])) < self.cluster._unit_shoot_range:
                    count += 1
            local_enemies.append((enemy, count, sum_distance))
        local_enemies.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        return local_enemies

    def get_center_position(self, alliance):
        position = (0, 0)
        if alliance == 'Self':
            units = self.sorted_agents
        else:
            units = self.sorted_enemies
        if len(units) == 0:
            return position
        for unit in units:
            position = tuple(map(lambda x, y: x + y, position, (unit['x'], unit['y'])))
        return (position[0] / len(units), position[1] / len(units))


    def action_ATK_nearest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        if len(units) > 0 and len(enemies) > 0:
            for unit in units:
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["attack"],
                    target_unit_tag=self.get_nearest_enemy((unit[1], unit[2]), enemies),
                    unit_tags=[unit[0]],
                    queue_command=False,
                ))))
            return action_list
        return r_pb.ActionRawUnitCommand(
                ability_id=actions["no_op"],
                queue_command=False,
            )

    def action_ATK_clu_nearest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        mp = self.get_center_position('Self')
        if len(units) > 0 and len(enemies) > 0:
            for clu in cluster_result[2]:
                local_enemies = self.get_local_enemy(clu[4], enemies)
                for unit in clu[4]:
                    action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                        ability_id=actions["attack"],
                        target_unit_tag=local_enemies[0][0][0],
                        unit_tags=[unit[0]],
                        queue_command=False,
                    ))))

    # TODO 重写step()
    def step(self, action_list):

        req_actions = sc_pb.RequestAction(actions=action_list)
        try:
            self._controller.actions(req_actions)
            self._controller.step(self._step_mul)
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}


        pass










