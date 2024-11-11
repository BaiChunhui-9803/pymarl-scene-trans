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
import src.utils.binich.math_utils as math_utils
import src.utils.binich.reward_utils as reward_utils

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
    "action_DEF_nearest": 6,
    "action_DEF_clu_nearest": 7,
    "action_MIX_gather": 8,
    "action_MIX_lure_remotest": 9,
    "action_MIX_lure_weakest": 10,
}


class CustomStarCraft2Env(StarCraft2Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.window_size = (860, 600)
        self.sorted_agents = None
        self.sorted_enemies = None
        self.featured_agents = None
        self.featured_enemies = None
        self.window_size = (960, 720)

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

    def get_clu_state(self, cluster_result):
        self.cluster.update(self.agents, self.enemies)
        cluster_list = [(item[2], item[3]) for item in cluster_result[2]]
        result = self.cluster.hashing(cluster_list)
        return result

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

    def get_nearest_enemy_pos(self, mp, enemies):
        min_dis = 99.
        min_pos = (enemies[0][1], enemies[0][2])
        for enemy in enemies:
            dis = distance((enemy[1], enemy[2]), mp)
            if dis < min_dis:
                min_dis = dis
                min_pos = (enemy[1], enemy[2])
        return min_pos


    def get_nearest_weakest_enemy(self, mp, enemies):
        min_dis = 99.
        min_tag = -1
        min_health = 9999
        for enemy in enemies:
            dis = distance((enemy[1], enemy[2]), mp)
            if enemy[3] < min_health or (enemy[3] == min_health and dis < min_dis):
                min_health = enemy[3]
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
                if distance((enemy[1], enemy[2]), (unit[1], unit[2])) < self.cluster.get_shoot_range():
                    count += 1
            local_enemies.append((enemy, count, sum_distance))
        local_enemies.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        return local_enemies

    def get_local_weak_enemy(self, local_agents, enemies):
        local_enemies = []
        for enemy in enemies:
            count = 0
            for agent in local_agents:
                if distance((enemy[1], enemy[2]), (agent[1], agent[2])) < self.cluster.get_shoot_range():
                    count += 1
            local_enemies.append((enemy, count, enemy[3]/enemy[4]))
        local_enemies.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        return local_enemies

    def get_threatening_enemy(self, mp, enemies):
        max_health = 0
        min_dis = 99.
        min_tag = -1
        for enemy in enemies:
            dis = distance((enemy[1], enemy[2]), mp)
            if enemy[3] > max_health or (enemy[3] == max_health and dis < min_dis):
                max_health = enemy[3]
                min_dis = dis
                min_tag = enemy[0]
        return min_tag

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

    def get_center_position_point_param(self, alliance, units):
        position = (0, 0)
        if alliance == 'Self':
            for unit in units:
                position = tuple(map(lambda x, y: x + y, position, (unit[1], unit[2])))
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
        return None

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
                return action_list
            return None

    def action_ATK_nearest_weakest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        mp = self.get_center_position('Self')
        if len(units) > 0 and len(enemies) > 0:
            for unit in units:
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["attack"],
                    target_unit_tag=self.get_nearest_weakest_enemy(mp, enemies),
                    unit_tags=[unit[0]],
                    queue_command=False,
                ))))
            return action_list
        return None

    def action_ATK_clu_nearest_weakest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        if len(units) > 0 and len(enemies) > 0:
            for clu in cluster_result[2]:
                local_enemy_list = self.get_local_weak_enemy(clu[4], enemies)
                for unit in clu[4]:
                    action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                        ability_id=actions["attack"],
                        target_unit_tag=local_enemy_list[0][0][0],
                        unit_tags=[unit[0]],
                        queue_command=False,
                    ))))
            return action_list
        return None

    def action_ATK_threatening(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        mp = self.get_center_position('Self')
        if len(units) > 0 and len(enemies) > 0:
            for unit in units:
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["attack"],
                    target_unit_tag=self.get_threatening_enemy(mp, enemies),
                    unit_tags=[unit[0]],
                    queue_command=False,
                ))))
            return action_list
        return None

    def action_DEF_nearest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        if len(units) > 0 and len(enemies) > 0:
            for unit in units:
                enemy = self.get_nearest_enemy_pos((unit[1], unit[2]), enemies)
                target = sc_common.Point2D(
                    x=2 * unit[1] - enemy[0], y=2 * unit[2] - enemy[1]
                )
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["move"],
                    target_world_space_pos=target,
                    unit_tags=[unit[0]],
                    queue_command=False,
                ))))
            return action_list
        return None

    def action_DEF_clu_nearest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        if len(units) > 0 and len(enemies) > 0:
            for clu in cluster_result[2]:
                clu_mp = clu[1]
                clu_enemy_tag_list = [self.get_nearest_enemy((item[1], item[2]), enemies) for item in clu[4]]
                clu_ep_lst = [(item[1], item[2]) for item in enemies if item[0] in clu_enemy_tag_list]
                clu_ep = tuple(sum(x) / len(clu_ep_lst) for x in zip(*clu_ep_lst))
                vec = tuple(3 * x - 3 * y for x, y in zip(clu_mp, clu_ep))
                clu_tp = tuple(map(lambda x, y: min(max((x + y), 0), 128), clu_mp, vec))
                if len(clu) > 0:
                    for unit in clu[4]:
                        action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                            ability_id=actions["move"],
                            target_world_space_pos=sc_common.Point2D(x=clu_tp[0], y=clu_tp[1]),
                            unit_tags=[unit[0]],
                            queue_command=False,
                        ))))
            return action_list
        return None

    # def action_MIX_gather(self, cluster_result):
    #     self.update(self.agents, self.enemies)
    #     units = self.featured_agents
    #     enemies = self.featured_enemies
    #     action_list = []
    #     mc = math_utils.find_min_circle([(unit[1], unit[2]) for unit in units])
    #     ec = math_utils.find_min_circle([(enemy[1], enemy[2]) for enemy in enemies])
    #     target = sc_common.Point2D(
    #         x=mc.x, y=mc.y
    #         # x=3*mc.x - 2*ec.x, y=3*mc.x- 2*ec.x
    #     )
    #     if len(units) > 0 and len(enemies) > 0:
    #         for unit in units:
    #             action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
    #                 ability_id=actions["move"],
    #                 target_world_space_pos=target,
    #                 unit_tags=[unit[0]],
    #                 queue_command=False,
    #             ))))
    #         return action_list
    #     return None

    def action_MIX_gather(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        mc = math_utils.find_min_circle([(unit[1], unit[2]) for unit in units])
        if len(units) > 0 and len(enemies) > 0:
            for clu in cluster_result[2]:
                if clu[3] < 0.6:
                    action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                        ability_id=actions["move"],
                        target_world_space_pos=sc_common.Point2D(x=clu[1][0], y=clu[1][1]),
                        unit_tags=[item[0] for item in clu[4]],
                        queue_command=False,
                    ))))
                else:
                    for unit in clu[4]:
                        enemy = self.get_nearest_enemy_pos((unit[1], unit[2]), enemies)
                        target = sc_common.Point2D(
                            x=enemy[0], y=enemy[1]
                        )
                        action_list.append(
                            sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                                ability_id=actions["move"],
                                target_world_space_pos=target,
                                unit_tags=[unit[0]],
                                queue_command=False,
                            ))))
            return action_list
        return None

    def action_MIX_lure_remotest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        mp = self.get_center_position('Self')
        ep = self.get_center_position('Enemy')
        if len(units) > 0 and len(enemies) > 0:
            separation_unit_list = []
            for clu in cluster_result[2]:
                if clu[2] < 0.5:
                    for unit in clu[4]:
                        separation_unit_list.append((unit, distance((unit[1], unit[2]), ep)))
            if len(separation_unit_list) > 1:
                sorted_list = sorted(separation_unit_list, key=lambda x: x[1])
                except_unit_tag = sorted_list[0][0][0]
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["move"],
                    target_world_space_pos=sc_common.Point2D(x=ep[0], y=ep[1]),
                    unit_tags=[except_unit_tag],
                    queue_command=False,
                ))))
            else:
                except_unit_tag = self.get_nearest_enemy(ep, units)
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["move"],
                    target_world_space_pos=sc_common.Point2D(x=ep[0], y=ep[1]),
                    unit_tags=[except_unit_tag],
                    queue_command=False,
                ))))
            units.remove([item for item in units if item[0] == except_unit_tag][0])
            mp_new = self.get_center_position_point_param('Self', units)
            for unit in units:
                if unit[0] != except_unit_tag:
                    action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                        ability_id=actions["move"],
                        target_world_space_pos=sc_common.Point2D(x=mp_new[0], y=mp_new[1]),
                        unit_tags=[unit[0]],
                        queue_command=False,
                    ))))
            return action_list
        return None

    def action_MIX_lure_weakest(self, cluster_result):
        self.update(self.agents, self.enemies)
        units = self.featured_agents
        enemies = self.featured_enemies
        action_list = []
        mp = self.get_center_position('Self')
        ep = self.get_center_position('Enemy')
        if len(units) > 0 and len(enemies) > 0:
            # swap param_1 and param_2 to get the nearest weakest unit to the enemy
            lure_uid = self.get_nearest_weakest_enemy(ep, units)
            back_pt = ((6/5*mp[0]-1/5*ep[0]), (6/5*mp[1]-1/5*ep[1]))
            if lure_uid > 0:
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["move"],
                    target_unit_tag=self.get_nearest_weakest_enemy(mp, enemies),
                    unit_tags=[lure_uid],
                    queue_command=False,
                ))))
            if [item[0] for item in units if item[0] != lure_uid]:
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["move"],
                    target_unit_tag=self.get_nearest_weakest_enemy(mp, enemies),
                    unit_tags=[item[0] for item in units if item[0] != lure_uid],
                    queue_command=True,
                ))))
                action_list.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=r_pb.ActionRawUnitCommand(
                    ability_id=actions["move"],
                    target_world_space_pos=sc_common.Point2D(x=back_pt[0], y=back_pt[1]),
                    unit_tags=[item[0] for item in units if item[0] != lure_uid],
                    queue_command=False,
                ))))
            return action_list
        return None


    # TODO 重写reward_battle()，使其支持多目标
    def reward_battle(self, args=None):
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if args.short_reward:
            short_reward = reward_utils.ShortTermReward()
            reward = short_reward.short_reward(self.agents,
                                               self.enemies,
                                               self.previous_ally_units,
                                               self.previous_enemy_units,
                                               self.cluster.get_shoot_range())
        else:
            reward = delta_ally - delta_enemy - delta_deaths
        return reward







    # TODO 重写step()
    def step(self, action_list, args=None):

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
        reward = self.reward_battle(args)
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        # binich - sum of health of all agents and enemies
        sum_health_agents, sum_health_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            sum_health_agents += al_unit.health
        for _e_id, e_unit in self.enemies.items():
            sum_health_enemies += e_unit.health

        info["sum_health_agents"] = sum_health_agents
        info["sum_health_enemies"] = sum_health_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        return reward, terminated, info










