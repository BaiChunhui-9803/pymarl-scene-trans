from smac.env import StarCraft2Env

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

from src.utils.binich.influence_map import InfluenceMap

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

attributes = [
    "alliance",
    "tag",
    "unit_type",
    "health",
    "health_max",
    "pos"
]




class CustomStarCraft2Env(StarCraft2Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.window_size = (860, 600)
        self.window_size = (1280, 960)

        self.im = InfluenceMap(self.n_agents)
        pass

    def get_agents_dict(self):
        agents = {}
        # print(self.agents)[]
        for agent_id in range(self.n_agents):
            agents[agent_id] = {}
            print(self.agents[agent_id])
            for attr in attributes:
                agents[agent_id][attr] = self.agents[agent_id][attr]
        return agents

    def get_enemies_dict(self):
        enemies = {}
        for enemy_id in range(self.n_enemies):
            enemies[enemy_id] = {}
            for attr in attributes:
                enemies[enemy_id][attr] = self.enemies(enemy_id)[attr]
        return enemies

    def get_im_state(self):
        agents = self.get_agents_dict()
        enemies = self.get_enemies_dict()
        self.im.update(agents, enemies)
        im_state = self.im.get_im_hash()
        return im_state

    def get_avail_actions(self):
        avail_actions = []
        self.get_im_state()

        # for agent_id in range(self.n_agents):
        #     avail_actions.append(self.get_avail_agent_actions(agent_id))

    def script_1(self):
        action_id = actions["attack"]
        target_tag = self.get_enemy_num_attributes()
        cmd = r_pb.ActionRawUnitCommand(
            ability_id=action_id,
            target_unit_tag=target_tag,
            unit_tags=[1],
            queue_command=False,
        )
        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action







