from smac.env import StarCraft2Env

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

from src.utils.binich.influence_map import InfluenceMap
from src.utils.binich.cluster import Cluster

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
    "script_1": 1,
}


class CustomStarCraft2Env(StarCraft2Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.window_size = (860, 600)
        self.window_size = (1280, 960)

        self.im = InfluenceMap(self.n_agents)
        self.cluster = Cluster(self.n_agents)
        pass


    # binich - custom method for getting state using influence map hashing
    def get_im_state(self):
        self.im.update(self.agents, self.enemies)
        im_state = self.im.get_im_hash()
        return im_state

    def get_original_state(self):
        self.im.update(self.agents, self.enemies)
        original_state = {'featured_agents': self.im.featured_agents, 'featured_enemies': self.im.featured_enemies}
        return original_state






    def get_avail_actions(self):
        avail_actions = {
            "avail_cluster_strengths": self.cluster.cluster_strengths,
            "avail_scripts": scripts
        }
        # self.cluster.update(self.agents, self.enemies)
        # cluster = self.cluster.kmeans(3)
        return avail_actions


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







