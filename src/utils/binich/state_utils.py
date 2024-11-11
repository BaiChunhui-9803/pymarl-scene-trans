import influence_map as im
import math_utils

class StateUtil:
    def __init__(self, unit_scale: int):
        self.unit_scale = unit_scale
        self.im = im.InfluenceMap(self.unit_scale)

    def get_im_state(self):
        im_state = self.im.get_im_hash()
        return im_state

    def get_clu_state(self):
        return 0
