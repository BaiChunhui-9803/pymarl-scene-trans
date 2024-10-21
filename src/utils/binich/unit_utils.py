

class UnitUtils:
    def __init__(self):
        self.player1_unit_type = 48
        self.player2_unit_type = 48

    def get_units(self, unit_type, alliance):
        if alliance == 'Self':
            return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
                    and unit.alliance == features.PlayerRelative.SELF]
        elif alliance == 'Enemy':
            return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
                    and unit.alliance == features.PlayerRelative.ENEMY]
