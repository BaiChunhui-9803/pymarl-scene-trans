from smac.env import StarCraft2Env

# from src.utils.binich import script_utils


class CustomStarCraft2Env(StarCraft2Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.window_size = (860, 600)
        self.window_size = (1280, 960)
        pass

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))


class Script:



    def __init__(self):
        pass



