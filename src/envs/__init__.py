from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

from .custom_starcraft2 import CustomStarCraft2Env

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["custom_sc2"] = partial(env_fn, env=CustomStarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
