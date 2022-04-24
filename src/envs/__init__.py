from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .overcooked import OvercookedPyMarl
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def overcooked_env_fn(env_config, seed):
    return OvercookedPyMarl.from_config(env_config)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["overcooked"] = overcooked_env_fn

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
