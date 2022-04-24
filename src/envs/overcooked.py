import sys
import os
sys.path.append("..")
import numpy as np
import gym
from overcooked_ai_py.mdp.actions import Action
from human_aware_rl.rllib.utils import get_base_ae, get_required_arguments
import torch as th
from .multiagentenv import MultiAgentEnv


class OvercookedPyMarl(MultiAgentEnv):
    supported_agents = ['ppo', 'bc']
    bc_schedule = self_play_bc_schedule = [(0, 0), (float('inf'), 0)]
    DEFAULT_CONFIG = {
        "mdp_params" : {
            "layout_name" : "coordination_ring",
            "rew_shaping_params" : {}
        },
        "env_params" : {
            "horizon" : 400
        },
        "multi_agent_params" : {
            "reward_shaping_factor" : 0.0,
            "reward_shaping_horizon" : 0,
            "bc_schedule" : self_play_bc_schedule,
            "use_phi" : True
        }
    }
    
    def __init__(self, base_env, use_phi=True, reward_shaping_factor=0.0, reward_shaping_horizon=0, bc_schedule=None, obs_pattern="compat", state_pattern="compat", seed=-1):
        self.n_agents = 2
        self.obs_pattern = obs_pattern
        self.state_pattern = state_pattern
        if bc_schedule:
            self.bc_schedule = bc_schedule
        self._validate_schedule(self.bc_schedule)
        
        self.base_env = base_env
        self.featurize_fn_map = {
            "raw": lambda state: self.base_env.lossless_state_encoding_mdp(state),
            "compat": lambda state: self.base_env.featurize_state_mdp(state)
        }
        # self._validate_featurize_fns(self.featurize_fn_map)
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.use_phi = use_phi
        self._seed = seed
        self.episode_limit = 400
        # self.anneal_bc_factor(0)
        self._configure_featurize_fn()
        self.reset()
        
    def _validate_schedule(self, schedule):
        timesteps = [p[0] for p in schedule]
        values = [p[1] for p in schedule]
        
        assert len(schedule) >= 2, "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0][0] == 0, "Schedule must start at timestep 0"
        assert all([t >=0 for t in timesteps]), "All timesteps in schedule must be non-negative"
        assert all([v >=0 and v <= 1 for v in values]), "All values in schedule must be between 0 and 1"
        assert sorted(timesteps) == timesteps, "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if (schedule[-1][0] < float('inf')):
            schedule.append((float('inf'), schedule[-1][1]))
        
    def _validate_featurize_fns(self, mapping):
        assert 'ppo' in mapping, "At least one ppo agent must be specified"
        for k, v in mapping.items():
            assert k in self.supported_agents, "Unsuported agent type in featurize mapping {0}".format(k)
            assert callable(v), "Featurize_fn values must be functions"
            assert len(get_required_arguments(v)) == 1, "Featurize_fn value must accept exactly one argument"
        
    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        
        featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(state)
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)
        
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)
        
    def _configure_featurize_fn(self):
        self.obs_fn = self.featurize_fn_map[self.obs_pattern]
        self.state_fn = self.featurize_fn_map[self.state_pattern]
        
    def _get_obs(self, state):
        return [self.obs_fn(state)[0], self.obs_fn(state)[1]]
    
    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]
    
    def get_obs_size(self):
        if self.obs_pattern == "raw":
            #TODO add support for cnn agent
            return 96
        elif self.obs_pattern == "compat":
            return 96
        else:
            raise NotImplementedError
    
    def get_state_size(self):
        if self.state_pattern == "raw":
            #TODO add support for cnn agent
            return 96
        elif self.state_pattern == "compat":
            return 96 * 2
        else:
            raise NotImplementedError
        
    def get_obs(self):
        return self._get_obs(self.base_env.state)
    
    def get_state(self):
        raw = self.state_fn(self.base_env.state)
        return np.concatenate([raw[0], raw[1]], axis=0)
    
    def get_avail_agent_actions(self, agent_id):
        return [1] * 6
    
    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
    
    def get_total_actions(self):
        return 6
    
    def reset(self, regen_mdp=True):
        self.base_env.reset(regen_mdp)
        return self.get_obs, self.get_state()
    
    def step(self, action_pair):
        # NOTE action_pair = (main_agent_aciton, other_agent_action)
        # action = [action_dict[self.curr_agents[0]], action_dict[self.curr_agents[1]]]
        action = [action_pair[0].item(), action_pair[1].item()]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        
        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=True)
            potential = info['phi_s_prime'] - info['phi_s']
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)
            dense_reward = info["shaped_r_by_agent"]
            

        shaped_reward_main = sparse_reward + self.reward_shaping_factor * dense_reward[0]
        shaped_reward_other = sparse_reward + self.reward_shaping_factor * dense_reward[1]
        
        rewards = [shaped_reward_main, shaped_reward_other]
        reward = sum(rewards) / len(rewards)
        dones = done
        infos = info
        del infos["phi_s"]
        del infos["phi_s_prime"]
        return reward, dones, infos
    
    def seed(self):
        #TODO add support for seed
        return self._seed
    
    def render(self):
        pass
    
    def close(self):
        pass
    
    def save_replay(self):
        #TODO add support on saving trajectories
        pass
    
    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
    
    @classmethod
    def from_config(cls, env_config):
        assert env_config and "env_params" in env_config and "multi_agent_params" in env_config
        assert "mdp_params" in env_config or "mdp_params_schedule_fn" in env_config, \
            "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "rew_shaping_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
        base_env = base_ae.env

        return cls(base_env=base_env, **multi_agent_params)