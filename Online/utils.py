import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
import math

import dmc2gym


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

from npf_utils import EnsembleLinearLayer
def ensemble_mlp(num_members, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [EnsembleLinearLayer(num_members, input_dim, output_dim)]
    else:
        mods = [EnsembleLinearLayer(num_members, input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [EnsembleLinearLayer(num_members, hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(EnsembleLinearLayer(num_members, hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()



metaworld_150 = ['faucet-close-v2',
 'basketball-v2',
 'handle-pull-v2',
 'door-unlock-v2',
 'button-press-wall-v2',
 'coffee-push-v2',
 'handle-press-side-v2',
 'drawer-close-v2',
 'plate-slide-v2',
 'stick-push-v2',
 'door-close-v2',
 'plate-slide-side-v2',
 'push-back-v2',
 'faucet-open-v2',
 'dial-turn-v2',
 'push-wall-v2',
 'button-press-topdown-wall-v2',
 'plate-slide-back-side-v2',
 'reach-wall-v2',
 'door-lock-v2',
 'soccer-v2',
 'drawer-open-v2',
 'window-open-v2',
 'handle-pull-side-v2',
 'sweep-into-v2',
 'sweep-v2',
 'handle-press-v2',
 'reach-v2',
 'window-close-v2',
 'button-press-topdown-v2',
 'pick-place-v2',
 'hand-insert-v2',
 'button-press-v2',
 'plate-slide-back-v2',
 'peg-insert-side-v2',
 'peg-unplug-side-v2',
 'push-v2',
 'coffee-button-v2',
 'door-open-v2']

metaworld_500 = ['coffee-pull-v2', 'hammer-v2', 'pick-place-wall-v2', 'stick-pull-v2', 'shelf-place-v2','assembly-v2']

from mtenv.envs.metaworld.env import get_list_of_func_to_make_envs as get_list_of_func_to_make_metaworld_envs
from mtrl.env.vec_env import MetaWorldVecEnv, VecEnv
def get_metawold_venv(benchmark, benchmark_name, env_id_to_task_map, mode):
    num_tasks = int(benchmark_name.replace("MT", ""))
    make_kwargs = {
        "benchmark": benchmark,
        "benchmark_name": benchmark_name,
        "env_id_to_task_map": env_id_to_task_map,
        "num_copies_per_env": 1,
        "should_perform_reward_normalization": True,
    }

    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
        **make_kwargs
    )
    env_metadata = {
        "ids": list(range(num_tasks)),
        "mode": [mode for _ in range(num_tasks)],
    }
    env = MetaWorldVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
    )

    return env, env_id_to_task_map