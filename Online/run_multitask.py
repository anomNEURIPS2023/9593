
# import torch.multiprocessing as mp
# try:
#     mp.set_start_method('forkserver', force=True)
#     print("forkserver init")
# except RuntimeError:
#     pass
import copy
import sys

from mtrl.env import builder as env_builder
import metaworld
from typing import Any, List, Optional, Tuple
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType
from mtrl.env.types import EnvType
import numpy as np
import os


class metaworld_env(object):
    def __init__(self):
        self.envs = None

    def get_env_metadata(self,
            env: EnvType,
            max_episode_steps: Optional[int] = None,
            ordered_task_list: Optional[List[str]] = None,
    ) -> EnvMetaDataType:
        """Method to get the metadata from an environment"""
        dummy_env = env.env_fns[0]().env
        metadata: EnvMetaDataType = {
            "observation_space": dummy_env.observation_space,
            "action_space": dummy_env.action_space,
            "ordered_task_list": ordered_task_list,
        }
        if max_episode_steps is None:
            metadata["max_episode_steps"] = dummy_env._max_episode_steps
        else:
            metadata["max_episode_steps"] = max_episode_steps
        return metadata

    def make(self):
        self.envs = {}
        mode = "train"
        benchmark_name = 'MT10'
        benchmark = metaworld.MT10()
        self.envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
            benchmark_name=benchmark_name, benchmark=benchmark, mode=mode, env_id_to_task_map=None
        )


        mode = "eval"
        self.envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
            benchmark_name=benchmark_name,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=env_id_to_task_map,
        )
        # In MT10 and MT50, the tasks are always sampled in the train mode.
        # For more details, refer https://github.com/rlworkgroup/metaworld

        self.max_episode_steps = 150
        # hardcoding the steps as different environments return different
        # values for max_path_length. MetaWorld uses 150 as the max length.
        self.metadata = self.get_env_metadata(
            env=self.envs["train"],
            max_episode_steps=self.max_episode_steps,
            ordered_task_list=list(env_id_to_task_map.keys()),
        )



def get_env_metadata(
        env: EnvType,
        max_episode_steps: Optional[int] = None,
        ordered_task_list: Optional[List[str]] = None,
) -> EnvMetaDataType:
    """Method to get the metadata from an environment"""
    try:
        dummy_env = env.env_fns[0]().env
    except:
        dummy_env = env.env_fns[0]()
    metadata: EnvMetaDataType = {
        "observation_space": dummy_env.observation_space,
        "action_space": dummy_env.action_space,
        "ordered_task_list": ordered_task_list,
    }
    if max_episode_steps is None:
        metadata["max_episode_steps"] = dummy_env._max_episode_steps
    else:
        metadata["max_episode_steps"] = max_episode_steps
    return metadata

def main():
    import argparse
    from hydra import compose, initialize
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', default=400, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_depth', default=3, type=int)
    parser.add_argument('--keep_ratio', default=0.05, type=float)
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    parser.add_argument('--continual_pruning', action='store_true', default=False)
    parser.add_argument('--grad_update_rule', default='pmean', type=str)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--num_tasks', default=10, type=int)
    parser.add_argument('--env', default='MT10', type=str)
    # important hypers
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--lr", default=0.001, type=float)              # OpenAI gym environment name
    parser.add_argument("--ips_threshold", default=800, type=float)     # # for mujoco it's episodic reward, for metaworld it's success rate
    parser.add_argument('--mask_init_method', default='random', type=str)
    parser.add_argument('--mask_update_mavg', default=1, type=int)
    parser.add_argument('--env_type', default='metaworld', type=str) # for halfcheetah --> multitask, for metaworld ---> metaworld
    parser.add_argument('--snip_itr', default=1, type=int)
    parser.add_argument('--experiment', default='test', type=str)
    parser.add_argument('--optimization_type', default='adamW', type=str)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--clip_grad', default=0, type=float)
    # for pretrained masks
    parser.add_argument('--pretrain_masks', action='store_true', default=False)
    parser.add_argument('--num_pretrain_steps', default=2000000, type=int)
    parser.add_argument('--success_eval', default='v2', type=str, help='v1 sets easy success')
    args = parser.parse_args()
    mode = "train"
    if args.env == 'MT10':
        benchmark_name = 'MT10'
        benchmark = metaworld.MT10(seed=args.seed)
        args.num_tasks = 10
    envs, env_id_to_task_map = env_builder.build_metaworld_vec_env(
        benchmark_name=benchmark_name, benchmark=benchmark, mode=mode, env_id_to_task_map=None
    )
    max_episode_steps = 150
    # hardcoding the steps as different environments return different
    # values for max_path_length. MetaWorld uses 150 as the max length.
    metadata = get_env_metadata(
        env=envs,
        max_episode_steps=max_episode_steps,
        ordered_task_list=list(env_id_to_task_map.keys()),
    )
    initialize(config_path="./config", job_name="train_pruned_venv")
    cfg = compose(config_name="train_pruned_venv")

    # update config
    args = vars(args)
    for key in args.keys():
        cfg[key] = args[key]
    if cfg.env_type == 'metaworld':
        pass
    else:
        sys.exit('select accurate `env_type` ')

    from train_multitask import execute_process
    execute_process(cfg, envs, metadata)

if __name__ == "__main__":
    main()


