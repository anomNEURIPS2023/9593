#!/usr/bin/env python3
# TODO:  neural pathway for multiple tasks
#  uses multiple parallel metaworld envs
import npf_utils
import replay_buffer
from mtrl.env import builder as env_builder
import metaworld
from typing import Any, List, Optional, Tuple
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType
from mtrl.env.types import EnvType
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

#import dmc2gym
import hydra
#import gym
import warnings
import gzip
import tensorflow as tf
import glob
warnings.filterwarnings('ignore', category=DeprecationWarning)


def save_keep_masks(path, keep_masks, task):
    if not os.path.exists(path):
        os.makedirs(path)
    M = []
    for layer_mask in keep_masks:
        M.append(layer_mask.cpu().numpy())
    np.save('{}/keep_masks_task{}'.format(path, task), np.array(M))

def make_agent(obs_dim, action_dim, action_range, cfg):
    cfg.obs_dim = obs_dim
    cfg.action_dim = action_dim
    cfg.action_range = action_range
    return hydra.utils.instantiate(cfg)

class Workspace(object):
    def __init__(self, cfg, env, metadata):

        self.cfg = cfg
        # set workdir
        self.set_work_dir()
        self.set_logger()
        # set seed
        self.set_seed()
        self.device = torch.device(cfg.device)
        self.env = env
        self.env_metadata = metadata
        self.agent = make_agent(self.env_metadata['observation_space'].shape[0],
                                self.env_metadata['action_space'].shape[0],
                                [float(self.env_metadata['action_space'].low.min()), float(self.env_metadata['action_space'].high.max())],
                                self.cfg.agent)
        self.agent.num_tasks = self.env.num_envs #len(self.env.ids)

        self.max_episode_steps = self.env_metadata[
            "max_episode_steps"
        ]  # maximum steps that the agent can take in one environment.

        self.action_space = self.env_metadata['action_space']
        self.action_scale_high = self.env_metadata['action_space'].high.max()
        self.action_scale_low = self.env_metadata['action_space'].high.min()

        self.replay_buffer = collections.defaultdict(dict)
        self.tmp_buffer = collections.defaultdict(dict)
        for task in range(self.agent.num_tasks):
            self.replay_buffer[f'task{task}'] = ReplayBuffer(self.env_metadata['observation_space'].shape,
                                              self.env_metadata['action_space'].shape,
                                              int(cfg.replay_buffer_capacity),
                                              self.device)
            self.tmp_buffer[f'task{task}'] = ReplayBuffer(self.env_metadata['observation_space'].shape,
                                              self.env_metadata['action_space'].shape,
                                              int(max(self.env_metadata['max_episode_steps']*self.cfg.snip_itr, self.cfg.batch_size)),
                                              self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        print(count_parameters(self.agent.actor))

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

    def make_env(self):
        from mtrl.env import builder as env_builder

        from typing import Any, List, Optional, Tuple
        from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType
        from mtrl.env.types import EnvType

        envs = {}
        mode = "train"
        benchmark_name = 'MT10'
        benchmark = metaworld.MT10()
        envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
            benchmark_name=benchmark_name, benchmark=benchmark, mode=mode, env_id_to_task_map=None
        )

        mode = "eval"
        envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
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
            env=envs["train"],
            max_episode_steps=self.max_episode_steps,
            ordered_task_list=list(env_id_to_task_map.keys()),
        )
        return envs

    def set_dummy_agent(self):
        from npf_utils import sync_weights
        self.dummy_agent = make_agent(self.env_metadata['observation_space'].shape[0],
                                self.env_metadata['action_space'].shape[0],
                                [float(self.env_metadata['action_space'].low.min()), float(self.env_metadata['action_space'].high.max())],
                                self.cfg.agent)

        print('NOTE: dummy agent is init as updated weight')
        self.dummy_agent.actor.load_state_dict(self.agent.actor.state_dict())
        self.dummy_agent.critic.load_state_dict(self.agent.critic.state_dict())
        self.dummy_agent.critic_target.load_state_dict(self.agent.critic_target.state_dict())

    def set_logger(self):
        self.logger = Logger(self.work_dir,
                             save_tb=self.cfg.log_save_tb,
                             log_frequency=self.cfg.log_frequency,
                             agent=self.cfg.agent_name)

    def set_work_dir(self):
        self.work_dir = os.getcwd()
        if self.cfg['pretrain_masks']:
            self.work_dir = self.work_dir + f'/run_parallel_{self.cfg["experiment"]}_pretrain_mask_{self.cfg["pretrain_masks"]}_{self.cfg["num_pretrain_steps"]}/algo={self.cfg["agent_name"]},env={self.cfg["env"]},' \
                                        f'env_type={self.cfg["env_type"]},seed={self.cfg["seed"]},' \
                                        f'batch_size={self.cfg.agent.batch_size},actor_hd={self.cfg.diag_gaussian_actor.hidden_dim},' \
                                        f'critic_hd={self.cfg.double_q_critic.hidden_dim},keep_ratio={self.cfg.keep_ratio},lr={self.cfg.lr},' \
                                        f'continual_pruning={self.cfg.continual_pruning},mavg={self.cfg.mask_update_mavg},' \
                                        f'mask_init_m={self.cfg.mask_init_method},ips_thrsh={self.cfg.ips_threshold},' \
                                        f'grad_update_rule={self.cfg.grad_update_rule},optim_type={self.cfg.optimization_type},wd={self.cfg.weight_decay},clip={self.cfg.clip_grad}'
        else:
            self.work_dir = self.work_dir + f'/run_parallel_{self.cfg["experiment"]}/algo={self.cfg["agent_name"]},env={self.cfg["env"]},' \
                                        f'env_type={self.cfg["env_type"]},seed={self.cfg["seed"]},' \
                                        f'batch_size={self.cfg.agent.batch_size},actor_hd={self.cfg.diag_gaussian_actor.hidden_dim},' \
                                        f'critic_hd={self.cfg.double_q_critic.hidden_dim},keep_ratio={self.cfg.keep_ratio},lr={self.cfg.lr},' \
                                        f'continual_pruning={self.cfg.continual_pruning},mavg={self.cfg.mask_update_mavg},' \
                                        f'mask_init_m={self.cfg.mask_init_method},ips_thrsh={self.cfg.ips_threshold},' \
                                        f'grad_update_rule={self.cfg.grad_update_rule},optim_type={self.cfg.optimization_type},wd={self.cfg.weight_decay},clip={self.cfg.clip_grad}'
        print(f'workspace: {self.work_dir}')

    def set_seed(self):
        utils.set_seed_everywhere(self.cfg.seed)


    def reset_episodic_storage(self):
        self.storage = {'observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'next_observations': [], 'episodic_returns': [], 'success': []}


    def evaluate(self):
        self.reset_episodic_storage()
        average_episode_reward = np.full(shape=self.agent.num_tasks, fill_value=0.0)
        average_success_rate = np.full(shape=self.agent.num_tasks, fill_value=0.0)
        average_episode_len = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_len = 0
            self.agent.reset()
            success = np.full(shape=self.agent.num_tasks, fill_value=0.0)

            done = False
            episode_reward = np.full(shape=self.agent.num_tasks, fill_value=0.0)

            print(f'eval itr {episode}')
            while not (done or episode_len >= self.max_episode_steps):
                with utils.eval_mode(self.agent):
                    action = self.agent.multiobs_act(obs['env_obs'], sample=False)
                    action = action.squeeze(1)
                obs, reward, done, info = self.env.step(action)
                if (episode_len + 1) == self.max_episode_steps:
                    done = True
                    print(f'reset episode {episode}')
                else:
                    done = False
                # update - each step
                episode_reward += reward
                average_episode_len += 1
                episode_len += 1
                if self.cfg.success_eval == 'v1':
                    success += np.asarray([x["success"] for x in info])
                elif self.cfg.success_eval == 'v2':
                    success = np.asarray([x["success"] for x in info])

            # update - each episode
            average_episode_reward += episode_reward
            if self.cfg.env_type == 'metaworld':
                average_success_rate += (success > 0).astype("float")
        # update - end of episodes
        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_len /= self.cfg.num_eval_episodes


        # update logger
        if self.cfg.env_type == 'metaworld':
            average_success_rate /= self.cfg.num_eval_episodes
            print(f'Average success rate {average_success_rate}')
        for task in range(self.agent.num_tasks):
            self.logger.log(f'eval/task{task}/success_rate', average_success_rate[task], self.step)
            self.logger.log(f'eval/task{task}/episode_reward', average_episode_reward[task], self.step)
        self.logger.log(f'eval/Overall/success_rate', np.sum(average_success_rate), self.step)

        self.logger.log(f'eval/episode_reward', average_episode_reward[0], self.step)
        self.logger.log('eval/episode_len', average_episode_len, self.step)
        self.logger.dump(self.step)

        if self.cfg.env_type == 'metaworld':
            return average_episode_reward, average_success_rate
        else:
            return average_episode_reward

    def quick_collect(self):
        done = False
        obs = self.env.reset()
        self.agent.reset()
        episode_step = 1
        episode_rw = np.full(shape=self.agent.num_tasks, fill_value=0.0)
        success = np.full(shape=self.agent.num_tasks, fill_value=0.0)
        episodes = 10
        for _ in range(episodes):
            while not done:
                with utils.eval_mode(self.agent):
                     action = self.agent.multiobs_act(obs['env_obs'], sample=True)
                     action = action.squeeze(1)

                # take a step
                next_obs, reward, done, info = self.env.step(action) #*self.env.action_scale_high)

                if self.cfg.success_eval == 'v1':
                    success += np.asarray([x["success"] for x in info])
                elif self.cfg.success_eval == 'v2':
                    success = np.asarray([x["success"] for x in info])

                if episode_step + 1 == self.max_episode_steps:
                    done = True
                else:
                    done = False
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done

                # collect samples
                for task in range(self.agent.num_tasks):
                    self.replay_buffer[f'task{task}'].add(np.array(obs['env_obs'][task, :]),
                                                          action[task, :],
                                                          reward[task],
                                                          np.array(next_obs['env_obs'][task, :]),
                                                          done,
                                                          done_no_max)
                obs = next_obs
                episode_step += 1
                episode_rw += reward

            # if done - reset()
            episode_step = 0
            done = False
            obs = self.env.reset()
            success = np.full(shape=self.agent.num_tasks, fill_value=0.0)

        print(f'expert_reward {episode_rw/episodes}')

    # updating the weight mask on ensemble model:
    def update_ensemble_mask(self, task=None):
        """
        if task is `None`, update the mask for all task, otherwise update specific task-mask
        """
        if task is None:
            for task in range(self.agent.num_tasks):
                layer = 0
                for idx, all_layer in enumerate(self.agent.ensemble_actor.modules()):
                    if isinstance(all_layer, npf_utils.EnsembleLinearLayer):
                        # # nn.Linear saves weights as transposed (W.T) https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
                        all_layer.weight_mask[task, ...] = \
                            copy.deepcopy(self.agent.keep_masks["actor"][f'task{task}'][layer].T)
                        layer += 1
                layer = 0
                for idx, all_layer in enumerate(self.agent.ensemble_critic.modules()):
                    if isinstance(all_layer, npf_utils.EnsembleLinearLayer):
                        all_layer.weight_mask[task, ...] = \
                            copy.deepcopy(self.agent.keep_masks["critic"][f'task{task}'][layer].T)
                        layer += 1
                layer = 0
                for idx, all_layer in enumerate(self.agent.ensemble_critic_target.modules()):
                    if isinstance(all_layer, npf_utils.EnsembleLinearLayer):
                        all_layer.weight_mask[task, ...] = \
                            copy.deepcopy(self.agent.keep_masks["critic"][f'task{task}'][layer].T)
                        layer += 1
        else:
            layer = 0
            for idx, all_layer in enumerate(self.agent.ensemble_actor.modules()):
                if isinstance(all_layer, npf_utils.EnsembleLinearLayer):
                    # # nn.Linear saves weights as transposed (W.T) https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
                    all_layer.weight_mask[task, ...] = \
                        copy.deepcopy(self.agent.keep_masks["actor"][f'task{task}'][layer].T)
                    layer += 1
            layer = 0
            for idx, all_layer in enumerate(self.agent.ensemble_critic.modules()):
                if isinstance(all_layer, npf_utils.EnsembleLinearLayer):
                    all_layer.weight_mask[task, ...] = \
                        copy.deepcopy(self.agent.keep_masks["critic"][f'task{task}'][layer].T)
                    layer += 1

            layer = 0
            for idx, all_layer in enumerate(self.agent.ensemble_critic_target.modules()):
                if isinstance(all_layer, npf_utils.EnsembleLinearLayer):
                    all_layer.weight_mask[task, ...] = \
                        copy.deepcopy(self.agent.keep_masks["critic"][f'task{task}'][layer].T)
                    layer += 1

    def init_network(self):
        # uses random expert samples
        self.quick_collect()

        # -----------------------------
        # configure pathway/find masks
        # -----------------------------
        import collections
        self.agent.keep_masks = collections.defaultdict(dict)
        self.agent.ensemble_keep_masks = collections.defaultdict(dict)
        from npf_utils import class_pathways
        self.pathways = class_pathways(self.cfg.keep_ratio, history_len=self.cfg.mask_update_mavg, num_tasks=self.agent.num_tasks)
        for task in range(self.agent.num_tasks):
            self.set_dummy_agent()
            self.agent.keep_masks["actor"][f'task{task}'], \
            self.agent.keep_masks["critic"][f'task{task}'] = self.pathways.get_masks(self.dummy_agent, self.replay_buffer[f'task{task}'], task=task)

        # updating the weight mask on ensemble model
        self.update_ensemble_mask()

    def configure_new_mask(self):
        """
        FIND MASK :
        """
        for task in range(self.agent.num_tasks):

            # update only if requires
            if self.iterative_pruning[task] and self.step > 0:
                self.set_dummy_agent()
                self.agent.keep_masks["actor"][f'task{task}'], \
                self.agent.keep_masks["critic"][f'task{task}'] = self.pathways.get_masks(self.dummy_agent,
                                                                                         self.tmp_buffer[f'task{task}'],
                                                                                         task=task)
                """
                UPDATE THE MASK IN W_E (ensemble model)
                """
                self.update_ensemble_mask(task=task)

            # find the the number of changed masks
            from npf_utils import common_weight
            init_w_counts = common_weight(self.agent.keep_masks['actor']['task0'],
                                          self.agent.keep_masks['actor']['task0'])
            self.logger.log('train/mask count', init_w_counts, self.step)


    def run(self):
        self.replay_buffer = replay_buffer.ReplayBuffer_VEnv(self.agent.num_tasks,
                                                            (self.agent.num_tasks, self.env_metadata['observation_space'].shape[0]),
                                                            (self.agent.num_tasks, self.env_metadata['action_space'].shape[0]),
                                                            int(self.cfg.replay_buffer_capacity),
                                                            self.device)

        best_ep_ret = 0
        episode, episode_reward, done = 0, np.full(shape=self.agent.num_tasks, fill_value=0.0), True
        start_time = time.time()
        activate_eval = True
        success = np.full(shape=self.agent.num_tasks, fill_value=0.0)
        self.iterative_pruning = np.repeat(True, self.agent.num_tasks)

        while self.step < self.cfg.num_train_steps:
            if done:
                success = (success > 0).astype("float")
                print(f'reset at {self.step} finding masks for {self.iterative_pruning}')
                for task in range(self.agent.num_tasks):
                    if (not self.cfg.continual_pruning and success[task] and episode_reward[task] >= self.cfg.ips_threshold):
                        self.iterative_pruning[task] = False
                        save_keep_masks(self.work_dir, self.agent.keep_masks['actor'][f'task{task}'], task)

                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if activate_eval:
                    self.logger.log('eval/episode', episode, self.step)
                    print(f'running evaluation at : {self.step}')
                    avg_eval_ret, avg_success_rate = self.evaluate()
                    self.agent.save(self.work_dir, step='final')  # saves the last evaluation
                    """Stop learning if reached to sucess of all"""
                    if np.mean(avg_success_rate) == 1:
                        self.cfg.num_train_steps = self.step
                    activate_eval = False

                for task in range(self.agent.num_tasks):
                    self.logger.log(f'train/task{task}/episode_reward', episode_reward[task], self.step)
                    self.logger.log(f'train/task{task}/success_rate', success[task], self.step)

                self.logger.log('train/episode_reward', episode_reward[0], self.step)
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = np.full(shape=self.agent.num_tasks, fill_value=0.0)
                success = np.full(shape=self.agent.num_tasks, fill_value=0.0)
                episode_step = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)

                if self.cfg.keep_ratio != 1:
                    """ UPDATE THE MASK """
                    self.configure_new_mask()

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = np.asarray([self.action_space.sample() for _ in range(self.agent.num_tasks)])  # (num_envs, action_dim)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.multiobs_act(obs['env_obs'], sample=True)
                    action = action.squeeze(1)

            """training update"""
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update_venv(self.replay_buffer, self.logger, self.step, clip_grad=self.cfg.clip_grad if self.cfg.clip_grad != 0 else None)
            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            if episode_step + 1 == self.max_episode_steps:
                done = True
            else:
                done = False
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward
            self.replay_buffer.add(np.array(obs['env_obs']), action, reward, np.array(next_obs['env_obs']),
                                   np.repeat(not done, self.agent.num_tasks), np.repeat(not done_no_max, self.agent.num_tasks))
            for task in range(self.agent.num_tasks):
                if self.iterative_pruning[task]:
                    self.tmp_buffer[f'task{task}'].add(obs['env_obs'][task, :],
                                                          action[task, :],
                                                          reward[task],
                                                          next_obs['env_obs'][task, :],
                                                          done,
                                                          done_no_max)


            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.cfg.success_eval == 'v1':
                success += np.asarray([x["success"] for x in info])
            elif self.cfg.success_eval == 'v2':
                success = np.asarray([x["success"] for x in info])

            # important: once done is reached, it checks if 'activate done' is already activated
            # allows to get eval at i.e. 5007 1024 steps
            if self.step % self.cfg.eval_frequency == 0:
                activate_eval = True

        # save the recorder only on the last eval
        self.video_recorder = VideoRecorder(self.work_dir)
        self.agent.save(self.work_dir, step='final')



    def pretrain(self):
        self.replay_buffer = replay_buffer.ReplayBuffer_VEnv(self.agent.num_tasks,
                                                            (self.agent.num_tasks, self.env_metadata['observation_space'].shape[0]),
                                                            (self.agent.num_tasks, self.env_metadata['action_space'].shape[0]),
                                                            int(self.cfg.replay_buffer_capacity),
                                                            self.device)

        best_ep_ret = 0
        episode, episode_reward, done = 0, np.full(shape=self.agent.num_tasks, fill_value=0.0), True
        start_time = time.time()
        activate_eval = True
        success = np.full(shape=self.agent.num_tasks, fill_value=0.0)
        self.iterative_pruning = np.repeat(True, self.agent.num_tasks)

        while self.step < self.cfg.num_pretrain_steps:
            if done:
                success = (success > 0).astype("float")
                print(f'reset at {self.step} finding masks for {self.iterative_pruning}')
                for task in range(self.agent.num_tasks):
                    if (not self.cfg.continual_pruning and success[task] and episode_reward[task] >= self.cfg.ips_threshold):
                        self.iterative_pruning[task] = False
                        save_keep_masks(self.work_dir, self.agent.keep_masks['actor'][f'task{task}'], task)

                # once we reached threshold for all the tasks:
                if sum(self.iterative_pruning==False) == self.agent.num_tasks:
                    self.cfg.num_pretrain_steps = self.step
                # evaluate agent periodically
                if activate_eval:
                     activate_eval = False
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = np.full(shape=self.agent.num_tasks, fill_value=0.0)
                success = np.full(shape=self.agent.num_tasks, fill_value=0.0)
                episode_step = 0
                episode += 1
                """ UPDATE THE MASK """

                def transfer_task_weight(model, ensembled_model, task):
                    '''transfer actor weight to ensemble model at it's initialization

                    '''
                    for layer, en_layer in zip(model.parameters(), ensembled_model.parameters()):
                        if len(layer.data.shape) == 1:  # bias
                            layer.data.copy_(en_layer.data[task, :].flatten())
                        else:  # weight
                            layer.data.copy_(en_layer.data[task,:].transpose(0, 1))


                for task in range(self.agent.num_tasks):
                    # update only if requires
                    if self.iterative_pruning[task] and self.step > 0:
                        self.dummy_agent = make_agent(self.env_metadata['observation_space'].shape[0],
                                                      self.env_metadata['action_space'].shape[0],
                                                      [float(self.env_metadata['action_space'].low.min()),
                                                       float(self.env_metadata['action_space'].high.max())],
                                                      self.cfg.agent)

                        print('NOTE: dummy agent is init as updated weight')
                        transfer_task_weight(self.dummy_agent.actor, self.agent.ensemble_actor, task)
                        transfer_task_weight(self.dummy_agent.critic, self.agent.ensemble_critic, task)
                        transfer_task_weight(self.dummy_agent.critic_target, self.agent.ensemble_critic_target, task)


                        self.agent.keep_masks["actor"][f'task{task}'], \
                        self.agent.keep_masks["critic"][f'task{task}'] = self.pathways.get_masks(self.dummy_agent,
                                                                                                 self.tmp_buffer[
                                                                                                     f'task{task}'],
                                                                                                 task=task)
                        """
                        UPDATE THE MASK IN W_E (ensemble model)
                        """
                        self.update_ensemble_mask(task=task)



            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = np.asarray([self.action_space.sample() for _ in range(self.agent.num_tasks)])  # (num_envs, action_dim)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.multiobs_act(obs['env_obs'], sample=True)
                    action = action.squeeze(1)

            """
            training update:
            """
            if self.step >= self.cfg.num_seed_steps:
                self.agent.pretrain_venv(self.replay_buffer, self.logger, self.step, self.iterative_pruning)
            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            if episode_step + 1 == self.max_episode_steps:
                done = True
            else:
                done = False
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward
            self.replay_buffer.add(np.array(obs['env_obs']),
                                   action,
                                   reward,
                                   np.array(next_obs['env_obs']),
                                   np.repeat(not done, self.agent.num_tasks),
                                   np.repeat(not done_no_max, self.agent.num_tasks))
            for task in range(self.agent.num_tasks):
                if self.iterative_pruning[task]:
                    self.tmp_buffer[f'task{task}'].add(obs['env_obs'][task, :],
                                                          action[task, :],
                                                          reward[task],
                                                          next_obs['env_obs'][task, :],
                                                          done,
                                                          done_no_max)


            obs = next_obs
            episode_step += 1
            self.step += 1
            if self.cfg.success_eval == 'v1':
                success += np.asarray([x["success"] for x in info])
            elif self.cfg.success_eval == 'v2':
                success = np.asarray([x["success"] for x in info])
            if self.step % self.cfg.eval_frequency == 0:
                activate_eval = True


    def run_after_pretrain(self):

        def init_avg_weight(model, ensembled_model):
            '''transfer actor weight to ensemble model at it's initialization

            '''

            for layer, en_layer in zip(model.parameters(), ensembled_model.parameters()):
                num_tasks = en_layer.shape[0]
                if len(layer.data.shape) == 1:  # bias
                    en_layer.data.copy_(en_layer.data.mean(0).repeat(num_tasks, 1, 1))
                    layer.data.copy_(en_layer.data[0, :].flatten())
                else:  # weight
                    en_layer.data.copy_(en_layer.data.mean(0).repeat(num_tasks, 1, 1))
                    layer.data.copy_(en_layer.data[0, :].transpose(0, 1))

        init_avg_weight(self.agent.actor, self.agent.ensemble_actor)
        init_avg_weight(self.agent.critic, self.agent.ensemble_critic)
        self.agent.critic_target.load_state_dict(self.agent.critic.state_dict())
        self.agent.ensemble_critic_target.load_state_dict(self.agent.ensemble_critic.state_dict())
        self.agent.init_optim()
        self.replay_buffer = replay_buffer.ReplayBuffer_VEnv(self.agent.num_tasks,
                                                            (self.agent.num_tasks, self.env_metadata['observation_space'].shape[0]),
                                                            (self.agent.num_tasks, self.env_metadata['action_space'].shape[0]),
                                                            int(self.cfg.replay_buffer_capacity),
                                                            self.device)
        self.step = 0
        episode, episode_reward, done = 0, np.full(shape=self.agent.num_tasks, fill_value=0.0), True
        start_time = time.time()
        activate_eval = True
        success = np.full(shape=self.agent.num_tasks, fill_value=0.0)

        while self.step < self.cfg.num_train_steps:
            if done:
                success = (success > 0).astype("float")
                print(f'reset at {self.step}')

                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if activate_eval:
                    self.logger.log('eval/episode', episode, self.step)
                    print(f'running evaluation at : {self.step}')
                    avg_eval_ret, avg_success_rate = self.evaluate()
                    self.agent.save(self.work_dir, step='final')  # saves the last evaluation
                    """Stop learning if reached to sucess of all"""
                    if np.mean(avg_success_rate) == 1:
                        self.cfg.num_train_steps = self.step
                    activate_eval = False

                for task in range(self.agent.num_tasks):
                    self.logger.log(f'train/task{task}/episode_reward', episode_reward[task], self.step)
                    self.logger.log(f'train/task{task}/success_rate', success[task], self.step)

                self.logger.log('train/episode_reward', episode_reward[0], self.step)
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = np.full(shape=self.agent.num_tasks, fill_value=0.0)
                success = np.full(shape=self.agent.num_tasks, fill_value=0.0)
                episode_step = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)


            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = np.asarray([self.action_space.sample() for _ in range(self.agent.num_tasks)])  # (num_envs, action_dim)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.multiobs_act(obs['env_obs'], sample=True)
                    action = action.squeeze(1)

            """training update"""
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update_venv(self.replay_buffer, self.logger, self.step, clip_grad=self.cfg.clip_grad if self.cfg.clip_grad != 0 else None)

            next_obs, reward, done, info = self.env.step(action)
            # allow infinite bootstrap
            if episode_step + 1 == self.max_episode_steps:
                done = True
            else:
                done = False
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward
            self.replay_buffer.add(np.array(obs['env_obs']), action, reward, np.array(next_obs['env_obs']),
                                   np.repeat(not done, self.agent.num_tasks), np.repeat(not done_no_max, self.agent.num_tasks))


            obs = next_obs
            episode_step += 1
            self.step += 1
            if self.cfg.success_eval == 'v1':
                success += np.asarray([x["success"] for x in info])
            elif self.cfg.success_eval == 'v2':
                success = np.asarray([x["success"] for x in info])
            if self.step % self.cfg.eval_frequency == 0:
                activate_eval = True

        # save the recorder only on the last eval
        self.video_recorder = VideoRecorder(self.work_dir)
        self.agent.save(self.work_dir, step='final')






def execute_process(cfg, env, metadata):
    workspace_exec = Workspace(cfg, env, metadata)
    if cfg.keep_ratio != 1:
        workspace_exec.init_network()
    # pretrain the masks and fine-tune the networks for multitasks
    if cfg.pretrain_masks:
        workspace_exec.pretrain()
        workspace_exec.run_after_pretrain()
    # regular training script
    else:
        workspace_exec.run()


