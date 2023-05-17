#!/usr/bin/env python3

# neural pathways for single task

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

import dmc2gym
import hydra
import gym
import warnings
import gzip
import tensorflow as tf
import glob
from npf_utils_single_task import class_pathways, common_weight, get_abs_sps
import collections


warnings.filterwarnings('ignore', category=DeprecationWarning)
try:
    from multitask_envs.walker2d import Walker2dEnv
except:
    print('failed to import walker2d')
try:
    from motion_imitation.envs import env_builder as env_builder
except:
    print('failed to import motion_imitation')
try:
    from multitask_envs.halfcheetah import HalfCheetahVelJumpEnv
except:
    print('failed to import halfcheetah')
try:
    import pybullet_envs
except:
    print('Could not import pybullet_envs')
    pass


def make_env(cfg):
    """Helper function to create environment"""
    if cfg.env_type == 'gym':
        env = gym.make(cfg.env)

    # metaworld
    elif cfg.env_type == 'metaworld':
        # for fixed goal
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{cfg.env}-goal-observable"]
        env = env_cls(seed=cfg.seed)
        env._max_episode_steps = 150

    # halfcheetah
    elif cfg.env_type == 'multitask':
        domain_name, task_name = cfg.env.split('_')[0], cfg.env.split('_')[1]
        if domain_name == 'walker2d':
            env = Walker2dEnv(backward=(task_name == 'backward'), jump=(task_name == 'jump'))
            env._max_episode_steps = 1000
        if domain_name == 'halfcheetah':
            # for forward_jump , backward_jump
            if 'jump' in task_name.split('_'):
                env = HalfCheetahVelJumpEnv(forward=(task_name == 'forward'), backward=(task_name == 'backward'), jump=True)
            elif 'goalvel' in task_name.split('_'):
                goal_vel = float(cfg.env.split('_')[-1])
                env = HalfCheetahVelJumpEnv(forward=True, goal_vel=goal_vel)
            # for forward, backward, jump
            else:
                env = HalfCheetahVelJumpEnv(forward=(task_name == 'forward'), backward=(task_name == 'backward'), jump=(task_name == 'jump'))
            env._max_episode_steps = 1000

    # quadrupod
    elif cfg.env_type == 'A1':
        motion_file = f"../motion_imitation/data/motions/{cfg.env}.txt"
        train_reset = False # a task where it learns to stand up from fall
        mode = 'test' # test or train ; not sure why it's important

        ENABLE_ENV_RANDOMIZER = True
        enable_env_rand = ENABLE_ENV_RANDOMIZER and (mode != "test") # I guess starts at random position to train better in onlineRL
        visualize = False
        real = False
        multitask = False
        realistic_sim = False
        env = env_builder.build_env("reset" if train_reset else "imitate",
                                    motion_files=[motion_file],
                                    num_parallel_envs=1,
                                    mode=mode,
                                    enable_randomizer=enable_env_rand,
                                    enable_rendering=visualize,
                                    use_real_robot=real,
                                    reset_at_current_position=multitask,
                                    realistic_sim=realistic_sim)
        env._max_episode_steps = 600
    else:
        print('choose correct env')

    env.seed(cfg.seed)
    env.action_scale_high = env.action_space.high.max()
    env.action_scale_low = env.action_space.high.min()
    return env

def make_agent(obs_dim, action_dim, action_range, cfg):
    cfg.obs_dim = obs_dim
    cfg.action_dim = action_dim
    cfg.action_range = action_range
    return hydra.utils.instantiate(cfg)

class Workspace(object):
    def __init__(self, cfg):

        self.cfg = cfg
        # set workdir
        self.set_work_dir()
        self.set_logger()
        # set seed
        self.set_seed()
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)
        self.agent = make_agent(self.env.observation_space.shape[0],
                                self.env.action_space.shape[0],
                                [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())],
                                self.cfg.agent)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        self.tmp_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(max(self.env._max_episode_steps, self.cfg.batch_size)),
                                          self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def set_agent(self):
        self.agent = make_agent(self.env.observation_space.shape[0],
                                self.env.action_space.shape[0],
                                [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())],
                                self.cfg.agent)

    def set_dummy_agent(self):
        from npf_utils_single_task import sync_weights
        self.dummy_agent = make_agent(self.env.observation_space.shape[0],
                                self.env.action_space.shape[0],
                                [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())],
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
        self.work_dir = self.work_dir + f'/algo={self.cfg["agent_name"]},env={self.cfg["env"]},' \
                                        f'env_type={self.cfg["env_type"]},seed={self.cfg["seed"]},' \
                                        f'bs={self.cfg.agent.batch_size},h_dim={self.cfg.diag_gaussian_actor.hidden_dim},' \
                                        f'h_depth={self.cfg.hidden_depth},kr={self.cfg.keep_ratio},lr={self.cfg.lr},' \
                                        f'continual_pruning={self.cfg.continual_pruning},m_mavg={self.cfg.mask_update_mavg},' \
                                        f'mask_init={self.cfg.mask_init_method},ips_thrsh={self.cfg.ips_threshold},iterative_pruning={self.cfg.iterative_pruning}'
        print(f'workspace: {self.work_dir}')

    def set_seed(self):
        utils.set_seed_everywhere(self.cfg.seed)

    def reset_episodic_storage(self):
        self.storage = {'observations': [],
                        'actions': [],
                        'rewards': [],
                        'terminals': [],
                        'next_observations': [],
                        'episodic_returns': [],
                        'success': []}

    def evaluate(self):
        self.reset_episodic_storage()
        average_episode_reward = 0
        average_episode_len = 0
        average_success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_len = 0
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0

            while not (done or episode_len >= self.env._max_episode_steps):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                self.storage['observations'].append(obs.astype(np.float32))
                self.storage['actions'].append(action.astype(np.float32))

                obs, reward, done, info = self.env.step(action*self.env.action_scale_high)
                if (episode_len + 1) == self.env._max_episode_steps:
                    done = True
                    print(f'reset episode {episode}')
                self.storage['next_observations'].append(obs.astype(np.float32))
                self.storage['rewards'].append(reward)
                self.storage['terminals'].append(int(done))

                self.video_recorder.record(self.env)
                episode_reward += reward
                average_episode_len += 1
                episode_len += 1
            self.storage['episodic_returns'].append(episode_reward)

            average_episode_reward += episode_reward
            if self.cfg.env_type == 'metaworld':
                average_success_rate += info['success']
                self.storage['success'].append(info['success'])

            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_len /= self.cfg.num_eval_episodes
        if self.cfg.env_type == 'metaworld':
            average_success_rate /= self.cfg.num_eval_episodes
            print(f'Average success rate {average_success_rate}')
            self.logger.log('eval/success_rate', average_success_rate, self.step)
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/episode_len', average_episode_len, self.step)

        self.logger.dump(self.step)
        if self.cfg.env_type == 'metaworld':
            return average_success_rate
        else:
            return average_episode_reward


    def quick_collect(self):
        done = False
        obs = self.env.reset()
        self.agent.reset()
        episode_step = 1
        episode_rw = 0
        episodes = 10
        for _ in range(episodes):
            while not done:
                # sample action:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                # take a step
                next_obs, reward, done, _ = self.env.step(action*self.env.action_scale_high)
                if episode_step + 1 == self.env._max_episode_steps: done = True
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
                # collect samples
                self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
                obs = next_obs
                episode_step +=1
                episode_rw += reward
            episode_step = 0
            done = False
            obs = self.env.reset()
        print(f'expert_reward {episode_rw/episodes}')

    def init_network(self):
        # uses random expert samples
        self.quick_collect()

        # ---------------------------------
        # configure pathway/ find masks
        # ---------------------------------
        keep_masks = collections.defaultdict(dict)
        # init dummy agent: this by default syncs weight of the "agent"
        self.set_dummy_agent()
        self.neural_pathway = class_pathways(self.cfg.keep_ratio, history_len=self.cfg.mask_update_mavg)
        keep_masks["actor"][f'task0'], keep_masks["critic"][f'task0'] = self.neural_pathway.get_masks(self.dummy_agent,
                                                                                                      self.replay_buffer)

        # load mask
        self.mask_agent(keep_masks)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(self.cfg.replay_buffer_capacity),
                                          self.device)


    def mask_agent(self, keep_masks):
        from npf_utils import load_keep_masks, monkey_patch, apply_prune_mask, unhook
        # unhook
        unhook(self.agent.actor)
        unhook(self.agent.critic)
        unhook(self.agent.critic_target)
        # for forward
        monkey_patch(self.agent.actor, keep_masks["actor"][f'task0'])
        monkey_patch(self.agent.critic, keep_masks["critic"][f'task0'])
        monkey_patch(self.agent.critic_target, keep_masks["critic"][f'task0'])
        # for backward
        apply_prune_mask(self.agent.actor, keep_masks["actor"]['task0'], fixed_weight=-1)
        apply_prune_mask(self.agent.critic, keep_masks["critic"]['task0'], fixed_weight=-1)



    def run(self):

        # initalize variables
        best_ep_ret = 0
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        activate_eval = False
        info = {}
        info['success'] = False

        while self.step < self.cfg.num_train_steps:
            if done:
                # check if we should stop configuring pathway: for single-task metaworld, stop if reached to success state
                if self.cfg.env_type == 'metaworld' and info['success']:
                    self.cfg.iterative_pruning = False

                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # -----------------------------
                # evaluate agent periodically
                # -----------------------------

                if activate_eval:
                    self.logger.log('eval/episode', episode, self.step)
                    avg_eval_ret = self.evaluate()
                    self.agent.save(self.work_dir, step='final')  # saves the last evaluation
                    if avg_eval_ret > best_ep_ret:
                        self.agent.save(self.work_dir, step='best')
                        best_ep_ret = avg_eval_ret
                    activate_eval = False

                    # check if we should stop configuring pathway: stop if performance reached to a threshold value
                    if self.cfg.env_type == 'metaworld':
                        if (not self.cfg.continual_pruning and avg_eval_ret >= self.cfg.ips_threshold):
                            self.cfg.iterative_pruning = False
                    else:
                        if (not self.cfg.continual_pruning and avg_eval_ret > self.cfg.ips_threshold):
                            self.cfg.iterative_pruning = False


                self.logger.log('train/episode_reward', episode_reward, self.step)
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)

                # ---------------------------------
                # configure pathway/ find masks
                # ---------------------------------

                if self.cfg.iterative_pruning and self.step > 0:
                    import collections
                    keep_masks = collections.defaultdict(dict)
                    self.set_dummy_agent()
                    keep_masks["actor"][f'task0'], keep_masks["critic"][f'task0'] = self.neural_pathway.get_masks(self.dummy_agent, self.tmp_buffer)
                    print(f'pruned param score: {self.neural_pathway.prune_param_score}')
                    self.logger.log('train/pruned param score', self.neural_pathway.prune_param_score, self.step)
                    self.mask_agent(keep_masks)

                    # find the the number of change masks

                    init_w_counts = common_weight(keep_masks['actor']['task0'], keep_masks['actor']['task0'])
                    self.logger.log('train/mask_counts', init_w_counts, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, info = self.env.step(action*self.env.action_scale_high)
            if episode_step + 1 == self.env._max_episode_steps:
                done = True
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
            self.tmp_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

            # important: once done is reached, it checks if 'activate done' is already activated
            # allows to get eval at i.e. 5007 1024 steps
            if self.step % self.cfg.eval_frequency == 0:
                activate_eval = True

        # save the recorder only on the last eval
        self.video_recorder = VideoRecorder(self.work_dir)
        self.agent.save(self.work_dir, step='final')




os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(config_path='./config', config_name='train_pruned')
def main(cfg):
    from train_pruned import Workspace as W
    workspace = W(cfg)
    workspace.init_network()
    workspace.run()


if __name__ == '__main__':
   main()
