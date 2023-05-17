import gym
import os
import tensorflow as tf
from data_utils import d4rl_collect_expert_data,  collect_expert_data
import numpy as np
import pickle
import gzip
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:import d4rl
except:
    print('failed importing d4rl')
    pass
try:from multitask_envs.walker2d import Walker2dEnv
except:
    print('failed importing walkerEnv')
    pass
try:from multitask_envs.halfcheetah import HalfCheetahVelJumpEnv
except:
    print('failed importing HalfcheetahVelJumpEnv')
    pass
try:from motion_imitation.quadrupod_multitask import get_A1, collect_A1_expert
except:
    print('failed importing get_A1')
    pass

MT39 = ['faucet-close-v2',
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
class make_env(object):
    def __init__(self, env_name, seed):
        self.task_name = env_name
        self.env = None
        self.initize_env(env_name, seed)

    def initize_env(self, env_name, seed):
        if env_name in ['pace', 'pace_backward', 'hopturn', 'sidesteps','spin']:
            self.env = get_A1(env_name)
            self.env._max_episode_steps = 600

        elif (env_name in ["reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2", "drawer-close-v2",
                          "button-press-topdown-v2", "peg-insert-side-v2", "window-open-v2", "window-close-v2"]) or (env_name in MT39):
            from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
            env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_name}-goal-observable"]
            self.env = env_cls(seed=0)          # TODO: need to fix to 0 since expert is collected for seed 0
            self.env._max_episode_steps = 150
        else:
            try:
                task, domain = env_name.split('_')[0], env_name.split('_')[1]  # for walker_jump
            except:
                task, domain, _ = env_name.split('-')  # for halfcheetah-expert-v2
            if task == 'walker2d' and domain in ['forward', 'backward', 'jump']:
                self.env = Walker2dEnv(backward=(domain == 'backward'), jump=(domain == 'jump'))
                self.env._max_episode_steps = 1000

            elif task == 'halfcheetah' and domain in ['forward', 'backward', 'jump']:
                # env_list = {'task0': HalfCheetahVelJumpEnv(forward=(domain == 'forward'), backward=(domain == 'backward'), jump=(domain == 'jump'))}
                # for forward_jump , backward_jump, jump
                if 'jump' in env_name.split('_'):
                    self.env = HalfCheetahVelJumpEnv(forward=(domain == 'forward'), backward=(domain == 'backward'), jump=True)
                # for forward, backward
                else:
                    self.env = HalfCheetahVelJumpEnv(forward=(domain == 'forward'), backward=(domain == 'backward'),
                                                jump=(domain == 'jump'))
                self.env._max_episode_steps = 1000
            elif task == 'halfcheetah' and domain in ['goalvel']:
                goal_vel = float(env_name.split('_')[-1])
                self.env = HalfCheetahVelJumpEnv(forward=True, goal_vel=goal_vel)
                self.env._max_episode_steps = 1000
            else:
                import gym
                self.env = gym.make(env_name)
                assert self.env._max_episode_steps == 1000

        self.env.seed(seed)
        self.env.task_name = self.task_name


def fillup_reply(replay_buffer, env, env_name, expert_data_type, task_name,buffer_size):
    print(f'collecting data for {env_name}')
    # -- collect data --
    if env_name in ["walker2d_multitask", "walker2d_forward", "walker2d_backward", "walker2d_jump",
                    "halfcheetah_multitask", "halfcheetah_forward", "halfcheetah_forward_jump", "halfcheetah_backward","halfcheetah_backward_jump", "halfcheetah_jump",
                    "halfcheetah_goalvel_multitask", "halfcheetah_goalvel_0.5", "halfcheetah_goalvel_1.0", "halfcheetah_goalvel_1.5","halfcheetah_goalvel_2.0",
                    "halfcheetah_goalvel_2.5","halfcheetah_goalvel_3.0",
                    "reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2", "drawer-close-v2",
                    "button-press-topdown-v2", "peg-insert-side-v2", "window-open-v2", "window-close-v2"
                    ]:
        # collect expert data
        data_type = expert_data_type
        data_dir = f"./multitask_envs/expert_dataset/{data_type}/{task_name}"
        expert_datasets = os.listdir(data_dir)
        print(expert_datasets)
        assert len(expert_datasets) == 1  # just single file
        filename = data_dir + '/' + expert_datasets[0]
        with tf.io.gfile.GFile(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                dataset = np.load(outfile, allow_pickle=True).item()
        collect_expert_data(dataset, env, replay_buffer, buffer_size=buffer_size, AR=1)

    elif env_name in ["pace", "pace_backward", "hopturn", "sidesteps", "quadrupod_multitask", "spin"]:
        # collect dataset
        data_dir = './motion_imitation/expert_dataset'
        filename = data_dir + '/' + env_name + '/storage'
        with tf.io.gfile.GFile(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                dataset = np.load(outfile, allow_pickle=True).item()
        collect_A1_expert(dataset, env, replay_buffer)
    else:
        dataset = env.unwrapped.get_dataset()
        d4rl_collect_expert_data(dataset, env, replay_buffer,
                                    buffer_size=buffer_size, AR=1,
                                    validation_set=False)


class ReplayBuffer(object):
    def __init__(self):
        self.storage = dict()
        self.buffer_size = 1000000
        self.ctr = 0

    def add(self, data):
        self.storage['observations'][self.ctr] = data[0]
        self.storage['next_observations'][self.ctr] = data[1]
        self.storage['actions'][self.ctr] = data[2]
        self.storage['rewards'][self.ctr] = data[3]
        self.storage['terminals'][self.ctr] = data[4]
        self.ctr += 1
        self.ctr = self.ctr % self.buffer_size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.buffer_size, size=batch_size)
        s = self.storage['observations'][ind]
        a = self.storage['actions'][ind]
        r = self.storage['rewards'][ind]
        s2 = self.storage['next_observations'][ind]
        d = self.storage['terminals'][ind]
        return (np.array(s), np.array(s2), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1))

    def normalize_states(self, eps=1e-3):
        mean = self.storage['observations'].mean(0, keepdims=True)
        std = self.storage['observations'].std(0, keepdims=True) + eps
        self.storage['observations'] = (self.storage['observations'] - mean)/std
        self.storage['next_observations'] = (self.storage['next_observations'] - mean)/std
        return mean, std

    def save(self, filename):
        np.save("./buffers/"+filename+".npy", self.storage)




