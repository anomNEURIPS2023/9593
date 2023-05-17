import sys
import gym
import numpy as np
import torch
import argparse
import os
import random
import utils
import algos
from logger import logger, setup_logger
import torch.nn.functional as F
import torch.nn as nn
import time
from data_utils import d4rl_collect_expert_data, collect_expert_data
import copy
import types
from collections import OrderedDict
from multitask_envs.halfcheetah import HalfCheetahVelJumpEnv
import tensorflow as tf
import gzip
from video import VideoRecorder
from motion_imitation.quadrupod_multitask import get_A1, collect_A1_expert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def unhook(model):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
                             model.modules())
    for layer in prunable_layers:
        layer.weight._backward_hooks = OrderedDict()


def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def monkey_patch(model, mask_layers):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
                             model.modules())
    for layer, mask in zip(prunable_layers, mask_layers):
        # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #   layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
        layer.weight_mask = mask
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)
def load_keep_masks(path):
    keep_masks = np.load(f'{path}/keep_masks.npy', allow_pickle=True).item()
    return keep_masks

def evaluate_policy(policy , eval_env, mean, std, eval_episodes=10, action_repeat=4, masks = None, logdir=None, save_video=False):
    if save_video:
        video_recorder = VideoRecorder(logdir)
    eval_init = time.time()
    avg_reward = 0.
    success_rate = 0
    all_rewards = []
    for t in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        cntr = 0
        ep_step = 0
        if save_video:  video_recorder.init(enabled=True)
        while not done:
            obs = (np.array(obs).reshape(1, -1) - mean) / std
            action = policy.select_action(obs)
            reward = 0
            for _ in range(action_repeat):
                obs, r, done, info = eval_env.step(action)
                reward += r
                if (ep_step+1) == eval_env._max_episode_steps:
                    done = True
                    print(f'eval timeout at : {ep_step}')
                if done:
                    print(f'Episode: {t} eval done at : {ep_step}')
                    break
            if save_video: video_recorder.record(eval_env)
            avg_reward += reward
            cntr += 1
            ep_step += 1

        if 'success' in info.keys():
            success_rate += info['success']
        all_rewards.append(avg_reward)
        if save_video:
            if ('success' in info.keys()):
                if (info['success'] == 1):
                    print(f'saving the video at : {logdir}/video')
                    video_recorder.save(f'{t}_{eval_env.task_name}.mp4')
                    save_video = False
            else:
                print(f'saving the video at : {logdir}')
                video_recorder.save(f'{t}_{eval_env.task_name}.mp4')
                save_video = False
    eval_time = time.time()-eval_init
    print(f'time took for evaluation {eval_time}')
    avg_reward /= eval_episodes
    success_rate /= eval_episodes
    success_rate = success_rate*100
    for j in range(eval_episodes-1, 1, -1):
        all_rewards[j] = all_rewards[j] - all_rewards[j-1]

    all_rewards = np.array(all_rewards)
    std_rewards = np.std(all_rewards)
    median_reward = np.median(all_rewards)

    def get_normalized_score(env, score):
         return (score - env.ref_random_score) / (env.ref_max_score - env.ref_random_score)

    try:
        d4rl_score = eval_env.get_normalized_score(avg_reward)
    except:
        d4rl_score = get_normalized_score(eval_env, avg_reward)

    if 'success' in info.keys():
        print ("---------------------------------------")
        print("Evaluation over %d episodes: %f | success rate: %f" % (eval_episodes, avg_reward, success_rate))
        print ("---------------------------------------")
    else:
        print ("---------------------------------------")
        print("Evaluation over %d episodes: %f | normalized score :%f " % (eval_episodes, avg_reward, d4rl_score))
        print ("---------------------------------------")
    return avg_reward, std_rewards, median_reward, d4rl_score, eval_time, success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="quadrupod_multitask")                          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                                      # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=float)                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)         # Max time steps to run environment for
    parser.add_argument("--version", default='0', type=str)                 # Basically whether to do min(Q), max(Q), mean(Q) over multiple Q networks for policy updates
    parser.add_argument('--algo_name', default="BC", type=str)         # Which algo to run (see the options below in the main function)
    parser.add_argument('--log_dir', default='./data_tmp', type=str)    # Logging directory
    parser.add_argument('--buffer_size', default=1000000, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument("--experiment_name", default='neural_pathway')
    parser.add_argument('--clip_grad', action="store_true", default=False)
    # pruning
    parser.add_argument("--snip_itr", default=1, type=int)
    parser.add_argument("--keep_ratio", default=0.05, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument("--fixed_pruned_weight", default=-1, type=int, help='if set 0 then all the non-masked-weights are set to be 0, otherwise it keeps the init weights')
    parser.add_argument('--expert_data_type', default='final', type=str, help=' final or best - we are using same seed expert  ')
    parser.add_argument("--activation", default='tanh', help='default tanh for mujoco [-1,1], use relu oterhwsie')
    # IQL
    parser.add_argument('--IQL_policy_type', default='gaussian', help=' `gaussian` or `deterministic`')
    parser.add_argument('--target_update_rule', default='pertwostep', help='choose among [`perstep`, `copy`, `pertwostep`]')
    parser.add_argument('--normalize_data', default=True, type=bool)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', action="store_true", default=False)


    args = parser.parse_args()
    init_time = time.time()
    # Use any random seed, and not the user provided seed
    seed = args.seed
    algo_name = args.algo_name
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # init envs
    if args.env_name == 'halfcheetah_multitask':
        task = 'halfcheetah_multitask'
        num_envs = 5
        env_name_list = {'task0': 'halfcheetah_forward', 'task1': 'halfcheetah_backward', 'task2': 'halfcheetah_jump',
                         'task3': 'halfcheetah_forward_jump', 'task4': 'halfcheetah_backward_jump'}
        env_list = {'task0': HalfCheetahVelJumpEnv(forward=True), 'task1': HalfCheetahVelJumpEnv(backward=True), 'task2': HalfCheetahVelJumpEnv(jump=True),
                    'task3': HalfCheetahVelJumpEnv(forward=True, jump=True), 'task4': HalfCheetahVelJumpEnv(backward=True, jump=True)}

        for key in env_list.keys():
            env_list[f'{key}'].seed(seed)
            env_list[f'{key}']._max_episode_steps = 1000
            env_list[f'{key}'].task_name = env_name_list[key]
    elif args.env_name == 'quadrupod_multitask':
        num_envs = 4
        env_name_list = {'task0': 'pace', 'task1': 'pace_backward', 'task2': 'hopturn', 'task3': 'sidesteps'}
        env_list = {'task0': get_A1('pace'), 'task1': get_A1('pace_backward'), 'task2': get_A1('hopturn'), 'task3': get_A1('sidesteps')}

        for key in env_list.keys():
            env_list[f'{key}'].seed(seed)
            env_list[f'{key}']._max_episode_steps = 600
            env_list[f'{key}'].task_name = env_name_list[key]
    elif args.env_name == 'halfcheetah_goalvel_multitask':
        num_envs = 6
        env_name_list = {'task0': 'halfcheetah_goalvel_0.5', 'task1': 'halfcheetah_goalvel_1.0',
                         'task2': 'halfcheetah_goalvel_1.5', 'task3': 'halfcheetah_goalvel_2.0',
                         'task4': 'halfcheetah_goalvel_2.5', 'task5': 'halfcheetah_goalvel_3.0'}
        env_list = {'task0': HalfCheetahVelJumpEnv(forward=True, goal_vel=0.5),
                    'task1': HalfCheetahVelJumpEnv(forward=True, goal_vel=1.0),
                    'task2': HalfCheetahVelJumpEnv(forward=True, goal_vel=1.5),
                    'task3': HalfCheetahVelJumpEnv(forward=True, goal_vel=2.0),
                    'task4': HalfCheetahVelJumpEnv(forward=True, goal_vel=2.5),
                    'task5': HalfCheetahVelJumpEnv(forward=True, goal_vel=3.0)}

        for key in env_list.keys():
            env_list[f'{key}'].seed(seed)
            env_list[f'{key}']._max_episode_steps = 1000
            env_list[f'{key}'].task_name = env_name_list[key]
    elif args.env_name == 'metaworld':
        env_name_list = {}
        env_list = {}
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        ENV_LIST = ["reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2", "drawer-close-v2",
                  "button-press-topdown-v2", "peg-insert-side-v2", "window-open-v2", "window-close-v2"]
        num_envs = len(ENV_LIST)
        for i, name in enumerate(ENV_LIST):
            env_name_list[f'task{i}'] = name

        for key, meta_task in env_name_list.items():
            env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{meta_task}-goal-observable"]
            env_list[f'{key}'] = env_cls(seed=0)
            env_list[f'{key}']._max_episode_steps = 150
            env_list[f'{key}'].task_name = meta_task
    elif args.env_name == 'metaworld50':
        env_name_list = {}
        env_list = {}
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        ENV_LIST = MT39
        num_envs = len(ENV_LIST)
        for i, name in enumerate(ENV_LIST):
            env_name_list[f'task{i}'] = name

        for key, meta_task in env_name_list.items():
            env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{meta_task}-goal-observable"]
            env_list[f'{key}'] = env_cls(seed=0)
            env_list[f'{key}']._max_episode_steps = 150
            env_list[f'{key}'].task_name = meta_task
    else:
        sys.exit('choose the right env_name')

    state_dim = env_list['task0'].observation_space.shape[0]
    action_dim = env_list['task0'].action_space.shape[0]
    max_action = torch.FloatTensor(env_list['task0'].action_space.high).to(device)
    min_action = torch.FloatTensor(env_list['task0'].action_space.low).to(device)
    print(state_dim, action_dim)
    print('Max action: ', max_action)
    print('Min action: ', min_action)

    # Load buffer
    num_trajectories = int(args.buffer_size / env_list['task0']._max_episode_steps)
    replay_buffer = {}
    state_mean = {}
    state_std = {}
    for task in range(num_envs):
        replay_buffer[f'task{task}'] = utils.ReplayBuffer()

        if (args.env_name in ["walker2d_multitask", "walker2d_forward", "walker2d_backward", "walker2d_jump",
                             "halfcheetah_multitask", "halfcheetah_forward", "halfcheetah_backward", "halfcheetah_jump",
                             "halfcheetah_goalvel_multitask", "metaworld", "metaworld50",
                             "reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2", "drawer-close-v2",
                             "button-press-topdown-v2", "peg-insert-side-v2", "window-open-v2", "window-close-v2"]) or (args.env_name in MT39):
            # collect expert data
            print(env_name_list[f'task{task}'])
            data_type = args.expert_data_type
            print(os.getcwd())
            data_dir = f"./multitask_envs/expert_dataset/{data_type}/{env_name_list[f'task{task}']}"
            expert_datasets = os.listdir(data_dir)
            print(expert_datasets)
            assert len(expert_datasets) == 1 # just single file
            filename = data_dir + '/' + expert_datasets[0]
            with tf.io.gfile.GFile(filename, 'rb') as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    dataset = np.load(outfile, allow_pickle=True).item()
            collect_expert_data(dataset, env_list[f'task{task}'], replay_buffer[f'task{task}'], buffer_size=args.buffer_size)

        elif args.env_name in ["pace", "pace_backward", "hopturn", "sidesteps", "quadrupod_multitask"]:
            # collect dataset
            data_dir = './motion_imitation/expert_dataset'
            filename = data_dir + '/' + env_name_list[f'task{task}'] + '/storage'
            with tf.io.gfile.GFile(filename, 'rb') as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    dataset = np.load(outfile, allow_pickle=True).item()
            collect_A1_expert(dataset, env_list[f'task{task}'], replay_buffer[f'task{task}'])
            args.activation = 'relu'
        else:
            dataset = env_list[f'task{task}'].unwrapped.get_dataset()
            d4rl_collect_expert_data(dataset, env_list[f'task{task}'], replay_buffer[f'task{task}'], buffer_size=args.buffer_size)
        state_mean[f'task{task}'], state_std[f'task{task}'] = replay_buffer[f'task{task}'].normalize_states()



    hparam_str_dict = dict(Exp=args.experiment_name, algo=args.algo_name, seed=args.seed, env=args.env_name,
                           keep_ratio=args.keep_ratio, batch_size=args.batch_size, buffer_size=args.buffer_size,
                           clip_grad=args.clip_grad)
    file_name = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in sorted(hparam_str_dict.keys())])
    print ("---------------------------------------")
    print ("Settings: " + file_name)
    print ("---------------------------------------")
    variant = hparam_str_dict
    setup_logger(file_name, variant=variant, log_dir=os.path.join(args.log_dir, file_name))
    if algo_name == 'BCQ':
        policy = algos.BCQ(state_dim, action_dim, max_action, activation=args.activation, discount=args.gamma, batch_size=args.batch_size)
    elif algo_name =='IQL':
        args.hidden_dim = 256
        policy = algos.IQL(state_dim, action_dim, max_action, min_action, args.hidden_dim, args.activation, discount=args.gamma, batch_size=args.batch_size, max_steps=1000000, policy_type=args.IQL_policy_type, clip_grad=args.clip_grad)
    elif algo_name =='IQL-v2':
        args.hidden_dim = 1024
        policy = algos.IQL(state_dim, action_dim, max_action, min_action, args.hidden_dim, args.activation, discount=args.gamma, batch_size=args.batch_size,
                           actor_grad_norm_lambda=args.actor_grad_norm_lambda, max_steps=1000000, policy_type=args.IQL_policy_type, clip_grad=args.clip_grad)
    elif algo_name == 'BCQ-v2':
        policy = algos.BCQ(state_dim, action_dim, max_action, activation=args.activation, cloning=True)


    # load policy and masks
    policy.load(directory=f'{args.log_dir}/{file_name}', filename='weight')
    # TODO
    policy.actor.eval()
    policy.critic.eval()
    policy.value.eval()

    keep_masks = load_keep_masks(os.path.join(args.log_dir, file_name))
    if args.env_name=='metaworld50':
        import collections
        update_mask = collections.defaultdict(dict)
        for i, env in enumerate(MT39):
            update_mask['actor'][f'task{i}'] = keep_masks['actor'][env]
            update_mask['critic'][f'task{i}'] = keep_masks['critic'][env]
            if args.algo_name in ['BCQ-v2','BCQ']:
                update_mask['vae'][f'task{i}'] = keep_masks['vae'][env]
            else:
                update_mask['value'][f'task{i}'] = keep_masks['value'][env]
        keep_masks = update_mask

    evaluations = []
    eval_time_list = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        for i in range(1):
            ret_eval, var_ret, median_ret, d4rl_score, eval_time, success_rate = {}, {}, {}, {}, {}, {}
            for task in range(num_envs):
                local_model = copy.deepcopy(policy)
                # unhook just to be safe
                unhook(local_model.actor)
                unhook(local_model.critic)
                if args.algo_name in ['IQL','IQL-v2']:
                    unhook(local_model.value)
                else:
                    unhook(local_model.vae)

                # only forward hook is required for evaluation
                monkey_patch(local_model.actor, keep_masks["actor"][f'task{task}'])
                monkey_patch(local_model.critic, keep_masks["critic"][f'task{task}'])
                if args.algo_name in ['IQL','IQL-v2']:
                    monkey_patch(local_model.value, keep_masks["value"][f'task{task}'])
                else:
                    monkey_patch(local_model.vae, keep_masks["vae"][f'task{task}'])

                ret_eval[f'task{task}'], var_ret[f'task{task}'], median_ret[f'task{task}'], d4rl_score[f'task{task}'], eval_time[f'task{task}'], success_rate[f'task{task}'] = evaluate_policy(local_model,
                                                                                                     env_list[f'task{task}'],
                                                                                                     state_mean[f'task{task}'], state_std[f'task{task}'],
                                                                                                     action_repeat=args.action_repeat, eval_episodes=args.eval_episodes, logdir=f'{args.log_dir}/{file_name}', save_video=args.save_video)


        print(f'eval return {ret_eval}')
        if args.env_name in ['metaworld', 'metaworld50']:
            for task in success_rate.keys():
                print(f'success rate {env_name_list[task], success_rate[task]}')
        break

