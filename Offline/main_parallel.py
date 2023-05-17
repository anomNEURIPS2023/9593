import sys

import gym
import numpy as np
import torch
import argparse
import os
import random
import algos
try:
   from multitask_envs import point_mass
except:
   print('could not import point mass')
   pass

import torch.nn.functional as F
import torch.nn as nn
import time
from neural_pathway import apply_mask
import copy
import types

from multitask_envs.halfcheetah import HalfCheetahVelJumpEnv
try:from multitask_envs.halfcheetah import HalfCheetahVelJumpEnv
except:
    print('failed importing HalfcheetahVelJumpEnv')
    pass
try:from motion_imitation.quadrupod_multitask import get_A1, collect_A1_expert
except:
    print('failed importing get_A1')
    pass

import torch.multiprocessing as mp
from collections import OrderedDict
from logger import setup_logger
from parallel_trainer import BCQ_train, IQL_train, get_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    mp.set_start_method('forkserver', force=True)
    print("forkserver init")
except RuntimeError:
    pass

def save_keep_masks(path, keep_masks):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save('{}/keep_masks'.format(path), dict(keep_masks))

def load_keep_masks(path):
    keep_masks = np.load(f'{path}/keep_masks.npy', allow_pickle=True).item()
    return keep_masks

def unhook(model):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear), model.modules())
    for layer in prunable_layers:
        layer.weight._backward_hooks = OrderedDict()

def mod_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

def mod_forward_linear(self, x):
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
            layer.forward = types.MethodType(mod_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(mod_forward_linear, layer)

def evaluate_policy(policy , eval_env, mean, std, eval_episodes=10, action_repeat=1, masks = None):
    if masks !=None:
        apply_mask(policy.actor, masks['actor_keep_masks'], fixed_weight=0)
        apply_mask(policy.critic, masks['critic_keep_masks'], fixed_weight=0)
        apply_mask(policy.vae, masks['vae_keep_masks'], fixed_weight=0)

    eval_init = time.time()
    avg_reward = 0.
    all_rewards = []
    success_rate = 0
    for t in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_step = 0
        while not done:
            obs = (np.array(obs).reshape(1, -1) - mean) / std
            action = policy.select_action(obs)
            reward = 0
            for _ in range(action_repeat):
                obs, r, done, info = eval_env.step(action)
                reward += r
                if (ep_step+1) == eval_env._max_episode_steps: done = True
                if done: break
            avg_reward += reward
            ep_step += 1

        if 'success' in info.keys():
            success_rate += info['success']
        all_rewards.append(avg_reward)
    eval_time = time.time()-eval_init
    print(f'time took for evaluation {eval_time}')
    avg_reward /= eval_episodes
    for j in range(eval_episodes-1, 1, -1):
        all_rewards[j] = all_rewards[j] - all_rewards[j-1]

    all_rewards = np.array(all_rewards)
    std_rewards = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    success_rate = success_rate / eval_episodes

    def get_normalized_score(env, score):
        try:
            return (score - env.ref_random_score) / (env.ref_max_score - env.ref_random_score)
        except:
            return (score - 0) / (env.ref_max_score - 0)
    try:
        d4rl_score = eval_env.get_normalized_score(avg_reward)
    except:
        d4rl_score = get_normalized_score(eval_env, avg_reward)
    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f | normalized score :%f | success rate: %f" % (eval_episodes, avg_reward, d4rl_score, success_rate))
    print ("---------------------------------------")
    return avg_reward, std_rewards, median_reward, d4rl_score , eval_time, success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah_multitask")                          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                                      # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=float)                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)         # Max time steps to run environment for
    parser.add_argument('--algo_name', default="IQL", type=str)             # Which algo to run (see the options below in the main function)
    parser.add_argument('--log_dir', default='./data_tmp/', type=str)    # Logging directory
    parser.add_argument('--cloning', default="False", type=str)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--experiment', type=str, default='None')
    parser.add_argument('--buffer_size', default=1000000, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument("--experiment_name", default='neural_pathway')
    # if clip grad
    parser.add_argument('--clip_grad', action="store_true", default=False)
    # pathways
    parser.add_argument("--snip_itr", default=1, type=int)
    parser.add_argument("--keep_ratio", default=0.05, type=float)
    parser.add_argument("--fixed_pruned_weight", default=-1, type=int, help='if set 0 then all the non-masked-weights are set to be 0, otherwise it keeps the init weights')
    # env
    parser.add_argument('--mujoco_different_tasks', action="store_true", default=False, help='if True, trains `cheetah` and `walker` ')
    parser.add_argument('--expert_data_type', default='final', type=str, help=' final or best - we are using same seed expert  ')
    parser.add_argument("--activation", default='tanh', help='default tanh for mujoco [-1,1], use relu oterhwsie')
    # IQL
    parser.add_argument('--IQL_policy_type', default='gaussian', help=' `gaussian` or `deterministic`')
    parser.add_argument("--separate_test_process", action="store_true", default=False)
    parser.add_argument("--actor_grad_norm_lambda", default=0.0, type=float)
    parser.add_argument('--target_update_rule', default='pertwostep', help='choose among [`perstep`, `copy`, `pertwostep`]')
    parser.add_argument('--normalize_data', default=True, type=bool)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument("--lr", default=0.001, type=float)



    args = parser.parse_args()
    assert args.algo_name in ['IQL', 'BCQ']

    init_time = time.time()
    seed = args.seed
    algo_name = args.algo_name

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # init envs
    if args.env_name == 'halfcheetah_multitask':
        num_envs = 5
        env_name_list = {'task0': 'halfcheetah_forward', 'task1': 'halfcheetah_backward', 'task2': 'halfcheetah_jump',
                         'task3': 'halfcheetah_forward_jump', 'task4': 'halfcheetah_backward_jump'}
        env_list = {'task0': HalfCheetahVelJumpEnv(forward=True), 'task1': HalfCheetahVelJumpEnv(backward=True),
                    'task2': HalfCheetahVelJumpEnv(jump=True),
                    'task3': HalfCheetahVelJumpEnv(forward=True, jump=True),
                    'task4': HalfCheetahVelJumpEnv(backward=True, jump=True)}

        for key in env_list.keys():
            env_list[f'{key}'].seed(seed)
            env_list[f'{key}']._max_episode_steps = 1000
            env_list[f'{key}'].task_name = env_name_list[key]

    elif args.env_name == 'quadrupod_multitask':
        num_envs = 4
        env_name_list = {'task0': 'pace', 'task1': 'pace_backward', 'task2': 'hopturn', 'task3': 'sidesteps'}
        env_list = {'task0': get_A1('pace'), 'task1': get_A1('pace_backward'), 'task2': get_A1('hopturn'),
                    'task3': get_A1('sidesteps')}

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
        import metaworld
        mT10 = metaworld.MT10()  # Construct the benchmark, sampling tasks
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
            env_list[f'{key}'] = env_cls(seed=0)   # IMPORTANT: expert data collected on seed 0 thus important to keep it fixed
            env_list[f'{key}']._max_episode_steps = 150
            env_list[f'{key}'].task_name = meta_task

    else:
        sys.exit('choose right env')

    # fix seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = env_list['task0'].observation_space.shape[0]
    action_dim = env_list['task0'].action_space.shape[0]
    # max_action = float(env_list['task0'].action_space.high[0])
    max_action = torch.FloatTensor(env_list['task0'].action_space.high).to(device)
    min_action = torch.FloatTensor(env_list['task0'].action_space.low).to(device)
    print(state_dim, action_dim)
    print('Max action: ', max_action)
    print('Min action: ', min_action)


    hparam_str_dict = dict(Exp=args.experiment_name, algo=args.algo_name, seed=args.seed, env=args.env_name,
                           batch_size=args.batch_size, buffer_size=args.buffer_size, keep_ratio=args.keep_ratio,
                           hidden_dim=args.hidden_dim, clip_grad=args.clip_grad)
    file_name = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in sorted(hparam_str_dict.keys())])
    variant = hparam_str_dict
    def _setup_logger(file_name, variant, log_dir):
        def execute():
            setup_logger(file_name, variant, log_dir)
        return execute

    print ("---------------------------------------")
    print ("Settings: " + file_name)
    print ("---------------------------------------")

    # assign relu at lasy layer 'quadrupod tasks'
    if args.env_name in ["pace", "pace_backward", "hopturn", "sidesteps", "quadrupod_multitask", "DeepMimic"]:
        args.activation = 'relu'

    # initialize global-model
    if args.algo_name == 'BCQ':
        # default:
        # policy = algos.BCQ(state_dim, action_dim, max_action, activation=args.activation, discount=args.gamma, batch_size=args.batch_size)
        if args.hidden_dim == 256:
            hidden_dim = {'actor': [256, 256], 'critic': [256, 256], 'vae': [256, 256]}
        elif args.hidden_dim == 400:
            hidden_dim = {'actor': [400, 300], 'critic': [400, 300], 'vae': [750, 750]}
        policy = algos.BCQ(state_dim,
                           action_dim,
                           max_action,
                           hidden_dim=hidden_dim,
                           activation=args.activation,
                           cloning=True)

    elif args.algo_name =='IQL':
        if args.hidden_dim == 256:
            hidden_dim = {'actor': 256, 'critic': [256, 256], 'value': 256}
        elif args.hidden_dim == 400:
            hidden_dim = {'actor': 400, 'critic': [400, 300], 'value': 400}
        elif args.hidden_dim == 1024:
            hidden_dim = {'actor': 1024, 'critic': [1024, 1024], 'value': 1024}
        policy = algos.IQL(state_dim,
                           action_dim,
                           max_action,
                           min_action,
                           hidden_dim=hidden_dim,
                           activation=args.activation,
                           discount=args.gamma,
                           batch_size=args.batch_size,
                           actor_grad_norm_lambda=args.actor_grad_norm_lambda,
                           max_steps=1000000,
                           policy_type=args.IQL_policy_type,
                           clip_grad=args.clip_grad)


    # local-model
    dummy_policy = copy.deepcopy(policy)
    # share the global model over the multi-process
    policy.actor.share_memory()
    policy.critic.share_memory()
    if args.algo_name in ['BCQ', 'BCQ-v2']:
        policy.vae.share_memory()
    elif args.algo_name in ['IQL', 'IQL-v2']:
        policy.value.share_memory()



    # for local process output
    env_num_list = list(range(num_envs))
    process_list = []
    file_info = {'name': file_name, 'variant': variant, 'log_dir': os.path.join(args.log_dir, file_name)}
    res_queue = {}
    env_queue = {}

    # trainer function
    if args.algo_name in ['BCQ', 'BCQ-v2']:
        train_func = BCQ_train
    elif args.algo_name in ['IQL', 'IQL-v2']:
        train_func = IQL_train

    # pre-define the optimizer
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=args.lr)
    if args.algo_name in ['BCQ']:
        vae_optimizer = torch.optim.Adam(policy.vae.parameters(), lr=args.lr)
        optimizer = [actor_optimizer, critic_optimizer, vae_optimizer]
    elif args.algo_name == 'IQL':
        value_optimizer = torch.optim.Adam(policy.value.parameters(), lr=args.lr)
        optimizer = [actor_optimizer, critic_optimizer, value_optimizer]

    # add all the trainer process
    for task in env_num_list:
        res_queue[f'task{task}'] = mp.Queue()
        p = mp.Process(target=train_func, args=(dummy_policy, policy, task, evaluate_policy, env_name_list[f'task{task}'], seed, args.keep_ratio, args.snip_itr, args.clip_grad,
                                                   res_queue[f'task{task}'], args.target_update_rule, optimizer, file_info, args.batch_size, args.max_timesteps, args.buffer_size))
        p.start()
        process_list.append(p)

    # process to collect local output
    p = mp.Process(target=get_data, args=(env_num_list, res_queue, file_info, args.algo_name))
    p.start()
    process_list.append(p)
    for process in process_list:
        process.join()

