import random
import numpy as np
import torch
from collections import deque


def d4rl_collect_expert_data(dataset, env, replay_buffer, buffer_size=2000000, AR=1):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], buffer_size)
    expert_states = all_obs[:N]
    expert_actions = all_act[:N]
    if 'next_observations' in dataset:
        expert_next_states = dataset['next_observations'][:N]
    else:
        expert_next_states = np.concatenate([all_obs[1:N, :], np.zeros_like(expert_states[0])[np.newaxis, :]], axis=0)
    expert_reward = dataset['rewards'][:N]
    expert_dones = dataset['terminals'][:N]
    expert_timeouts = dataset['timeouts'][:N]
    expert_dones[np.where(dataset['timeouts'][:N] == 1)] = True

    expert_states_traj = [[]]
    expert_actions_traj = [[]]
    expert_rewards_traj = [[]]
    expert_next_states_traj = [[]]
    expert_dones_traj = [[]]
    expert_next_actions_traj = [[]]
    Z = AR
    action_que = deque(maxlen=Z)
    traj_terminate = Z

    for i in range(expert_states.shape[0]-(Z-1)):
        if traj_terminate<Z:
            traj_terminate+=1
        else:
            expert_states_traj[-1].append(expert_states[i])
            r = 0
            action_que = deque(maxlen=Z)
            for j in range(Z):
                action_que.append(expert_actions[i+j]) # this will add from t+0, t+1, t+2, t+3
                r+=expert_reward[i+j]                  # accumulate rewards
            expert_rewards_traj[-1].append(r)
            expert_actions_traj[-1].append(np.mean(action_que, axis=0))
            expert_next_states_traj[-1].append(expert_next_states[i+Z-1])
            expert_dones_traj[-1].append(expert_dones[i+Z-1])

        if (expert_dones[i + Z - 1]) or (expert_timeouts[i+Z-1]):
            expert_states_traj.append([])
            expert_actions_traj.append([])
            expert_rewards_traj.append([])
            expert_next_states_traj.append([])
            expert_dones_traj.append([])
            traj_terminate = 0

    expert_states_traj =[expert_states_traj[i] for i in range(len(expert_states_traj)) if len(expert_states_traj[i])!=0]
    expert_actions_traj = [expert_actions_traj[i] for i in range(len(expert_actions_traj)) if len(expert_actions_traj[i]) != 0]
    expert_rewards_traj = [expert_rewards_traj[i] for i in range(len(expert_rewards_traj)) if len(expert_rewards_traj[i]) != 0]
    expert_next_states_traj= [expert_next_states_traj[i] for i in range(len(expert_next_states_traj)) if len(expert_next_states_traj[i]) != 0]
    expert_dones_traj = [expert_dones_traj[i] for i in range(len(expert_dones_traj)) if len(expert_dones_traj[i]) != 0]

    # expert next action
    for per_traj_actions in expert_actions_traj:
        for i in range(len(per_traj_actions)-1):
            expert_next_actions_traj[-1].append(per_traj_actions[i+1])
        expert_next_actions_traj[-1].append(np.zeros_like(per_traj_actions[i + 1]))
        expert_next_actions_traj.append([])
    expert_next_actions_traj = [expert_next_actions_traj[i] for i in range(len(expert_next_actions_traj)) if len(expert_next_actions_traj[i]) != 0]


    shuffle_inds = list(range(len(expert_states_traj)))
    random.shuffle(shuffle_inds)
    expert_states_traj = [expert_states_traj[i] for i in shuffle_inds]
    expert_actions_traj = [expert_actions_traj[i] for i in shuffle_inds]
    expert_rewards_traj = [expert_rewards_traj[i] for i in shuffle_inds]
    expert_next_states_traj = [expert_next_states_traj[i] for i in shuffle_inds]
    expert_dones_traj = [expert_dones_traj[i] for i in shuffle_inds]
    expert_next_actions_traj = [expert_next_actions_traj[i] for i in shuffle_inds]


    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    traj_reward = [sum(expert_rewards_traj[i]) for i in range(len(expert_rewards_traj)) if len(expert_rewards_traj[i]) != 0]
    env.ref_max_score = max(traj_reward)
    env.ref_min_score = min(traj_reward)
    env.avg_score = sum(traj_reward)/len(traj_reward)

    print('len : {}'.format(len(expert_states_traj)))
    replay_buffer.storage['observations'] = concat_trajectories(expert_states_traj).astype(np.float32)
    replay_buffer.storage['actions'] = concat_trajectories(expert_actions_traj).astype(np.float32)
    replay_buffer.storage['rewards'] = concat_trajectories(expert_rewards_traj).reshape(-1, 1).astype(np.float32)
    replay_buffer.storage['next_observations'] = concat_trajectories(expert_next_states_traj).astype(np.float32)
    replay_buffer.storage['terminals'] = concat_trajectories(expert_dones_traj).reshape(-1, 1).astype(np.float32)
    replay_buffer.storage['next_actions'] = concat_trajectories(expert_next_actions_traj).astype(np.float32)
    replay_buffer.buffer_size = min(concat_trajectories(expert_states_traj).shape[0], buffer_size) - 1



def collect_expert_data(dataset, env, replay_buffer, buffer_size=2000000, AR=1):
    all_obs = np.array(dataset['observations'])
    all_act = np.array(dataset['actions'])
    N = min(all_obs.shape[0], buffer_size)
    expert_states = all_obs[:N]
    expert_actions = all_act[:N]
    #expert_next_states = np.concatenate([all_obs[1:N, :], np.zeros_like(expert_states[0])[np.newaxis, :]], axis=0)
    expert_next_states = np.array(dataset['next_observations'][:N])
    expert_reward = np.array(dataset['rewards'][:N])
    expert_dones = np.array(dataset['terminals'][:N])
    if 'timeouts' in dataset.keys():
        expert_timeouts = np.array(dataset['timeouts'][:N])
        expert_dones[np.where(dataset['timeouts'][:N] == 1)] = True
    else:
        expert_timeouts = np.array(dataset['terminals'][:N])

    expert_states_traj = [[]]
    expert_actions_traj = [[]]
    expert_rewards_traj = [[]]
    expert_next_states_traj = [[]]
    expert_dones_traj = [[]]
    expert_next_actions_traj = [[]]
    Z = AR
    action_que = deque(maxlen=Z)
    traj_terminate = Z

    for i in range(expert_states.shape[0]-(Z-1)):
        if traj_terminate < Z:
            traj_terminate += 1
        else:
            expert_states_traj[-1].append(expert_states[i])
            r = 0
            action_que = deque(maxlen=Z)
            for j in range(Z):
                action_que.append(expert_actions[i+j]) # this will add from t+0, t+1, t+2, t+3
                r += expert_reward[i+j]                  # accumulate rewards
            expert_rewards_traj[-1].append(r)
            expert_actions_traj[-1].append(np.mean(action_que, axis=0))
            expert_next_states_traj[-1].append(expert_next_states[i+Z-1])
            expert_dones_traj[-1].append(expert_dones[i+Z-1])

        if (expert_dones[i + Z - 1]) or (expert_timeouts[i+Z-1]):
            expert_states_traj.append([])
            expert_actions_traj.append([])
            expert_rewards_traj.append([])
            expert_next_states_traj.append([])
            expert_dones_traj.append([])
            traj_terminate = 0

    expert_states_traj =[expert_states_traj[i] for i in range(len(expert_states_traj)) if len(expert_states_traj[i])!=0]
    expert_actions_traj = [expert_actions_traj[i] for i in range(len(expert_actions_traj)) if len(expert_actions_traj[i]) != 0]
    expert_rewards_traj = [expert_rewards_traj[i] for i in range(len(expert_rewards_traj)) if len(expert_rewards_traj[i]) != 0]
    expert_next_states_traj= [expert_next_states_traj[i] for i in range(len(expert_next_states_traj)) if len(expert_next_states_traj[i]) != 0]
    expert_dones_traj = [expert_dones_traj[i] for i in range(len(expert_dones_traj)) if len(expert_dones_traj[i]) != 0]
    # expert next action
    for per_traj_actions in expert_actions_traj:
        for i in range(len(per_traj_actions)-1):
            expert_next_actions_traj[-1].append(per_traj_actions[i+1])
        expert_next_actions_traj[-1].append(np.zeros_like(per_traj_actions[i + 1]))
        expert_next_actions_traj.append([])
    expert_next_actions_traj = [expert_next_actions_traj[i] for i in range(len(expert_next_actions_traj)) if len(expert_next_actions_traj[i]) != 0]

    # shuffle trajectory
    shuffle_inds = list(range(len(expert_states_traj)))
    random.shuffle(shuffle_inds)
    expert_states_traj = [expert_states_traj[i] for i in shuffle_inds]
    expert_actions_traj = [expert_actions_traj[i] for i in shuffle_inds]
    expert_rewards_traj = [expert_rewards_traj[i] for i in shuffle_inds]
    expert_next_states_traj = [expert_next_states_traj[i] for i in shuffle_inds]
    expert_dones_traj = [expert_dones_traj[i] for i in shuffle_inds]
    expert_next_actions_traj = [expert_next_actions_traj[i] for i in shuffle_inds]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    traj_reward = [sum(expert_rewards_traj[i]) for i in range(len(expert_rewards_traj)) if len(expert_rewards_traj[i]) != 0]
    env.ref_max_score = max(traj_reward)
    env.ref_min_score = min(traj_reward)
    env.avg_score = sum(traj_reward)/len(traj_reward)
    env.ref_random_score = dataset['info']['random_ep_ret']

    print('len : {}'.format(len(expert_states_traj)))
    replay_buffer.storage['observations'] = concat_trajectories(expert_states_traj).astype(np.float32)
    replay_buffer.storage['actions'] = concat_trajectories(expert_actions_traj).astype(np.float32)
    replay_buffer.storage['rewards'] = concat_trajectories(expert_rewards_traj).reshape(-1, 1).astype(np.float32)
    replay_buffer.storage['next_observations'] = concat_trajectories(expert_next_states_traj).astype(np.float32)
    replay_buffer.storage['terminals'] = concat_trajectories(expert_dones_traj).reshape(-1, 1).astype(np.float32)
    replay_buffer.storage['next_actions'] = concat_trajectories(expert_next_actions_traj).astype(np.float32)
    replay_buffer.buffer_size = min(concat_trajectories(expert_states_traj).shape[0], buffer_size) - 1
