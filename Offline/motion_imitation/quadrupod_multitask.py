
from motion_imitation.envs import env_builder
import os
import numpy as np
def get_A1(env_name):
    #print(os.getcwd())
    motion_file = f"./motion_imitation/data/motions/{env_name}.txt"
    train_reset = False  # a task where it learns to stand up from fall
    mode = 'test'  # test or train ; not sure why it's important

    ENABLE_ENV_RANDOMIZER = True
    enable_env_rand = ENABLE_ENV_RANDOMIZER and (
                mode != "test")  # I guess starts at random position to train better in onlineRL
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
    return env


def collect_A1_expert(dataset, env, replay_buffer):
    try:
        env.ref_max_score = dataset['info']['max_ep_ret']
    except:
        env.ref_max_score = dataset['info']['mean_ep_ret']

        #env.ref_min_score = min(traj_reward)
    env.avg_score = dataset['info']['mean_ep_ret']
    try:
        env.ref_random_score = dataset['info']['random_ep_ret']
    except:
        env.ref_random_score = 0
    replay_buffer.storage['observations'] = np.array(dataset['observations']).astype(np.float32)
    replay_buffer.storage['actions'] = np.array(dataset['actions']).astype(np.float32)
    replay_buffer.storage['rewards'] = np.array(dataset['rewards']).reshape(-1, 1).astype(np.float32)
    replay_buffer.storage['next_observations'] = np.array(dataset['next_observations']).astype(np.float32)
    replay_buffer.storage['terminals'] = np.array(dataset['terminals']).reshape(-1, 1).astype(np.float32)
    #replay_buffer.storage['next_actions'] = concat_trajectories(expert_next_actions_traj).astype(np.float32)
    replay_buffer.storage['true_Q'] = np.array(dataset['rewards']).reshape(-1, 1).astype(np.float32)
    replay_buffer.buffer_size = replay_buffer.storage['observations'].shape[0] - 1