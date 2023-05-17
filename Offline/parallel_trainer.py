import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict
import types
from neural_pathway import configure_pathway, apply_mask
import copy
import time
import utils
from utils import make_env, fillup_reply
import collections

from logger import logger, setup_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def is_metaworld(env_name):
    ENV_LIST = ["reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2", "drawer-close-v2",
                  "button-press-topdown-v2", "peg-insert-side-v2", "window-open-v2", "window-close-v2"]
    return env_name in ENV_LIST


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


def unhook(model):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
                             model.modules())
    for layer in prunable_layers:
        layer.weight._backward_hooks = OrderedDict()

def mod_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def mod_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def monkey_patch(model, mask_layers):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
                             model.modules())
    for layer, mask in zip(prunable_layers, mask_layers):
        layer.weight_mask = mask
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(mod_forward_conv2d, layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(mod_forward_linear, layer)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad



def save_keep_masks(path, keep_masks):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save('{}/keep_masks'.format(path), dict(keep_masks))

def get_data(env_num_list, res_queue, file_info, algo_name):
    setup_logger(file_info['name'], variant=file_info['variant'], log_dir=file_info['log_dir'])
    keep_masks = collections.defaultdict(dict)
    done_mask_info = False

    while not done_mask_info:
        for task in env_num_list:
            info = res_queue[f'task{task}'].get()
            if info is not None:
                keep_masks['actor'][f'task{task}'] = info['masks']['actor']
                keep_masks['critic'][f'task{task}'] = info['masks']['critic']

                if algo_name in ['BCQ', 'BC']:
                    keep_masks['vae'][f'task{task}'] = info['masks']['vae']
                elif algo_name in ['IQL']:
                    keep_masks['value'][f'task{task}'] = info['masks']['value']
                else:
                    pass

            else:
                break
        done_mask_info = True
    # save keep_masks
    save_keep_masks(file_info['log_dir'], keep_masks)

    while True:
        for task in env_num_list:
            info = res_queue[f'task{task}'].get()
            if info is not None:
                logger.record_tabular(f'Eval/ENV{task}/AverageReturn', info['return'])
                logger.record_tabular(f'Eval/ENV{task}/D4RL_normalized_score', info['normalized_score'])
                logger.record_tabular(f'Eval/ENV{task}/SuccessRate', info['success_rate'])
                logger.record_tabular(f'Eval/ENV{task}/Iteration', info['itr'])
                logger.record_tabular(f'Eval/ENV{task}/Time', info['time'])
                logger.record_tabular(f'Eval/ENV{task}/ExpertReturn', info['expert_avg_score'])


            else:
                break
        logger.dump_tabular()


def get_data_seq(num_process, res_queue, file_info, algo_name):
    setup_logger(file_info['name'], variant=file_info['variant'], log_dir=file_info['log_dir'])
    keep_masks = collections.defaultdict(dict)
    done_mask_info = False

    while not done_mask_info:
        for task in range(num_process):
            info = res_queue[f'process_{task}'].get()
            if info is not None:
                for z in range(len(info['masks']['actor'].keys())):
                    keep_masks['actor'][f'{info["env_list_"][z]}'] = info['masks']['actor'][f'task_{z}']
                    keep_masks['critic'][f'{info["env_list_"][z]}'] = info['masks']['critic'][f'task_{z}']

                    if algo_name in ['BCQ', 'BC']:
                        keep_masks['vae'][f'{info["env_list_"][z]}'] = info['masks']['vae'][f'task_{z}']
                    elif algo_name in ['IQL']:
                        keep_masks['value'][f'{info["env_list_"][z]}'] = info['masks']['value'][f'task_{z}']
                    else:
                        pass

            else:
                break
        done_mask_info = True
    # save keep_masks
    save_keep_masks(file_info['log_dir'], keep_masks)


    while True:
        for task in range(num_process):
            info = res_queue[f'process_{task}'].get()
            if info is not None:
                for z in info.keys():
                    logger.record_tabular(f'Eval/{z}/AverageReturn', info[z]['return'])
                    logger.record_tabular(f'Eval/{z}/D4RL_normalized_score', info[z]['normalized_score'])
                    logger.record_tabular(f'Eval/{z}/SuccessRate', info[z]['success_rate'])
                    logger.record_tabular(f'Eval/{z}/Iteration', info[z]['itr'])
                    logger.record_tabular(f'Eval/{z}/Time', info[z]['time'])
                    logger.record_tabular(f'Eval/{z}/ExpertReturn', info[z]['expert_avg_score'])


            else:
                break
        logger.dump_tabular()


def BCQ_train(model, shared_model, task, evaluate, env_name, seed, keep_ratio, snip_itr, clip_grad, res_queue, target_update_rule='perstep', optimizer=None, file_info=None,
              batch_size=100, max_timesteps=1000000, buffer_size=None, eval_freq=5000,  fixed_pruned_weight=-1):


    def check_if_update_target(itr):
        if target_update_rule=='perstep':
            return True
        elif target_update_rule=='pertwostep':
            return True if (itr%2==0 and itr!=0)else False

    def update_target(model, target, itr, tau):
        if target_update_rule == 'copy' :
            target.load_state_dict(model.state_dict())
        else:
            if check_if_update_target(itr):
                for param, target_param in zip(model.parameters(), target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    env = make_env(env_name, seed).env
    torch.manual_seed(seed)
    np.random.seed(seed)

    # replay buffer
    replay_buffer = utils.ReplayBuffer()

    print(f'collecting data for: {env_name}')
    fillup_reply(replay_buffer, env, env_name, expert_data_type='final',
                 task_name=env.task_name, buffer_size=buffer_size)
    state_mean, state_std = replay_buffer.normalize_states()

    print(f'getting pruning masks for: {env_name}')
    # get masks
    keep_masks = collections.defaultdict(dict)
    keep_masks["actor"], keep_masks["critic"], keep_masks["vae"] = configure_pathway(shared_model, keep_ratio, replay_buffer, iterations=snip_itr, vae=True, value_func=False)
    info = {'masks': keep_masks, 'task': task}
    res_queue.put(info)



    # init shared optimizer
    if optimizer is not None:
        actor_optimizer, critic_optimizer, vae_optimizer = optimizer
    else:
        actor_optimizer = torch.optim.Adam(shared_model.actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(shared_model.critic.parameters(), lr=5e-6)
        vae_optimizer = torch.optim.Adam(shared_model.vae.parameters(), lr=1e-4)

    init_t = time.time()

    training_iters = 0
    while training_iters < max_timesteps:
        print('task: {} | iter: {}'.format(task, training_iters))
        for n in range(int(eval_freq)):
            # ----- specify the neurons that are to be updated ----

            # --- Stage A: Sync with the shared model
            model.actor.load_state_dict(shared_model.actor.state_dict())
            model.critic.load_state_dict(shared_model.critic.state_dict())
            model.vae.load_state_dict(shared_model.vae.state_dict())

            # TODO update target


            tau = 0.005
            update_target(model.critic, model.critic_target, training_iters, tau)
            update_target(model.actor, model.actor_target, training_iters, tau)


            #  --- Stage B ---
            # step 1: unhook
            unhook(model.actor)
            unhook(model.critic)
            unhook(model.vae)

            # step 2: for gradient update
            # this backward hook does an inplace operation
            # thus require to unhook

            apply_mask(model.actor, keep_masks["actor"], fixed_weight=fixed_pruned_weight)
            apply_mask(model.critic, keep_masks["critic"], fixed_weight=fixed_pruned_weight)
            apply_mask(model.vae, keep_masks["vae"], fixed_weight=fixed_pruned_weight)

            # step 3: for forward :
            # monkey patch does not do any inplace operation
            # it multiply weights with specific mask,
            # thus no unhook is required for forward
            monkey_patch(model.actor, keep_masks["actor"])
            monkey_patch(model.critic, keep_masks["critic"])
            monkey_patch(model.vae, keep_masks["vae"])

            # TODO: target network
            # monkey_patch(model.actor_target, keep_masks["actor"])
            # monkey_patch(model.critic_target, keep_masks["critic"])
            #model.actor_target.load_state_dict(model.actor_target.state_dict())
            #model.critic_target.load_state_dict(model.critic_target.state_dict())

            #  --- (end Stage B) ---

            # --- update network ----

            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)


            # Variational Auto-Encoder Training
            recon, mean, std = model.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            vae_optimizer.zero_grad()
            vae_loss.backward()
            if clip_grad: torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 40)
            ensure_shared_grads(model.vae, shared_model.vae)
            vae_optimizer.step()
            # update vae
            unhook(model.vae)
            model.vae.load_state_dict(shared_model.vae.state_dict())
            monkey_patch(model.vae, keep_masks["vae"])

            # Critic Training
            with torch.no_grad():
                # Duplicate state 10 times
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)

                # Compute value of perturbed actions sampled from the VAE
                if model.use_cloning:
                    target_Q1, target_Q2 = model.critic_target(state_rep, model.vae.decode(state_rep))
                else:
                    target_Q1, target_Q2 = model.critic_target(state_rep, model.actor_target(state_rep, model.vae.decode(state_rep)))

                # Soft Clipped Double Q-learning
                target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

                target_Q = reward + done * model.discount * target_Q

            current_Q1, current_Q2 = model.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)


            critic_optimizer.zero_grad()
            critic_loss.backward()
            if clip_grad: torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 40)
            ensure_shared_grads(model.critic, shared_model.critic)
            critic_optimizer.step()
            # update critic
            unhook(model.critic)
            model.critic.load_state_dict(shared_model.critic.state_dict())
            monkey_patch(model.critic, keep_masks["critic"])

            if training_iters%2==0:
                # Pertubation Model / Action Training
                with torch.no_grad():
                    sampled_actions = model.vae.decode(state)
                perturbed_actions = model.actor(state, sampled_actions)

                # Update through DPG
                actor_loss = -model.critic.q1(state, perturbed_actions).mean()


                actor_optimizer.zero_grad()
                actor_loss.backward()
                if clip_grad: torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 40)
                ensure_shared_grads(model.actor, shared_model.actor)
                actor_optimizer.step()
                # update actor
                unhook(model.actor)
                model.actor.load_state_dict(shared_model.actor.state_dict())
                monkey_patch(model.actor, keep_masks["actor"])

            training_iters += 1

        # #  -- eval --
        # # Stage A: Sync with the shared model
        # model.actor.load_state_dict(shared_model.actor.state_dict())
        # model.critic.load_state_dict(shared_model.critic.state_dict())
        # model.vae.load_state_dict(shared_model.vae.state_dict())

        # Stage B: model.
        # only forward hook is required for evaluation
        # monkey_patch(model.actor, keep_masks["actor"])
        # monkey_patch(model.critic, keep_masks["critic"])
        # monkey_patch(model.vae, keep_masks["vae"])
        #
        #
        # ret_eval, var_ret, median_ret, d4rl_score, \
        # eval_time, success_rate = evaluate(model, env, state_mean, state_std, action_repeat=1)
        dummy = copy.deepcopy(shared_model)
        monkey_patch(dummy.actor, keep_masks["actor"])
        monkey_patch(dummy.critic, keep_masks["critic"])
        monkey_patch(dummy.vae, keep_masks["vae"])



        ret_eval, var_ret, median_ret, d4rl_score, \
        eval_time, success_rate = evaluate(dummy, env, state_mean, state_std, action_repeat=1)
        times_spent = (time.time() - init_t) / (3600)
        print('\n Time {} ----- task: {} ----- return : {}'.format(times_spent, task, ret_eval))
        info = {'return': ret_eval, 'normalized_score': d4rl_score, 'success_rate':success_rate, 'expert_avg_score': env.avg_score, 'itr': training_iters, 'time': times_spent}
        res_queue.put(info)
        if task == 0:
            shared_model.save(filename='weight', directory=file_info['log_dir'])



def IQL_train(model, shared_model, task, evaluate, env_name, seed, keep_ratio, snip_itr, clip_grad, res_queue, target_update_rule='perstep', optimizer=None, file_info=None,
              batch_size=100, max_timesteps=1000000, buffer_size=None, eval_freq=5000,  fixed_pruned_weight=-1):


    env = make_env(env_name, seed).env
    torch.manual_seed(seed)
    np.random.seed(seed)

    # replay buffer
    replay_buffer = utils.ReplayBuffer()

    print(f'collecting data for: {env_name}')
    fillup_reply(replay_buffer, env, env_name, expert_data_type='final',
                 task_name=env.task_name, buffer_size=buffer_size)
    state_mean, state_std = replay_buffer.normalize_states()

    print(f'getting pruning masks for: {env_name}')
    # get masks
    keep_masks = collections.defaultdict(dict)
    keep_masks["actor"], keep_masks["critic"], keep_masks["value"] = configure_pathway(shared_model, keep_ratio, replay_buffer, iterations=snip_itr, vae=False, value_func=True)
    info = {'masks': keep_masks, 'task': task}
    res_queue.put(info)


    # init shared optimizer
    if optimizer is not None:
        actor_optimizer, critic_optimizer, value_optimizer = optimizer
    else:
        actor_optimizer = torch.optim.Adam(shared_model.actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(shared_model.critic.parameters(), lr=3e-4)
        value_optimizer = torch.optim.Adam(shared_model.value.parameters(), lr=3e-4)


    # IQL parameters
    policy_lr_schedule = CosineAnnealingLR(actor_optimizer, max_timesteps)
    discount = 0.99
    beta = 3.0
    EXP_ADV_MAX = 100.
    alpha = 0.005
    tau_x = 0.7

    init_t = time.time()

    training_iters = 0
    while training_iters < max_timesteps:
        print('task: {} | iter: {}'.format(task, training_iters))
        for n in range(int(eval_freq)):
            # ----- specify the neurons that are to be updated ----

            # --- Stage A: Sync with the shared model
            model.actor.load_state_dict(shared_model.actor.state_dict())
            model.critic.load_state_dict(shared_model.critic.state_dict())
            model.value.load_state_dict(shared_model.value.state_dict())

            tau = 0.005
            for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            #  --- Stage B ---
            # step 1: unhook
            unhook(model.actor)
            unhook(model.critic)
            unhook(model.value)

            # step 2: for gradient update
            # this backward hook does an inplace operation
            # thus require to unhook

            apply_mask(model.actor, keep_masks["actor"], fixed_weight=fixed_pruned_weight)
            apply_mask(model.critic, keep_masks["critic"], fixed_weight=fixed_pruned_weight)
            apply_mask(model.value, keep_masks["value"], fixed_weight=fixed_pruned_weight)

            # step 3: for forward :
            # monkey patch does not do any inplace operation
            # it multiply weights with specific mask,
            # thus no unhook is required for forward
            monkey_patch(model.actor, keep_masks["actor"])
            monkey_patch(model.critic, keep_masks["critic"])
            monkey_patch(model.value, keep_masks["value"])

            #  --- (end Stage B) ---

            # --- update network ----

            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)

            # Update value function
            with torch.no_grad():
                target_q = torch.min(*model.critic_target(state, action))
                next_v = model.value(next_state).reshape(-1,1)

            # v, next_v = compute_batched(self.vf, [observations, next_observations])
            v = model.value(state)
            adv = target_q - v
            v_loss = asymmetric_l2_loss(adv, tau_x)

            value_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            ensure_shared_grads(model.value, shared_model.value)
            value_optimizer.step()
            unhook(model.value)
            model.value.load_state_dict(shared_model.value.state_dict())
            monkey_patch(model.value, keep_masks["value"])

            # Update Q function
            true_Q = reward + done * discount * next_v.detach()
            current_Q1, current_Q2 = model.critic(state, action)
            critic_loss = (F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q))/2

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if clip_grad: torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 40)
            ensure_shared_grads(model.critic, shared_model.critic)
            critic_optimizer.step()
            # update critic
            unhook(model.critic)
            model.critic.load_state_dict(shared_model.critic.state_dict())
            monkey_patch(model.critic, keep_masks["critic"])


            # Pertubation Model / Action Training
            # Update policy
            exp_adv = torch.exp(beta * adv.detach()).clamp(max=EXP_ADV_MAX)
            policy_out = model.actor(state)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(action)
                #bc_losses = bc_losses.sum(-1)
            elif torch.is_tensor(policy_out):
                assert policy_out.shape == action.shape
                bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
            else:
                raise NotImplementedError
            actor_loss = torch.mean(exp_adv * bc_losses)


            actor_optimizer.zero_grad()
            actor_loss.backward()
            if clip_grad: torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 40)
            ensure_shared_grads(model.actor, shared_model.actor)
            actor_optimizer.step()
            policy_lr_schedule.step()
            # update actor
            unhook(model.actor)
            model.actor.load_state_dict(shared_model.actor.state_dict())
            monkey_patch(model.actor, keep_masks["actor"])

            training_iters += 1

        # #  -- eval --
        # # Stage A: Sync with the shared model
        # model.actor.load_state_dict(shared_model.actor.state_dict())
        # model.critic.load_state_dict(shared_model.critic.state_dict())
        # model.vae.load_state_dict(shared_model.vae.state_dict())

        # Stage B: model.
        # only forward hook is required for evaluation
        # monkey_patch(model.actor, keep_masks["actor"])
        # monkey_patch(model.critic, keep_masks["critic"])
        # monkey_patch(model.value, keep_masks["value"])
        #
        #
        # ret_eval, var_ret, median_ret, d4rl_score, \
        # eval_time, success_rate = evaluate(model, env, state_mean, state_std, action_repeat=1)
        dummy = copy.deepcopy(shared_model)
        monkey_patch(dummy.actor, keep_masks["actor"])
        monkey_patch(dummy.critic, keep_masks["critic"])
        monkey_patch(dummy.value, keep_masks["value"])


        ret_eval, var_ret, median_ret, d4rl_score, \
        eval_time, success_rate = evaluate(dummy, env, state_mean, state_std, action_repeat=1)

        times_spent = (time.time() - init_t) / (3600)
        print('\n Time {} ----- task: {} ----- return : {}'.format(times_spent, task, ret_eval))



        info = {'return': ret_eval, 'normalized_score': d4rl_score, 'success_rate':success_rate, 'expert_avg_score': env.avg_score, 'itr': training_iters, 'time': times_spent}

        res_queue.put(info)

        if task==0:
            shared_model.save(filename='weight', directory=file_info['log_dir'])





def BCQ_train_10(model, shared_model, process_ID, evaluate, env_list_, seed, keep_ratio, snip_itr, clip_grad, res_queue, target_update_rule='perstep', optimizer=None, file_info=None,
              batch_size=100, max_timesteps=1000000, eval_freq=5000,  fixed_pruned_weight=-1):

    env_name_list = {}
    for i,x in enumerate(env_list_):
        env_name_list[f'task_{i}'] = x
    #setup_logger(file_info['name'], variant=file_info['variant'], log_dir=file_info['log_dir'])
    # init env
    num_envs = len(env_name_list.keys())
    torch.manual_seed(seed)
    np.random.seed(seed)

    def check_if_update_target(itr):
        if target_update_rule=='perstep':
            return True
        elif target_update_rule=='pertwostep':
            return True if (itr%2==0 and itr!=0)else False

    def update_target(model, target, itr, tau):
        if target_update_rule == 'copy':
            target.load_state_dict(model.state_dict())
        else:
            if check_if_update_target(itr):
                for param, target_param in zip(model.parameters(), target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    env_list = {}
    replay_buffer = {}
    state_mean = {}
    state_std = {}
    keep_masks = collections.defaultdict(dict)

    for i in range(num_envs):
        # get env
        env_list[f'task_{i}'] = make_env(env_name_list[f'task_{i}'], seed).env
        # get replay
        replay_buffer[f'task_{i}'] = utils.ReplayBuffer()
        fillup_reply(replay_buffer[f'task_{i}'], env_list[f'task_{i}'], env_name_list[f'task_{i}'],
                     expert_data_type='final', task_name=env_list[f'task_{i}'].task_name, buffer_size=1000000)
        state_mean[f'task_{i}'], state_std[f'task_{i}'] = replay_buffer[f'task_{i}'].normalize_states()

        # get mask
        #print(f'getting pruning masks for: {env_name}')
        keep_masks["actor"][f'task_{i}'], \
        keep_masks["critic"][f'task_{i}'], \
        keep_masks["vae"][f'task_{i}'] = configure_pathway(shared_model, keep_ratio, replay_buffer[f'task_{i}'], iterations=snip_itr, vae=True, value_func=False)

    # TODO : MUST UNCOMMENT
    info = {'masks': keep_masks, 'task': process_ID, 'env_list_':env_list_}
    res_queue.put(info)

    # init shared optimizer
    if optimizer is not None:
        actor_optimizer, critic_optimizer, vae_optimizer = optimizer
    else:
        actor_optimizer = torch.optim.Adam(shared_model.actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(shared_model.critic.parameters(), lr=5e-6)
        vae_optimizer = torch.optim.Adam(shared_model.vae.parameters(), lr=1e-4)

    init_t = time.time()

    training_iters = 0
    while training_iters < max_timesteps:
        print('task: {} | iter: {}'.format(process_ID, training_iters))
        for n in range(int(eval_freq)):

            for i in range(num_envs):

                # ----- specify the neurons that are to be updated ----

                # --- Stage A: Sync with the shared model
                model.actor.load_state_dict(shared_model.actor.state_dict())
                model.critic.load_state_dict(shared_model.critic.state_dict())
                model.vae.load_state_dict(shared_model.vae.state_dict())

                # TODO update target


                tau = 0.005
                update_target(model.critic, model.critic_target, training_iters, tau)
                update_target(model.actor, model.actor_target, training_iters, tau)
                # for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                #
                # for param, target_param in zip(model.actor.parameters(), model.actor_target.parameters()):
                #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                #  --- Stage B ---
                # step 1: unhook
                unhook(model.actor)
                unhook(model.critic)
                unhook(model.vae)

                # step 2: for gradient update
                # this backward hook does an inplace operation
                # thus require to unhook

                apply_mask(model.actor, keep_masks["actor"][f'task_{i}'], fixed_weight=fixed_pruned_weight)
                apply_mask(model.critic, keep_masks["critic"][f'task_{i}'], fixed_weight=fixed_pruned_weight)
                apply_mask(model.vae, keep_masks["vae"][f'task_{i}'], fixed_weight=fixed_pruned_weight)

                # step 3: for forward :
                # monkey patch does not do any inplace operation
                # it multiply weights with specific mask,
                # thus no unhook is required for forward
                monkey_patch(model.actor, keep_masks["actor"][f'task_{i}'])
                monkey_patch(model.critic, keep_masks["critic"][f'task_{i}'])
                monkey_patch(model.vae, keep_masks["vae"][f'task_{i}'])

                #  --- (end Stage B) ---

                # --- update network ----

                # Sample replay buffer / batch
                state_np, next_state_np, action, reward, done = replay_buffer[f'task_{i}'].sample(batch_size)
                state           = torch.FloatTensor(state_np).to(device)
                action          = torch.FloatTensor(action).to(device)
                next_state      = torch.FloatTensor(next_state_np).to(device)
                reward          = torch.FloatTensor(reward).to(device)
                done            = torch.FloatTensor(1 - done).to(device)


                # Variational Auto-Encoder Training
                recon, mean, std = model.vae(state, action)
                recon_loss = F.mse_loss(recon, action)
                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss = recon_loss + 0.5 * KL_loss

                vae_optimizer.zero_grad()
                vae_loss.backward()
                if clip_grad: torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 40)
                ensure_shared_grads(model.vae, shared_model.vae)
                vae_optimizer.step()
                # update vae
                unhook(model.vae)
                model.vae.load_state_dict(shared_model.vae.state_dict())
                monkey_patch(model.vae, keep_masks["vae"][f'task_{i}'])

                # Critic Training
                with torch.no_grad():
                    # Duplicate state 10 times
                    state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)

                    # Compute value of perturbed actions sampled from the VAE
                    if model.use_cloning:
                        target_Q1, target_Q2 = model.critic_target(state_rep, model.vae.decode(state_rep))
                    else:
                        target_Q1, target_Q2 = model.critic_target(state_rep, model.actor_target(state_rep, model.vae.decode(state_rep)))

                    # Soft Clipped Double Q-learning
                    target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
                    target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

                    target_Q = reward + done * model.discount * target_Q

                current_Q1, current_Q2 = model.critic(state, action)
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)


                critic_optimizer.zero_grad()
                critic_loss.backward()
                if clip_grad: torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 40)
                ensure_shared_grads(model.critic, shared_model.critic)
                critic_optimizer.step()
                # update critic
                unhook(model.critic)
                model.critic.load_state_dict(shared_model.critic.state_dict())
                monkey_patch(model.critic, keep_masks["critic"][f'task_{i}'])

                if training_iters%2==0:
                    # Pertubation Model / Action Training
                    with torch.no_grad():
                        sampled_actions = model.vae.decode(state)
                    perturbed_actions = model.actor(state, sampled_actions)

                    # Update through DPG
                    actor_loss = -model.critic.q1(state, perturbed_actions).mean()


                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if clip_grad: torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 40)
                    ensure_shared_grads(model.actor, shared_model.actor)
                    actor_optimizer.step()
                    # update actor
                    unhook(model.actor)
                    model.actor.load_state_dict(shared_model.actor.state_dict())
                    monkey_patch(model.actor, keep_masks["actor"][f'task_{i}'])

            training_iters += 1

        # #  -- eval --
        # # Stage A: Sync with the shared model
        # model.actor.load_state_dict(shared_model.actor.state_dict())
        # model.critic.load_state_dict(shared_model.critic.state_dict())
        # model.vae.load_state_dict(shared_model.vae.state_dict())

        # Stage B: model.
        # only forward hook is required for evaluation
        # monkey_patch(model.actor, keep_masks["actor"])
        # monkey_patch(model.critic, keep_masks["critic"])
        # monkey_patch(model.vae, keep_masks["vae"])
        #
        #
        # ret_eval, var_ret, median_ret, d4rl_score, \
        # eval_time, success_rate = evaluate(model, env, state_mean, state_std, action_repeat=1)
        h_info={}
        for i in range(num_envs):
            dummy = copy.deepcopy(shared_model)
            monkey_patch(dummy.actor, keep_masks["actor"][f'task_{i}'])
            monkey_patch(dummy.critic, keep_masks["critic"][f'task_{i}'])
            monkey_patch(dummy.vae, keep_masks["vae"][f'task_{i}'])

            ret_eval, var_ret, median_ret, d4rl_score, \
            eval_time, success_rate = evaluate(dummy, env_list[f'task_{i}'], state_mean[f'task_{i}'], state_std[f'task_{i}'], action_repeat=1)
            times_spent = (time.time() - init_t) / (3600)
            print('\n Time {} ----- task: {} ----- return : {}'.format(times_spent, process_ID, ret_eval))
            info = {'return': ret_eval, 'normalized_score': d4rl_score, 'success_rate':success_rate,
                    'expert_avg_score': env_list[f'task_{i}'].avg_score, 'itr': training_iters, 'time': times_spent}
            _t_name = env_list_[i]
            h_info[f"{_t_name}"] = info

        res_queue.put(h_info)
        if process_ID == 0:
            shared_model.save(filename='weight', directory=file_info['log_dir'])



def IQL_train_10(model, shared_model, process_ID, evaluate, env_list_, seed, keep_ratio, snip_itr, clip_grad, res_queue, target_update_rule='perstep', optimizer=None, file_info=None,
              batch_size=100, max_timesteps=1000000, eval_freq=5000, fixed_pruned_weight=-1):

    #setup_logger(file_info['name'], variant=file_info['variant'], log_dir=file_info['log_dir'])
    # init env

    env_name_list = {}
    for i,x in enumerate(env_list_):
        env_name_list[f'task_{i}'] = x
    #setup_logger(file_info['name'], variant=file_info['variant'], log_dir=file_info['log_dir'])
    # init env
    num_envs = len(env_name_list.keys())
    torch.manual_seed(seed)
    np.random.seed(seed)

    def check_if_update_target(itr):
        if target_update_rule == 'perstep':
            return True
        elif target_update_rule == 'pertwostep':
            return True if (itr % 2 == 0 and itr != 0)else False

    def update_target(model, target, itr, tau):
        if target_update_rule == 'copy':
            target.load_state_dict(model.state_dict())
        else:
            if check_if_update_target(itr):
                for param, target_param in zip(model.parameters(), target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    env_list = {}
    replay_buffer = {}
    state_mean = {}
    state_std = {}
    keep_masks = collections.defaultdict(dict)

    for i in range(num_envs):
        # get env
        env_list[f'task_{i}'] = make_env(env_name_list[f'task_{i}'], seed).env
        # get replay
        replay_buffer[f'task_{i}'] = utils.ReplayBuffer()
        fillup_reply(replay_buffer[f'task_{i}'], env_list[f'task_{i}'], env_name_list[f'task_{i}'],
                     expert_data_type='final', task_name=env_list[f'task_{i}'].task_name, buffer_size=1000000)
        state_mean[f'task_{i}'], state_std[f'task_{i}'] = replay_buffer[f'task_{i}'].normalize_states()

        # get mask
        # print(f'getting pruning masks for: {env_name}')
        keep_masks["actor"][f'task_{i}'], \
        keep_masks["critic"][f'task_{i}'], \
        keep_masks["value"][f'task_{i}'] = configure_pathway(shared_model, keep_ratio, replay_buffer[f'task_{i}'], iterations=snip_itr,
                                              vae=False, value_func=True)

    # TODO : MUST UNCOMMENT
    info = {'masks': keep_masks, 'task': process_ID, 'env_list_': env_list_}
    res_queue.put(info)

    # init shared optimizer
    if optimizer is not None:
        actor_optimizer, critic_optimizer, value_optimizer = optimizer
    else:
        actor_optimizer = torch.optim.Adam(shared_model.actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(shared_model.critic.parameters(), lr=3e-4)
        value_optimizer = torch.optim.Adam(shared_model.value.parameters(), lr=3e-4)


    # IQL parameters
    policy_lr_schedule = CosineAnnealingLR(actor_optimizer, max_timesteps)
    discount = 0.99
    beta = 3.0
    EXP_ADV_MAX = 100.
    alpha = 0.005
    tau_x = 0.7

    init_t = time.time()

    training_iters = 0
    while training_iters < max_timesteps:
        print('task: {} | iter: {}'.format(process_ID, training_iters))
        for n in range(int(eval_freq)):

            for i in range(num_envs):
                # ----- specify the neurons that are to be updated ----

                # --- Stage A: Sync with the shared model
                model.actor.load_state_dict(shared_model.actor.state_dict())
                model.critic.load_state_dict(shared_model.critic.state_dict())
                model.value.load_state_dict(shared_model.value.state_dict())

                tau = 0.005
                for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                #  --- Stage B ---
                # step 1: unhook
                unhook(model.actor)
                unhook(model.critic)
                unhook(model.value)

                # step 2: for gradient update
                # this backward hook does an inplace operation
                # thus require to unhook

                apply_mask(model.actor, keep_masks["actor"][f'task_{i}'], fixed_weight=fixed_pruned_weight)
                apply_mask(model.critic, keep_masks["critic"][f'task_{i}'], fixed_weight=fixed_pruned_weight)
                apply_mask(model.value, keep_masks["value"][f'task_{i}'], fixed_weight=fixed_pruned_weight)

                # step 3: for forward :
                # monkey patch does not do any inplace operation
                # it multiply weights with specific mask,
                # thus no unhook is required for forward
                monkey_patch(model.actor, keep_masks["actor"][f'task_{i}'])
                monkey_patch(model.critic, keep_masks["critic"][f'task_{i}'])
                monkey_patch(model.value, keep_masks["value"][f'task_{i}'])

                # TODO: target network
                # monkey_patch(model.actor_target, keep_masks["actor"])
                # monkey_patch(model.critic_target, keep_masks["critic"])
                #model.actor_target.load_state_dict(model.actor_target.state_dict())
                #model.critic_target.load_state_dict(model.critic_target.state_dict())

                #  --- (end Stage B) ---

                # --- update network ----

                # Sample replay buffer / batch
                state_np, next_state_np, action, reward, done = replay_buffer[f'task_{i}'].sample(batch_size)
                state           = torch.FloatTensor(state_np).to(device)
                action          = torch.FloatTensor(action).to(device)
                next_state      = torch.FloatTensor(next_state_np).to(device)
                reward          = torch.FloatTensor(reward).to(device)
                done            = torch.FloatTensor(1 - done).to(device)


                # Update value function
                with torch.no_grad():
                    target_q = torch.min(*model.critic_target(state, action))
                    next_v = model.value(next_state).reshape(-1,1)

                # v, next_v = compute_batched(self.vf, [observations, next_observations])
                v = model.value(state)
                adv = target_q - v
                v_loss = asymmetric_l2_loss(adv, tau_x)

                value_optimizer.zero_grad(set_to_none=True)
                v_loss.backward()
                ensure_shared_grads(model.value, shared_model.value)
                value_optimizer.step()
                unhook(model.value)
                model.value.load_state_dict(shared_model.value.state_dict())
                monkey_patch(model.value, keep_masks["value"][f'task_{i}'])

                # Update Q function
                true_Q = reward + done * discount * next_v.detach()
                current_Q1, current_Q2 = model.critic(state, action)
                critic_loss = (F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q))/2

                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if clip_grad: torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 40)
                ensure_shared_grads(model.critic, shared_model.critic)
                critic_optimizer.step()
                # update critic
                unhook(model.critic)
                model.critic.load_state_dict(shared_model.critic.state_dict())
                monkey_patch(model.critic, keep_masks["critic"][f'task_{i}'])


                # Pertubation Model / Action Training
                # Update policy
                exp_adv = torch.exp(beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                policy_out = model.actor(state)
                if isinstance(policy_out, torch.distributions.Distribution):
                    bc_losses = -policy_out.log_prob(action)
                    bc_losses = bc_losses.sum(-1)
                elif torch.is_tensor(policy_out):
                    assert policy_out.shape == action.shape
                    bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
                else:
                    raise NotImplementedError
                actor_loss = torch.mean(exp_adv * bc_losses)


                actor_optimizer.zero_grad()
                actor_loss.backward()
                if clip_grad: torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 40)
                ensure_shared_grads(model.actor, shared_model.actor)
                actor_optimizer.step()
                policy_lr_schedule.step()
                # update actor
                unhook(model.actor)
                model.actor.load_state_dict(shared_model.actor.state_dict())
                monkey_patch(model.actor, keep_masks["actor"][f'task_{i}'])

            training_iters += 1

        # #  -- eval --
        # # Stage A: Sync with the shared model
        # model.actor.load_state_dict(shared_model.actor.state_dict())
        # model.critic.load_state_dict(shared_model.critic.state_dict())
        # model.vae.load_state_dict(shared_model.vae.state_dict())

        # Stage B: model.
        # only forward hook is required for evaluation
        # monkey_patch(model.actor, keep_masks["actor"])
        # monkey_patch(model.critic, keep_masks["critic"])
        # monkey_patch(model.value, keep_masks["value"])
        #
        #
        # ret_eval, var_ret, median_ret, d4rl_score, \
        # eval_time, success_rate = evaluate(model, env, state_mean, state_std, action_repeat=1)
        h_info = {}
        for i in range(num_envs):
            dummy = copy.deepcopy(shared_model)
            monkey_patch(dummy.actor, keep_masks["actor"][f'task_{i}'])
            monkey_patch(dummy.critic, keep_masks["critic"][f'task_{i}'])
            monkey_patch(dummy.value, keep_masks["value"][f'task_{i}'])

            ret_eval, var_ret, median_ret, d4rl_score, \
            eval_time, success_rate = evaluate(dummy, env_list[f'task_{i}'], state_mean[f'task_{i}'],
                                               state_std[f'task_{i}'], action_repeat=1)
            times_spent = (time.time() - init_t) / (3600)
            print('\n Time {} ----- task: {} ----- return : {}'.format(times_spent, process_ID, ret_eval))
            info = {'return': ret_eval, 'normalized_score': d4rl_score, 'success_rate': success_rate,
                    'expert_avg_score': env_list[f'task_{i}'].avg_score, 'itr': training_iters, 'time': times_spent}
            _t_name = env_list_[i]
            h_info[f"{_t_name}"] = info

        res_queue.put(h_info)
        if process_ID == 0:
            shared_model.save(filename='weight', directory=file_info['log_dir'])
