import collections
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import npf_utils
from agent import Agent
import utils
from npf_utils import unhook, apply_prune_mask, monkey_patch
import hydra

def tie_weights(src, trg):
    assert type(src) == type(trg)
    if hasattr(src, "weight"):
        trg.weight = src.weight
    if hasattr(src, "bias"):
        trg.bias = src.bias

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def get_grads(model, idx=0, layer_wise=False):
    """ if layer_wise, return will be a vector of norm-grad of each layer,
    otherwise will return a scaler of total grad_norm"""
    if layer_wise:
        grads = []
        for layer in model.modules():
            if isinstance(layer, npf_utils.EnsembleLinearLayer):
                grads.append(np.float32((layer.weight.grad[idx, ...].norm(2) + 1e-12).detach().cpu().numpy())) # grad_norm += grad.norm(2, dim=(1, 2)) + 1e-12
        return np.sum(grads), grads
    else:
        grads = 0.0
        for layer in model.modules():
            if isinstance(layer, npf_utils.EnsembleLinearLayer):
                grads += (layer.weight.grad[idx, ...].norm(2) + 1e-12).detach().cpu().numpy() # grad_norm += grad.norm(2, dim=(1, 2)) + 1e-12
        return np.float32(grads)



def transfer_grad(model, ensembled_model, cat = 'sum'):
    '''compute avg grad and transfer to `model`

    '''

    l = 0
    if cat == 'stochastic':
        n_tasks = list(ensembled_model.modules())[-1].num_members
        selected_task = np.random.randint(0, n_tasks)
    for layer, en_layer in zip(model.modules(), ensembled_model.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            num_tasks = en_layer.weight.shape[0]
            # sum
            if cat == 'sum':
                wgrads = torch.sum(en_layer.weight.grad * en_layer.weight_mask, dim=0)
                bgrads = torch.sum(en_layer.bias.grad, dim=0)
            # mean
            elif cat == 'mean':
                wgrads = torch.mean(en_layer.weight.grad*en_layer.weight_mask, dim=0) #TODO: important to mask grad
                bgrads = torch.mean(en_layer.bias.grad, dim=0)
            # euclidean distance
            elif cat == 'ED':
                wgrads = (en_layer.weight.grad * en_layer.weight_mask).pow(2).sum(0).sqrt()
                bgrads = (en_layer.bias.grad).pow(2).sum(0).sqrt()
            elif cat =='pmean': # proper mean
                div = en_layer.weight_mask[1:].sum(0)
                div[div == 0] = 1
                wgrads = torch.sum(en_layer.weight.grad * en_layer.weight_mask, dim=0) / div
                bgrads = torch.sum(en_layer.bias.grad, dim=0)
            elif cat == 'stochastic':
                wgrads = (en_layer.weight.grad * en_layer.weight_mask)[selected_task]
                bgrads = en_layer.bias.grad[selected_task]
            elif cat == 'true_stochastic':
                # common (en_layer.weight_mask.sum(0)==1).sum()
                common_mask = (en_layer.weight_mask.sum(0) == 1).repeat(num_tasks, 1, 1)
                random_mask = (en_layer.weight_mask.sum(0) > 1).repeat(num_tasks, 1, 1)*(torch.rand(en_layer.weight_mask.shape) > 0.5).to('cuda')
                wgrads = (en_layer.weight.grad * common_mask + en_layer.weight.grad * random_mask).sum(0)
                bgrads = torch.mean(en_layer.bias.grad, dim=0)
                # the common masks + ( mask wheres multiple weights active) * ( random activation 0 or 1 )
            en_layer.weight.grad = copy.deepcopy(wgrads.unsqueeze(0).repeat(num_tasks, 1, 1))
            en_layer.bias.grad = copy.deepcopy(bgrads.unsqueeze(0).repeat(num_tasks, 1, 1))
            layer.weight.grad = copy.deepcopy(wgrads.T)
            layer.bias.grad = copy.deepcopy(bgrads.flatten())
            l += 1



def transfer_weight_at_init(model, ensembled_model):
    '''transfer actor weight to ensemble model at it's initialization

    '''
    for layer, en_layer in zip(model.parameters(), ensembled_model.parameters()):
        num_tasks = en_layer.shape[0]
        if len(layer.data.shape) == 1: # bias
            en_layer.data.copy_(layer.data.unsqueeze(0).repeat(num_tasks, 1, 1))
        else: # weight
            en_layer.data.copy_(layer.data.transpose(0, 1).unsqueeze(0).repeat(num_tasks, 1, 1))

    # sanity check if all tranfered
    for layer, en_layer in zip(model.parameters(), ensembled_model.parameters()):
        num_tasks = en_layer.shape[0]
        if len(layer.data.shape) == 1: # bias
            assert (en_layer.data != layer.data.unsqueeze(0).repeat(num_tasks, 1, 1)).sum() == 0
        else: # weight
            assert (en_layer.data != layer.data.transpose(0, 1).unsqueeze(0).repeat(num_tasks, 1, 1)).sum() == 0

def sync_model_weight(model, ensembled_model):
    '''transfer actor weight to ensemble model at it's initialization

    '''
    for layer, en_layer in zip(model.parameters(), ensembled_model.parameters()):
        if len(layer.data.shape) == 1:
            layer.data.copy_(en_layer.data[0].squeeze(0))
        else:
            layer.data.copy_(en_layer.data[0].transpose(0, 1))




def sanity_check(model, ensembled_model):
    for layer, en_layer in zip(model.modules(), ensembled_model.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            num_tasks = en_layer.weight.shape[0]
            assert torch.sum(en_layer.weight.data) == torch.sum(layer.weight.transpose(0, 1).unsqueeze(0).repeat(num_tasks, 1, 1))
            # check if no weight is set to zero
            if not torch.sum(en_layer.weight.data != 0) == en_layer.weight.shape[0] * en_layer.weight.shape[1] * en_layer.weight.shape[2]:
                print(f'SANITY CHECK ENSEMBLED-LAYER: found 0 in {torch.sum(en_layer.weight.data == 0)} / {en_layer.weight.shape[0] * en_layer.weight.shape[1] * en_layer.weight.shape[2]}')
            if not torch.sum(layer.weight.data != 0) == layer.weight.shape[0] * layer.weight.shape[1]:
                print(f'SANITY CHECK ENSEMBLED-LAYER: found 0 in {torch.sum(layer.weight.data == 0)}/{layer.weight.shape[0] * layer.weight.shape[1]}')

def check_if_weights_copied_to_ensemble_properly(model, ensembled_model):
    for layer, en_layer in zip(model.parameters(), ensembled_model.parameters()):
        num_tasks = en_layer.shape[0]
        for task in range(num_tasks):
            try:
                assert torch.sum(en_layer.data[task, ...]) == torch.sum(layer.data.T)
            except:
                pass

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, ensemble_critic_cfg, ensemble_actor_cfg,
                 num_tasks, grad_update_rule, keep_ratio,
                 optimization_type='adamW',
                 weight_decay=0, activate_task_encoder=False):
        super().__init__()

        self.num_tasks = num_tasks
        self.grad_update_rule = grad_update_rule
        self.keep_ratio = keep_ratio
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.critic = critic_cfg.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = actor_cfg.to(self.device)
        # initialize all the ensemble model as the main model
        # ensemble critic:
        self.ensemble_critic = ensemble_critic_cfg.to(self.device)
        transfer_weight_at_init(self.critic, self.ensemble_critic)
        self.ensemble_critic_target = copy.deepcopy(self.ensemble_critic)
        self.ensemble_critic_target.load_state_dict(self.ensemble_critic.state_dict())
        # ensemble actor:
        self.ensemble_actor = ensemble_actor_cfg.to(self.device)
        transfer_weight_at_init(self.actor, self.ensemble_actor)
        self.log_alpha = torch.tensor(np.repeat(np.log(init_temperature), self.num_tasks).reshape(-1, 1)).to(self.device).to(dtype=torch.float32)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim
        self.optimization_type = optimization_type
        self.alpha_betas = alpha_betas
        self.alpha_lr = alpha_lr
        self.weight_decay = weight_decay
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_betas = actor_betas
        self.critic_betas = critic_betas
        # initialize optimizer
        self.init_optim()

    def init_optim(self):
        assert self.optimization_type in ['adam', 'adamW']
        if self.optimization_type == 'adam':
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas, weight_decay=self.weight_decay)
            # optimizers
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)
            # optimizer for ensembles
            self.ensemble_actor_optimizer = torch.optim.Adam(self.ensemble_actor.parameters(), lr=self.actor_lr, betas=self.actor_betas, weight_decay=self.weight_decay)
            self.ensemble_critic_optimizer = torch.optim.Adam(self.ensemble_critic.parameters(), lr=self.critic_lr, betas=self.critic_betas, weight_decay=self.weight_decay)

        elif self.optimization_type == 'adamW':
            self.log_alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas)
            # optimizers
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)
            # optimizer for ensembles
            self.ensemble_actor_optimizer = torch.optim.AdamW(self.ensemble_actor.parameters(), lr=self.actor_lr, betas=self.actor_betas)
            self.ensemble_critic_optimizer = torch.optim.AdamW(self.ensemble_critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)



        self.train()
        self.critic_target.train()
        self.ensemble_critic.train()

    def copy_conv_weights_from(self, source, target):
        for src, trg in zip(source, target):  # type: ignore[call-overload]
            tie_weights(src=src, trg=trg)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.ensemble_actor.train(training)
        self.ensemble_critic.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def compute_norm(self, loss, net):
        grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True, grad_outputs=torch.ones_like(loss))
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.norm(2, dim=(1, 2)) + 1e-12
        return grad_norm
    def norm_penalty(self, loss, net):
        loss = loss + 10 * self.compute_norm(loss, net)
        return loss

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def multiobs_act(self, obs, sample=False):
        # TODO: https://github.com/facebookresearch/mbrl-lib/blob/9b7534d4186225d9780f62909b9d2353f95af778/mbrl/models/gaussian_mlp.py#L262
        # https://github.com/facebookresearch/mbrl-lib/blob/9b7534d4186225d9780f62909b9d2353f95af778/mbrl/models/gaussian_mlp.py#L185
        # forward func: https://github.com/facebookresearch/mbrl-lib/blob/9b7534d4186225d9780f62909b9d2353f95af778/mbrl/models/gaussian_mlp.py#L140
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(1) # convert (num_task, obs_dim) --> ((num_task, batch_size, obs_dim)) , here batch_size=1
        dist = self.ensemble_actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return utils.to_np(action)

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()
        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if step != 0:
            logger.log('train_critic/train_critic_grad_norm', self.compute_norm(critic_loss, self.critic).detach().cpu().numpy(),step)
        logger.log('train_critic/loss', critic_loss.detach().cpu().numpy(), step)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        logger.log('train_actor/loss', actor_loss.detach().cpu().numpy(), step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean().detach().cpu().numpy(), step)
        # optimize the actor
        self.actor_optimizer.zero_grad()
        if step != 0:
            logger.log('train_actor/train_actor_grad_norm', self.compute_norm(actor_loss, self.actor).detach().cpu().numpy(),step)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.log(logger, step)
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss.detach().cpu().numpy(), step)
            logger.log('train_alpha/value', self.alpha.detach().cpu().numpy(), step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
        logger.log('train/batch_reward', reward.mean(), step)

    def update_init_configure(self, replay_buffer, itr=1, task=0):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size*itr)
        # ----- update critic -----
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha[task].detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()
        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ----  update actor  -----
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha[task].detach() * log_prob - actor_Q).mean()
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def sanity_check(self):
        print('checking actor')
        sanity_check(self.actor, self.ensemble_actor)
        print('checking critic')
        sanity_check(self.critic, self.ensemble_critic)

    def update_venv(self, replay_buffer, logger, step, norm_reg=False, clip_grad=None):
        # (num_tasks, batch_size, obs_dim)
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
        # ----- update critic -----
        dist = self.ensemble_actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        with torch.no_grad():
            target_Q1, target_Q2 = self.ensemble_critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - (self.alpha.detach().expand(-1, self.batch_size).unsqueeze(-1) * log_prob)
            target_Q = reward + ((not_done * self.discount) * target_V)
            target_Q = target_Q.detach()
        # get current Q estimates
        current_Q1, current_Q2 = self.ensemble_critic(obs, action)
        critic_loss = (current_Q1 - target_Q).pow(2).mean(dim=(1, 2)) + (current_Q2 - target_Q).pow(2).mean(dim=(1, 2))
        # Optimize the critic
        if norm_reg: critic_loss = self.norm_penalty(critic_loss, self.ensemble_critic)

        self.ensemble_critic_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward(torch.ones_like(critic_loss))
        if self.keep_ratio != 1:
            transfer_grad(self.critic.Q1, self.ensemble_critic.Q1, cat=self.grad_update_rule)
            transfer_grad(self.critic.Q2, self.ensemble_critic.Q2, cat=self.grad_update_rule)
        if clip_grad!=None:
            torch.nn.utils.clip_grad_norm_(self.ensemble_critic.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), clip_grad)
        self.ensemble_critic_optimizer.step()
        """Chcek Ensemble Weights: At this point all the weights in ensemble model should be same"""
        if self.keep_ratio != 1:
            self.critic_optimizer.step()

        if step % self.actor_update_frequency == 0:
            # ----  update actor  -----
            dist = self.ensemble_actor(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_Q1, actor_Q2 = self.ensemble_critic(obs, action)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach().expand(-1, self.batch_size).unsqueeze(-1) * log_prob - actor_Q).mean(dim=(1, 2))
            # optimize the actor
            if norm_reg: actor_loss = self.norm_penalty(actor_loss, self.ensemble_actor)
            self.ensemble_actor_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            actor_loss.backward(torch.ones_like(actor_loss))
            if self.keep_ratio != 1:
                transfer_grad(self.actor, self.ensemble_actor, cat=self.grad_update_rule)
            if clip_grad != None:
                torch.nn.utils.clip_grad_norm_(self.ensemble_actor.parameters(), clip_grad)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), clip_grad)
            self.ensemble_actor_optimizer.step()
            if self.keep_ratio != 1:
                self.actor_optimizer.step()

            if self.learnable_temperature:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (self.alpha.expand(-1, self.batch_size).unsqueeze(-1) * (-log_prob - self.target_entropy).detach()).mean(dim=(1, 2))
                alpha_loss.backward(torch.ones_like(alpha_loss))
                self.log_alpha_optimizer.step()

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.ensemble_critic, self.ensemble_critic_target, self.critic_tau)
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)




    def pretrain_venv(self, replay_buffer, logger, step, iterative_pruning):
        # (num_tasks, batch_size, obs_dim)
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
        # ----- update critic -----
        dist = self.ensemble_actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        with torch.no_grad():
            target_Q1, target_Q2 = self.ensemble_critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - (self.alpha.detach().expand(-1, self.batch_size).unsqueeze(-1) * log_prob)
            target_Q = reward + ((not_done * self.discount) * target_V)
            target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.ensemble_critic(obs, action)
        critic_loss = (current_Q1 - target_Q).pow(2).mean(dim=(1, 2)) + (current_Q2 - target_Q).pow(2).mean(dim=(1, 2))


        self.ensemble_critic_optimizer.zero_grad()
        # if some of the task-weights are freezed
        if False in iterative_pruning:
            for idx, cr_loss in enumerate(critic_loss):
                # critic update if still weight are updating
                if iterative_pruning[idx]:
                    if idx != (self.num_tasks-1):
                        cr_loss.backward(retain_graph=True)
                    else:
                        cr_loss.backward()
        # if updating for all tasks
        else:
            critic_loss.backward(torch.ones_like(critic_loss))
        self.ensemble_critic_optimizer.step()
        # ----  update actor  -----
        if step % self.actor_update_frequency == 0:

            dist = self.ensemble_actor(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_Q1, actor_Q2 = self.ensemble_critic(obs, action)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach().expand(-1, self.batch_size).unsqueeze(-1) * log_prob - actor_Q).mean(dim=(1, 2))


            self.ensemble_actor_optimizer.zero_grad()
            # if some of the task-weights are freezed
            if False in iterative_pruning:
                for idx, ac_loss in enumerate(actor_loss):
                    # update if still weight are updating
                    if iterative_pruning[idx]:
                        if idx != (self.num_tasks - 1):
                            ac_loss.backward(retain_graph=True)
                        else:
                            ac_loss.backward()
            # if updating for all tasks
            else:
                actor_loss.backward(torch.ones_like(actor_loss))
            self.ensemble_actor_optimizer.step()

            if self.learnable_temperature:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (self.alpha.expand(-1, self.batch_size).unsqueeze(-1) * (-log_prob - self.target_entropy).detach()).mean(dim=(1, 2))
                # if some of the task-weights are freezed
                if False in iterative_pruning:
                    for idx, al_loss in enumerate(alpha_loss):
                        # update if still weight are updating
                        if iterative_pruning[idx]:
                            if idx != (self.num_tasks - 1):
                                al_loss.backward(retain_graph=True)
                            else:
                                al_loss.backward()
                # if updating for all tasks
                else:
                    alpha_loss.backward(torch.ones_like(alpha_loss))
                self.log_alpha_optimizer.step()

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.ensemble_critic, self.ensemble_critic_target, self.critic_tau)
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(torch.load('%s/critic_%s.pt' % (model_dir, step)))
