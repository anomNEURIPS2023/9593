import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import logger
from logger import create_stats_ordered_dict
import copy
from torch.distributions import MultivariateNormal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Actor used in BCQ"""
    def __init__(self, state_dim, action_dim, max_action, hidden_dims=[400, 300], threshold=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], action_dim)
        
        self.max_action = max_action
        self.threshold = threshold

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.threshold * self.max_action * torch.tanh(self.l3(a))
        return torch.max(torch.min(self.max_action, a + action), -self.max_action)

class Critic(nn.Module):
    """Regular critic used in off-policy RL"""
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300]):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l5 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l6 = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class VAE(nn.Module):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dims=[750, 750], activation='tanh'):
        super(VAE, self).__init__()
        # encoder
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dims[0])  # self.e1 = nn.Linear(state_dim, 750)
        self.e2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mean = nn.Linear(hidden_dims[1], latent_dim)
        self.log_std = nn.Linear(hidden_dims[1], latent_dim)
        # decoder
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dims[0])  # self.d1 = nn.Linear(latent_dim, 750)
        self.d2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.d3 = nn.Linear(hidden_dims[1], action_dim)  # self.d3 = nn.Linear(750, state_dim)
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.activation = activation
    def forward(self, state, action):  # forward (self, state)
        z = F.relu(self.e1(torch.cat([state, action], 1)))  # z = F.relu(self.e1(state))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device)
        u = self.decode(state, z)
        return u, mean, std
    def decode(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        if self.activation == 'tanh':
            return self.max_action * torch.tanh(self.d3(a))
        elif self.activation == 'relu':
            return torch.max(torch.min(self.max_action, self.d3(a)), -self.max_action)
    def decode_bc(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        if self.activation == 'tanh':
            return self.max_action * torch.tanh(self.d3(a))
        elif self.activation == 'relu':
            return torch.max(torch.min(self.max_action, self.d3(a)), -self.max_action)
    def decode_bc_test(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25, 0.25)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        if self.activation == 'tanh':
            return self.max_action * torch.tanh(self.d3(a))
        elif self.activation == 'relu':
            return torch.max(torch.min(self.max_action, self.d3(a)), -self.max_action)


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim, activation='tanh', cloning=False, discount=0.99, batch_size=100, actor_grad_norm_lambda=0,
                 clip_grad=False):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim['actor']).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim['actor']).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, hidden_dim['critic']).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim['critic']).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dims=hidden_dim['vae'], activation=activation, ).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.use_cloning = cloning
        self.discount = discount
        self.activation = activation
        self.actor_grad_norm_lambda = actor_grad_norm_lambda
        self.clip_grad = clip_grad
        self.alpha = 0.005

    def update_exponential_moving_average(self, target, source, alpha):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

    def policy_loss_(self, state, perturbed_actions, y=None):

        # Update through DPG
        actor_loss = self.critic.q1(state, perturbed_actions).mean()

        return actor_loss

    def sample_action(self, state):
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state, sampled_actions)
        return perturbed_actions

    def select_action(self, state):
        if self.use_cloning:
            return self.select_action_cloning(state)       
        with torch.no_grad():
                state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
                action = self.actor(state, self.vae.decode(state))
                q1 = self.critic.q1(state, action)
                ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def select_action_cloning(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.vae.decode_bc_test(state)
        return action[0].cpu().data.numpy().flatten()

    def compute_normed_loss(self, loss, net):
        ac_grad_norm= []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
                ac_grad_norm.append(p.grad.flatten())
        ac_grad_norm = torch.cat(ac_grad_norm).norm(2).sum() + 1e-12
        loss = loss + self.actor_grad_norm_lambda * ac_grad_norm
        return loss

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, using_snip=False):
        for it in range(iterations):
            # print ('Iteration : ', it)
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)


            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            if self.actor_grad_norm_lambda:
                vae_loss = self.compute_normed_loss(vae_loss, self.vae)
            elif self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), .25)
            self.vae_optimizer.step()

            # Critic Training
            with torch.no_grad():
                # Duplicate state 10 times
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)
                
                # Compute value of perturbed actions sampled from the VAE
                if self.use_cloning:
                    target_Q1, target_Q2 = self.critic_target(state_rep, self.vae.decode(state_rep))
                else:
                    target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep, self.vae.decode(state_rep)))

                # Soft Clipped Double Q-learning 
                target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

                target_Q = reward + done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)



            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.actor_grad_norm_lambda:
                critic_loss = self.compute_normed_loss(critic_loss, self.critic)
            elif self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .25)
            self.critic_optimizer.step()


            # Pertubation Model / Action Training
            with torch.no_grad():
                sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)
            action_divergence = ((sampled_actions - perturbed_actions)**2).sum(-1)

            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.actor_grad_norm_lambda:
                actor_loss = self.compute_normed_loss(actor_loss, self.actor)
            elif self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .25)
            self.actor_optimizer.step()

            if not using_snip:
                # Update Target Networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename)))


# -------------------- IQL ------------------
from torch.optim.lr_scheduler import CosineAnnealingLR
class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, min_action, activation, hidden_dim=256, n_hidden=2):
        super().__init__()

        self.activation = activation
        if activation == 'relu':
            self.latent = mlp([obs_dim, *([hidden_dim] * n_hidden), hidden_dim], output_activation=nn.ReLU)
            self.out_layer = mlp([hidden_dim, *([hidden_dim] * 1), act_dim], output_activation=None)
        elif activation == 'tanh':
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim], output_activation=nn.Tanh)

        self.max_action = max_action
        self.min_action = min_action
    def forward(self, obs):
        if self.activation == 'tanh':
            return torch.max(torch.min(self.max_action, self.net(obs)), self.min_action)
        elif self.activation == 'relu':
            return torch.max(torch.min(self.max_action, self.out_layer(self.latent(obs))), self.min_action)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()


class IQL(object):
    def __init__(self, state_dim, action_dim, max_action, min_action, hidden_dim=None, activation='tanh', discount=0.99, batch_size=256,
                 actor_grad_norm_lambda=0, max_steps=1000000, policy_type='gaussian', clip_grad=False):
        self.policy_type = policy_type
        if policy_type == 'gaussian':
            self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim['actor']).to(device)
            self.actor_target = copy.deepcopy(self.actor).requires_grad_(False).to(device)
        elif policy_type == 'deterministic':
            self.actor = DeterministicPolicy(state_dim, action_dim, max_action, min_action, activation=activation, hidden_dim=hidden_dim['actor']).to(device)
            self.actor_target = copy.deepcopy(self.actor).requires_grad_(False).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim, hidden_dims=hidden_dim['critic']).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.value = ValueFunction(state_dim, hidden_dim['value']).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters())
        self.max_action = max_action
        self.action_dim = action_dim
        self.policy_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.discount = discount
        self.beta = 3.0
        self.EXP_ADV_MAX = 100.
        self.alpha = 0.005
        self.tau = 0.7
        self.actor_grad_norm_lambda = actor_grad_norm_lambda
        self.clip_grad = clip_grad

    def policy_loss_(self, state, perturbed_actions, y=None):
        # Update through DPG
        actor_loss = self.critic.q1(state, perturbed_actions).mean()
        return actor_loss

    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        if self.policy_type == 'deterministic':
            return self.actor(obs).cpu().data.numpy().flatten()
        else:
            dist = self.actor(obs)
            return dist.mean.cpu().detach().numpy().flatten() if deterministic else dist.sample().cpu().detach().numpy().flatten()

    def update_exponential_moving_average(self, target, source, alpha):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

    def asymmetric_l2_loss(self, u, tau):
        return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)

    def compute_normed_loss(self, loss, net):
        ac_grad_norm= []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
                ac_grad_norm.append(p.grad.flatten())
        ac_grad_norm = torch.cat(ac_grad_norm).norm(2).sum() + 1e-12
        loss = loss + self.actor_grad_norm_lambda * ac_grad_norm
        return loss

    def train(self, replay_buffer, iterations, batch_size=256, discount=0.99, tau=0.005, using_snip=False):
        for it in range(iterations):
            state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))
                next_v = self.value(next_state).reshape(-1,1)

            # Update value function
            v = self.value(state)
            adv = target_q - v
            v_loss = self.asymmetric_l2_loss(adv, self.tau)
            self.value_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 40)
            self.value_optimizer.step()

            # Update Q function
            true_Q = reward + done * discount * next_v.detach()
            current_Q1, current_Q2 = self.critic(state, action)
            q_loss = (F.mse_loss(current_Q1, true_Q) + F.mse_loss(current_Q2, true_Q))/2
            self.critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40)
            self.critic_optimizer.step()

            # Update policy
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.EXP_ADV_MAX)
            policy_out = self.actor(state)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(action)
                #bc_losses = bc_losses.sum(-1)
            elif torch.is_tensor(policy_out):
                assert policy_out.shape == action.shape
                bc_losses = torch.sum((policy_out - action) ** 2, dim=1)
            else:
                raise NotImplementedError
            actor_loss = torch.mean(exp_adv * bc_losses)


            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
            self.actor_optimizer.step()
            self.policy_lr_schedule.step()


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.value.state_dict(), '%s/%s_value.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.value.load_state_dict(torch.load('%s/%s_value.pth' % (directory, filename)))




