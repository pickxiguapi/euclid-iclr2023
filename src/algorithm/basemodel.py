import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.policy = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)

        self.apply(h.orthogonal_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = h.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)

        self.apply(h.orthogonal_init)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q1 = self._Q1(x)
        q2 = self._Q2(x)

        return q1, q2


class TOLD(nn.Module):
    """TOLD Model"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda')
        # encoder
        if cfg.use_encoder:
            self._encoder = h.enc(cfg)
            self._encoder_target = h.enc(cfg)
            self._encoder_target.load_state_dict(self._encoder.state_dict())
        else:
            self._encoder = nn.Identity()
            self._encoder_target = nn.Identity()

        # dynamic model
        self._dynamics = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)

        # reward model
        self._reward = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)

        # actor
        self.actor = Actor(cfg).to(self.device)

        # critic
        self.critic = Critic(cfg).to(self.device)
        self.critic_target = Critic(cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self.critic._Q1, self.critic._Q2]:
            h.set_requires_grad(m, enable)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def encoder(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def encoder_target(self, obs):
        """Target encoder"""
        return self._encoder_target(obs)

    def train(self, training=True):
        self.training = training
        if self.cfg.use_encoder:
            self._encoder.train(training)
            self._encoder_target.train(training)
        self._dynamics.train(training)
        self._reward.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'encoder': self._encoder.state_dict(),
                'encoder_target': self._encoder_target.state_dict(),
                'dynamic': self._dynamics.state_dict(),
                'reward': self._reward.state_dict(),
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict()}

    def load(self, d):
        self._encoder.load_state_dict(d['encoder'])
        self._encoder_target.load_state_dict(d['encoder_target'])
        self._dynamics.load_state_dict(d['dynamic'])
        self._reward.load_state_dict(d['reward'])
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        self.critic_target.load_state_dict(d['critic_target'])

    def load_except_critic(self, d):
        self._encoder.load_state_dict(d['encoder'])
        self._encoder_target.load_state_dict(d['encoder_target'])
        self._dynamics.load_state_dict(d['dynamic'])
        self._reward.load_state_dict(d['reward'])
        self.actor.load_state_dict(d['actor'])

    def load_except_critic_and_actor(self, d):
        self._encoder.load_state_dict(d['encoder'])
        self._encoder_target.load_state_dict(d['encoder_target'])
        self._dynamics.load_state_dict(d['dynamic'])
        self._reward.load_state_dict(d['reward'])

    def load_except_model(self, d):
        self._encoder.load_state_dict(d['encoder'])
        self._encoder_target.load_state_dict(d['encoder_target'])
        self._reward.load_state_dict(d['reward'])
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        self.critic_target.load_state_dict(d['critic_target'])


class BaseModel(object):
    """Base Model"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.stddev_schedule = 0.2
        self.aug = nn.Identity()
        self.std = h.linear_schedule(cfg.std_schedule, 0)

        self.model = TOLD(cfg).cuda()

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model.actor.parameters(), lr=self.cfg.lr)

        self.model.train()

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.model.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        if not self.cfg.use_critic:
            if not self.cfg.use_actor:
                self.model.load_except_critic_and_actor(d)
            else:
                self.model.load_except_critic(d)
        else:
            if not self.cfg.use_model:
                self.model.load_except_model(d)
            else:
                self.model.load(d)

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            z = self.model.encoder(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                dist = self.model.actor(z, self.cfg.min_std)
                pi_actions[t] = dist.sample(clip=0.3)
                z, _ = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.encoder(obs).repeat(self.cfg.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2 * torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device),
                                  -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                        score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.critic(z, self.model.actor(z, self.cfg.min_std).sample(clip=0.3)))
        return G

    @torch.no_grad()
    def act_pi(self, obs, step, eval_mode=False):
        # Select action using policy pi
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.model.encoder(obs)

        stddev = h.schedule(self.stddev_schedule, step)
        dist = self.model.actor(z, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.cfg.seed_steps:
                action.uniform_(-1, 1)
        return action

    def update_critic(self, obs, action, reward, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = h.schedule(self.stddev_schedule, step)
            dist = self.model.actor(next_obs, stddev)
            next_action = dist.sample(clip=0.3)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.cfg.discount * target_V)

        Q1, Q2 = self.model.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.mean().item()

        # optimize critic
        self.optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optim.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        # freeze critic
        self.model.track_q_grad(False)

        stddev = h.schedule(self.stddev_schedule, step)
        dist = self.model.actor(obs, stddev)
        action = dist.sample(clip=0.3)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.model.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.pi_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.pi_optim.step()

        # unfreeze critic
        self.model.track_q_grad(True)

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.model.encoder(obs)

    def update_model(self, obs, action, reward, next_obs, step):
        metrics = dict()
        # Representation
        z = self.model.encoder(self.aug(obs))
        # Predictions
        next_z_pred, reward_pred = self.model.next(z, action)
        with torch.no_grad():
            next_z = self.model.encoder_target(self.aug(next_obs))

        # Losses
        rho = self.cfg.rho
        consistency_loss = rho * torch.mean(h.mse(next_z_pred, next_z), dim=1, keepdim=True)
        reward_loss = rho * h.mse(reward_pred, reward)

        # Optimize model
        total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + self.cfg.reward_coef * reward_loss.clamp(max=1e4)
        weighted_loss = total_loss.mean()
        self.optim.zero_grad(set_to_none=True)
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        metrics['consistency_loss'] = float(consistency_loss.mean().item())
        metrics['reward_loss'] = float(reward_loss.mean().item())
        metrics['total_loss'] = float(total_loss.mean().item())
        metrics['grad_norm'] = float(grad_norm)

        return metrics

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.actor(z, self.cfg.min_std).sample(clip=0.3)
            Q = torch.min(*self.model.critic(z, a))
            pi_loss += -Q.mean() * (self.cfg.rho ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.encoder(next_obs)
        td_target = reward + self.cfg.discount * \
                    torch.min(*self.model.critic_target(next_z, self.model.actor(next_z, self.cfg.min_std).sample(clip=0.3)))
        return td_target

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        metrics = dict()

        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Representation
        z = self.model.encoder(self.aug(obs))
        zs = [z.detach()]

        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.cfg.horizon):
            # Predictions
            Q1, Q2 = self.model.critic(z, action[t])
            z, reward_pred = self.model.next(z, action[t])
            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                next_z = self.model.encoder_target(next_obs)
                td_target = self._td_target(next_obs, reward[t])
            zs.append(z.detach())

            # Losses
            rho = (self.cfg.rho ** t)
            consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
            reward_loss += rho * h.mse(reward_pred, reward[t])
            value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
            priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

        # Optimize model
        total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                     self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                     self.cfg.value_coef * value_loss.clamp(max=1e4)
        weighted_loss = (total_loss * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy
        pi_loss = self.update_pi(zs)

        # update critic target
        if step % self.cfg.update_freq == 0:
            # update critic target
            h.ema(self.model.critic, self.model.critic_target, self.cfg.tau)
            if self.cfg.use_encoder:
                h.ema(self.model._encoder, self.model._encoder_target, self.cfg.tau)

        self.model.eval()

        metrics['consistency_loss'] = float(consistency_loss.mean().item())
        metrics['reward_loss'] = float(reward_loss.mean().item())
        metrics['value_loss'] = float(value_loss.mean().item())
        metrics['pi_loss'] = pi_loss
        metrics['total_loss'] = float(total_loss.mean().item())
        metrics['weighted_loss'] = float(weighted_loss.mean().item())
        metrics['grad_norm'] = float(grad_norm)

        return metrics
