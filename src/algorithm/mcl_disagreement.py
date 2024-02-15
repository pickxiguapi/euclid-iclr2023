import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.mcl_basemodel import BaseModel
import algorithm.helper as h


class Disagreement(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, n_models=5):
        super().__init__()
        self.ensemble = nn.ModuleList([
            nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim),
                          nn.ReLU(), nn.Linear(hidden_dim, obs_dim))
            for _ in range(n_models)
        ])

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        errors = []
        for model in self.ensemble:
            next_obs_hat = model(torch.cat([obs, action], dim=-1))
            model_error = torch.norm(next_obs - next_obs_hat,
                                     dim=-1,
                                     p=2,
                                     keepdim=True)
            errors.append(model_error)

        return torch.cat(errors, dim=1)

    def get_disagreement(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        preds = []
        for model in self.ensemble:
            next_obs_hat = model(torch.cat([obs, action], dim=-1))
            preds.append(next_obs_hat)
        preds = torch.stack(preds, dim=0)
        return torch.var(preds, dim=0).mean(dim=-1)


class DisagreementAgent(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        print("Running agent: MCL Disagreement")
        self.cfg = cfg
        self.obs_dim = cfg.obs_dim
        self.latent_dim = cfg.latent_dim
        self.action_dim = cfg.action_dim
        self.hidden_dim = cfg.mlp_dim
        self.device = cfg.device
        self.lr = cfg.lr
        self.reward_free = cfg.reward_free
        self.use_encoder = cfg.use_encoder

        self.disagreement = Disagreement(self.obs_dim, self.action_dim,
                                         self.hidden_dim).to(self.device)

        # optimizers
        self.disagreement_opt = torch.optim.Adam(
            self.disagreement.parameters(), lr=self.lr)

        self.disagreement.train()

    def update_disagreement(self, obs, action, next_obs, step):
        metrics = dict()

        error = self.disagreement(obs, action, next_obs)

        loss = error.mean()

        self.disagreement_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.disagreement_opt.step()

        metrics['disagreement_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        reward = self.disagreement.get_disagreement(obs, action,
                                                    next_obs).unsqueeze(1)
        return reward

    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.cfg.update_every_steps != 0:
            return metrics

        # sample
        obs, next_obses, action, extr_reward, idxs, weights = replay_buffer.sample()
        next_obs = next_obses[0]
        action = action[0]
        extr_reward = extr_reward[0]

        # avoid similarity representation
        # obs = self.aug_and_encode(obs)
        # with torch.no_grad():
        #     next_obs = self.aug_and_encode(next_obs)

        # update Disagreement
        if self.reward_free:
            metrics.update(
                self.update_disagreement(obs, action, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs,
                                                       step)

            metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        # update model
        metrics.update(self.update_model(obs, action, reward, next_obs, step))

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if not self.use_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        h.ema(self.model.critic, self.model.critic_target, self.cfg.tau)
        if self.cfg.use_encoder:
            h.ema(self.model._encoder, self.model._encoder_target, self.cfg.tau)

        return metrics

