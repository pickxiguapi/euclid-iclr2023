import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env

from algorithm.helper import Episode, ReplayBuffer
import logger

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'pretrain_cfgs', 'pretrain_logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video, task):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video: video.init(env, enabled=(i == 0))
        while not done:
            action = agent.act_pi(obs, step=step, eval_mode=True)
            if task == "humanoid-stand":
                action = torch.clamp(action, min=-0.95, max=0.95)
            obs, reward, done, _ = env.step(action.cpu().detach().numpy())
            ep_reward += reward
            if video: video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)


def train(cfg):
    """Pretraining script for EUCLID. Requires a CUDA-enabled device. Reward free."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    task = cfg.task
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, buffer = make_env(cfg), ReplayBuffer(cfg)
    if cfg.agent == 'disagreement':
        from algorithm.disagreement import DisagreementAgent
        agent = DisagreementAgent(cfg)
    else:
        from algorithm.base import BaseModel
        agent = BaseModel(cfg)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):

        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.act_pi(obs, step=step, eval_mode=False)
            if task == "humanoid-stand":  # avoid boundary bugs
                action = torch.clamp(action, min=-0.95, max=0.95)
            obs, reward, done, _ = env.step(action.cpu().detach().numpy())
            episode += (obs, action, reward, done)
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video, task)
            L.log(common_metrics, category='eval')

        if cfg.save_model and env_step in cfg.snapshot:
            L.save_snapshot(agent, env_step)

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
