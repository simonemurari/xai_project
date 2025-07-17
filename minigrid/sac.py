# sac_reacher.py
# Adapted SAC script to mirror C51 style for Reacher-v2
import os
import random
import time
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config_sac import Args
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import wandb

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.prod(env.single_observation_space.shape)
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.prod(env.single_observation_space.shape)
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        # rescale
        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

if __name__ == "__main__":
    # load args
    args = tyro.cli(Args)
    # setup names and logging
    start_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    run_name = f"SAC_{args.env_id}__seed={args.seed}__{start_time}"
    # wandb init
    wandb.tensorboard.patch(root_logdir=f"SAC/runs/{run_name}")
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        group=f"SAC_{args.env_id}_seed{args.seed}"
    )
    writer = SummaryWriter(f"SAC/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # seeding
    print(f"File: {os.path.basename(__file__)}, seed={args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "continuous action space only"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    # entropy
    if args.autotune:
        target_entropy = -np.prod(envs.single_action_space.shape).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    obs, _ = envs.reset(seed=args.seed)

    # training loop
    episodes = 0
    print_step = args.print_step
    returns_episode = []
    lengths_episode = []
    for global_step in tqdm(range(int(args.total_timesteps)), desc="Training", colour="blue"):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # record
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ret = info['episode']['r']
                    length = info['episode']['l']
                    returns_episode.append(ret)
                    lengths_episode.append(length)
                    episodes += 1
                    break
        # buffer
        real_next = next_obs.copy()
        for i, t in enumerate(truncations):
            if t:
                real_next[i] = infos['final_observation'][i]
        rb.add(obs, real_next, actions, rewards, terminations, infos)
        obs = next_obs
        # train
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            # compute target
            with torch.no_grad():
                next_a, next_logpi, _ = actor.get_action(data.next_observations)
                q1_next = qf1_target(data.next_observations, next_a)
                q2_next = qf2_target(data.next_observations, next_a)
                min_q_next = torch.min(q1_next, q2_next) - alpha * next_logpi
                target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_q_next.view(-1)
            # q loss
            q1_pred = qf1(data.observations, data.actions).view(-1)
            q2_pred = qf2(data.observations, data.actions).view(-1)
            q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            # delayed policy
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, logpi, _ = actor.get_action(data.observations)
                    q1_pi = qf1(data.observations, pi)
                    q2_pi = qf2(data.observations, pi)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * logpi - min_q_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if args.autotune:
                        with torch.no_grad(): _, logpi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (logpi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
            # target networks
            if global_step % args.target_network_frequency == 0:
                for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
        # logging
        if global_step % print_step == 0 and episodes > 0:
            mean_r = np.mean(returns_episode[-episodes:])
            mean_l = np.mean(lengths_episode[-episodes:])
            print(f"global_step={global_step}, episodes={episodes}, mean_return={mean_r:.2f}, mean_length={mean_l:.2f}")
            episodes = 0
    envs.close()
    writer.close()