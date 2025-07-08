# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/C51rtv3/#c51py
import random
import time
from datetime import datetime
import sys
from tqdm import tqdm
import tyro
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import Args
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import warnings
import os
from collections import namedtuple

load_dotenv(".env")
WANDB_KEY = os.getenv("WANDB_KEY")
warnings.filterwarnings("ignore")
print(
    f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}, device = {Args.device}"
)

# Add the same RuleAugmentedReplayBufferSamples namedtuple as c51_rules_training_v3_ws.py
RuleAugmentedReplayBufferSamples = namedtuple(
    "RuleAugmentedReplayBufferSamples",
    [
        "observations",
        "actions",
        "next_observations",
        "dones",
        "rewards",
        "rule_suggestions",
    ],
)


# Add the RuleAugmentedReplayBuffer class to extend the ReplayBuffer to also store rule-suggested actions
class RuleAugmentedReplayBuffer(ReplayBuffer):
    """Extends ReplayBuffer to also store rule-suggested actions"""

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        handle_timeout_termination=True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            handle_timeout_termination=handle_timeout_termination,
        )
        # Add buffer for rule suggestions - initialize with None (-1 as sentinel value)
        self.rule_suggestions = np.full((self.buffer_size,), None, dtype=object)

    def add(self, obs, next_obs, actions, rewards, dones, infos, rule_suggestions=None):
        # Batch operations instead of loop
        batch_size = len(obs)
        indices = np.arange(self.pos, self.pos + batch_size) % self.buffer_size

        # Use NumPy's vectorized operations
        self.observations[indices] = np.array(obs)
        self.next_observations[indices] = np.array(next_obs)
        self.actions[indices] = np.array(actions)
        self.rewards[indices] = np.array(rewards)
        self.dones[indices] = np.array(dones)

        if rule_suggestions is not None:
            self.rule_suggestions[indices] = [
                rs if rs is not None else [-1] for rs in rule_suggestions
            ]

        self.pos = (self.pos + batch_size) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        """Sample data and also return rule suggestions"""
        # Call the parent's sample method to get the base fields
        batch = super().sample(batch_size)

        # Generate our own batch indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Get rule suggestions for the sampled indices
        rule_suggestions = self.rule_suggestions[batch_inds]
        # Convert -1 back to None for consistency
        rule_suggestions_list = [
            None if (r == -1 or r == [-1]) else r for r in rule_suggestions
        ]

        # Create a new namedtuple with all the fields including rule_suggestions
        return RuleAugmentedReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=batch.rewards,
            rule_suggestions=rule_suggestions_list,
        )


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            env_id,
            params={
                "generation": "random",
                "environment_seed": seed,
                "use_one_hot_vector_states": True,
            },
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.observation_space, spaces.Discrete):
            n = env.observation_space.n
            env.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(n,), dtype=np.float32
            )
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms, device=Args.device))
        self.n = env.single_action_space.n
        self.rule_pmf = self.rule_distribution()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n * n_atoms),
        )

    def rule_distribution(self):
        """
        Hybrid rule distribution: baseline 0.05 everywhere, smooth peak up to 0.5.
        The peak is at the rightmost atom (highest return).
        """
        n = self.n_atoms
        device = self.atoms.device

        # Baseline
        baseline = 0.0
        weights = torch.full((n,), baseline, device=device)

        # Gaussian-like peak at the rightmost atom
        peak_height = 0.8
        peak_pos = n - 1  # rightmost
        peak_width = n // 16  # controls spread; adjust as needed

        # Add the peak
        idxs = torch.arange(n, device=device)
        peak = torch.exp(-0.5 * ((idxs - peak_pos) / peak_width) ** 2)
        peak = peak / peak.max() * (peak_height - baseline)
        weights += peak

        return weights.view(1, 1, n)

    def get_action(
        self, x, stored_rule_actions=None, action=None, skip=False, epsilon=1.0
    ):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        if skip is True:
            q_values = (pmfs * self.atoms).sum(2)
            if action is None:
                action = torch.argmax(q_values, 1)
            return action, pmfs[torch.arange(len(x)), action]

        suggested_actions = (
            [self.env.envs[0].guide_agent()]
            if stored_rule_actions is None
            else stored_rule_actions
        )

        combined_pmfs = pmfs.clone()

        # Create a mask for rule-suggested actions
        rule_mask = torch.zeros(len(x), self.n, device=pmfs.device, dtype=torch.bool)
        for i, actions in enumerate(suggested_actions):
            if actions:
                valid_actions = [act for act in actions if act is not None]
                if valid_actions:
                    rule_mask[i, valid_actions] = True

        # Apply rule influence vectorized
        rule_multiplier = 1 + (epsilon * self.rule_pmf[0, 0])
        combined_pmfs[rule_mask] *= rule_multiplier

        # Renormalize
        combined_pmfs = combined_pmfs / combined_pmfs.sum(dim=2, keepdim=True)

        q_values = (combined_pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, dim=1)

        return action, combined_pmfs[torch.arange(len(x)), action], suggested_actions


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    args = tyro.cli(Args)
    run_name = f"C51rtv3_{args.env_id}__seed={args.seed}__{start_datetime}"
    if args.track:
        import wandb

        # wandb.login(key=WANDB_KEY)
        wandb.tensorboard.patch(root_logdir=f"C51rtv3/runs/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"OfficeWorld-C51rtv3_{args.exploration_fraction}_{args.run_code}",
        )
    writer = SummaryWriter(f"C51rtv3/runs/{run_name}/train")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []
    len_episodes_returns = 0
    print_step = args.print_step

    # TRY NOT TO MODIFY:
    print(
        f"File: {os.path.basename(__file__)}, using seed {args.seed} and exploration fraction {args.exploration_fraction}"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    q_network = QNetwork(
        envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max
    ).to(device)
    optimizer = optim.Adam(
        q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size
    )
    target_network = QNetwork(
        envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = RuleAugmentedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()
    print(
        f"Starting training for {args.total_timesteps} timesteps on {args.env_id} with print_step={print_step}"
    )
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in tqdm(range(args.total_timesteps), colour="green"):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            suggested_actions = envs.envs[0].guide_agent()
            weights = [0.2] * envs.single_action_space.n
            for action in suggested_actions:
                weights[action] = 0.8
            weights = weights / np.sum(weights)
            actions = np.random.choice(
                envs.single_action_space.n, size=envs.num_envs, p=weights
            )
        else:
            actions, pmf, suggested_actions = q_network.get_action(
                torch.Tensor(obs).float().to(device), skip=False, epsilon=epsilon
            )
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                writer.add_scalar("episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("epsilon", epsilon, global_step)
                episodes_returns.append(info["episode"]["r"])
                episodes_lengths.append(info["episode"]["l"])
                if global_step >= print_step:
                    old_len_episodes_returns = len_episodes_returns
                    len_episodes_returns = len(episodes_returns)
                    print_num_eps = len_episodes_returns - old_len_episodes_returns
                    mean_ep_return = np.mean(episodes_returns[-print_num_eps:])
                    mean_ep_lengths = np.mean(episodes_lengths[-print_num_eps:])
                    tot_mean_return = np.mean(episodes_returns)
                    tot_mean_length = np.mean(episodes_lengths)
                    if (print_num_eps >= 3500 and mean_ep_return < 0.1) or (
                        mean_ep_return < -5
                        and global_step / args.total_timesteps >= 0.5
                    ):
                        tqdm.write(
                            f"Stopping early after {(global_step / args.total_timesteps) * 100:.2f}% of timesteps"
                        )
                        tqdm.write(
                            f"global_step={global_step}, mean_return_last_{print_num_eps}_episodes={mean_ep_return}, tot_mean_ret={tot_mean_return}, mean_length_last_{print_num_eps}_episodes={mean_ep_lengths}, tot_mean_len={tot_mean_length}, epsilon={epsilon:.2f}"
                        )
                        sys.exit(0)
                    tqdm.write(
                        f"global_step={global_step}, mean_return_last_{print_num_eps}_episodes={mean_ep_return}, tot_mean_ret={tot_mean_return}, mean_length_last_{print_num_eps}_episodes={mean_ep_lengths}, tot_mean_len={tot_mean_length}, epsilon={epsilon:.2f}"
                    )
                    print_step += args.print_step
                break
            elif dones[0] is True:
                # If the first environment is done, print the info
                tqdm.write(
                    f"global_step={global_step}, info={info}, infos={infos}, rewards={rewards}, dones={dones}, epsilon={epsilon:.2f}"
                )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        # for idx, d in enumerate(dones):
        #     print(f'infos, idx={idx}, d={d}, infos={infos[idx]}, dones={dones}')
        #     if d:
        #         real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos, [suggested_actions])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(
                        data.next_observations.float(), skip=True
                    )
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (
                        1 - data.dones
                    )
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs, _ = q_network.get_action(
                    data.observations.float(),
                    stored_rule_actions=data.rule_suggestions,
                    action=data.actions.flatten(),
                    skip=False,
                    epsilon=epsilon,
                )
                loss = (
                    -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(
                        -1
                    )
                ).mean()

                if global_step % 10000 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # Final print

    old_len_episodes_returns = len_episodes_returns
    len_episodes_returns = len(episodes_returns)
    print_num_eps = len_episodes_returns - old_len_episodes_returns
    mean_ep_return = np.mean(episodes_returns[-print_num_eps:])
    mean_ep_lengths = np.mean(episodes_lengths[-print_num_eps:])
    tot_mean_return = np.mean(episodes_returns)
    tot_mean_length = np.mean(episodes_lengths)
    tqdm.write(
        f"global_step={global_step}, mean_return_last_{print_num_eps}_episodes={mean_ep_return}, tot_mean_ret={tot_mean_return}, mean_length_last_{print_num_eps}_episodes={mean_ep_lengths}, tot_mean_len={tot_mean_length}, epsilon={epsilon:.2f}"
    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.c51_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "C51rtv3",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
