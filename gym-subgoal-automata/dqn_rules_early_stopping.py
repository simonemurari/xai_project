# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from tqdm import tqdm
import tyro
import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config_dqn import Args
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import os
import warnings
from dotenv import load_dotenv
load_dotenv(".env")
WANDB_KEY = os.getenv("WANDB_KEY")
warnings.filterwarnings("ignore")
print(
    f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}, device = {Args.device}"
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
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    args = tyro.cli(Args)
    run_name = f"OfficeWorld-DQN_{args.env_id}__seed={args.seed}__{start_datetime}"
    if args.track:
        import wandb

        wandb.tensorboard.patch(root_logdir=f"DQN/runs/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"OfficeWorld-DQN_{args.exploration_fraction}_{args.run_code}",
        )
    writer = SummaryWriter(f"DQN/runs/{run_name}/train")
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

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
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
            q_values = q_network(torch.Tensor(obs).float().to(device))
            suggested_actions = envs.envs[0].guide_agent()
            weights = [0.2] * envs.single_action_space.n
            for action in suggested_actions:
                weights[action] = 0.8
            q_values = q_values * (1 + (epsilon * torch.Tensor(weights).to(device)))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.True
        # print(f"global_step={global_step}, actions={actions}, epsilon={epsilon:.2f}")
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                writer.add_scalar(
                    "episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "episodic_length", info["episode"]["l"], global_step
                )
                writer.add_scalar("epsilon", epsilon, global_step)
                episodes_returns.append(info["episode"]["r"])
                episodes_lengths.append(info["episode"]["l"])
                if global_step >= print_step:
                    old_len_episodes_returns = len_episodes_returns
                    len_episodes_returns = len(episodes_returns)
                    print_num_eps = len_episodes_returns - old_len_episodes_returns
                    mean_ep_return = np.mean(episodes_returns[-print_num_eps :])
                    mean_ep_lengths = np.mean(episodes_lengths[-print_num_eps :])
                    tot_mean_return = np.mean(episodes_returns)
                    tot_mean_length = np.mean(episodes_lengths)
                    if (print_num_eps >= 11000 and mean_ep_return < 0.1) or (mean_ep_return < -5 and global_step / args.total_timesteps >= 0.5):
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

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        # for idx, d in enumerate(dones):
        #     if d:
        #         real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations.float()).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = q_network(data.observations.float()).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 10000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "SPS",
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

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

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
                "DQN",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
