# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config_dqn import Args
import minigrid
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
load_dotenv(".env")
WANDB_KEY = os.getenv("WANDB_KEY")
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(
    f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}, device = {Args.device}"
)


def make_env(env_id, seed, n_keys, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, n_keys=n_keys)
            env = gym.wrappers.FlattenObservation(
                gym.wrappers.FilterObservation(env, filter_keys=["image", "direction"])
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

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
        self.n = env.single_action_space.n

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"DQNbase_{args.env_id}__{args.exp_name}__{args.seed}__{start_datetime}"
    if args.track:
        import wandb

        # wandb.login(key=WANDB_KEY)
        wandb.tensorboard.patch(root_logdir=f"DQNbase/runs/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"DQNbase_{args.exploration_fraction}_{args.run_code}",
        )
    writer = SummaryWriter(f"DQNbase/runs/{run_name}/train")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []
    len_episodes_returns = 0

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, args.n_keys, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
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
        handle_timeout_termination=False,
    )
    start_time = time.time()
    print_step = args.print_step

    print(
        f"Starting training for {args.total_timesteps} timesteps on {args.env_id} with {args.n_keys} keys, with print_step={print_step}"
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in tqdm(range(args.total_timesteps), colour="green"):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar(
                        "episodic_return", float(info["episode"]["r"][0]), global_step
                    )
                    writer.add_scalar(
                        "episodic_length", float(info["episode"]["l"][0]), global_step
                    )
                    episodes_returns.append(float(info["episode"]["r"][0]))
                    episodes_lengths.append(float(info["episode"]["l"][0]))
                    if global_step >= print_step:
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
                        print_step += args.print_step

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations.float()).max(
                        dim=1
                    )
                    td_target = data.rewards.flatten() + args.gamma * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = (
                    q_network(data.observations.float())
                    .gather(1, data.actions)
                    .squeeze()
                )
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

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data
                        + (1.0 - args.tau) * target_network_param.data
                    )

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
