# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51py
import os
import random
import time
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Args
from tqdm import tqdm
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Base C51 algorithm 

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env, filter_keys=['image']))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n
        self.network = nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, self.n * n_atoms)
        )

    def get_action(self, x, action=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"C51_{args.env_id}__seed={args.seed}__{start_datetime}"
    if args.track:
        import wandb

        wandb.tensorboard.patch(root_logdir=f"C51/runs/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group="C51"
        )
    writer = SummaryWriter(f"C51/runs/{run_name}/train")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    print_step = args.print_step
    print(f"Starting training for {args.total_timesteps} timesteps on {args.env_id}, with print_step={print_step}")
    for global_step in tqdm(range(args.total_timesteps), colour='green'):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, pmf = q_network.get_action(torch.Tensor(obs).float().to(device))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    if global_step >= print_step:
                        tqdm.write(f"global_step={global_step}, episodic_return={info['episode']['r'][0]}, episodic_length={info['episode']['l'][0]}, exploration_rate={epsilon}")
                        print_step += args.print_step
                    writer.add_scalar("episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("episodic_length", info["episode"]["l"], global_step)
                    episodes_returns.append(info["episode"]["r"])

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
                    _, next_pmfs = target_network.get_action(data.next_observations.float())
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
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
                _, old_pmfs = q_network.get_action(data.observations.float(), data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                if global_step % 10000 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    plt.plot(episodes_returns)
    plt.title(f'C51 on {args.env_id} - Return over {args.total_timesteps} timesteps')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    path = f'C51/{args.env_id}_c51_{args.total_timesteps}_{start_datetime}'
    if not os.path.exists("C51/"):
        os.makedirs("C51/")
    os.makedirs(path)
    plt.savefig(f"{path}/{args.env_id}_c51_{args.total_timesteps}_{start_datetime}.png")
    plt.close()
    with open(f"{path}/c51_args.txt", "w") as f:
        for key, value in vars(args).items():
            if key == "env_id":
                f.write("# C51 Algorithm specific arguments\n")
            if key == "sigmoid_shift" or key == "sigmoid_scale" or key == "distribution":
                continue
            f.write(f"{key}: {value}\n")

    if args.save_model:
        model_path = f"{path}/c51_model.pt"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        from baseC51.c51_eval import evaluate
        eval_episodes=100000
        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=eval_episodes,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0
        )
        writer = SummaryWriter(f"C51/runs/{run_name}/eval")
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("episodic_return", episodic_return, idx)

        plt.figure()
        plt.plot(episodic_returns)
        plt.title(f'C51Eval on {args.env_id} - Return over {eval_episodes} episodes')
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(f"{path}/{args.env_id}_c51_{args.total_timesteps}_{start_datetime}_eval.png")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "C51", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()