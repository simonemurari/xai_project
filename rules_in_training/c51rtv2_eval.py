import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from argparse import Namespace
from typing import Callable
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import torch
from baseC51.c51 import QNetwork, make_env
from config import Args
import tyro
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Evaluation of a C51rtv2 model on a MiniGrid environment, rules used only during training

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device,
    epsilon: float = 0.00,
    capture_video: bool = False,
    write: bool = False
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    model = Model(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    model.load_state_dict(model_data["model_weights"])
    model = model.to(device)
    model.eval()
    if write:
        writer = SummaryWriter(f"C51rtv2/runs_eval/{run_name}/eval")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    
    obs, _ = envs.reset()
    episodic_returns = []
    for i in range(eval_episodes):
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = model.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                if i % 500 == 0:
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                if write:
                    writer.add_scalar("episodic_return", info["episode"]["r"], i)
                    writer.add_scalar("episodic_length", info["episode"]["l"], i)
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # Change these paths to the paths of the models you want to evaluate
    # path_5x5 = "MiniGrid-DoorKey-5x5-v0_c51rtv2_100000_2025_02_14-19_15_05" # training not done yet
    # path_6x6 = "MiniGrid-DoorKey-6x6-v0_c51rtv2_1000000_2025_02_15-23_01_06" # training not done yet
    path_8x8 = "MiniGrid-DoorKey-8x8-v0_c51rtv2_1000000_2025_02_19-21_27_02"
    path = path_8x8
    model_path = f"C51rtv2/{path}/c51rtv2_model.pt"
    args = tyro.cli(Args)
    run_name = f"C51rtv2Eval_trainedon={path.split('_')[0]}__testedon={args.env_id}__seed={args.seed}__{start_datetime}"
    eval_episodes = 100000
    if args.track:
            import wandb
            wandb.tensorboard.patch(root_logdir=f"C51rtv2/runs_eval/{run_name}/eval")
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
    episodic_returns = evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=eval_episodes,
        run_name=run_name,
        Model=QNetwork,
        device="cpu",
        capture_video=False,
        write=True
    )

    plt.plot(episodic_returns)
    plt.title(f"C51rtv2Eval on {args.env_id} - Return over {eval_episodes} episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(
        f"C51rtv2/{path}/{args.env_id}_c51rtv2eval_{eval_episodes}_{start_datetime}.png"
    )
    print('Evaluation done!')
