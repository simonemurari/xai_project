import random
from argparse import Namespace
from typing import Callable
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import torch


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
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    model = Model(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    model.load_state_dict(model_data["model_weights"])
    model = model.to(device)
    model.eval()

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
                if i % 50 == 0:
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from c51 import QNetwork, make_env
    from config import Args
    from datetime import datetime
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    path_5x5 = "MiniGrid-DoorKey-5x5-v0_c51_100000_2025_01_29-00_43_27"
    path_6x6 = "MiniGrid-DoorKey-6x6-v0_c51_200000_2025_01_30-00_02_08"
    path_8x8 = "MiniGrid-DoorKey-8x8-v0_c51_1000000_2025_01_30-12_11_39"
    path = path_6x6
    model_path = f"C51/{path}/c51_model.pt"
    eval_episodes = 10000
    episodic_returns = evaluate(
        model_path,
        make_env,
        Args.env_id,
        eval_episodes=eval_episodes,
        run_name="eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
    )

    plt.plot(episodic_returns)
    plt.title(f"C51Eval on {Args.env_id} - Return over {eval_episodes} episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.savefig(
        f"C51/{path}/{Args.env_id}_c51eval_{Args.total_timesteps}_{start_datetime}.png"
    )
    print('Evaluation done!')
