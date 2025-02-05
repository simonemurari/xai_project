import random
from argparse import Namespace
from typing import Callable
import os
import gymnasium as gym
from minigrid.core.constants import IDX_TO_COLOR
import numpy as np
import torch
from c51_rules_training import make_env, QNetwork
from config import Args
from matplotlib import pyplot as plt
from datetime import datetime


def get_observables(raw_obs_batch):
    """
    Vectorized version of get_observables that processes entire batch at once
    Args:
        raw_obs_batch: shape (batch_size, H*W*C) or (batch_size, H, W, C)
    Returns:
        List of observation lists for each item in batch
    """
    DOOR_STATES = ["open", "closed", "locked"]
    view_size = 7
    mid = (view_size - 1) // 2
    batch_size = raw_obs_batch.shape[0]
    
    if isinstance(raw_obs_batch, torch.Tensor):
        raw_obs_batch = raw_obs_batch.cpu().numpy()
    
    # Reshape to (batch_size, H, W, C)
    img_batch = raw_obs_batch.reshape(batch_size, view_size, view_size, 3)
    
    # Pre-compute offsets once
    i, j = np.meshgrid(np.arange(view_size), np.arange(view_size), indexing='ij')
    offset_x = i - mid
    offset_y = np.abs(j - (view_size - 1))
    
    batch_obs = []
    
    # Process each batch item
    for img in img_batch:
        obs = []
        item_first = img[..., 0]
        item_second = img[..., 1]
        
        # Find all object positions at once
        key_positions = np.where(item_first == 5)
        door_positions = np.where(item_first == 4)
        goal_positions = np.where(item_first == 8)
        wall_positions = np.where(item_first == 2)
        
        # Process keys
        for k_i, k_j in zip(*key_positions):
            color = IDX_TO_COLOR.get(item_second[k_i, k_j])
            obs.append(("key", [color, offset_x[k_i, k_j], offset_y[k_i, k_j]]))
            if k_i == mid and k_j == view_size - 1:
                obs.append(("carryingKey", [color]))
        
        # Process doors
        for d_i, d_j in zip(*door_positions):
            color = IDX_TO_COLOR.get(item_second[d_i, d_j])
            obs.append(("door", [color, offset_x[d_i, d_j], offset_y[d_i, d_j]]))
            obs.append((DOOR_STATES[2], [color]))
        
        # Process goals
        for g_i, g_j in zip(*goal_positions):
            obs.append(("goal", [offset_x[g_i, g_j], offset_y[g_i, g_j]]))
        
        # Process walls
        for w_i, w_j in zip(*wall_positions):
            obs.append(("wall", [offset_x[w_i, w_j], offset_y[w_i, w_j]]))
            
        batch_obs.append(obs)
    
    return batch_obs

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
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
            weights = model.get_suggested_action(get_observables(obs))
            weights = weights.cpu().numpy()[0]
            actions = np.random.choice(range(model.n), p=weights)
            actions = np.array([actions])
        else:
            actions, _ = model.get_action(torch.Tensor(obs).to(device), global_step=i, total_timesteps=eval_episodes)
            actions = actions.cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                if i % 50 == 0:
                    print(
                        f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                    )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    eval_episodes = 10000
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    path_c51_5x5 = "MiniGrid-DoorKey-5x5-v0_c51_100000_2025_01_29-00_43_27"
    path_c51_6x6 = "MiniGrid-DoorKey-6x6-v0_c51_200000_2025_01_30-00_02_08"
    path_c51_8x8 = "MiniGrid-DoorKey-8x8-v0_c51_1000000_2025_01_30-12_11_39"
    path = path_c51_8x8
    model_path = f"C51/{path}/c51_model.pt"  # change this to the path of the model you want to evaluate
    if os.path.exists(f"C51re/{path}") is False:
        os.makedirs(f"C51re/{path}")
    run_name = f"C51re/{path}/{Args.env_id}_c51re_{Args.total_timesteps}_{start_datetime}"
    episodic_returns = evaluate(
        model_path,
        make_env,
        Args.env_id,
        eval_episodes=eval_episodes,
        run_name=f"{run_name}-eval",
        Model=QNetwork,
        device=Args.device,
        epsilon=0.00,
    )

    plt.plot(episodic_returns)
    plt.title(f"C51 Rules Eval on {Args.env_id} - Return over {eval_episodes} episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.savefig(
        f"C51re/{path}/{Args.env_id}_c51re_{Args.total_timesteps}_{start_datetime}.png"
    )
    print('Evaluation done!')
