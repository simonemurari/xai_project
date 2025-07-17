from dataclasses import dataclass
import os
from dotenv import load_dotenv
import torch

load_dotenv(".env")

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

@dataclass
class Args:
    env_id: str = 'Reacher-v2'
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = WANDB_PROJECT_NAME
    wandb_entity: str = WANDB_ENTITY
    capture_video: bool = False
    num_envs: int = 1
    total_timesteps: int = 1000000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: float = 5e3
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    print_step: int = 1000
