from dataclasses import dataclass
import os
from dotenv import load_dotenv
import torch

load_dotenv(".env")

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

@dataclass
class Args:
    # General arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    
    seed: int = 6
    """seed of the experiment"""

    num_envs: int = 1
    """the number of parallel game environments"""

    run_code: str = "8x8_v2"
    """the group of the run"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device: str = "cpu"
    """the device to run the experiment on (set it to cuda if using GPU)"""

    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""

    wandb_project_name: str = WANDB_PROJECT_NAME
    """the wandb's project name"""

    wandb_entity: str = WANDB_ENTITY
    """the entity (team) of wandb's project"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    upload_model: bool = False
    """whether to upload the saved model to huggingface"""

    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    print_step: int = 200000
    """the frequency to printout the training progress"""

    sigmoid_shift: float = 0.75
    """the sigmoid shift value for the C51 algorithm"""

    sigmoid_scale: float = 25
    """the sigmoid scale value for the C51 algorithm"""

    # C51 Algorithm specific arguments
    size_env: int = 8
    """the size of the environment (5, 6, 8, 16)"""

    env_id: str = f"MiniGrid-DoorKey-{size_env}x{size_env}-v0"
    """the id of the environment"""
    
    total_timesteps: int = 3_000_000
    """total timesteps of the experiments"""
    
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""

    n_atoms: int = 51
    """the number of atoms"""

    v_min: float = -1
    """the return lower bound"""

    v_max: float = 1
    """the return upper bound"""

    buffer_size: int = 500_000
    """the replay memory buffer size"""

    gamma: float = 0.99
    """the discount factor gamma"""

    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""

    batch_size: int = 128
    """the batch size of sample from the replay memory"""

    start_e: float = 1
    """the starting epsilon for exploration"""

    end_e: float = 0.05
    """the ending epsilon for exploration"""

    exploration_fraction: float = 0.3
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""

    learning_starts: int = 50_000 
    """timestep to start learning"""

    train_frequency: int = 2
    """the frequency of training"""

