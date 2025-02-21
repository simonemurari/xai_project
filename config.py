from dataclasses import dataclass
import os
import torch
import os
from dotenv import load_dotenv

load_dotenv(".env")

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

@dataclass
class Args:
    # General arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    
    seed: int = 21
    """seed of the experiment"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """the device to run the experiment on (set it to cuda if using GPU)"""

    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""

    wandb_project_name: str = WANDB_PROJECT_NAME
    """the wandb's project name"""

    wandb_entity: str = WANDB_ENTITY
    """the entity (team) of wandb's project"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    upload_model: bool = False
    """whether to upload the saved model to huggingface"""

    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    print_step: int = 100000
    """the frequency to printout the training progress"""

    sigmoid_shift: float = 0.75 # Used only in rules_training_v2 and rules_training_v3
    """the shift of the sigmoid function for the distribution in rules_training_v2 and rules_training_v3"""

    sigmoid_scale: float = 50 # Used only in rules_training_v2 and rules_training_v3
    """ the scale of the sigmoid function for the distribution in rules_training_v2 and rules_training_v3"""

    # C51 Algorithm specific arguments
    size_env: int = 8
    """the size of the environment (5, 6, 8, 16)"""

    env_id: str = f"MiniGrid-DoorKey-{size_env}x{size_env}-v0"
    """the id of the environment"""
    
    total_timesteps: int = 1000000 # 5x5: 100k, 6x6: 200k, 8x8: 1kk
    """total timesteps of the experiments"""
    
    learning_rate: float =  2e-4 # 5x5 and 6x6: 0.0005, 8x8: 0.0001
    """the learning rate of the optimizer"""

    num_envs: int = 1
    """the number of parallel game environments"""

    n_atoms: int = 51
    """the number of atoms"""

    v_min: float = 0
    """the return lower bound"""

    v_max: float = 1
    """the return upper bound"""

    buffer_size: int = 250000 # 5x5 and 6x6: 50000, 8x8: 100000
    """the replay memory buffer size"""

    gamma: float = 0.95 # 5x5 and 6x6: 0.95, 8x8: 0.999
    """the discount factor gamma"""

    target_network_frequency: int = 200 # 5x5 and 6x6: 200, 8x8: 500
    """the timesteps it takes to update the target network"""

    batch_size: int = 64
    """the batch size of sample from the replay memory"""

    start_e: float = 1
    """the starting epsilon for exploration"""

    end_e: float = 0.01
    """the ending epsilon for exploration"""

    exploration_fraction: float = 0.75 # 5x5 and 6x6: 0.3, 8x8: 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""

    learning_starts: int = 10000 # 5x5 and 6x6: 5000, 8x8: 50000
    """timestep to start learning"""

    train_frequency: int = 2 # 5x5 and 6x6: 2, 8x8: 4
    """the frequency of training"""
