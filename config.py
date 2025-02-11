from dataclasses import dataclass
import os
import torch

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

    wandb_project_name: str = "c51"
    """the wandb's project name"""

    wandb_entity: str = "simone-murari3"
    """the entity (team) of wandb's project"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    upload_model: bool = False
    """whether to upload the saved model to huggingface"""

    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # C51 Algorithm specific arguments
    env_id: str = "MiniGrid-DoorKey-6x6-v0"
    """the id of the environment"""
    
    total_timesteps: int = 100000 # 5x5: 100k, 6x6: 200k, 8x8: 1kk
    """total timesteps of the experiments"""
    
    learning_rate: float =  0.0005 # 5x5 and 6x6: 0.0005, 8x8: 0.0001
    """the learning rate of the optimizer"""

    num_envs: int = 1
    """the number of parallel game environments"""

    n_atoms: int = 51
    """the number of atoms"""

    v_min: float = -10
    """the return lower bound"""

    v_max: float = 10
    """the return upper bound"""

    buffer_size: int = 30000 # 5x5 and 6x6: 50000, 8x8: 100000
    """the replay memory buffer size"""

    gamma: float = 0.95 # 5x5 and 6x6: 0.95, 8x8: 0.999
    """the discount factor gamma"""

    target_network_frequency: int = 200 # 5x5 and 6x6: 200, 8x8: 500
    """the timesteps it takes to update the target network"""

    batch_size: int = 32
    """the batch size of sample from the replay memory"""

    start_e: float = 1
    """the starting epsilon for exploration"""

    end_e: float = 0.05
    """the ending epsilon for exploration"""

    exploration_fraction: float = 0.25 # 5x5 and 6x6: 0.3, 8x8: 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""

    learning_starts: int = 5000 # 5x5 and 6x6: 5000, 8x8: 50000
    """timestep to start learning"""

    train_frequency: int = 2 # 5x5 and 6x6: 2, 8x8: 4
    """the frequency of training"""