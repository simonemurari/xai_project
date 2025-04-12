# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51py
import os
os.environ['OMP_NUM_THREADS'] = '1'
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
import torch.multiprocessing as mp
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from minigrid.core.constants import IDX_TO_COLOR
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import itertools

# Import the necessary constants from minigrid
from minigrid.core.constants import (
    OBJECT_TO_IDX,
    IDX_TO_COLOR,
    STATE_TO_IDX
)
# Create inverse mapping from state index to state name string
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
# Define a default/unknown state string
UNKNOWN_STATE_STR = "unknown_state"

# Create NumPy array for faster color mapping
_MAX_COLOR_IDX = max(IDX_TO_COLOR.keys()) if IDX_TO_COLOR else -1
_COLOR_MAP_NP = np.array([IDX_TO_COLOR.get(i, 'unknown') for i in range(_MAX_COLOR_IDX + 1)])

# Object indices from constants
KEY_IDX = OBJECT_TO_IDX['key']
DOOR_IDX = OBJECT_TO_IDX['door']
GOAL_IDX = OBJECT_TO_IDX['goal']
WALL_IDX = OBJECT_TO_IDX['wall']

# print(f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}")

# Rules applied to the C51 algorithm only during training (v2)

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env, filter_keys=['image', 'direction']))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

def _plot_pmfs(
    pmfs_rules, pmfs, combined_pmfs, action_index, n_categories, alpha, title_prefix
):
    """
    Plots the PMFs for a rule-based action, a network action, and their combined probabilities.

      Args:
          pmfs_rules (torch.Tensor): PMF from rules of shape (1, num_actions, num_categories).
          pmfs (torch.Tensor): PMF from the neural network (1, num_actions, num_categories).
          combined_pmfs (torch.Tensor): Combined PMF, result of multiplying pmfs and pmfs_rules (1, num_actions, num_categories).
          action_index (int): The index of the action to plot.
          n_categories (int): Number of categories (atoms) in the PMFs
          title_prefix (string): A prefix to add to the titles of the plots

    """
    # Make sure the input tensors are in CPU
    pmfs_rules = pmfs_rules.cpu().detach().numpy()
    pmfs = pmfs.cpu().detach().numpy()
    combined_pmfs = combined_pmfs.cpu().detach().numpy()

    # Extract the PMFs for the specified action
    rule_pmf = pmfs_rules[0, action_index]
    network_pmf = pmfs[0, action_index]
    combined_pmf = combined_pmfs[0, action_index]

    # Map the range 0 to 51 to 0 to 1
    x = np.linspace(Args.v_min, Args.v_max, n_categories)
    # Create subplots
    _, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the rule-based PMF
    axs[0].bar(x, rule_pmf, label="Rule PMF", color="skyblue")
    axs[0].set_title(f"{title_prefix} Rule PMF - Action {action_index} - alpha={alpha:.2f}")
    axs[0].set_xlim(Args.v_min, Args.v_max)
    axs[0].set_xlabel("Return")
    axs[0].set_ylabel("Probability")
    axs[0].legend()

    # Plot the neural network PMF
    axs[1].bar(x, network_pmf, label="Network PMF", color="salmon")
    axs[1].set_title(f"{title_prefix} Network PMF - Action {action_index} - alpha={alpha:.2f}")
    axs[0].set_xlim(Args.v_min, Args.v_max)
    axs[1].set_xlabel("Return")
    axs[1].set_ylabel("Probability")
    axs[1].legend()

    # Plot the combined PMF
    axs[2].bar(x, combined_pmf, label="Combined PMF", color="lightgreen")
    axs[2].set_title(f"{title_prefix} Combined PMF - Action {action_index} - alpha={alpha:.2f}")
    axs[0].set_xlim(Args.v_min, Args.v_max)
    axs[2].set_xlabel("Return")
    axs[2].set_ylabel("Probability")
    axs[2].legend()

    plt.tight_layout()
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    plt.savefig(f"plots/{title_prefix}_pmfs_{action_index}_{alpha:.2f}.png")
    plt.close()


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms, device=Args.device))
        self.n = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n * n_atoms),
        )
        self.rule_pmf = self.rule_distribution().view(1, 1, self.n_atoms) # Pre-compute rule PMF


    def get_action(self, x, action=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        # Only use the image part
        x = x[:, 4:]
        observables = get_observables_optimized_with_constants(x)
        rule_mask = self.get_rule_mask_v2(observables).unsqueeze(2)
        rule_pmfs = rule_mask * self.rule_pmf
        combined_pmfs = (1 - epsilon) * pmfs + epsilon * rule_pmfs
        q_values = (combined_pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        # for act in range(self.n):
        #     if rule_mask[0, act].sum() > 0 and epsilon > 0.5 and epsilon < 0.8:
        #         _plot_pmfs(
        #             rule_pmfs, pmfs, combined_pmfs, act, self.n_atoms, epsilon, "rule_based"
        #         )
        #         asfasffsa
        return action, pmfs[torch.arange(len(x)), action]
    

    def get_rule_mask_v2(self, batch_of_observables):
        """
        Optimized vectorized version that processes multiple observations in parallel.
        Uses correct door states provided by the fixed get_observables.
        """
        device = Args.device # Assuming Args.device is accessible
        batch_size = len(batch_of_observables)
        if batch_size == 0:
            return torch.zeros(0, self.n, device=device) # Handle empty batch

        # Pre-allocate tensors
        rule_mask = torch.zeros(batch_size, self.n, device=device)

        # Action indices from MiniGrid (standard DoorKeyEnv)
        # 0: left, 1: right, 2: forward, 3: pickup, 4: drop, 5: toggle, 6: done
        ACTION_LEFT = 0
        ACTION_RIGHT = 1
        ACTION_FORWARD = 2
        ACTION_PICKUP = 3
        ACTION_TOGGLE = 5

        for batch_idx, observables in enumerate(batch_of_observables):
            # State tracking for the current observation
            carrying_key_color = None
            front_is_clear = True # Assume clear unless wall or closed/locked door is seen
            front_door_state = None # ('open'/'closed'/'locked', color)
            front_door_color = None
            front_is_key = False
            goal_pos = None # (x, y)
            can_turn_left = True
            can_turn_right = True

            # Single pass through observations to gather relevant state
            for name, args in observables:
                match name:
                    case "carryingKey":
                        carrying_key_color = args[0]
                    case "key":
                         # Check if key is directly in front
                        if args[1] == 0 and args[2] == 1:
                            front_is_key = True
                            front_is_clear = False # Can't move forward onto key tile directly
                    case "door":
                        # Check if door is directly in front
                        if args[1] == 0 and args[2] == 1:
                            front_door_color = args[0]
                            # State will be handled by the state observation below
                    case "goal":
                        goal_pos = (args[0], args[1]) # Store goal relative position
                    case "wall":
                        x, y = args
                        # Check for wall directly in front
                        if x == 0 and y == 1:
                            front_is_clear = False
                        # Check for wall blocking left turn (relative to agent)
                        elif x == -1 and y == 0:
                            can_turn_left = False
                        # Check for wall blocking right turn (relative to agent)
                        elif x == 1 and y == 0:
                            can_turn_right = False
                    # Check for ACTUAL door states
                    case "open" | "closed" | "locked":
                        state_name = name
                        state_color = args[0]
                        # Check if this state corresponds to the door in front
                        if front_door_color == state_color: # Assumes colors are unique enough ID for doors in view
                           front_door_state = (state_name, state_color)
                           if state_name != "open":
                                front_is_clear = False # Blocked if closed or locked

            # --- Apply Rules Based on Gathered State ---

            # Rule 1: pickup(X) :- key(X), ..., notcarrying
            # Simplified: Suggest pickup if key is in front and not carrying any key
            if front_is_key and carrying_key_color is None:
                rule_mask[batch_idx, ACTION_PICKUP] = 1
                # continue # Prioritize pickup

            # Rule 2: open(X) :- door(X), locked(X), carryingKey(Z), samecolor(X,Z)
            # Suggest toggle if door is in front, it's locked, and carrying matching key
            if front_door_state and front_door_state[0] == "locked" and \
               carrying_key_color == front_door_state[1]:
                 rule_mask[batch_idx, ACTION_TOGGLE] = 1
                #  continue # Prioritize unlocking

            # Rule 3: goto :- goal(X), unlocked
            # Navigate towards goal if visible and path seems clear or can be opened
            if goal_pos:
                goal_x, goal_y = goal_pos
                # If goal is directly in front and path is clear
                if goal_x == 0 and goal_y == 1 and front_is_clear:
                     rule_mask[batch_idx, ACTION_FORWARD] = 1
                # Else, navigate towards goal direction if possible
                elif goal_x < 0 and can_turn_left: # Goal is left, turn left if possible
                     rule_mask[batch_idx, ACTION_LEFT] = 1
                elif goal_x > 0 and can_turn_right: # Goal is right, turn right if possible
                     rule_mask[batch_idx, ACTION_RIGHT] = 1
                elif front_is_clear: # Aligned with goal or cannot turn, move forward if clear
                     rule_mask[batch_idx, ACTION_FORWARD] = 1
                # (Implicit else: if goal is visible but blocked, suggest nothing for goto)

            # Note: No explicit rule for closing doors or dropping keys is included here.
            # If multiple rules suggest actions, the `continue` statements create a priority:
            # Pickup > Unlock > Goto

        return rule_mask
    
    
    def rule_distribution(self):
        """
        Computes a proper distribution over the 51 atoms,
        increasing slowly from 0 to 1, reaching the highest value only at 1.
        Returns a PyTorch tensor.
        """
        rule_shift = Args.sigmoid_shift
        rule_scale = Args.sigmoid_scale

        # Compute the standard sigmoid over the atoms
        s = torch.sigmoid(rule_scale * (self.atoms - rule_shift))
        
        rule_pmf = s / s.sum()
        # rule_pmf = torch.clone(rule_pmf).detach()

        return rule_pmf
    
    def plot_rule_distribution(self):
        rule_pmf_np = self.rule_pmf.squeeze().numpy()  # Convert to NumPy for plotting
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, 1, 51)
        plt.bar(x, rule_pmf_np, width=(1 / 51) * 0.8)
        plt.xlabel("Return Value (Atom)")
        plt.ylabel("Probability")
        plt.xlim(-0.05, 1.05)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"plots/rule_pmf_v2_{Args.sigmoid_shift}_{Args.sigmoid_scale}_{Args.run_code}.png")  # Save the plot
        plt.close()

    def get_rule_mask(self, batch_of_observables):
        """
        Optimized vectorized version that processes multiple observations in parallel.
        """
        device = Args.device
        batch_size = len(batch_of_observables)
        
        # Pre-allocate tensors
        rule_mask = torch.zeros(batch_size, self.n, device=device)
        
        for batch_idx, observables in enumerate(batch_of_observables):
            # State tracking with minimal variables
            carrying_key = None
            door_state = None
            has_door_front = False
            has_key_front = False
            wall_state = [False, False, False]  # left, right, front
            goal_data = [0, False]  # x, is_front
            door_is_open = False  # Track if we observe an open door in this obs
            
            # Single pass through observations
            for name, args in observables:
                match name:
                    case "door" if args[1:] == [0, 1]:
                        has_door_front = True
                        door_color = args[0]
                    case "key" if args[1:] == [0, 1]:
                        has_key_front = True
                    case "carryingKey":
                        carrying_key = args[0]
                    case "goal":
                        goal_data[0] = args[0]
                        goal_data[1] = args[1] == 1
                    case "wall":
                        x, y = args
                        if y == 0:
                            if x == -1:
                                wall_state[0] = True
                            elif x == 1:
                                wall_state[1] = True
                            elif x == 0 and y == 1:
                                wall_state[2] = True
                    case state if state in ["open", "closed", "locked"]:
                        door_state = (state, args[0])
                        # Only mark door_is_open if we actually see an open door
                        if state == "open":
                            door_is_open = True
            
            # Fast action selection with minimal branching
            if not carrying_key and has_key_front:
                rule_mask[batch_idx, 3] = 1  # Rule 1: pickup key
            
            # Door interaction - Rule 2: open door
            elif door_state and door_state[0] == "locked" and carrying_key:
                if has_door_front and carrying_key == door_color:
                    rule_mask[batch_idx, 5] = 1
            
            # Movement logic - Rule 3: go to goal when door open
            elif goal_data[1] and (door_is_open or (door_state and door_state[0] == "open")):
                # Move forward when goal is in front and any door is open
                rule_mask[batch_idx, 2] = 1
            elif goal_data[0]:  # Navigate toward goal
                if goal_data[0] < 0 and not wall_state[0]:
                    rule_mask[batch_idx, 0] = 1  # Turn left
                elif goal_data[0] > 0 and not wall_state[1]:
                    rule_mask[batch_idx, 1] = 1  # Turn right
                elif not wall_state[2]:
                    rule_mask[batch_idx, 2] = 1  # Move forward
        
        return rule_mask


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



def get_observables_optimized_with_constants(raw_obs_batch):
    """
    Highly vectorized version of get_observables processing the entire batch,
    using official minigrid constants and CORRECTLY reading door states.
    Args:
        raw_obs_batch: shape (batch_size, H*W*C) or (batch_size, H, W, C)
                       Can be NumPy array or PyTorch Tensor. H, W assumed to be 7.
                       C=3 assumed (Object Type, Color, State).
    Returns:
        List of observation lists for each item in batch
    """
    VIEW_SIZE = 7 # Still assumes a 7x7 view based on original functions
    MID = (VIEW_SIZE - 1) // 2

    if not isinstance(raw_obs_batch, np.ndarray):
        if isinstance(raw_obs_batch, torch.Tensor):
             if raw_obs_batch.requires_grad:
                  raw_obs_batch = raw_obs_batch.detach()
             raw_obs_batch = raw_obs_batch.cpu().numpy()
        else:
             raw_obs_batch = np.array(raw_obs_batch)

    batch_size = raw_obs_batch.shape[0]
    if batch_size == 0:
        return []

    # Ensure shape is (batch_size, H, W, C=3)
    try:
        img_batch = raw_obs_batch.reshape(batch_size, VIEW_SIZE, VIEW_SIZE, 3)
    except ValueError as e:
        raise ValueError(f"Input shape {raw_obs_batch.shape} cannot be reshaped to ({batch_size}, {VIEW_SIZE}, {VIEW_SIZE}, 3)") from e

    # Separate channels for the whole batch
    item_first_batch = img_batch[..., 0]  # Object types
    item_second_batch = img_batch[..., 1] # Colors
    item_third_batch = img_batch[..., 2]  # States <--- IMPORTANT for doors

    # --- Pre-compute offsets ---
    i_coords, j_coords = np.meshgrid(np.arange(VIEW_SIZE), np.arange(VIEW_SIZE), indexing='ij')
    offset_x = i_coords - MID
    offset_y = np.abs(j_coords - (VIEW_SIZE - 1)) # Transformed y-coord

    # --- Find all object indices across the entire batch ---
    key_b_idx, key_i, key_j = np.where(item_first_batch == KEY_IDX)
    door_b_idx, door_i, door_j = np.where(item_first_batch == DOOR_IDX)
    goal_b_idx, goal_i, goal_j = np.where(item_first_batch == GOAL_IDX)
    wall_b_idx, wall_i, wall_j = np.where(item_first_batch == WALL_IDX)

    # --- Prepare list to gather all observations ---
    all_obs_with_batch_idx = []

    # --- Process Keys ---
    if key_b_idx.size > 0:
        key_offs_x = offset_x[key_i, key_j]
        key_offs_y = offset_y[key_i, key_j]
        key_colors_idx = item_second_batch[key_b_idx, key_i, key_j].astype(int)
        key_colors = np.where((key_colors_idx >= 0) & (key_colors_idx < len(_COLOR_MAP_NP)),
                              _COLOR_MAP_NP[key_colors_idx], 'unknown')

        for b, c, ox, oy in zip(key_b_idx, key_colors, key_offs_x, key_offs_y):
            all_obs_with_batch_idx.append((b, ("key", [c, ox, oy])))

        is_carrying_mask = (key_i == MID) & (key_j == VIEW_SIZE - 1)
        if np.any(is_carrying_mask):
            carrying_b_idxs = key_b_idx[is_carrying_mask]
            carrying_colors = key_colors[is_carrying_mask]
            for b, c in zip(carrying_b_idxs, carrying_colors):
                all_obs_with_batch_idx.append((b, ("carryingKey", [c])))

    # --- Process Doors ---
    if door_b_idx.size > 0:
        door_offs_x = offset_x[door_i, door_j]
        door_offs_y = offset_y[door_i, door_j]
        door_colors_idx = item_second_batch[door_b_idx, door_i, door_j].astype(int)
        door_states_idx = item_third_batch[door_b_idx, door_i, door_j].astype(int) # <<< Get state index
        door_colors = np.where((door_colors_idx >= 0) & (door_colors_idx < len(_COLOR_MAP_NP)),
                               _COLOR_MAP_NP[door_colors_idx], 'unknown')

        # Convert state indices to state strings using the mapping
        door_state_strs = [IDX_TO_STATE.get(idx, UNKNOWN_STATE_STR) for idx in door_states_idx]

        for b, c, ox, oy, state_str in zip(door_b_idx, door_colors, door_offs_x, door_offs_y, door_state_strs):
            all_obs_with_batch_idx.append((b, ("door", [c, ox, oy])))
            # Append the ACTUAL state observed
            all_obs_with_batch_idx.append((b, (state_str, [c]))) # <<< Use actual state_str

    # --- Process Goals ---
    if goal_b_idx.size > 0:
        goal_offs_x = offset_x[goal_i, goal_j]
        goal_offs_y = offset_y[goal_i, goal_j]
        for b, ox, oy in zip(goal_b_idx, goal_offs_x, goal_offs_y):
            all_obs_with_batch_idx.append((b, ("goal", [ox, oy])))

    # --- Process Walls ---
    if wall_b_idx.size > 0:
        wall_offs_x = offset_x[wall_i, wall_j]
        wall_offs_y = offset_y[wall_i, wall_j]
        for b, ox, oy in zip(wall_b_idx, wall_offs_x, wall_offs_y):
            all_obs_with_batch_idx.append((b, ("wall", [ox, oy])))

    # --- Group observations by batch index ---
    all_obs_with_batch_idx.sort(key=lambda x: x[0])

    final_batch_obs = [[] for _ in range(batch_size)]
    for batch_idx, group in itertools.groupby(all_obs_with_batch_idx, key=lambda x: x[0]):
        if 0 <= batch_idx < batch_size:
             final_batch_obs[batch_idx].extend([obs for _, obs in group])

    return final_batch_obs


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
        item_third = img[..., 2]
        
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
            obs.append((DOOR_STATES[int(item_third[d_i, d_j])], [color]))
        
        # Process goals
        for g_i, g_j in zip(*goal_positions):
            obs.append(("goal", [offset_x[g_i, g_j], offset_y[g_i, g_j]]))
        
        # Process walls
        for w_i, w_j in zip(*wall_positions):
            obs.append(("wall", [offset_x[w_i, w_j], offset_y[w_i, w_j]]))
            
        batch_obs.append(obs)
    
    return batch_obs

# Set process affinity to specific cores
def set_process_affinity(process_idx, num_processes):
    """Pin process to specific cores to avoid contention"""
    try:
        import psutil
        process = psutil.Process()
        num_cores = psutil.cpu_count(logical=True)
        
        # Allocate cores per process
        cores_per_process = num_cores // num_processes
        
        # Calculate core range for this process
        start_core = process_idx * cores_per_process
        end_core = start_core + cores_per_process
        
        # Create CPU affinity mask
        cpu_list = list(range(start_core, end_core))
        process.cpu_affinity(cpu_list)
        
        print(f"Process {os.getpid()}: Pinned to cores {cpu_list}")
    except (ImportError, AttributeError):
        # If psutil isn't available or doesn't support affinity
        print(f"Process {os.getpid()}: Couldn't set CPU affinity")

def train_c51_rules(args, seed, process_idx, num_processes):
    """Main training function for C51 with rules algorithm"""
    # Set process affinity to specific cores
    set_process_affinity(process_idx, num_processes)
    
    # Set number of threads for this process
    torch.set_num_threads(max(1, mp.cpu_count() // num_processes))
    
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # Set the seed for this process
    args.seed = seed

    run_name = f"C51rtv2_{args.env_id}__seed{seed}__{start_datetime}"
    
    if args.track:
        import wandb
        wandb.tensorboard.patch(root_logdir=f"C51rtv2/runs_rules_training/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"C51rtv2_{args.sigmoid_shift}_{args.sigmoid_scale}_{args.exploration_fraction}_mp_{args.run_code}",
        )
    
    # Only write to TensorBoard periodically to reduce I/O overhead
    writer = SummaryWriter(f"C51rtv2/runs_rules_training/{run_name}/train", 
                         max_queue=100, 
                         flush_secs=30)
    
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []

    # TRY NOT TO MODIFY: seeding
    print(f'Process {os.getpid()}: File: {os.path.basename(__file__)}, using seed {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = args.device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    q_network = QNetwork(
        envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max
    ).to(device)
    # q_network.plot_rule_distribution()  # Call the plot function to save the plot
    # asfsaffsa
    optimizer = optim.Adam(
        q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size
    )
    target_network = QNetwork(
        envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max
    ).to(device)
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
    obs, _ = envs.reset(seed=seed)
    print_step = args.print_step
    print(f"Process {os.getpid()}: Starting training with seed {seed} for {args.total_timesteps} timesteps")
    
    # Pre-allocate observation tensor for efficiency
    obs_tensor = torch.zeros((1, *envs.single_observation_space.shape), dtype=torch.float32, device=device)
    
    # Use tqdm with minimal updates to reduce overhead
    for global_step in tqdm(range(args.total_timesteps), 
                          desc=f"Seed {seed}", 
                          position=process_idx,
                          colour=['blue', 'green', 'red'][process_idx % 3],
                          miniters=10000,
                          maxinterval=5):
                            
        # ALGO LOGIC: put action logic here
        global epsilon  # Make epsilon available to QNetwork.get_action
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Reuse pre-allocated tensor
            obs_tensor.copy_(torch.as_tensor(obs, dtype=torch.float32))
            with torch.no_grad():
                actions, pmf = q_network.get_action(obs_tensor)
                actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("episodic_return", info["episode"]["r"][0], global_step)
                    writer.add_scalar("episodic_length", info["episode"]["l"][0], global_step)
                    episodes_returns.append(info["episode"]["r"])
                    episodes_lengths.append(info["episode"]["l"])
                    if global_step >= print_step:
                        mean_ep_return = np.mean(episodes_returns[-args.print_step:])
                        mean_ep_lengths = np.mean(episodes_lengths[-args.print_step:])
                        tqdm.write(f"seed = {seed}: global_step={global_step}, mean_return_print_step={mean_ep_return:.4f}, mean_length_print_step={mean_ep_lengths} eps={epsilon:.4f}")
                        print_step += args.print_step

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos and idx < len(infos.get("final_observation", [])):
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(
                        data.next_observations.float(),
                    )
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

                _, old_pmfs = q_network.get_action(
                    data.observations.float(), data.actions.flatten()
                )
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                if global_step % 10000 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # optimize the model
                optimizer.zero_grad(set_to_none=True)  # More efficient
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # Save results
    plt.figure()
    plt.plot(episodes_returns)
    plt.title(f'C51rtv2 on {args.env_id} - Seed {seed} - Return over {args.total_timesteps} timesteps')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    path = f'C51rtv2/{args.env_id}_c51rtv2_{args.total_timesteps}_seed{seed}_{start_datetime}'
    if not os.path.exists("C51rtv2/"):
        os.makedirs("C51rtv2/")
    os.makedirs(path)
    plt.savefig(f"{path}/{args.env_id}_c51rtv2_{args.total_timesteps}_seed{seed}_{start_datetime}.png")
    plt.close()
    
    with open(f"{path}/c51rtv2_args.txt", "w") as f:
        for key, value in vars(args).items():
            if key == "env_id":
                f.write("# C51 Algorithm specific arguments\n")
            f.write(f"{key}: {value}\n")

    if args.save_model:
        model_path = f"{path}/c51rtv2_model.pt"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
            "seed": seed,
        }
        torch.save(model_data, model_path)
        print(f"Process {os.getpid()}: Model saved to {model_path}")
        
        # Only perform evaluation in the main process or last process
        if process_idx == num_processes - 1:
            from baseC51.c51_eval import QNetwork as QNetworkEval
            from c51rtv2_eval import evaluate
            eval_episodes=100000
            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=eval_episodes,
                run_name=f"{run_name}-eval",
                Model=QNetworkEval,
                device=device,
                epsilon=0
            )
            writer = SummaryWriter(f"C51rtv2/runs_rules_training/{run_name}/eval")
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("episodic_return", episodic_return, idx)

            plt.plot(episodic_returns)
            plt.title(f'C51rtv2 Eval on {args.env_id} - Return over {eval_episodes} episodes')
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.ylim(0, 1)
            plt.grid(True)
            plt.savefig(f"{path}/{args.env_id}_c51rtv2_{eval_episodes}_{start_datetime}_eval.png")

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "C51rtv2", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
    
    return episodes_returns, path

if __name__ == "__main__":
    # Set up multiprocessing
    if sys.platform != 'win32':
        mp.set_start_method('forkserver', force=True)
    else:
        mp.set_start_method('spawn', force=True)
    
    print(f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}, device = {Args.device}")
    args = tyro.cli(Args)
    
    # Add shared memory lock for GPU access if using CUDA
    if torch.cuda.is_available() and args.device == "cuda":
        lock = mp.Lock()
    else:
        lock = None
    
    # Seeds to use for different processes
    seeds = [6, 21, 42]
    num_processes = len(seeds)
    
    print(f"Starting C51rtv2 training with {num_processes} processes, seeds: {seeds}")
    
    # Start multiple processes for training
    processes = []
    for idx, seed in enumerate(seeds):
        p = mp.Process(target=train_c51_rules, args=(args, seed, idx, num_processes))
        p.start()
        # Sleep briefly between process launches to prevent resource contention
        time.sleep(3)
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All training processes completed!")