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
from minigrid.core.constants import IDX_TO_COLOR
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Rules applied to the C51 algorithm only during training (SUM)

# Pre-computed constant arrays for observation processing
DOOR_STATES = ["open", "closed", "locked"]
VIEW_SIZE = 7
MID_POINT = (VIEW_SIZE - 1) // 2

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            env = gym.wrappers.FlattenObservation(
                gym.wrappers.FilterObservation(env, filter_keys=['image', 'direction'])
            )
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
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n * n_atoms),
        )
                
        self.conf_level = 0.8  # confidence level of the rules

    def get_action(self, x, action=None, epsilon=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if epsilon is not None:
            # Normalize Q-values to [0,1] for fair blending with rules
            min_vals = torch.amin(q_values, dim=1, keepdim=True)
            max_vals = torch.amax(q_values, dim=1, keepdim=True)
            q_values = (q_values - min_vals) / (max_vals - min_vals + 1e-8)
            
            # Process image part only for rules
            x_img = x[:, 4:]
            observables = get_observables(x_img)
            weights = self.get_suggested_action(observables)
            combined_values = weights * q_values
            
            if action is None:
                action = torch.argmax(combined_values, 1)
        else:
            if action is None:
                action = torch.argmax(q_values, 1)

        return action, pmfs[torch.arange(len(x)), action]

    def get_suggested_action(self, batch_of_observables):
        """
        Optimized vectorized version that handles multiple doors and goals properly.
        This version allows movement toward the goal even when doors aren't visible.
        """
        device = self.atoms.device
        batch_size = len(batch_of_observables)
        
        # Pre-allocate tensors
        weights = torch.full((batch_size, self.n), 1 - self.conf_level, device=device)
        
        for batch_idx, observables in enumerate(batch_of_observables):
            # State tracking with minimal variables
            carrying_key = None
            visible_doors = {}  # Track all doors {color: (state, position)}
            has_door_front = False
            door_front_color = None
            has_key_front = False
            wall_state = [False, False, False]  # left, right, front
            goal_data = [0, False]  # x, is_front
            path_blocked = False   # Flag to track if path to goal is blocked
            
            # Parse observations
            for name, args in observables:
                match name:
                    case "door" if args[1:] == [0, 1]:
                        has_door_front = True
                        door_front_color = args[0]
                        visible_doors[args[0]] = {"pos": args[1:], "state": None}
                    case "door":
                        # Track all doors
                        visible_doors[args[0]] = {"pos": args[1:], "state": None}
                    case "key" if args[1:] == [0, 1]:
                        has_key_front = True
                    case "carryingKey":
                        carrying_key = args[0]
                    case "goal":
                        goal_data[0] = args[0]  # x offset
                        goal_data[1] = args[1] == 1  # is front
                    case "wall":
                        x, y = args
                        if y == 0:  # Same y-level
                            if x == -1:
                                wall_state[0] = True  # Left wall
                            elif x == 1:
                                wall_state[1] = True  # Right wall
                        elif x == 0 and y == 1:  # Directly in front
                            wall_state[2] = True  # Front wall
                    case state if state in ["open", "closed", "locked"]:
                        # Store door state with the door
                        if args[0] in visible_doors:
                            visible_doors[args[0]]["state"] = state
            
            # Check if the path to goal is blocked by a locked door
            if has_door_front and door_front_color in visible_doors:
                door_info = visible_doors[door_front_color]
                if door_info.get("state") == "locked":
                    if not carrying_key or carrying_key != door_front_color:
                        path_blocked = True
            
            # Rule 1: Pick up key if in front and not carrying one
            if not carrying_key and has_key_front:
                weights[batch_idx, 3] = self.conf_level
            
            # Rule 2: Open/unlock door if locked and have matching key
            elif has_door_front and door_front_color in visible_doors:
                door_info = visible_doors[door_front_color]
                if door_info.get("state") == "locked" and carrying_key:
                    if carrying_key == door_front_color:
                        weights[batch_idx, 5] = self.conf_level
            
            # Rule 3: Navigate toward goal when visible, unless path is blocked
            elif goal_data[1] and not path_blocked and not wall_state[2]:
                # Goal is directly in front and path is clear - move forward
                weights[batch_idx, 2] = self.conf_level
            elif goal_data[0] and not path_blocked:
                # Goal is visible with x-offset - navigate toward it
                if goal_data[0] < 0 and not wall_state[0]:  # Goal is to the left
                    weights[batch_idx, 0] = self.conf_level
                elif goal_data[0] > 0 and not wall_state[1]:  # Goal is to the right
                    weights[batch_idx, 1] = self.conf_level
                elif not wall_state[2]:  # No wall in front
                    weights[batch_idx, 2] = self.conf_level
        
        # Normalize weights to sum to 1.0 per observation
        weights = weights / weights.sum(1, keepdim=True)
        
        return weights


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def get_observables(raw_obs_batch):
    """
    Vectorized version of get_observables that processes entire batch at once
    Args:
        raw_obs_batch: shape (batch_size, H*W*C) or (batch_size, H, W, C)
    Returns:
        List of observation lists for each item in batch
    """
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

def train_c51(args, seed):
    """
    Single-process training function for C51 algorithm with rules
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False
    
    # Create a timestamped run name
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    run_name = f"C51rtPRODUCT_{args.env_id}_seed{seed}_{start_datetime}"
    
    # Set up device (CPU/GPU)
    device = args.device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup logging
    if args.track:
        import wandb
        wandb.tensorboard.patch(root_logdir=f"C51rtPRODUCT/runs_rules_training/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"C51rtPRODUCT_{args.run_code}",
        )
    
    writer = SummaryWriter(f"C51rtPRODUCT/runs_rules_training/{run_name}/train",
                         max_queue=100,
                         flush_secs=30)
                         
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []
    
    print(f'Using seed {seed}, device {device}')

    # Create environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, seed, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    # Initialize networks
    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Create replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Start the environment
    obs, _ = envs.reset(seed=seed)
    print_step = args.print_step
    print(f"Starting training with seed={seed}, device={device}")
    
    # Pre-allocate observation tensor
    obs_tensor = torch.zeros((1, *envs.single_observation_space.shape), dtype=torch.float32, device=device)
    
    # Training loop
    progress_bar = tqdm(
        range(args.total_timesteps),
        desc="Training",
        colour='blue',
        miniters=5000,
        maxinterval=10
    )
    
    for global_step in progress_bar:
        # Action selection with epsilon-greedy
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_tensor.copy_(torch.as_tensor(obs, dtype=torch.float32))
            with torch.no_grad():
                actions, _ = q_network.get_action(obs_tensor, epsilon=epsilon)
                actions = actions.cpu().numpy()

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episode statistics
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
                        mean_return = np.mean(episodes_returns[-args.print_step:])
                        mean_len = np.mean(episodes_lengths[-args.print_step:])
                        progress_bar.set_postfix_str(
                            f"mean_return={mean_return:.2f}, mean_len={mean_len:.2f}, epsilon={epsilon:.2f}"
                        )
                        print_step += args.print_step

        # Handle episode termination
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Add to replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training update
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            
            with torch.no_grad():
                # Get target distribution
                _, next_pmfs = target_network.get_action(data.next_observations.float())
                next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                
                # Distribution projection
                delta_z = target_network.atoms[1] - target_network.atoms[0]
                tz = next_atoms.clamp(args.v_min, args.v_max)
                b = (tz - args.v_min) / delta_z
                l = b.floor().clamp(0, args.n_atoms - 1)
                u = b.ceil().clamp(0, args.n_atoms - 1)
                
                # Calculate projected distribution
                d_m_l = (u + (l == u).float() - b) * next_pmfs
                d_m_u = (b - l) * next_pmfs
                target_pmfs = torch.zeros_like(next_pmfs)
                
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

            # Current network processing
            _, old_pmfs = q_network.get_action(data.observations.float(), data.actions.flatten())
            loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

            # Logging
            if global_step % 10000 == 0:
                writer.add_scalar("losses/loss", loss.item(), global_step)
                writer.add_scalar("SPS", int(global_step / (time.time() - start_time)), global_step)

            # Gradient step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # Save results
    model_path = f"C51rtPRODUCT/{args.env_id}_c51rtPRODUCT_{args.total_timesteps}_seed{seed}_{start_datetime}"
    if not os.path.exists("C51rtPRODUCT/"):
        os.makedirs("C51rtPRODUCT/")
    os.makedirs(model_path, exist_ok=True)

    # Plot return curve
    plt.figure()
    plt.plot(episodes_returns)
    plt.title(f'C51rtPRODUCT on {args.env_id} - Return over {args.total_timesteps} timesteps (seed {seed})')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.savefig(f"{model_path}/{args.env_id}_c51rtPRODUCT_{args.total_timesteps}_seed{seed}.png")
    plt.close()

    if args.save_model:
        model_file = f"{model_path}/c51rtPRODUCT_model.pt"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_file)
        print(f"Model saved to {model_file}")
        
        # Evaluate if requested
        if args.evaluate:
            from baseC51.c51_eval import QNetwork as QNetworkEval
            from c51rtPRODUCT_eval import evaluate
            eval_episodes=100
            episodic_returns = evaluate(
                model_file,
                make_env,
                args.env_id,
                eval_episodes=eval_episodes,
                run_name=f"{run_name}-eval",
                Model=QNetworkEval,
                device=device,
                epsilon=0
            )
            
            eval_writer = SummaryWriter(f"C51rtPRODUCT/runs_rules_training/{run_name}/eval")
            for idx, episodic_return in enumerate(episodic_returns):
                eval_writer.add_scalar("episodic_return", episodic_return, idx)
            eval_writer.close()

    # Cleanup
    envs.close()
    writer.close()
    
    return episodes_returns, model_path

if __name__ == "__main__":
    # Configure for performance
    torch.backends.cudnn.benchmark = False  # More consistent behavior
    
    # Parse arguments
    args = tyro.cli(Args)
    
    # Get seed from command line or use default
    seed = args.seed
    print(f"Training C51 Rules (PRODUCT) with seed: {seed}")
    
    # Run training
    returns, model_path = train_c51(args, seed)
    
    print(f"Training completed! Model saved to {model_path}")
    print(f"Final mean return for last 10 episodes: {np.mean(returns[-10:]):.2f}")