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
import minigrid
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
from dotenv import load_dotenv
load_dotenv(".env")
WANDB_KEY = os.getenv("WANDB_KEY")
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}, device = {Args.device}")

# Pre-computed constant arrays for observation processing
DOOR_STATES = ["open", "closed", "locked"]
VIEW_SIZE = 7
MID_POINT = (VIEW_SIZE - 1) // 2

# Pre-computed meshgrid for faster observation processing
OFFSETS_X, OFFSETS_Y = np.meshgrid(
    np.arange(VIEW_SIZE) - MID_POINT,
    np.abs(np.arange(VIEW_SIZE) - (VIEW_SIZE - 1)),
    indexing="ij",
)

def _plot_pmfs(pmfs, action_index, n_categories, alpha, title_prefix, episode_step=0, plot_type="exploit", suggested_action=None):
    """
    Plots the PMFs for the network showing which action was suggested by rules.
    """
    # Make sure the input tensors are in CPU
    pmfs = pmfs.cpu().detach().numpy()

    # Extract the PMF for the specified action
    pmf = pmfs[0, action_index]

    # Map the range to value range
    x = np.linspace(Args.v_min, Args.v_max, n_categories)
    
    # Create subplot
    plt.figure(figsize=(8, 5))
    
    # Color based on whether this action was suggested
    color = "red" if action_index == suggested_action else "skyblue"
    label = "Network PMF (SUGGESTED)" if action_index == suggested_action else "Network PMF"
    
    # Plot the network PMF
    plt.plot(x, pmf, label=label, color=color, linewidth=2)
    plt.title(f"PMF_{title_prefix}_EP={episode_step}_Eps={alpha:.2f}_Act={action_index}")
    plt.xlim(Args.v_min - 0.05, Args.v_max + 0.05)
    plt.ylim(0, 1.05 * max(pmf.max(), 0.01))
    plt.xlabel("Return")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory structure based on epsilon
    plot_dir = Path(f"VBaseplots/{args.run_code}/epsilon_{alpha:.2f}/{plot_type}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot with a detailed filename
    plt.savefig(plot_dir / f"{title_prefix}_EPstep={episode_step}.png")
    plt.close()

def make_env(env_id, n_keys, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, n_keys=n_keys, render_mode="rgb_array")
            env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env, filter_keys=['image', 'direction']))
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
        
        # --- Plotting State ---
        self.plotting_epsilons = {0.8, 0.6, 0.4, 0.2}
        self.plotted_epsilons = set()
        self.is_plotting_episode = False
        self.exploit_plot_step_count = 0
        self.train_plot_step_count = 0
        self.plotting_episode_epsilon = 0
        self.max_plotting_steps = 10
        self.episode_steps_count = 0  # Track steps in the current episode for plotting
        
        # Define action mappings (adjust as needed based on your environment)
        self.action_map = {
            "left": 0,  # Turn left
            "right": 1,  # Turn right
            "forward": 2,  # Move forward
            "pickup": 3,  # Pickup object
            "drop (UNUSED)": 4,  # Drop object
            "toggle": 5,  # Open door
            "done (UNUSED)": 6,  # End episode
        }
        # Create a reverse map for easy lookup of action names
        self.action_id_to_name = {v: k for k, v in self.action_map.items()}
        
    def get_action(self, x, action=None, epsilon=1.0, global_step=0, is_exploit_step=False):
        """Enhanced action selection with rule visualization"""
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        
        if action is None:
            action = torch.argmax(q_values, 1)
        
        # --- Episodic Plotting Activation ---
        if is_exploit_step and not self.is_plotting_episode:
            for plot_eps in self.plotting_epsilons:
                if epsilon <= plot_eps and plot_eps not in self.plotted_epsilons:
                    self.is_plotting_episode = True
                    self.plotting_episode_epsilon = plot_eps
                    self.plotted_epsilons.add(plot_eps)
                    # Reset counters for the new plotting episode
                    self.exploit_plot_step_count = 0
                    self.train_plot_step_count = 0
                    print(f"--- Starting plotting episode for epsilon {plot_eps} (Global Step: {global_step}) ---")
                    break

        if not self.is_plotting_episode: # Only check if not already plotting
            for plot_eps in self.plotting_epsilons:
                if epsilon <= plot_eps and plot_eps not in self.plotted_epsilons:
                    self.is_plotting_episode = True
                    self.plotting_episode_epsilon = plot_eps
                    self.plotted_epsilons.add(plot_eps)
                    print(f"--- Starting plotting episode for epsilon {plot_eps} (Global Step: {global_step}) ---")
                    break

        # --- In-Episode Plotting Execution ---
        if self.is_plotting_episode:
            # Only process observables when we're actually plotting
            suggested_actions = None
            if len(x) > 0:
                try:
                    suggested_actions = self._get_rule_suggestions_batch(self.get_observables(x[:, 4:]))
                except Exception as e:
                    print(f"Error processing observables: {e}")
                    suggested_actions = [None] * len(x)
            
            # Only plot if there are actual valid action suggestions (not None and not -1)
            has_valid_suggestions = (suggested_actions and 
                                any(action is not None and action != -1 and action >= 0 
                                    for action in suggested_actions))
            
            if has_valid_suggestions:
                plot_this_step = False
                plot_type = ""
                current_plot_step = 0

                if is_exploit_step and self.exploit_plot_step_count < self.max_plotting_steps:
                    plot_this_step = True
                    plot_type = "exploit"
                    current_plot_step = self.exploit_plot_step_count
                    self.exploit_plot_step_count += 1
                
                elif not is_exploit_step and self.train_plot_step_count < self.max_plotting_steps:
                    plot_this_step = True
                    plot_type = "training"
                    current_plot_step = self.train_plot_step_count
                    self.train_plot_step_count += 1

                if plot_this_step:
                    plot_idx = 0  # Plot the first item in the batch
                    suggested_action = suggested_actions[plot_idx] if (suggested_actions and 
                                                                    suggested_actions[plot_idx] is not None and 
                                                                    suggested_actions[plot_idx] >= 0) else -1

                    # Only proceed if we have a valid suggestion for this specific observation
                    if suggested_action >= 0:

                        # Plot PMF for every action
                        for act_id in range(self.n):
                            action_name = self.action_id_to_name.get(act_id, f"Action_{act_id}")
                            is_suggested = "SUGGESTED" if act_id == suggested_action else "NOT_SUGGESTED"
                            title_prefix = f"GLstep={global_step}_{is_suggested}_{action_name}"

                            _plot_pmfs(
                                pmfs,
                                act_id,
                                self.n_atoms,
                                self.plotting_episode_epsilon,
                                title_prefix,
                                episode_step=self.episode_steps_count,
                                plot_type=plot_type,
                                suggested_action=suggested_action,
                            )
                        
                        # --- Custom Logging (Exploit vs. Training) ---
                        plot_dir = Path(f"VBaseplots/{args.run_code}/epsilon_{self.plotting_episode_epsilon:.2f}/{plot_type}")
                        plot_dir.mkdir(parents=True, exist_ok=True)

                        if plot_type == "exploit":
                            try:
                                grid_img = self.env.envs[0].render()
                                plt.imsave(plot_dir / f"GLstep={global_step}_EPstep={self.episode_steps_count}_GRID.png", grid_img)
                            except Exception as e:
                                print(f"Error saving grid image: {e}")
                        
                        elif plot_type == "training":
                            try:
                                obs_for_log = self.get_observables(x[plot_idx].unsqueeze(0).cpu().numpy()[:, 4:])
                                log_filename = plot_dir / f"GLstep={global_step}_EPstep={self.episode_steps_count}_OBSERVABLE.txt"
                                with open(log_filename, "w") as f:
                                    import json
                                    f.write(json.dumps(obs_for_log, indent=2, default=str))
                            except Exception as e:
                                print(f"Error saving observable log: {e}")

                        print(f"--- Plotting step {current_plot_step + 1}/{self.max_plotting_steps} ({plot_type}) for epsilon = {self.plotting_episode_epsilon} with suggestion: {suggested_action} ---")
        return action, pmfs[torch.arange(len(x)), action]

    def _get_rule_suggestions_batch(self, batch_observables):
        """Get what rules would suggest for each observation in the batch (for visualization only)"""
        rule_actions = []
        
        for observables in batch_observables:
            # Parse observables
            keys = [o for o in observables if o[0] == "key"]
            doors = [o for o in observables if o[0] == "door"]
            goals = [o for o in observables if o[0] == "goal"]
            walls = [o for o in observables if o[0] == "wall"]
            carrying_keys = [o for o in observables if o[0] == "carryingKey"]
            locked_doors = [o for o in observables if o[0] == "locked"]
            closed_doors = [o for o in observables if o[0] == "closed"]

            # Rule 1: pickup(X) :- key(X), samecolor(X,Y), door(Y), notcarrying
            if keys and doors and not carrying_keys:
                for key in keys:
                    key_color = key[1][0]
                    matching_doors = [door for door in doors if door[1][0] == key_color]
                    if matching_doors:
                        # Check if key is directly in front
                        key_x, key_y = key[1][1], key[1][2]
                        if key_x == 0 and key_y == 1:  # Key is directly in front
                            rule_actions.append(self.action_map["pickup"])
                            break
                        else:
                            # Move towards the key with wall avoidance
                            action = self._navigate_towards(key_x, key_y, walls)
                            rule_actions.append(action)
                            break
                else:
                    rule_actions.append(None)  # No applicable key found

            # Rule 2: open(X) :- door(X), locked(X), key(Z), carryingKey(Z), samecolor(X,Z)
            elif doors and locked_doors and carrying_keys:
                carrying_key_color = carrying_keys[0][1][0]

                # Check locked doors first (priority)
                matching_doors_to_open = []
                if locked_doors:
                    for door in doors:
                        door_color = door[1][0]
                        if door_color == carrying_key_color:
                            for locked in locked_doors:
                                if locked[1][0] == door_color:
                                    matching_doors_to_open.append(door)

                if matching_doors_to_open:
                    door = matching_doors_to_open[0]
                    door_x, door_y = door[1][1], door[1][2]
                    if door_x == 0 and door_y == 1:  # Door is directly in front
                        rule_actions.append(self.action_map["toggle"])
                    else:
                        # Move towards the door with wall avoidance
                        action = self._navigate_towards(door_x, door_y, walls)
                        rule_actions.append(action)
                else:
                    rule_actions.append(None)

            # Rule 3: goto :- goal(X), unlocked
            elif goals:
                goal = goals[0]
                goal_x, goal_y = goal[1][0], goal[1][1]

                # Check if there's a clear path to the goal (no closed/locked doors in the way)
                blocked_by_door = False

                # Simple check: if we see a closed/locked door that's between us and the goal
                direction_to_goal = (
                    1 if goal_x > 0 else (-1 if goal_x < 0 else 0),
                    1 if goal_y > 0 else (-1 if goal_y < 0 else 0),
                )

                # Only consider a door blocking if it's in the same general direction as the goal
                for door in doors:
                    door_x, door_y = door[1][1], door[1][2]
                    door_direction = (
                        1 if door_x > 0 else (-1 if door_x < 0 else 0),
                        1 if door_y > 0 else (-1 if door_y < 0 else 0),
                    )

                    door_color = door[1][0]
                    # Check if the door is in the same general direction as the goal
                    same_direction = (
                        direction_to_goal[0] == door_direction[0]
                        and direction_to_goal[1] == door_direction[1]
                    )

                    # Check if door is closer than the goal
                    door_distance = abs(door_x) + abs(door_y)
                    goal_distance = abs(goal_x) + abs(goal_y)
                    door_is_closer = door_distance < goal_distance

                    # Check if the door is closed or locked
                    door_is_closed = any(cd[1][0] == door_color for cd in closed_doors)
                    door_is_locked = any(ld[1][0] == door_color for ld in locked_doors)

                    if (
                        same_direction
                        and door_is_closer
                        and (door_is_closed or door_is_locked)
                    ):
                        blocked_by_door = True
                        break

                if not blocked_by_door:
                    if goal_x == 0 and goal_y == 1:  # Goal is directly in front
                        rule_actions.append(self.action_map["forward"])
                    else:
                        # Move towards the goal with wall avoidance
                        action = self._navigate_towards(goal_x, goal_y, walls)
                        rule_actions.append(action)
                else:
                    rule_actions.append(None)
            else:
                rule_actions.append(None)  # No applicable rule

        return rule_actions

    def _navigate_towards(self, target_x, target_y, walls=None):
        """
        Improved navigation helper that avoids walls when moving towards a target
        """
        # If no walls, use simpler navigation
        if not walls:
            if target_y > 0:  # Target is in front
                return self.action_map["forward"]
            elif target_x < 0:  # Target is to the left
                return self.action_map["left"]
            elif target_x > 0:  # Target is to the right
                return self.action_map["right"]
            else:  # Target is behind, turn around
                return self.action_map["right"]

        # Check if there's a wall directly in front
        wall_in_front = any(w[1][0] == 0 and w[1][1] == 1 for w in walls)

        # Determine the relative position of the target
        if target_y > 0:  # Target is in front
            if not wall_in_front:
                return self.action_map["forward"]
            else:
                # Wall blocking forward movement, turn to find another path
                return (
                    self.action_map["left"]
                    if target_x <= 0
                    else self.action_map["right"]
                )
        elif target_x < 0:  # Target is to the left
            return self.action_map["left"]
        elif target_x > 0:  # Target is to the right
            return self.action_map["right"]
        else:  # Target is behind
            # Choose a turn direction based on wall presence
            wall_to_left = any(w[1][0] == -1 and w[1][1] == 0 for w in walls)
            if wall_to_left:
                return self.action_map["right"]
            else:
                return self.action_map["left"]

    # Optimize observation processing with NumPy
    def get_observables(self, raw_obs_batch):
        """
        Highly optimized version of get_observables that processes entire batch at once
        """
        batch_size = raw_obs_batch.shape[0]

        # Convert to NumPy once if needed
        if isinstance(raw_obs_batch, torch.Tensor):
            raw_obs_batch = raw_obs_batch.cpu().numpy()

        # Reshape efficiently with pre-computed shape
        try:
            img_batch = raw_obs_batch.reshape(batch_size, VIEW_SIZE, VIEW_SIZE, 3)
        except ValueError:
            # Handle case where dimensions don't match by taking only the image part
            img_batch = raw_obs_batch[:, : VIEW_SIZE * VIEW_SIZE * 3].reshape(
                batch_size, VIEW_SIZE, VIEW_SIZE, 3
            )

        # Process batch items in parallel
        batch_obs = []

        # Process each batch item with minimal Python overhead
        for img in img_batch:
            obs = []
            item_first = img[..., 0]
            item_second = img[..., 1]
            item_third = img[..., 2]

            # Find all object positions efficiently with NumPy
            key_positions = np.where(item_first == 5)
            door_positions = np.where(item_first == 4)
            goal_positions = np.where(item_first == 8)
            wall_positions = np.where(item_first == 2)

            # Vectorized processing for keys
            for k_i, k_j in zip(*key_positions):
                color = IDX_TO_COLOR.get(item_second[k_i, k_j])
                obs.append(("key", [color, OFFSETS_X[k_i, k_j], OFFSETS_Y[k_i, k_j]]))
                if k_i == MID_POINT and k_j == VIEW_SIZE - 1:
                    obs.append(("carryingKey", [color]))

            # Vectorized processing for doors
            for d_i, d_j in zip(*door_positions):
                color = IDX_TO_COLOR.get(item_second[d_i, d_j])
                obs.append(("door", [color, OFFSETS_X[d_i, d_j], OFFSETS_Y[d_i, d_j]]))
                # Get the door state from the third channel
                door_state_idx = int(item_third[d_i, d_j])
                obs.append((DOOR_STATES[door_state_idx], [color]))

            # Vectorized processing for goals and walls
            for g_i, g_j in zip(*goal_positions):
                obs.append(("goal", [OFFSETS_X[g_i, g_j], OFFSETS_Y[g_i, g_j]]))

            for w_i, w_j in zip(*wall_positions):
                obs.append(("wall", [OFFSETS_X[w_i, w_j], OFFSETS_Y[w_i, w_j]]))

            batch_obs.append(obs)

        return batch_obs

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"C51_plotting_{args.env_id}__seed={args.seed}__{start_datetime}"
    if args.track:
        import wandb
        # wandb.login(key=WANDB_KEY)
        wandb.tensorboard.patch(root_logdir=f"C51/runs/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"C51_plotting_{args.exploration_fraction}_{args.run_code}",
        )
    writer = SummaryWriter(f"C51/runs/{run_name}/train")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []
    len_episodes_returns = 0
    
    # TRY NOT TO MODIFY: seeding
    print(f'File: {os.path.basename(__file__)}, using seed {args.seed} and exploration fraction {args.exploration_fraction}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.n_keys, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
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
    print(
        f"Starting training for {args.total_timesteps} timesteps on {args.env_id} with {args.n_keys} keys, with print_step={print_step}"
    )
    for global_step in tqdm(range(args.total_timesteps), colour='green'):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            is_exploit_step = False
        else:
            actions, pmf = q_network.get_action(torch.Tensor(obs).float().to(device), epsilon=epsilon, global_step=global_step, is_exploit_step=True)
            actions = actions.cpu().numpy()
            is_exploit_step = True

        q_network.episode_steps_count += 1

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # --- Reset plotting state on episode end ---
        if terminations[0] or truncations[0]:
            q_network.is_plotting_episode = False
            q_network.episode_steps_count = 0

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
                        old_len_episodes_returns = len_episodes_returns
                        len_episodes_returns = len(episodes_returns)
                        print_num_eps = len_episodes_returns - old_len_episodes_returns
                        mean_ep_return = np.mean(episodes_returns[-print_num_eps:])
                        mean_ep_lengths = np.mean(episodes_lengths[-print_num_eps:])
                        tot_mean_return = np.mean(episodes_returns)
                        tot_mean_length = np.mean(episodes_lengths)
                        tqdm.write(
                            f"global_step={global_step}, mean_return_last_{print_num_eps}_episodes={mean_ep_return}, tot_mean_ret={tot_mean_return}, mean_length_last_{print_num_eps}_episodes={mean_ep_lengths}, tot_mean_len={tot_mean_length}, epsilon={epsilon:.2f}"
                        )
                        print_step += args.print_step

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
                        
                # Add plotting for training steps
                if q_network.is_plotting_episode:
                    _, train_pmfs = q_network.get_action(
                        data.observations.float()[:1], 
                        epsilon=epsilon, 
                        global_step=global_step, 
                        is_exploit_step=False
                    )
                        
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