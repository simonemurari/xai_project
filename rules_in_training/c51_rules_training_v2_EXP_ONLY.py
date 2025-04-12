import os
import random
import time
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Args
from tqdm import tqdm
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

# Rules applied to the C51 algorithm only during training (v2)

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



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            env = gym.wrappers.FlattenObservation(
                gym.wrappers.FilterObservation(env, filter_keys=["image", "direction"])
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


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

        # Define action mappings (adjust as needed based on your environment)
        self.action_map = {
            "left": 0,  # Turn left
            "right": 1,  # Turn right
            "forward": 2,  # Move forward
            "pickup": 3,  # Pickup object
            "toggle": 5,  # Open door
        }


    def get_action(self, x, action=None, observables=None, epsilon=0.0, global_step=None):
        """
        Enhanced action selection that combines C51 distribution with rule-based guidance
        using proper policy shaping (multiplying distributions)
        """
        
        batch_size = len(x)

        # Get distributional Q-values from the network
        logits = self.network(x)
        pmfs = torch.softmax(logits.view(batch_size, self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)

        # If no observables provided, use standard C51 logic
        if observables is None:
            if action is None:
                action = torch.argmax(q_values, 1)
            return action, pmfs[torch.arange(batch_size), action]

        # Apply rule-based action guidance to the batch
        rule_action = self._apply_rules_batch(observables)[0]

        # Rule influence parameter
        rule_influence = 0.8 * epsilon + 0.2

        dist = torch.ones((self.n), device=x.device) * (1 - rule_influence) / (self.n - 1)
        dist[rule_action] = rule_influence
        
        # Normalize distribution
        dist = dist / dist.sum()
        
        # Shape the policy by multiplying Q-values with rule distribution
        shaped_q_values = q_values * dist
        final_action = torch.argmax(shaped_q_values, dim=1)
           
        return final_action, pmfs[torch.arange(batch_size), final_action]

    def _apply_rules_batch(self, batch_observables):
        """Apply rules to each environment observation in the batch"""
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

        Args:
            target_x: Relative x-coordinate of the target
            target_y: Relative y-coordinate of the target
            walls: List of wall observations with their positions
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


def train_c51(args, seed):
    """
    Single-seed training function
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False

    # Set number of threads appropriately
    torch.set_num_threads(4)  # Adjust based on your CPU

    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    run_name = f"C51rtv2_{args.env_id}_seed{seed}_{start_datetime}"

    # Setup logging only if tracking is enabled
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
            group=f"C51rtv2EXPONLY_{args.exploration_fraction}_{args.run_code}",
        )

    # Set up TensorBoard writer
    writer = SummaryWriter(
        f"C51rtv2/runs_rules_training/{run_name}/train", max_queue=100, flush_secs=30
    )

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []

    # Print seed info
    print(f"File: {os.path.basename(__file__)}, using seed {seed}")
    
    device = args.device
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed(seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                seed,
                i,
                args.capture_video and i == 0,
                run_name,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    q_network = QNetwork(
        envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max
    ).to(device)
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

    # Start the game
    obs, _ = envs.reset(seed=seed)
    print_step = args.print_step
    print(f"Starting training with seed {seed}, device {device}")

    # Pre-allocate observation tensor for efficiency
    obs_tensor = torch.zeros(
        (1, *envs.single_observation_space.shape), dtype=torch.float32, device=device
    )

    # Create tqdm progress bar
    progress_bar = tqdm(
        range(args.total_timesteps),
        desc=f"Training seed {seed}",
        colour="blue",
        miniters=5000,
        maxinterval=10,
    )
    
    for global_step in progress_bar:
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        # Action selection
        if random.random() < epsilon:
            obs_tensor.copy_(torch.as_tensor(obs, dtype=torch.float32))
            obs_img = q_network.get_observables(obs_tensor[:, 4:])
            # Apply rule-based action guidance to the batch
            rule_action = q_network._apply_rules_batch(obs_img)[0]

            # Rule influence parameter
            rule_influence = 0.8 * epsilon + 0.2

            dist = torch.ones((q_network.n), device=Args.device) * (1 - rule_influence) / (q_network.n - 1)
            dist[rule_action] = rule_influence
            
            # Normalize distribution
            dist = dist / dist.sum()

            actions = np.random.choice(
                q_network.n, p=dist.cpu().numpy(), size=(envs.num_envs,)
            )
        else:
            with torch.no_grad():
                actions, _ = q_network.get_action(
                    obs_tensor, observables=None, epsilon=epsilon,
                    global_step=global_step
                )
                actions = actions.cpu().numpy()

        # Execute the game and log data
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record rewards for plotting purposes
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
                        mean_return = np.mean(episodes_returns[-args.print_step :])
                        mean_len = np.mean(episodes_lengths[-args.print_step :])
                        progress_bar.set_postfix({
                            "mean_return": f"{mean_return:.2f}",
                            "mean_len": f"{mean_len:.2f}",
                            "epsilon": f"{epsilon:.2f}",
                            "rule_influence": f"{0.8 * epsilon + 0.2:.2f}"
                        })
                        print_step += args.print_step

        # Save data to replay buffer; handle final_observation
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # Update current observation
        obs = next_obs

        # ALGO LOGIC: training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    # Target network processing
                    _, next_pmfs = target_network.get_action(
                        data.next_observations.float(), observables=None
                    )
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (
                        1 - data.dones
                    )

                    # Projection with optimized tensor operations
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)
                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)

                    # Optimized distribution projection
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)

                    # Vectorized index_add for better performance
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                # Current network processing
                _, old_pmfs = q_network.get_action(
                    data.observations.float(), data.actions.flatten(), observables=None
                )
                loss = (
                    -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(
                        -1
                    )
                ).mean()

                # Log less frequently for better performance
                if global_step % 10000 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # Optimize more efficiently
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network when needed
                if global_step % args.target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())

    # Save results
    model_path = f"C51rtv2/{args.env_id}_c51rtv2_{args.total_timesteps}_seed{seed}_{start_datetime}"
    if not os.path.exists("C51rtv2/"):
        os.makedirs("C51rtv2/")
    os.makedirs(model_path, exist_ok=True)

    if args.save_model:
        model_file = f"{model_path}/c51rtv2_model.pt"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_file)
        print(f"Model saved to {model_file}")

        # Evaluation if requested
        if args.evaluate:
            from baseC51.c51_eval import QNetwork as QNetworkEval
            from c51rtv2_eval import evaluate

            eval_episodes = 100
            episodic_returns = evaluate(
                model_file,
                make_env,
                args.env_id,
                eval_episodes=eval_episodes,
                run_name=f"{run_name}-eval",
                Model=QNetworkEval,
                device=device,
                epsilon=0,
            )

            eval_writer = SummaryWriter(f"C51rtv2/runs_rules_training/{run_name}/eval")
            for idx, episodic_return in enumerate(episodic_returns):
                eval_writer.add_scalar("episodic_return", episodic_return, idx)
            eval_writer.close()

    envs.close()
    writer.close()
    
    return episodes_returns, model_path


if __name__ == "__main__":

    # Parse arguments
    args = tyro.cli(Args)
    
    # Use the seed from args
    seed = args.seed
    print(f"Starting C51 Rules Training with seed: {seed}")
    
    # Run training with a single seed
    returns, model_path = train_c51(args, seed)
    
    print(f"Training completed for seed {seed}!")
    print(f"Model saved to: {model_path}")
    print(f"Final mean return (last 10 episodes): {np.mean(returns[-10:]):.2f}")