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
from collections import namedtuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
# print(f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}")

# Rules applied to the C51 algorithm only during training (v3)

# Add the same RuleAugmentedReplayBufferSamples namedtuple as c51_rules_training_v3_ws.py
RuleAugmentedReplayBufferSamples = namedtuple(
    "RuleAugmentedReplayBufferSamples",
    [
        "observations",
        "actions",
        "next_observations",
        "dones",
        "rewards",
        "rule_suggestions",
    ],
)

# print(f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}")

# Rules applied to the C51 algorithm only during training (v3)

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


# Add the RuleAugmentedReplayBuffer class to extend the ReplayBuffer to also store rule-suggested actions
class RuleAugmentedReplayBuffer(ReplayBuffer):
    """Extends ReplayBuffer to also store rule-suggested actions"""

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        handle_timeout_termination=True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            handle_timeout_termination=handle_timeout_termination,
        )
        # Add buffer for rule suggestions - initialize with None (-1 as sentinel value)
        self.rule_suggestions = np.full((self.buffer_size,), -1, dtype=np.int8)

    def add(self, obs, next_obs, actions, rewards, dones, infos, rule_suggestions=None):
        # Batch operations instead of loop
        batch_size = len(obs)
        indices = np.arange(self.pos, self.pos + batch_size) % self.buffer_size

        # Use NumPy's vectorized operations
        self.observations[indices] = np.array(obs)
        self.next_observations[indices] = np.array(next_obs)
        self.actions[indices] = np.array(actions)
        self.rewards[indices] = np.array(rewards)
        self.dones[indices] = np.array(dones)

        if rule_suggestions is not None:
            self.rule_suggestions[indices] = np.array(
                [rs if rs is not None else -1 for rs in rule_suggestions]
            )

        self.pos = (self.pos + batch_size) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        """Sample data and also return rule suggestions"""
        # Call the parent's sample method to get the base fields
        batch = super().sample(batch_size)

        # Generate our own batch indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Get rule suggestions for the sampled indices
        rule_suggestions = self.rule_suggestions[batch_inds]
        # Convert -1 back to None for consistency
        rule_suggestions_list = [None if r == -1 else r for r in rule_suggestions]

        # Create a new namedtuple with all the fields including rule_suggestions
        return RuleAugmentedReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=batch.rewards,
            rule_suggestions=rule_suggestions_list,
        )
    

def _plot_pmfs(
    original_pmfs, modified_pmfs, action_index, n_categories, alpha, title_prefix, rule_influence=1.0, episode_step=0, plot_type="exploit"
):
    """
    Plots the PMFs for original network action, modified action, and their combined probabilities.
    """
    # Make sure the input tensors are in CPU
    original_pmfs = original_pmfs.cpu().detach().numpy()
    modified_pmfs = modified_pmfs.cpu().detach().numpy()

    # Extract the PMFs for the specified action
    original_pmf = original_pmfs[0, action_index]
    modified_pmf = modified_pmfs[0, action_index]

    # Map the range to value range
    x = np.linspace(Args.v_min, Args.v_max, n_categories)
    # Create subplots
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the original network PMF
    axs[0].plot(x, original_pmf, label="Original PMF", color="skyblue")
    axs[0].set_title(f"PMF_{title_prefix}_EP={episode_step}_Eps={alpha:.2f}_Act={action_index}")
    axs[0].set_xlim(Args.v_min - 0.05, Args.v_max + 0.05)
    axs[0].set_ylim(0, 1.05 * max(original_pmf.max(), modified_pmf.max(), 0.01))
    axs[0].set_xlabel("Return")
    axs[0].set_ylabel("Probability")
    axs[0].legend()

    # Plot the modified PMF (after rule application)
    axs[1].plot(x, modified_pmf, label="Modified PMF", color="salmon")
    axs[1].set_title(f"NEWPMF_{title_prefix}_EP={episode_step}_Eps={alpha:.2f}_Act={action_index}")
    axs[1].set_xlim(Args.v_min - 0.05, Args.v_max + 0.05)
    axs[1].set_ylim(0, 1.05 * max(original_pmf.max(), modified_pmf.max(), 0.01))
    axs[1].set_xlabel("Return")
    axs[1].set_ylabel("Probability")
    axs[1].legend()
    plt.tight_layout()
    
    # Create directory structure based on epsilon and rule influence
    plot_dir = Path(f"V2aplots/{args.run_code}/epsilon_{alpha:.2f}/rule_influence_{rule_influence:.2f}/{plot_type}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot with a detailed filename
    plt.savefig(plot_dir / f"{title_prefix}_EPstep={episode_step}.png")
    plt.close()


def make_env(env_id, seed, n_keys, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, n_keys=n_keys, render_mode="rgb_array")
            env = gym.wrappers.FlattenObservation(
                gym.wrappers.FilterObservation(env, filter_keys=["image", "direction"])
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
            "drop (UNUSED)": 4,  # Drop object (not used in this context)
            "toggle": 5,  # Open door
            "done (UNUSED)": 6,  # Mark as done (not used in this context)
        }
        # Create a reverse map for easy lookup of action names
        self.action_id_to_name = {v: k for k, v in self.action_map.items()}
        self.conf_level = 0.8

    def get_action(self, x, stored_rule_actions=None, action=None, skip=False, epsilon=1.0, rule_influence=0.5, global_step=0, is_exploit_step=False):
        """
        Vectorized action selection with rule guidance for improved performance.
        The logic is functionally identical to the original but avoids slow Python loops.
        """
        batch_size = x.shape[0]
        device = x.device

        # Get distributional Q-values from the network
        logits = self.network(x)
        pmfs = torch.softmax(logits.view(batch_size, self.n, self.n_atoms), dim=2)

        if skip:
            q_values = (pmfs * self.atoms).sum(2)
            if action is None:
                action = torch.argmax(q_values, 1)
            return action, pmfs[torch.arange(batch_size), action]

        # Get rule suggestions (this part still involves a Python loop in _apply_rules_batch)
        rule_actions_list = (
            self._apply_rules_batch(self.get_observables(x[:, 4:]))
            if stored_rule_actions is None
            else stored_rule_actions
        )

        # Convert rule suggestions (which can contain None) into a tensor for masking
        # Use -1 as a sentinel value for 'no rule applied'
        rule_actions_tensor = torch.tensor(
            [r if r is not None else -1 for r in rule_actions_list],
            dtype=torch.long,
            device=device,
        )

        # Create a boolean mask for batch items that have a rule suggestion
        has_rule_mask = rule_actions_tensor != -1

        # If no rules were triggered in the entire batch, we can return early
        if not torch.any(has_rule_mask):
            q_values = (pmfs * self.atoms).sum(2)
            if action is None:
                action = torch.argmax(q_values, dim=1)
            return action, pmfs[torch.arange(batch_size), action], rule_actions_list

        # --- Vectorized PMF Shifting ---
        combined_pmfs = pmfs.clone()
        shift_amount = int(epsilon * rule_influence * self.n_atoms)

        if shift_amount > 0:
            # Create a mask for actions that should be penalized. This is True where:
            # 1. A rule exists for the batch item (has_rule_mask)
            # 2. The action is NOT the one suggested by the rule
            all_actions_grid = torch.arange(self.n, device=device).expand(batch_size, self.n)
            actions_to_penalize_mask = has_rule_mask.unsqueeze(1) & (
                all_actions_grid != rule_actions_tensor.unsqueeze(1)
            )

            # Select only the PMFs that need to be shifted
            pmfs_to_shift = combined_pmfs[actions_to_penalize_mask]

            if pmfs_to_shift.numel() > 0:
                # Perform the left shift vectorially
                if shift_amount < self.n_atoms:
                    # Create zero padding for the right side of the distribution
                    padding = torch.zeros(
                        (pmfs_to_shift.shape[0], shift_amount), device=device
                    )
                    # Concatenate the sliced end of the PMF with the padding
                    shifted_pmfs = torch.cat(
                        (pmfs_to_shift[:, shift_amount:], padding), dim=1
                    )
                else:
                    # If shift amount is too large, move all probability mass to the first atom
                    shifted_pmfs = torch.zeros_like(pmfs_to_shift)
                    shifted_pmfs[:, 0] = pmfs_to_shift.sum(dim=1)

                # Place the modified PMFs back into the main tensor
                combined_pmfs[actions_to_penalize_mask] = shifted_pmfs

         # --- Episodic Plotting Activation (only during exploitation) ---
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
        # --- In-Episode Plotting Execution ---
        if self.is_plotting_episode:
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
                suggested_action = rule_actions_tensor[plot_idx].item()
                
                original_pmf_for_plot = pmfs[plot_idx].unsqueeze(0)
                modified_pmf_for_plot = combined_pmfs[plot_idx].unsqueeze(0)

                # Plot PMF for every action
                for act_id in range(self.n):
                    action_name = self.action_id_to_name.get(act_id, f"Action_{act_id}")
                    is_suggested = "SUGGESTED" if act_id == suggested_action else "NOT_SUGGESTED"
                    title_prefix = f"GLstep={global_step}_{is_suggested}_{action_name}"

                    _plot_pmfs(
                        original_pmf_for_plot,
                        modified_pmf_for_plot,
                        act_id,
                        self.n_atoms,
                        self.plotting_episode_epsilon,
                        title_prefix,
                        rule_influence,
                        episode_step=self.episode_steps_count,
                        plot_type=plot_type,
                    )
                
                # --- Custom Logging (Exploit vs. Training) ---
                plot_dir = Path(f"V2aplots/{args.run_code}/epsilon_{self.plotting_episode_epsilon:.2f}/rule_influence_{rule_influence:.2f}/{plot_type}")
                plot_dir.mkdir(parents=True, exist_ok=True)

                if plot_type == "exploit":
                    grid_img = self.env.envs[0].render()
                    plt.imsave(plot_dir / f"GLstep={global_step}_EPstep={self.episode_steps_count}_GRID.png", grid_img)
                
                elif plot_type == "training":
                    obs_for_log = self.get_observables(x[plot_idx].unsqueeze(0).cpu().numpy()[:, 4:])
                    log_filename = plot_dir / f"GLstep={global_step}_EPstep={self.episode_steps_count}_OBSERVABLE.txt"
                    with open(log_filename, "w") as f:
                        import json
                        f.write(json.dumps(obs_for_log, indent=2, default=str))

                print(f"--- Plotting step {current_plot_step + 1}/{self.max_plotting_steps} ({plot_type}) for epsilon = {self.plotting_episode_epsilon} ---")

        # --- Vectorized Normalization ---
        # Normalize the distributions only for batch items where a rule was applied.
        # This ensures that each action's probability distribution sums to 1.
        pmfs_with_rules = combined_pmfs[has_rule_mask]
        sums = pmfs_with_rules.sum(dim=2, keepdim=True)
        combined_pmfs[has_rule_mask] = pmfs_with_rules / sums

        # --- Final Calculations ---
        q_values = (combined_pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, dim=1)

        return action, combined_pmfs[torch.arange(batch_size), action], rule_actions_list


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
    
    def _get_weights(self, batch_of_observables):
        """
        Apply rules to observations and return action weights for each observation
        """
        device = self.atoms.device
        batch_size = len(batch_of_observables)
        
        # Pre-allocate tensor with default low confidence
        weights = torch.full((batch_size, self.n), 1 - self.conf_level, device=device)
        
        for batch_idx, observables in enumerate(batch_of_observables):
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
                            weights[batch_idx, 3] = self.conf_level  # pickup action
                            break
                        else:
                            # Move towards the key with wall avoidance
                            action = self._navigate_towards(key_x, key_y, walls)
                            weights[batch_idx, action] = self.conf_level
                            break
            
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
                        weights[batch_idx, 5] = self.conf_level  # toggle action
                    else:
                        # Move towards the door with wall avoidance
                        action = self._navigate_towards(door_x, door_y, walls)
                        weights[batch_idx, action] = self.conf_level
            
            # Rule 3: goto :- goal(X), unlocked
            elif goals:
                goal = goals[0]
                goal_x, goal_y = goal[1][0], goal[1][1]

                # Check if path to goal is blocked by closed/locked doors
                blocked_by_door = False
                
                # Direction to goal
                direction_to_goal = (
                    1 if goal_x > 0 else (-1 if goal_x < 0 else 0),
                    1 if goal_y > 0 else (-1 if goal_y < 0 else 0),
                )

                # Check if any door blocks the path
                for door in doors:
                    door_x, door_y = door[1][1], door[1][2]
                    door_direction = (
                        1 if door_x > 0 else (-1 if door_x < 0 else 0),
                        1 if door_y > 0 else (-1 if door_y < 0 else 0),
                    )

                    door_color = door[1][0]
                    
                    # Check if door is in same direction and closer than goal
                    same_direction = (
                        direction_to_goal[0] == door_direction[0] and 
                        direction_to_goal[1] == door_direction[1]
                    )
                    
                    door_distance = abs(door_x) + abs(door_y)
                    goal_distance = abs(goal_x) + abs(goal_y)
                    door_is_closer = door_distance < goal_distance

                    # Check door state
                    door_is_closed = any(cd[1][0] == door_color for cd in closed_doors)
                    door_is_locked = any(ld[1][0] == door_color for ld in locked_doors)

                    if (same_direction and door_is_closer and 
                        (door_is_closed or door_is_locked)):
                        blocked_by_door = True
                        break

                if not blocked_by_door:
                    if goal_x == 0 and goal_y == 1:  # Goal is directly in front
                        weights[batch_idx, 2] = self.conf_level  # forward action
                    else:
                        # Move towards the goal with wall avoidance
                        action = self._navigate_towards(goal_x, goal_y, walls)
                        weights[batch_idx, action] = self.conf_level
        
        # Normalize weights to sum to 1.0 per observation
        weights = weights / weights.sum(1, keepdim=True)
        
        return weights

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


if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"C51rulesV2a_{args.env_id}__seed{args.seed}__{start_datetime}"
    if args.track:
        import wandb
        wandb.tensorboard.patch(root_logdir=f"C51rulesV2a/runs_rules_training/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"C51rulesV2a_ri{args.rule_influence}_{args.run_code}",
        )
    writer = SummaryWriter(f"C51rulesV2a/runs_rules_training/{run_name}/train")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
        [
            make_env(args.env_id, args.seed + i, args.n_keys, i, args.capture_video, run_name)
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

    # Use RuleAugmentedReplayBuffer instead of ReplayBuffer
    rb = RuleAugmentedReplayBuffer(
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

    # Create tqdm progress bar
    progress_bar = tqdm(
        range(args.total_timesteps),
        desc=f"Training seed {args.seed}",
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
        if random.random() < epsilon:
            obs_img = q_network.get_observables(obs[:, 4:])
            weights = q_network._get_weights(obs_img).squeeze().cpu().numpy()
            actions = np.random.choice(q_network.n, p=weights, size=(envs.num_envs,))
            # Get rule suggestions even for exploration actions
            rule_actions = q_network._apply_rules_batch(
                obs_img
            )
        else:
            # Updated to use the new get_action method that returns rule_actions
            actions, pmf, rule_actions = q_network.get_action(
                torch.Tensor(obs).float().to(device), skip=False, epsilon=epsilon,
                rule_influence=args.rule_influence, global_step=global_step,
                is_exploit_step=True
            )
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            else:
                actions = np.array([actions])
        q_network.episode_steps_count += 1

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # --- Reset plotting state on episode end ---
        if terminations[0] or truncations[0]:
            q_network.episode_steps_count = 0
            if q_network.is_plotting_episode:
                q_network.is_plotting_episode = False
                q_network.exploit_plot_step_count = 0
                q_network.train_plot_step_count = 0
                print("--- Episode ended, resetting plotting state. ---")

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
                        mean_ep_return = np.mean(episodes_returns[-print_num_eps :])
                        mean_ep_lengths = np.mean(episodes_lengths[-print_num_eps :])
                        tot_mean_return = np.mean(episodes_returns)
                        tot_mean_length = np.mean(episodes_lengths)
                        tqdm.write(
                            f"global_step={global_step}, mean_return_last_{print_num_eps}_episodes={mean_ep_return}, tot_mean_ret={tot_mean_return}, mean_length_last_{print_num_eps}_episodes={mean_ep_lengths}, tot_mean_len={tot_mean_length}, epsilon={epsilon:.2f}"
                        )
                        print_step += args.print_step

        # Update to include rule_actions in rb.add
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos, rule_actions)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(
                        data.next_observations.float(), skip=True
                    )
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (
                        1 - data.dones
                    )
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

                _, old_pmfs, _ = q_network.get_action(
                    data.observations.float(),
                    stored_rule_actions=data.rule_suggestions,
                    action=data.actions.flatten(),
                    skip=False,
                    epsilon=epsilon,
                    rule_influence=args.rule_influence,
                    global_step=global_step,
                    is_exploit_step=False
                )
                loss = (
                    -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(
                        -1
                    )
                ).mean()

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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    plt.plot(episodes_returns)
    plt.title(f'C51rulesV2a on {args.env_id} - Return over {args.total_timesteps} timesteps')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    path = f'C51rulesV2a/{args.env_id}_C51rules_{args.total_timesteps}_{start_datetime}'
    if not os.path.exists("C51rulesV2a/"):
        os.makedirs("C51rulesV2a/")
    os.makedirs(path)
    plt.savefig(f"{path}/{args.env_id}_C51rules_{args.total_timesteps}_{start_datetime}.png")
    plt.close()
    with open(f"{path}/C51rules_args.txt", "w") as f:
        for key, value in vars(args).items():
            if key == "env_id":
                f.write("# C51 Algorithm specific arguments\n")
            f.write(f"{key}: {value}\n")

    if args.save_model:
        model_path = f"{path}/C51rules_model.pt"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        from baseC51.c51_eval import QNetwork as QNetworkEval
        from C51rules_eval import evaluate
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
        writer = SummaryWriter(f"C51rulesV2a/runs_rules_training/{run_name}/eval")
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("episodic_return", episodic_return, idx)

        plt.plot(episodic_returns)
        plt.title(f'C51rulesV2a Eval on {args.env_id} - Return over {eval_episodes} episodes')
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(f"{path}/{args.env_id}_C51rules_{eval_episodes}_{start_datetime}_eval.png")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "C51rulesV2a", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
