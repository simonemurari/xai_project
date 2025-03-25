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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)  # Remove render_mode to avoid overhead
            env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env, filter_keys=['image', 'direction']))
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
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n * n_atoms),
        )
    def get_action(self, x, action=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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


def train_c51(args, seed, process_idx, num_processes):
    """Main training function for C51 algorithm"""
    # Set process affinity to specific cores
    set_process_affinity(process_idx, num_processes)
    
    # Set number of threads for this process
    torch.set_num_threads(max(1, mp.cpu_count() // num_processes))
    
    start_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    run_name = f"C51_{args.env_id}__seed={seed}__{start_datetime}"
    if args.track:
        import wandb
        wandb.tensorboard.patch(root_logdir=f"C51/runs/{run_name}/train")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"C51_{args.exploration_fraction}_mp_{args.run_code}",
        )
    
    # Only write to TensorBoard periodically to reduce I/O overhead
    writer = SummaryWriter(f"C51/runs/{run_name}/train", 
                         max_queue=100, 
                         flush_secs=30)
    
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []
    
    print(f'Process {os.getpid()}: File: {os.path.basename(__file__)}, using seed {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.device

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, seed, 0, args.capture_video, run_name)])
    
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
    obs, _ = envs.reset(seed=seed)
    print_step = args.print_step
    print(f"Process {os.getpid()}: Starting training with seed {seed} using device {device} for {args.total_timesteps} timesteps")
    print(f"Process {os.getpid()}: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    # Pre-allocate observation tensor for efficiency
    obs_tensor = torch.zeros((1, *envs.single_observation_space.shape), dtype=torch.float32, device=device)
    
    # Use tqdm with minimal updates to reduce overhead
    for global_step in tqdm(range(args.total_timesteps), 
                          desc=f"Seed {seed}", 
                          position=process_idx,
                          colour=['blue', 'green', 'red'][process_idx % 3],
                          miniters=10000,
                          maxinterval=20):
        
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
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
                        tqdm.write(f"seed: {seed}, global_step={global_step}, episodic_return_mean_last_print_step={mean_ep_return}, episodic_length_mean_last_print_step={mean_ep_lengths}, exploration_rate={epsilon:.2f}")
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
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
                _, old_pmfs = q_network.get_action(data.observations.float(), data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                # Less frequent logging to reduce overhead
                if global_step % 10000 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("SPS", int(global_step / (time.time() - start_time)), global_step)

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
    plt.title(f'C51 on {args.env_id} - Seed {seed} - Return over {args.total_timesteps} timesteps')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    path = f'C51/{args.env_id}_c51_{args.total_timesteps}_seed{seed}_{start_datetime}'
    if not os.path.exists("C51/"):
        os.makedirs("C51/")
    os.makedirs(path)
    plt.savefig(f"{path}/{args.env_id}_c51_{args.total_timesteps}_seed{seed}_{start_datetime}.png")
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
            "seed": seed,
        }
        torch.save(model_data, model_path)
        print(f"Process {os.getpid()}: Model saved to {model_path}")
        
    envs.close()
    writer.close()
    
    return episodes_returns, path

if __name__ == "__main__":
    # Set up multiprocessing
    mp.set_start_method('forkserver', force=True)
    
    print(f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}, device = {Args.device}")
    args = tyro.cli(Args)

    # Add shared memory lock for GPU access if using CUDA
    if torch.cuda.is_available() and args.device == "cuda":
        # Stagger process starts by 10 seconds to avoid CUDA initialization contention
        lock = mp.Lock()
    else:
        lock = None
    
    # Seeds to use for different processes
    seeds = [6, 21, 42]
    num_processes = len(seeds)

    print(f"Starting C51 training with {num_processes} processes, seeds: {seeds}")
    
    # Start multiple processes for training
    processes = []
    for idx, seed in enumerate(seeds):
        p = mp.Process(target=train_c51, args=(args, seed, idx, num_processes))
        p.start()
        # Sleep briefly between process launches to prevent resource contention
        time.sleep(3)
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All training processes completed!")