import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'      # Better for AMD than MKL
os.environ['VECLIB_MAXIMUM_THREADS'] = '1' 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA determinism
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
            env = gym.make(env_id)
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
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n * n_atoms),
        )
        
        # Initialize weights properly for better training stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
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


# Improved process affinity setting
def set_process_affinity(process_idx, num_processes):
    """Pin process to specific cores optimized for Ryzen 7840HS"""
    try:
        import psutil
        process = psutil.Process()
        
        # Ryzen 7840HS has 8 physical cores (0-7) with SMT threads (8-15)
        physical_cores = 8
        
        if num_processes <= 4:
            # With 4 or fewer processes, assign 2 physical cores per process
            cores_per_process = 2
            start_core = (process_idx * cores_per_process) % physical_cores
            
            # For each physical core, also assign its SMT thread
            core_list = [start_core, start_core + physical_cores]
            for i in range(1, cores_per_process):
                next_core = (start_core + i) % physical_cores
                core_list.extend([next_core, next_core + physical_cores])
                
            process.cpu_affinity(core_list)
            print(f"Process {os.getpid()}: Using cores {core_list}")
            
        else:
            # With 5+ processes, assign 1 physical core per process
            # Use modulo to wrap around if more processes than cores
            physical_core = process_idx % physical_cores
            smt_thread = physical_core + physical_cores
            process.cpu_affinity([physical_core, smt_thread])
            print(f"Process {os.getpid()}: Using core {physical_core} and thread {smt_thread}")
            
    except Exception as e:
        print(f"Process {os.getpid()}: Couldn't set CPU affinity: {e}")


# Assign process to a specific GPU or portion of a GPU
def setup_gpu_for_process(process_idx, num_processes, args):
    """Configure GPU resources for this process"""
    if not torch.cuda.is_available() or args.device == "cpu":
        return "cpu"
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        return "cpu"
    
    # If multiple GPUs available, distribute processes across them
    if num_gpus > 1:
        gpu_id = process_idx % num_gpus
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        print(f"Process {os.getpid()}: Using dedicated GPU {gpu_id}")
    else:
        # Single GPU - set memory limits to avoid OOM
        gpu_id = 0
        device = "cuda:0"
        torch.cuda.set_device(gpu_id)
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # Limit memory usage based on number of processes
            mem_fraction = 0.9 / num_processes  # Reserve 10% for system
            torch.cuda.set_per_process_memory_fraction(mem_fraction, gpu_id)
            print(f"Process {os.getpid()}: Using GPU {gpu_id} with {mem_fraction:.1%} memory limit")
    
    # Additional CUDA settings for determinism
    if args.torch_deterministic:
        torch.cuda.manual_seed_all(args.seed + process_idx)
    
    return device


def train_c51(args, seed, process_idx, num_processes):
    """Main training function for C51 algorithm with complete process isolation"""
    # Completely isolate this process
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set process affinity to specific cores
    set_process_affinity(process_idx, num_processes)
    
    # Set number of threads for this process
    torch.set_num_threads(2 if num_processes <= 4 else 1)  # Adjust based on workload
    
    # Create a unique run name with seed and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"C51_{args.env_id}_seed{seed}_{timestamp}_pid{os.getpid()}"
    
    # Configure device (CPU/GPU)
    device = setup_gpu_for_process(process_idx, num_processes, args)
    
    if args.track:
        import wandb
        # Use unique run name for each process
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
    
    # Use dedicated directories for each process
    writer = SummaryWriter(f"C51/runs/{run_name}/train", 
                         max_queue=100, 
                         flush_secs=30)
    
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    episodes_returns = []
    episodes_lengths = []
    
    # Set all random seeds for complete determinism
    print(f'Process {os.getpid()}: Using seed {seed} on device {device}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False

    # Create environment with process-specific paths
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, seed, 0, args.capture_video and process_idx == 0, run_name)])
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize networks with proper seeding
    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Create replay buffer for this process
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
    print(f"Process {os.getpid()}: Starting training with seed={seed}, device={device}")
    
    # Pre-allocate observation tensor for efficiency
    obs_tensor = torch.zeros((1, *envs.single_observation_space.shape), dtype=torch.float32, device=device)
    
    # Training loop with deterministic execution
    progress_bar = tqdm(
        range(args.total_timesteps), 
        desc=f"Seed {seed}", 
        position=process_idx,
        colour=['blue', 'green', 'red', 'yellow', 'magenta'][process_idx % 5],
        miniters=10000,
        maxinterval=20
    )
    
    for global_step in progress_bar:
        # Action selection with epsilon-greedy
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            obs_tensor.copy_(torch.as_tensor(obs, dtype=torch.float32))
            with torch.no_grad():
                actions, pmf = q_network.get_action(obs_tensor)
                actions = actions.cpu().numpy()

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record episodic statistics
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("episodic_return", info["episode"]["r"][0], global_step)
                    writer.add_scalar("episodic_length", info["episode"]["l"][0], global_step)
                    episodes_returns.append(info["episode"]["r"])
                    episodes_lengths.append(info["episode"]["l"])
                    if global_step >= print_step:
                        mean_return = np.mean(episodes_returns[-args.print_step:])
                        mean_length = np.mean(episodes_lengths[-args.print_step:])
                        tqdm.write(f"seed: {seed}, step={global_step}, return={mean_return:.2f}, len={mean_length:.1f}, ε={epsilon:.2f}")
                        print_step += args.print_step

        # Handle episode termination
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Add to replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training logic
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            # Sample from replay buffer
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
            
            # Get current distribution
            _, old_pmfs = q_network.get_action(data.observations.float(), data.actions.flatten())
            
            # Calculate cross-entropy loss
            loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

            # Occasional logging
            if global_step % 10000 == 0:
                writer.add_scalar("losses/loss", loss.item(), global_step)
                old_val = (old_pmfs * q_network.atoms).sum(1)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("SPS", int(global_step / (time.time() - start_time)), global_step)

            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Update target network periodically
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # Save results to unique directory
    results_dir = f'C51/{args.env_id}_c51_{args.total_timesteps}_seed{seed}_{timestamp}_pid{os.getpid()}'
    if not os.path.exists("C51/"):
        os.makedirs("C51/")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot learning curve
    plt.figure()
    plt.plot(episodes_returns)
    plt.title(f'C51 on {args.env_id} - Seed {seed} - Return over {args.total_timesteps} timesteps')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.savefig(f"{results_dir}/{args.env_id}_c51_{args.total_timesteps}_seed{seed}.png")
    plt.close()
    
    # Save configuration
    with open(f"{results_dir}/c51_args.txt", "w") as f:
        f.write(f"# Process ID: {os.getpid()}, Seed: {seed}, Device: {device}\n")
        for key, value in vars(args).items():
            if key == "env_id":
                f.write("# C51 Algorithm specific arguments\n")
            if key == "sigmoid_shift" or key == "sigmoid_scale" or key == "distribution":
                continue
            f.write(f"{key}: {value}\n")

    # Save model if requested
    if args.save_model:
        model_path = f"{results_dir}/c51_model.pt"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
            "seed": seed,
            "process_id": os.getpid(),
        }
        torch.save(model_data, model_path)
        print(f"Process {os.getpid()}: Model saved to {model_path}")
        
    # Clean up
    envs.close()
    writer.close()
    
    return episodes_returns, results_dir

if __name__ == "__main__":
    # Initialize multiprocessing method based on platform for best stability
    # Linux/Windows
    mp.set_start_method('forkserver', force=True)
    
    print(f"Torch: {torch.__version__}, cuda ON: {torch.cuda.is_available()}, device = {Args.device}")
    args = tyro.cli(Args)

    # Seeds to use for different processes
    # seeds = [6, 21, 42]  # Adjust as needed
    
    # You can safely add more seeds - the behavior should now be consistent
    seeds = [6, 21, 42, 47235, 31241]
    
    num_processes = len(seeds)

    print(f"Starting C51 training with {num_processes} processes, seeds: {seeds}")
    
    # Longer staggered process launch to avoid contention
    processes = []
    for idx, seed in enumerate(seeds):
        # Pass seed explicitly into arguments
        args_copy = tyro.cli(Args)
        args_copy.seed = seed
        
        p = mp.Process(target=train_c51, args=(args_copy, seed, idx, num_processes))
        p.start()
        # Wait longer between launches to completely avoid initialization conflicts
        time.sleep(10 + idx * 5)  # 10s base + 5s per previous process
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All training processes completed!")