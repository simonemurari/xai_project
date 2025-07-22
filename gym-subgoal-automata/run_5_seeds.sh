#!/bin/bash
for seed in 23 6 89 47235 31241
do
   uv run dqn_rules.py --seed $seed --task "PatrolABC" --run_code "PatrolABC_v1" --batch_size 128 --buffer_size 100000 --exploration_fraction 0.3 --gamma 0.95 --total_timesteps 500000 --learning_starts 5000 --end_e 0.05
done