#!/bin/bash
for seed in 47235 6 23 89 31241
do
   uv run c51.py --seed $seed --task "DeliverCoffee" --run_code "DeliverCoffee" --batch_size 32 --buffer_size 50000 --exploration_fraction 0.4 --gamma 0.99 --total_timesteps 300000 --learning_starts 5000 --end_e 0.05
done
for seed in 47235 6 23 89 31241
do
   uv run c51.py --seed $seed --task "DeliverCoffeeAndMail" --run_code "DeliverCoffeeAndMail" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.2 --gamma 0.95 --total_timesteps 600000 --learning_starts 5000 --end_e 0.01
done
for seed in 47235 6 23 89 31241
do
   uv run c51.py --seed $seed --task "PatrolAB" --run_code "PatrolAB" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.4 --gamma 0.99 --total_timesteps 300000 --learning_starts 5000 --end_e 0.05
done
for seed in 47235 6 23 89 31241
do
   uv run c51.py --seed $seed --task "PatrolABC" --run_code "PatrolABC" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.6 --gamma 0.95 --total_timesteps 600000 --learning_starts 5000 --end_e 0.05
done
cd ..
cd minigrid
./run_5_seeds_delayed.sh