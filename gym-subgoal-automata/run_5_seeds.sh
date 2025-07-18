#!/bin/bash
for seed in 29
do
   uv run c51_rules.py --seed $seed --task "DeliverCoffeeAndMail" --run_code "DeliverCoffeeAndMail_1" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.2 --gamma 0.95 --total_timesteps 600000 --learning_starts 5000 --end_e 0.01
done
for seed in 29
do
   uv run c51_rules.py --seed $seed --task "PatrolAB" --run_code "PatrolAB_v2" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.4 --gamma 0.99 --total_timesteps 300000 --learning_starts 5000 --end_e 0.05
done
for seed in 29
do
   uv run c51_rules.py --seed $seed --task "PatrolABC" --run_code "PatrolABC_v1" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.6 --gamma 0.95 --total_timesteps 600000 --learning_starts 5000 --end_e 0.05
done

for seed in 29
do
   uv run dqn_rules.py --seed $seed --task "DeliverCoffee" --run_code "DeliverCoffee" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.1 --gamma 0.99 --total_timesteps 250000 --learning_starts 1000 --end_e 0.05
done
for seed in 29
do
   uv run dqn_rules.py --seed $seed --task "DeliverCoffeeAndMail" --run_code "DeliverCoffeeAndMail" --batch_size 32 --buffer_size 100000 --exploration_fraction 0.2 --gamma 0.95 --total_timesteps 500000 --learning_starts 5000 --end_e 0.05
done
for seed in 29
do
   uv run dqn_rules.py --seed $seed --task "PatrolAB" --run_code "PatrolAB" --batch_size 32 --buffer_size 50000 --exploration_fraction 0.1 --gamma 0.99 --total_timesteps 250000 --learning_starts 1000 --end_e 0.05
done
for seed in 29 6 89 47235 31241
do
   uv run dqn_rules.py --seed $seed --task "PatrolABC" --run_code "PatrolABC_v1" --batch_size 128 --buffer_size 100000 --exploration_fraction 0.3 --gamma 0.95 --total_timesteps 500000 --learning_starts 5000 --end_e 0.05
done