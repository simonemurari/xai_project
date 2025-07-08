for seed in 31241 23 6 89 47235
do
   uv run c51_rules.py --seed $seed
done
# for seed in 6 89 31241 47235
# do
#    uv run dqn_rules.py --seed $seed --exploration_fraction 0.05 --run_code "PatrolABC_v5"
# done
# for seed in 47235 23 6 89 31241
# do
#    uv run c51_rules.py --seed $seed
# done
# ./find_params.py
# cd ..
# cd minigrid
# ./run_5_seeds_delayed.sh