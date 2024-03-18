#!/bin/sh
# exp params
env="SwarmRescue"
map_name="Easy"
algo="mappo"
exp="render"
seed=1

# football params
n_agents=3
n_targets=3
model_dir="/home/mip012/Documents/Code/train-swarm/onpolicy/scripts/train/results/3_mousquetairs"
 
# --save_videos is preferred instead of --save_gifs 
# because .avi file is much smaller than .gif file

echo "render ${render_episodes} episodes"

CUDA_VISIBLE_DEVICES=0 python render/render_swarm.py --env_name ${env} --map__name ${map_name} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} --n_agents ${n_agents}  --n_targets ${n_targets} --ifi 0.2 \
--use_render --render_episodes 3 --n_rollout_threads 1 --model_dir ${model_dir} --user_name "minhpham" --use_wandb \
# --save_videos