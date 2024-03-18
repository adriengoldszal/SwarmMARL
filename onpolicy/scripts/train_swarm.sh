#!/bin/sh
env="multi-agent"
map_name="Easy"
n_targets=3
n_agents=3
algo="mappo" #"mappo" "ippo"
exp="3v3_test_algo"
seed_max=5
scenario="easy_map"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in  `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ./train/train_swarm.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map_name} --n_agents ${n_agents} --n_targets ${n_targets} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 512 --num_env_steps 1000000 \
    --ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "minhpham1606" --user_name "minhpham1606" \
    --log_interval 1 --save_interval 10 --layer_N 1 --hidden_size 64 \
    --com_mode "market" \
    --use_conflict_reward \
    # --use_wandb
 
done
