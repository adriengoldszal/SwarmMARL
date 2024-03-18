#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket

# third-party packages
import numpy as np
import setproctitle
import torch

# code repository sub-packages
from onpolicy.config import get_config
from src.swarmrl.src.swarm_env.multi_env.multi_agent_gym import MultiSwarmEnv
from src.swarmrl.src.swarm_env.multi_env.multi_agent_comm import MultiSwarmEnv as EnvCom
from src.swarmrl.src.swarm_env.multi_env.multi_agent_market import (
    MultiSwarmEnv as EnvMarket,
)
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, NormalizeReward
from onpolicy.utils.util import find_and_construct_path
import json
import tqdm


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SwarmRescue":
                kwargs = {
                    "map_name": all_args.map_name,
                    "n_agents": all_args.n_agents,
                    "n_targets": all_args.n_targets,
                }
                env = EnvCom(**kwargs)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        "--map_name",
        type=str,
        default="Easy",
        help="which scenario to run on.",
    )
    parser.add_argument("--n_agents", type=int, default=3, help="number of drones.")

    parser.add_argument("--n_targets", type=int, default=3)
    parser.add_argument(
        "--save_videos",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--video_dir", type=str, default="", help="directory to save videos."
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args=None):
    parser = get_config()
    all_args = parse_args(args, parser)
    model_paths = {}
    total = 0
    for algo, id_list in models.items():
        for idx in id_list:
            model_paths[algo] = model_paths.get(algo, [])
            model_paths[algo].append(find_and_construct_path(base_dir=base, id=idx))
            total += 1

    print(model_paths)

    pbar = tqdm.tqdm(total=total)

    infos = {}
    for algo, paths in model_paths.items():
        all_args.render_episodes = total_episodes
        all_args.algorithm_name = algo
        all_args.model_dir = paths[0]
        all_args.use_render = True
        all_args.n_rollout_threads = 1
        all_args.env_name = "SwarmRescue"
        all_args.use_wandb = False

        if all_args.algorithm_name == "rmappo":
            all_args.use_centralized_V = True
            all_args.use_recurrent_policy = True
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "mappo":
            print("Using using mappo")
            all_args.use_centralized_V = True
            all_args.use_recurrent_policy = False
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "ippo":
            print("Using ippo")
            all_args.use_centralized_V = False
            all_args.use_naive_recurrent_policy = True
        else:
            raise NotImplementedError

        # assert all_args.use_render, "u need to set use_render be True"

        # cuda
        if all_args.cuda and torch.cuda.is_available():
            print("choose to use gpu...")
            device = torch.device("cuda:0")
            torch.set_num_threads(all_args.n_training_threads)
            if all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        else:
            print("choose to use cpu...")
            device = torch.device("cpu")
            torch.set_num_threads(all_args.n_training_threads)

        # seed
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)
        np.random.seed(all_args.seed)

        # env init
        envs = make_train_env(all_args)
        num_agents = all_args.n_agents

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": None,
            "num_agents": num_agents,
            "device": device,
            "run_dir": Path(paths[0]).parent.absolute(),
        }

        # run experiments
        if all_args.share_policy:
            from onpolicy.runner.shared.swarm_runner import SwarmRunner as Runner
        else:
            raise Exception("No runner available for seprated policy")
        runner = Runner(config)
        for path in paths:
            try:
                if path is None:
                    print("None path")
                    continue
                all_args.model_dir = path

                assert (
                    all_args.n_rollout_threads == 1
                ), "only support to use 1 env to render."
                # runner.render()

                runner.restore(path)
                ret, length, conf = runner.custom_eval()
                # print(ret, length, conf)

                infos[algo] = infos.get(algo, {})
                infos[algo]["return"] = infos[algo].get("return", [])
                infos[algo]["len"] = infos[algo].get("len", [])
                infos[algo]["conf"] = infos[algo].get("conf", [])
                infos[algo]["return"].append(ret)
                infos[algo]["len"].append(length)
                infos[algo]["conf"].append(conf)

                pbar.update()
            except Exception as e:
                print("Error in ", path)
                print(e)

        # post process

    for algo, info in infos.items():
        infos[algo]["mean_ret"] = np.mean(infos[algo]["return"])
        infos[algo]["std_ret"] = np.std(infos[algo]["return"])
        infos[algo]["mean_len"] = np.mean(infos[algo]["len"])
        infos[algo]["std_len"] = np.std(infos[algo]["len"])
        infos[algo]["mean_conf"] = np.mean(infos[algo]["conf"])
        infos[algo]["std_conf"] = np.std(infos[algo]["conf"])

    with open(f"{print_base}/exp_2_1.json", "w") as json_file:
        json.dump(infos, json_file)
    envs.close()


total_episodes = 50

# ippo, mappo wth reward conflict, no communication
models_no_com = {
    "ippo": ["r2jlbdg2", "ug89mfth", "jfa13pqc"],  # with reward conflict
    "mappo": ["51ebajzc", "ukoo391p", "wl97viw7"],  # with reward conflict
    "rmappo": ["11g1y5jb", "ru5qj91k", "xigi6nc7"],  # with reward conflict
}
base = "/users/eleves-b/2021/minh.pham/thesis/train-swarm/onpolicy/scripts/results/multi-agent/Easy"
print_base = "/users/eleves-b/2021/minh.pham/thesis/train-swarm/onpolicy/scripts/render"

models_target = {
    "ippo": ["ruaw8ay5", "lyg0qq3y", "42o1e3xa"],
    "mappo": ["02wtk5s9", "kmn3xl43", "xnvb8ay5"],
}

models_market = {
    "ippo": ["edommbem", "fc9xf5nv", "86lxahh9"],
    "mappo": ["golpodkb", "54spd0eb", "q49tnag3"],
}

models = models_target

if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
