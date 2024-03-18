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
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, NormalizeReward


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SwarmRescue":
                kwargs = {
                    "map_name": all_args.map_name,
                    "n_agents": all_args.n_agents,
                    "n_targets": all_args.n_targets,
                }
                env = MultiSwarmEnv(**kwargs)
                env = NormalizeReward(env)
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


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print(
            "u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo."
        )
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert all_args.use_render, "u need to set use_render be True"
    assert not (
        all_args.model_dir == None or all_args.model_dir == ""
    ), "set model_dir first"
    assert all_args.n_rollout_threads == 1, "only support to use 1 env to render."

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

    # run dir and video dir
    run_dir = Path(all_args.model_dir).parent.absolute()
    if all_args.save_videos and all_args.video_dir == "":
        video_dir = run_dir / "videos"
        all_args.video_dir = str(video_dir)

        if not video_dir.exists():
            os.makedirs(str(video_dir))

    setproctitle.setproctitle(
        "-".join(
            [
                all_args.env_name,
                all_args.map_name,
                all_args.algorithm_name,
                all_args.experiment_name,
            ]
        )
        + "@"
        + all_args.user_name
    )

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
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.swarm_runner import SwarmRunner as Runner
    else:
        raise Exception("No runner available for seprated policy")
    runner = Runner(config)
    runner.render()

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
