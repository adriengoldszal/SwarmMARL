#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config

from src.swarmrl.src.swarm_env.multi_env.multi_agent_comm import MultiSwarmEnv as EnvCom
from src.swarmrl.src.swarm_env.multi_env.multi_agent_gym import (
    MultiSwarmEnv as EnvNoCom,
)
from src.swarmrl.src.swarm_env.multi_env.multi_agent_market import (
    MultiSwarmEnv as EnvMarket,
)
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, Monitor
from guppy import hpy


"""Train script for Swarm Rescue."""


def make_train_env(all_args, env_type):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "multi-agent":
                config = {
                    "map_name": all_args.map_name,
                    "n_targets": all_args.n_targets,
                    "n_agents": all_args.n_agents,
                    "use_exp_map": all_args.use_exp_map,
                }
                env = env_type(**config)
                env = Monitor(env)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.unwrapped.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, env_type):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SwarmRescue":
                config = {
                    "map_name": all_args.map_name,
                    "n_targets": all_args.n_targets,
                    "n_agents": all_args.n_agents,
                    "use_exp_map": all_args.use_exp_map,
                }
                env = env_type(**config)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    parser.add_argument(
        "--map_name", type=str, default="Easy", help="Which map to run on"
    )
    parser.add_argument(
        "--n_targets", type=int, default=3, help="number of wounded persons"
    )
    parser.add_argument("--n_agents", type=int, default=3, help="number of drones")

    parser.add_argument("--use_conflict_reward", action="store_true", default=False)

    parser.add_argument(
        "--com_mode", type=str, default="none", help="type of communication to use"
    )
    parser.add_argument(
        "--use_exp_map",
        action="store_true",
        default=False,
        help="Use explore map in env",
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
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    if all_args.com_mode == "target":
        print("Using communication")
        env_type = EnvCom
    elif all_args.com_mode == "market":
        print("Using marketbased")
        env_type = EnvMarket
    elif all_args.com_mode == "none":
        print("Not using communication")
        env_type = EnvNoCom
    else:
        raise Exception("No compatible com mode")

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

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.map_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, env_type)
    eval_envs = make_eval_env(all_args, env_type) if all_args.use_eval else None
    n_agents = all_args.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": n_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.swarm_runner import SwarmRunner as Runner
    else:
        raise Exception("Do not support separate policy")

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
