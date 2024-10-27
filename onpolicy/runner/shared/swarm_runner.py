import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from memory_profiler import profile
import os 

def _t2n(x):
    return x.detach().cpu().numpy()


class SwarmRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(SwarmRunner, self).__init__(config)

    # @profile
    def run(self):
        self.warmup()

        start = time.time()
        iterations = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        pbar = tqdm(total=self.num_env_steps, position=1)
        total_num_steps = 0

        episodes_return = []
        episodes_len = []
        episodes_conf = []
        ep_count = 0
        env_infos = {}

        for iteration in range(iterations):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(iteration, iterations)
            conflict_per_ep = []
            for step in range(self.episode_length):
                total_num_steps += self.n_rollout_threads
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)
                # print("Action env ", actions_env)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

                pbar.update(self.n_rollout_threads)

                if any([x[0] for x in dones]):
                    # for info in infos["final_info"]:
                    for info in infos:
                        if info and "episode" in info:
                            # env_infos["charts/episodic_return"] = info["episode"]["r"]
                            # env_infos["charts/episodic_length"] = info["episode"]["l"]
                            # env_infos["charts/episodic_conflict"] = info["episode"]["c"]

                            if ep_count < 100:
                                # print("info", info["episode"]["r"])
                                episodes_return.append(max(info["episode"]["r"], -2000))
                                episodes_len.append(info["episode"]["l"])
                                episodes_conf.append(info["episode"]["c"])
                            else:
                                episodes_return[ep_count % 100] = max(
                                    info["episode"]["r"], -2000
                                )
                                episodes_len[ep_count % 100] = info["episode"]["l"]
                                episodes_conf[ep_count % 100] = info["episode"]["c"]

                            env_infos["average_episode_return"] = np.mean(
                                episodes_return
                            )
                            env_infos["average_episode_len"] = np.mean(episodes_len)
                            env_infos["average_episode_conf"] = np.mean(episodes_conf)

                            if total_num_steps % 1000 == 0:
                                print(
                                    f"total_num_steps={total_num_steps}",
                                    "average episode return is {}".format(
                                        env_infos["average_episode_return"]
                                    ),
                                    "average episode len is {}".format(
                                        env_infos["average_episode_len"]
                                    ),
                                    "average ep conflict is {}".format(
                                        env_infos["average_episode_conf"]
                                    ),
                                )
                            ep_count = ep_count + 1

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # save model
            if iteration % self.save_interval == 0 or iteration == iterations - 1:
                self.save()

            # log information
            if iteration % self.log_interval == 0:
                end = time.time()
                print(env_infos)
                self.custom_log_env(env_infos, total_num_steps)
                env_infos = {}
                print(
                    "\n Map {} Algo {} Exp {} updates {}/{} iterations, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.map_name,
                        self.algorithm_name,
                        self.experiment_name,
                        iteration,
                        iterations,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                self.log_train(train_infos, total_num_steps)

            # eval  
            if iteration % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        pbar.close()

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = (
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        actions_env = [actions[idx] for idx in range(self.n_rollout_threads)]

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        print("Eval is called")
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        #Add frames for gif saving
        print(f"Eval GIF save directory (run_dir): {self.run_dir}")
        all_frames = []
        #Only save if both arguments are true
        save_eval_gifs = self.save_eval_gifs and self.use_wandb
        
        if save_eval_gifs:
            print("Attempting to save gifs...")
            image = self.eval_envs.render('rgb_array')[0]
            all_frames.append(image)
        
        for eval_step in range(self.episode_length):
            calc_start = time.time()
            
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            eval_actions_env = [
                eval_actions[idx] for idx in range(self.n_eval_rollout_threads)
            ]

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env
            )
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )
            
            if save_eval_gifs:
                image = self.eval_envs.render('rgb_array')[0]
                all_frames.append(image)
                # Control frame timing if needed
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < self.all_args.ifi:
                    time.sleep(self.all_args.ifi - elapsed)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

        if save_eval_gifs:
            # Create temporary gif file
            gif_path = os.path.join(self.run_dir, f'eval_{total_num_steps}.gif')
            imageio.mimsave(gif_path, all_frames, duration=self.all_args.ifi)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "eval_animation": wandb.Video(gif_path, fps=1/self.all_args.ifi, format="gif"),
                "step": total_num_steps
            })
            print("Saved gif to wandb")
            
            
    @torch.no_grad()
    def render(self):
        print(f"save_videos flag: {self.all_args.save_videos}")
        print(f"video_dir: {self.all_args.video_dir}")
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_videos:
                print("Attempting to save video...")
                image = envs.render("rgb_array")[0]
                all_frames.append(image)
            else:
                envs.render("human")
                print("save_videos is False, not saving video")

            print(f"Number of frames captured: {len(all_frames)}")
            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []
            episode_conflicts = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                actions_env = [actions[idx] for idx in range(self.n_rollout_threads)]

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)
                episode_conflicts.append(np.sum(infos[0]["conflict_count"]) // 2)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if self.all_args.save_videos:
                    image = envs.render("rgb_array")[0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

                if any([any(x) for x in dones]):

                    print(
                        f"Ep return: {np.sum(episode_rewards)}, Ep len: {len(episode_rewards)}, Conflicts: {np.sum(episode_conflicts)}"
                    )
                    episode_rewards = []

                    if self.all_args.save_videos:
                        path = f"{str(self.all_args.video_dir)}/episode_{episode}.gif"
                        imageio.mimsave(
                            path,
                            all_frames,
                            duration=self.all_args.ifi,
                        )
                        print(f"Gif save to: {path}")
                    all_frames = []
                    break

    @torch.no_grad()
    def custom_eval(self):
        """Visualize the env."""
        envs = self.envs

        returns = []
        lengths = []
        conflicts = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []
            episode_conflicts = []

            for step in range(self.episode_length):
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                actions_env = [actions[idx] for idx in range(self.n_rollout_threads)]

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards[0][0][0])
                episode_conflicts.append(np.sum(infos[0]["conflict_count"]) // 2)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if any([any(x) for x in dones]):
                    # print(
                    #     f"Ep return: {np.sum(episode_rewards)}, Ep len: {len(episode_rewards)}, Conflicts: {np.sum(episode_conflicts)}"
                    # )
                    returns.append(np.sum(episode_rewards))
                    lengths.append(step + 1)
                    conflicts.append(np.sum(episode_conflicts))
                    episode_rewards = []
                    episode_conflicts = []
                    break
        return np.mean(returns), np.mean(lengths), np.mean(conflicts)
