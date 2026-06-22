"""
Module: src.train
Description: PPO training to defeat a Mega Man 1 boss from pixels.

The recipe (hyperparameters + reward function) is THE SAME for every boss and is fixed in the
constants below. The CLI only exposes what changes per boss:
  --tag --state --n-hits  (+ --align-bonus and --episode-frames for the Yellow Devil)
See the README for the per-boss table.
"""

import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

try:
    from src.env import make_venv
except ImportError:
    from env import make_venv

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

# ----------------------------------------------------------------------------------------------
# STANDARD RECIPE (same for every boss). See the design notes in the README.
# ----------------------------------------------------------------------------------------------
D = 0.05                      # reward scale: +d per hit on the boss, -d per damage taken
DAMAGE_PENALTY_MULT = 2.0     # TRAINING penalizes damage 2x (forces dodging); evaluation uses 1.0 (real)
WIN_BONUS = 0.5               # terminal win bonus during TRAINING
FRAMESKIP = 4                 # frames per action (training and evaluation)
NORM_REWARD = True            # normalize the reward only (not the observation) — eval-safe

# PPO hyperparameters
LR = 2.5e-4                   # initial learning rate (decays linearly to 0)
ENT_COEF = 5e-3
TARGET_KL = 0.05
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEF = 1.0
MAX_GRAD_NORM = 0.5
N_STEPS = 1024
BATCH_SIZE = 128
N_EPOCHS = 4
N_EVAL_EPISODES = 1
SEED = 666


def linear_schedule(initial_value: float):
    """
    Linear learning-rate schedule: decays from `initial_value` to 0 as training progresses.
    SB3 calls the function with progress_remaining going from 1.0 (start) to 0.0 (end of training).
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class CustomEvalCallback(EvalCallback):
    """
    EvalCallback subclass that prints the mean reward and standard deviation with 6 decimals.
    """
    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync env normalization if applicable
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            self._is_success_buffer = []

            # Run the evaluation
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            # Custom print with 6 decimals!
            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.6f} +/- {std_reward:.6f}")
                print(f"Episode length: {mean_ep_length:.6f} +/- {std_ep_length:.6f}")

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.6f}%")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


def main():
    """
    Parse the few per-boss parameters, build the train/eval envs with the standard recipe,
    set up the checkpoint/eval callbacks with win-based early stopping and run training.
    """
    # cuDNN optimization for fixed-shape inputs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Train PPO to defeat a boss (standard recipe; see the README).")
    parser.add_argument('--tag', type=str, default='run', help='Experiment name; namespaces logs/checkpoints/models')
    parser.add_argument('--state', type=str, default='YellowDevil-boss', help='Fight save state (e.g. Bombman-boss, YellowDevil-boss)')
    parser.add_argument('--n-hits', type=int, default=14, help='Hits to kill the boss (buster=14, Cutman=10, Elec/Ice=28). Sets the early-stop threshold')
    parser.add_argument('--align-bonus', type=float, default=0.0, help='PBRS aiming bonus (Yellow Devil only: 0.10). 0 = off')
    parser.add_argument('--episode-frames', type=int, default=7200, help='Frames per episode (Yellow Devil uses 14400)')
    parser.add_argument('--timesteps', type=int, default=40_000_000, help='Timestep ceiling; early-stop cuts it short on a win')
    parser.add_argument('--n-envs', type=int, default=16, help='Parallel environments')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path to resume training from')
    parser.add_argument('--visualize', action='store_true', help='Open the emulator window for one of the environments during training')
    args = parser.parse_args()

    render_mode = "human" if args.visualize else None

    # Reward: +d per hit, -d per damage. The boss dies after exactly n_hits hits, so the maximum
    # is n_hits*d (all hits, 0 damage = flawless win). The early-stop threshold sits just below it,
    # at (n_hits-0.5)*d, robustly capturing only the no-damage win.
    reward_threshold = (args.n_hits - 0.5) * D

    # Resolve paths relative to the project root, namespaced by tag
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    checkpoints_dir = os.path.join(project_root, "checkpoints", args.tag)
    models_dir = os.path.join(project_root, "models", f"{args.tag}_best")
    logs_dir = os.path.join(project_root, "logs", args.tag)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # TRAINING env: 2x damage penalty, win bonus and (YD only) aim shaping.
    env = make_venv(n_envs=args.n_envs, state=args.state, render_mode=render_mode, d=D,
                    damage_penalty_mult=DAMAGE_PENALTY_MULT, win_bonus=WIN_BONUS,
                    frameskip=FRAMESKIP, norm_reward=NORM_REWARD, align_bonus=args.align_bonus,
                    max_episode_frames=args.episode_frames)
    # EVAL env: REAL task (no shaping — 1x penalty, no win bonus, no align), same frameskip/state/
    # episode length, so best_model and early-stop reflect a real win.
    eval_env = make_venv(n_envs=1, state=args.state, render_mode=None, force_subproc=True, d=D,
                         frameskip=FRAMESKIP, norm_reward=NORM_REWARD,
                         max_episode_frames=args.episode_frames)
    if NORM_REWARD:
        eval_env.training = False
        eval_env.norm_reward = False

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 500000 // args.n_envs),
        save_path=checkpoints_dir,
        name_prefix=args.tag,
    )
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        eval_freq=max(1, 100000 // args.n_envs),
        n_eval_episodes=N_EVAL_EPISODES,
        callback_on_new_best=stop_callback,
        verbose=1,
    )

    learning_rate = linear_schedule(LR)

    # Initialize or resume the model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        custom_objects = {
            "learning_rate": learning_rate,
            "lr_schedule": learning_rate,  # lr_schedule is what the algo actually uses in train()
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "target_kl": TARGET_KL,
        }
        model = PPO.load(args.checkpoint, env=env, device=device, custom_objects=custom_objects)
        # On resume, reset tensorboard_log to the current tag.
        model.tensorboard_log = logs_dir
    else:
        print(f"Starting training from scratch (tag={args.tag})...")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            target_kl=TARGET_KL,
            policy_kwargs={"share_features_extractor": False},
            seed=SEED,
            device=device,
            tensorboard_log=logs_dir,
        )

    print(f"Training {args.tag}: state={args.state} n_hits={args.n_hits} "
          f"(early-stop threshold={reward_threshold:.3f}), up to {args.timesteps} steps")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="PPO",
        reset_num_timesteps=False if args.checkpoint else True,
    )

    # Save the FINAL model in addition to the eval best_model.
    model.save(os.path.join(models_dir, "final_model"))
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
