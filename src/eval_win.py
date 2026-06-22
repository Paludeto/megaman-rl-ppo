"""
Module: src.eval_win
Description: Evaluate a saved model (.zip). Loads the model, runs N episodes on the real task
             (no reward shaping) and reports win rate, mean reward and mean episode length.
             A win is detected when boss_hp reaches 0.

Usage:
  python src/eval_win.py --model models/<tag>_best/best_model.zip --state Bombman-boss
  python src/eval_win.py --model models/yd_best/best_model.zip --state YellowDevil-boss \
      --episodes 10 --episode-frames 14400
"""

import os
import argparse
import numpy as np

try:
    from src.env import make_venv
except ImportError:
    from env import make_venv

# Must match the training recipe (src/train.py).
D = 0.05
FRAMESKIP = 4


def run_episode(model, env, deterministic):
    """Run one episode and return metrics, including the lowest boss_hp seen (0 = boss killed)."""
    obs = env.reset()
    state = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    total_reward = 0.0
    steps = 0
    min_boss = 28
    final_hp = 0
    while not done:
        action, state = model.predict(obs, state=state, episode_start=episode_starts, deterministic=deterministic)
        obs, reward, dones, infos = env.step(action)
        done = bool(dones[0])
        episode_starts = dones
        total_reward += float(reward[0])
        steps += 1
        inf = infos[0]
        bhp = inf.get('boss_hp', None)
        hp = inf.get('hp', None)
        if isinstance(bhp, (int, float, np.integer, np.floating)):
            min_boss = min(min_boss, int(bhp))
        if isinstance(hp, (int, float, np.integer, np.floating)):
            final_hp = int(hp)
        if steps > 5000:  # safety cap
            break
    return {
        "reward": total_reward,
        "steps": steps,
        "min_boss": min_boss,
        "final_hp": final_hp,
        "win": min_boss <= 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model against a boss.")
    parser.add_argument('--model', type=str, required=True, help='Model path (.zip)')
    parser.add_argument('--state', type=str, default='YellowDevil-boss', help='Fight save state')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--deterministic', action='store_true', help='Greedy policy (no sampling)')
    parser.add_argument('--episode-frames', type=int, default=7200,
                        help='Frames per episode (Yellow Devil uses 14400; must match training)')
    args = parser.parse_args()

    path = args.model
    if not os.path.exists(path) and os.path.exists(path + '.zip'):
        path = path + '.zip'
    if not os.path.exists(path):
        raise SystemExit(f"Model not found: {path}")

    from stable_baselines3 import PPO

    env = make_venv(n_envs=1, state=args.state, render_mode=None, d=D,
                    frameskip=FRAMESKIP, max_episode_frames=args.episode_frames)
    model = PPO.load(path, env=env)

    mode = "deterministic" if args.deterministic else "stochastic"
    print(f"Evaluating {os.path.basename(path)} on {args.state} — {args.episodes} episodes ({mode})")

    results = []
    for i in range(args.episodes):
        r = run_episode(model, env, deterministic=args.deterministic)
        results.append(r)
        print(f"  ep {i + 1:>2}: reward={r['reward']:.4f}  min_boss={r['min_boss']:>2}  "
              f"final_hp={r['final_hp']:>2}  win={r['win']}  steps={r['steps']}")
    env.close()

    wins = sum(int(r['win']) for r in results)
    rewards = np.array([r['reward'] for r in results], dtype=float)
    steps = np.array([r['steps'] for r in results], dtype=float)
    print("-" * 60)
    print(f"Wins:         {wins}/{args.episodes} ({100 * wins / args.episodes:.0f}%)")
    print(f"Mean reward:  {rewards.mean():.4f} +/- {rewards.std():.4f}")
    print(f"Mean length:  {steps.mean():.0f} agent-steps")


if __name__ == '__main__':
    main()
