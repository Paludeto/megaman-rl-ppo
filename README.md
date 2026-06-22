# Mega Man (NES) — Defeating Bosses with PPO

*Read this in another language: **English** · [Português](README.pt-BR.md)*

Deep reinforcement learning (PPO, `CnnPolicy`, **no LSTM**) that learns to defeat *Mega Man*
(NES) bosses using **only the screen pixels** as observation. The same set of hyperparameters
and the same generic reward function beat the **Yellow Devil** and the **6 Robot Masters**
(Bombman, Cutman, Elecman, Fireman, Gutsman, Iceman).

- **Observation:** RGB frame → 84×84 grayscale → stack of 4 frames (`84×84×4`).
- **Actions:** `MULTI_DISCRETE` (direction, A, B).
- **Algorithm:** PPO + GAE (stable-baselines3), fixed seed (666).
- **Generic reward:** `r = d · (hits − m · damage)`, with `d = 0.05`. No win bonus at evaluation
  time. A win is detected from the `boss_health` RAM (address 1729, generic across bosses).

---

## Reproduce in 4 steps (TL;DR)

```bash
# 1. environment
conda env create -f environment.yml && conda activate megaman-ppo

# 2. ROM (you provide it — see section 2)
cp /path/to/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes

# 3. train a boss (e.g. Bombman)
python src/train.py --tag bombman --state Bombman-boss --n-hits 14

# 4. evaluate the trained model
python src/eval_win.py --model models/bombman_best/best_model.zip --state Bombman-boss --episodes 10
```

---

## 1. Requirements

Use **conda**:

```bash
conda env create -f environment.yml   # creates the "megaman-ppo" env (Python 3.12) and installs everything
conda activate megaman-ppo
```

`environment.yml` installs the dependencies by reusing `requirements.txt`. `stable-baselines3`
pulls in compatible `torch`, `numpy`, `gymnasium` and `cloudpickle`. An NVIDIA GPU is strongly
recommended (training is feasible on CPU, but slow).

## 2. ROM (required — you provide your own)

> ⚠️ For legal reasons, **the ROM is not distributed in this repository.** You need a legally
> obtained copy of *Mega Man (NES)*.

Place the file as `rom.nes` inside the integration folder and check the hash:

```bash
cp /path/to/your/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes
sha1sum custom_integrations/MegaMan-v1-Nes/rom.nes
# must be: 2f88381557339a14c20428455f6991c1eb902c99
```

The integration files (`data.json`, `scenario.json`, `metadata.json`, `rom.sha`) and the save
states for each boss (`*.state`) are already included — only `rom.nes` is missing.

## 3. Layout

```
src/
  wrappers.py       # frameskip, reward (BossWrapper), 84×84 preprocessing (WarpFrame)
  env.py            # make_env / make_venv: registers the custom integration and vectorizes
  train.py          # PPO training (CLI parameterized by --tag)
  eval_win.py       # evaluate a saved model: win rate, mean reward and mean length
custom_integrations/MegaMan-v1-Nes/   # stable-retro integration (without rom.nes)
environment.yml     # conda environment
requirements.txt    # dependencies (reused by environment.yml)
```

Outputs generated at runtime (git-ignored): `logs/<tag>`, `checkpoints/<tag>`,
`models/<tag>_best/best_model.zip`.

## 4. Train

The **recipe is fixed** (PPO hyperparameters + reward function) and lives in the constants at the
top of `src/train.py`. The CLI exposes only what changes per boss — in the general case, three flags:

```bash
python src/train.py --tag <tag> --state <STATE> --n-hits <N>
```

**Early-stop** ends training as soon as the evaluation reward crosses the win threshold
`(n_hits − 0.5)·d`. Per-boss parameters:

| Boss         | `--state`            | `--n-hits` | Extra                                       |
|--------------|----------------------|:----------:|---------------------------------------------|
| Yellow Devil | `YellowDevil-boss`   | 14         | `--align-bonus 0.10 --episode-frames 14400` |
| Bombman      | `Bombman-boss`       | 14         | —                                           |
| Cutman       | `Cutman-boss`        | 10         | —                                           |
| Gutsman      | `Gutsman-boss`       | 14         | —                                           |
| Elecman      | `Elecman-boss`       | 28         | —                                           |
| Iceman       | `Iceman-boss`        | 28         | —                                           |
| Fireman      | `Fireman-boss`       | 14         | —                                           |

Yellow Devil is the only special case — it needs the aiming *reward shaping* (`--align-bonus`)
and longer episodes:

```bash
python src/train.py --tag yd --state YellowDevil-boss --n-hits 14 \
  --align-bonus 0.10 --episode-frames 14400
```

Optional flags: `--timesteps` (ceiling, default 40 M), `--n-envs` (default 16), `--checkpoint`
(resume), `--visualize`. Follow training with TensorBoard:

```bash
tensorboard --logdir logs/
```

### Reproduce the 6 Robot Masters in one go

```bash
declare -A NHITS=( [Bombman-boss]=14 [Cutman-boss]=10 [Gutsman-boss]=14 \
                   [Elecman-boss]=28 [Iceman-boss]=28 [Fireman-boss]=14 )
for st in "${!NHITS[@]}"; do
  python src/train.py --tag "${st%-boss}" --state "$st" --n-hits "${NHITS[$st]}"
done
```

## 5. Evaluate

```bash
python src/eval_win.py --model models/<tag>_best/best_model.zip --state <STATE> --episodes 10
```

Loads the saved model, runs the episodes on the real task (no *shaping*) and prints the win rate,
mean reward and mean length. Use `--deterministic` for the greedy policy. For the Yellow Devil add
`--episode-frames 14400` (same as training).

## 6. Expected results

The flawless evaluation reward = `n_hits · d` (a win without taking damage). Early-stop only fires
at that ceiling, so 5 of the 7 bosses converge to the exact flawless value; Iceman and Fireman win
**robustly** (consistently, but taking some damage).

| Boss         | Eval reward     | Win rate | Approx. cost (steps to plateau) |
|--------------|:---------------:|----------|:-------------------------------:|
| Gutsman      | 0.70 (flawless) | 100%     | ~1.3 M                          |
| Iceman       | 1.30 (robust)   | 100%     | ~1.9 M                          |
| Fireman      | 0.55 (robust)   | 100%     | ~2.0 M                          |
| Bombman      | 0.70 (flawless) | 100%     | ~2.5 M                          |
| Cutman       | 0.50 (flawless) | 100%     | ~2.8 M                          |
| Elecman      | 1.40 (flawless) | 100%     | ~2.9 M                          |
| Yellow Devil | 0.70 (flawless) | 100%     | ~15.7 M                         |

On a GPU, the Robot Masters converge in minutes to ~1 h each; the Yellow Devil is the costliest
(~15 M steps). The seed is fixed (666), but the emulator/policy are stochastic — the numbers above
are the target, with small variations between runs.

## 7. Adding a new boss

`n_hits = ceil(28 / damage_per_shot)` (boss HP = 28). E.g.: *buster* (2 damage) → 14; a weapon
dealing 4 damage → 7. To train a new boss: create its initial `.state`, find the per-shot damage
of the weapon used, compute `n_hits` and use the base recipe from section 4.

## 8. Design notes

- **No LSTM and no curriculum:** experiments showed both are dispensable. What matters is the
  aligned reward (for the Yellow Devil, `--align-bonus` brings the shot height toward the eye via
  *potential-based reward shaping*) and long enough episodes.
- **Reward normalization** (fixed in the recipe) affects only the reward, not the observation, so
  it is eval-safe. Evaluation always uses the real task, with no *shaping*.

## 9. Troubleshooting

- **ROM hash differs** from `2f88381…`: you have another revision/dump; the integration expects
  exactly this ROM. Without the correct `rom.nes`, `stable_retro.make` fails.
- **Evaluation doesn't win** a model that won during training: make sure the eval `--episode-frames`
  is identical to training (for the Yellow Devil, `14400`).

## License

See [LICENSE](LICENSE). The *Mega Man* ROM is **not** included and is not covered by this license.
