"""
Module: src.env
Description: Provides utilities for creating and vectorizing the Mega Man NES environment
             with the custom wrappers for the Yellow Devil RL task.
"""

import os

import stable_retro
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

try:
    from src.wrappers import ActionSkipWrapper, FrameskipWrapper, BossWrapper, WarpFrame, RenderModeWrapper
except ImportError:
    from wrappers import ActionSkipWrapper, FrameskipWrapper, BossWrapper, WarpFrame, RenderModeWrapper

# Custom integrations directory (at the project root). Registered ONCE per process: SubprocVecEnv
# re-imports this module in every worker, so each process registers its own.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CUSTOM_INTEGRATIONS_DIR = os.path.join(_PROJECT_ROOT, "custom_integrations")
_integrations_registered = False


def _register_custom_integrations():
    """Register the custom integrations path with stable-retro (idempotent per process)."""
    global _integrations_registered
    if not _integrations_registered:
        stable_retro.data.add_custom_integration(_CUSTOM_INTEGRATIONS_DIR)
        _integrations_registered = True


def make_env(game="MegaMan-v1-Nes", state="YellowDevil-boss", render_mode=None, record=False, actual_render_mode=None, d=0.05, unlimited_ammo=False, win_bonus=0.0, invincible=False, bonus_hp=0, damage_penalty_mult=1.0, frameskip=4, survival_bonus=0.0, waste_penalty=0.0, aim_bonus=0.0, post_kill_frames=0, ammo_budget=0, fire_from_action=False, align_bonus=0.0, max_episode_frames=7200):
    """
    Creates a single instance of the Mega Man retro environment wrapped with the custom
    RL preprocessing steps.

    Parameters:
        game (str): Name of the game in stable-retro integrations.
        state (str): Name of the initial save state to load on reset.
        render_mode (str or None): The public render mode for Gymnasium (e.g. 'human', 'rgb_array').
        record (str or bool): The folder directory path to save .bk2 recordings of the gameplay.
        actual_render_mode (str or None): The actual emulator render mode used internally.
        d (float): Flat reward scale passed to BossWrapper for hits/damage.

    Returns:
        gym.Env: Wrapped Mega Man retro environment instance.
    """
    _register_custom_integrations()

    if actual_render_mode is None:
        actual_render_mode = render_mode

    env = stable_retro.make(
        game=game,
        state=state,
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=stable_retro.Actions.MULTI_DISCRETE,
        obs_type=stable_retro.Observations.IMAGE,
        render_mode=actual_render_mode,
        record=record,
    )

    env = ActionSkipWrapper(env)
    env = FrameskipWrapper(env, skip=frameskip)
    # max_episode_steps scales with the frameskip to keep ~the same in-game time per episode
    # (skip=4 -> 1800 agent-steps = 7200 frames; skip=2 -> 3600 agent-steps = 7200 frames).
    env = TimeLimit(env, max_episode_steps=max_episode_frames // frameskip)
    env = BossWrapper(env, d=d, unlimited_ammo=unlimited_ammo, win_bonus=win_bonus, invincible=invincible, bonus_hp=bonus_hp, damage_penalty_mult=damage_penalty_mult, survival_bonus=survival_bonus, waste_penalty=waste_penalty, aim_bonus=aim_bonus, post_kill_frames=post_kill_frames, ammo_budget=ammo_budget, fire_from_action=fire_from_action, align_bonus=align_bonus)
    env = WarpFrame(env)

    # Override the public render_mode so the VecEnv consistency check passes
    env = RenderModeWrapper(env, render_mode)
    return env


def make_venv(n_envs=8, game="MegaMan-v1-Nes", state="YellowDevil-boss", render_mode=None, record=False, force_subproc=False, d=0.05, unlimited_ammo=False, win_bonus=0.0, invincible=False, bonus_hp=0, damage_penalty_mult=1.0, frameskip=4, survival_bonus=0.0, norm_reward=False, waste_penalty=0.0, aim_bonus=0.0, post_kill_frames=0, ammo_budget=0, fire_from_action=False, align_bonus=0.0, max_episode_frames=7200):
    """
    Creates a vectorized environment with multiple parallel instances of make_env.
    Applies Frame Stacking (4 frames), Transposition to PyTorch format (C, H, W),
    and sets up basic metrics monitoring.

    Parameters:
        n_envs (int): Number of parallel environments.
        game (str): Name of the game in stable-retro.
        state (str): Name of the initial save state to load.
        render_mode (str or None): Render mode for the environments.
        record (str or bool): Record directory path for the first environment instance.
        force_subproc (bool): If True, forces using SubprocVecEnv even for n_envs=1.
        d (float): Flat reward scale passed through to BossWrapper for hits/damage.

    Returns:
        VecMonitor: Vectorized, stacked, transposed, and monitored environment wrapper.
    """
    # Environment/reward knobs shared by all parallel envs (render_mode/record vary per rank and
    # are passed separately). Defining them here avoids duplicating the parameter list.
    env_kwargs = dict(
        game=game, state=state, d=d, unlimited_ammo=unlimited_ammo, win_bonus=win_bonus,
        invincible=invincible, bonus_hp=bonus_hp, damage_penalty_mult=damage_penalty_mult,
        frameskip=frameskip, survival_bonus=survival_bonus, waste_penalty=waste_penalty,
        aim_bonus=aim_bonus, post_kill_frames=post_kill_frames, ammo_budget=ammo_budget,
        fire_from_action=fire_from_action, align_bonus=align_bonus,
        max_episode_frames=max_episode_frames,
    )

    def make_thunk(rank):
        # If render_mode is "human", only rank 0 actually renders the emulator (avoids opening
        # multiple windows); every env still exposes render_mode publicly (VecEnv validation).
        actual_render_mode = render_mode if (rank == 0 or render_mode != "human") else None
        return lambda: make_env(render_mode=render_mode, record=(record if rank == 0 else False),
                                actual_render_mode=actual_render_mode, **env_kwargs)

    if n_envs > 1 or force_subproc:
        venv = SubprocVecEnv([make_thunk(i) for i in range(n_envs)])
    else:
        venv = DummyVecEnv([make_thunk(0)])

    # Stack 4 frames, transpose image to (C, H, W) and monitor hp/boss_hp
    venv = VecFrameStack(venv, n_stack=4, channels_order='last')
    venv = VecTransposeImage(venv)
    venv = VecMonitor(venv, info_keywords=("hp", "boss_hp"))
    # Reward normalization (running std). Reward only (norm_obs=False), so it does not affect the
    # policy's observations/actions — it is eval-safe and the stats can be fresh per stage.
    if norm_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    return venv
