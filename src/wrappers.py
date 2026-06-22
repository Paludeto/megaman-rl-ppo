"""
Module: src.wrappers
Description: Custom Gym wrappers to process observations, adjust fire cadences,
             handle dynamic frameskipping during blink states, and compute flat rewards
             for Mega Man 1 boss battles.
"""

import gymnasium as gym
import numpy as np
import cv2

class ActionSkipWrapper(gym.Wrapper):
    """
    Alternates sending the shoot input (B button) when it is held by the agent,
    to respect the NES Mega Man firing cadence and avoid wasted inputs.
    """
    def __init__(self, env):
        """
        Initializes the action skip wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
        """
        super().__init__(env)
        self.b_state = 0

    def reset(self, **kwargs):
        """
        Resets the internal toggle state for the shoot button and the environment.

        Returns:
            tuple: Initial observation and environment info dictionary.
        """
        self.b_state = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Processes a single step in the environment. If the agent holds the shoot (B)
        button, alternates releasing it on alternate frames to ensure projectiles fire.

        Parameters:
            action (np.ndarray or list): MultiDiscrete actions of shape (3,).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # MultiDiscrete([2, 5, 2]) action for this integration:
        #   action[0] = B (SHOOT), action[1] = D-pad, action[2] = A (JUMP).
        # The NES firing cadence requires releasing+pressing B for each projectile -> we toggle B
        # (action[0]) so a policy that "holds B" can still fire in bursts. A (jump) is NOT toggled:
        # toggling A would chop the jump and Mega Man wouldn't reach the YD eye (the aiming bottleneck).
        modified_action = action.copy() if hasattr(action, 'copy') else list(action)
        if modified_action[0] == 1:
            if self.b_state == 1:
                modified_action[0] = 0
            self.b_state = 1 - self.b_state
        else:
            self.b_state = 0
        return self.env.step(modified_action)


class FrameskipWrapper(gym.Wrapper):
    """
    Repeats the action for 4 frames. If the agent is flashing (invisible) after taking damage,
    continues repeating the action until the agent becomes visible again, inferring visibility
    from the periodic NES blink pattern derived from blink_counter.
    """
    def __init__(self, env, skip=4):
        """
        Initializes the dynamic frameskip wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
            skip (int): The number of base frames to skip.
        """
        super().__init__(env)
        self._skip = skip
        # frame_sink: for LIVE RECORDING. If set to a list, every emulated frame (raw RGB + audio)
        # is appended here, capturing ALL frames (not just the last of the frameskip) with the real
        # game running (includes RAM writes such as unlimited ammo). None = no capture.
        self.frame_sink = None

    def _capture(self, obs):
        if self.frame_sink is not None:
            try:
                self.frame_sink.append((obs.copy(), self.unwrapped.em.get_audio().copy()))
            except Exception:
                pass

    def step(self, action):
        """
        Applies frameskip by repeating the action for `skip` steps, and then continues
        skipping frames dynamically while the player character is invisible (blinking).

        Parameters:
            action (np.ndarray or list): Action vector to repeat.

        Returns:
            tuple: (observation, aggregated_reward, terminated, truncated, final_info)
        """
        total_reward = 0.0
        terminated = False
        truncated = False

        # Step for the fixed frame-skip amount
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._capture(obs)
            total_reward += reward
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info

        # Continue stepping if Mega Man is in an invisible blink state.
        # Safety cap: never lock the loop if the blink never resolves to visible.
        max_blink_steps = 60
        blink_steps = 0
        while True:
            blink_counter = info.get('blink_counter', 0)
            # Periodic NES Mega Man blink pattern (invisible if remainder is 1 or 2)
            visibility = 'i' if (blink_counter % 4 in (1, 2)) and blink_counter > 0 else 'v'
            if visibility == 'v':
                break

            obs, reward, terminated, truncated, info = self.env.step(action)
            self._capture(obs)
            total_reward += reward
            if terminated or truncated:
                break

            blink_steps += 1
            if blink_steps >= max_blink_steps:
                break

        return obs, total_reward, terminated, truncated, info


class BossWrapper(gym.Wrapper):
    """
    Custom wrapper for the boss fight scenario. Implements the flat reward function
    and the termination condition.
    """
    def __init__(self, env, d=0.05, unlimited_ammo=False, win_bonus=0.0, invincible=False, bonus_hp=0, damage_penalty_mult=1.0, survival_bonus=0.0, waste_penalty=0.0, aim_bonus=0.0, post_kill_frames=0, ammo_budget=0, fire_from_action=False, align_bonus=0.0):
        """
        Initializes the boss reward and termination wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
            d (float): Flat reward scale for dealing damage and penalty for taking damage.
            unlimited_ammo (bool): Curriculum. If True, refills weapon_energy=28 in RAM every step
                and disables ammo-based termination, letting the agent learn to AIM without the
                weapon's ammo cap. Use it to pre-train aiming, then fine-tune with real ammo
                (unlimited_ammo=False).
            win_bonus (float): Terminal bonus added to the reward when the boss dies (boss_health=0).
                Reinforces the (sparse) "finish the kill" signal. 0.0 = original behavior.
        """
        super().__init__(env)
        self.d = d
        self.unlimited_ammo = unlimited_ammo
        self.win_bonus = win_bonus
        self.invincible = invincible
        self.bonus_hp = bonus_hp  # Curriculum: starting HP > 28 (still lethal, but a buffer to learn dodging)
        self.damage_penalty_mult = damage_penalty_mult  # >1 forces the agent to dodge (penalizes damage more than it rewards a hit)
        self.survival_bonus = survival_bonus  # dense + per step alive: strong/continuous signal to learn dodging
        # waste_penalty: - for shooting while the YD eye is CLOSED (shot doomed to miss).
        # aim_bonus: + for shooting while the eye is OPEN (RAM addr 1395 == 0). Dense aiming signal that
        # solves the timing credit-assignment: teaches WHEN to shoot without relying only on sparse hits.
        self.waste_penalty = waste_penalty
        self.aim_bonus = aim_bonus
        # post_kill_frames: for RECORDING. After the boss dies, keep stepping N frames (instead of
        # ending immediately) to capture the death/explosion animation in the .bk2. 0 = normal behavior.
        self.post_kill_frames = post_kill_frames
        self._post_kill_left = None
        # ammo_budget: AMMO CURRICULUM. >0 = grants N effective shots per episode (refills while
        # _shots_fired < ammo_budget; then stops -> ammo runs out -> ammo termination). Bridge from
        # unlimited ammo to real ammo, teaching shot discipline. 0 = off.
        self.ammo_budget = ammo_budget
        self._shots_fired = 0
        # fire_from_action: detect the shot from the agent's ACTION (B button == action[0]) instead of
        # the weapon_energy decrease. REQUIRED for the standard BUSTER, which does NOT consume
        # weapon_energy (stays at 28) — without it the dense aiming signal (aim_bonus/waste_penalty) is dead.
        self.fire_from_action = fire_from_action
        # align_bonus: dense POSITIONAL aiming signal. Rewards the agent for matching the shot height
        # (Mega Man's y) to the YD EYE height (see EYE_Y_ADDR below, reverse-engineered from RAM).
        # Breaks the buster bootstrap wall (1 projectile + ~1-frame window -> a random shot NEVER hits
        # on its own) by giving a "rise to the eye" gradient before the first hit.
        self.align_bonus = align_bonus
        # YD eye Y: RAM 0x0601 (1537) — validated by reverse engineering (pink pupil 224,0,88 detected
        # by CV, corr 1.000, 0.1px residual at 4 heights). It tracks Mega Man's y (1536) in an object-Y
        # table. eye_y_px ≈ 1.0*reg - 6.9 (same coordinate system as the player).
        self.EYE_Y_ADDR = 1537
        self.EYE_Y_A = 1.0
        self.EYE_Y_B = -6.9
        self.align_gamma = 0.99         # PBRS gamma (matches the training gamma)
        self._prev_align_phi = 0.0
        self.prev_health = 0
        self.prev_boss_health = 0
        self.prev_lives = 0
        self.prev_weapon_energy = 28
        self.prev_eye_state = 128  # 128 = eye closed

    def _align_phi(self, info):
        """Positional aiming potential: φ = align_bonus·exp(-(err/18)²), err in pixels between Mega
        Man's shot height and the YD eye height (RAM EYE_Y_ADDR)."""
        try:
            eye_y_reg = float(self.unwrapped.get_ram()[self.EYE_Y_ADDR])
            eye_px = self.EYE_Y_A * eye_y_reg + self.EYE_Y_B
            gun_px = float(info.get('y', 148)) - 12.0   # the shot leaves at chest height (~y - 12)
            err = abs(eye_px - gun_px)
            return self.align_bonus * float(np.exp(-(err / 18.0) ** 2))
        except Exception:
            return 0.0

    @staticmethod
    def _info_dict(health, boss_health, lives, weapon_energy, eye_state, info):
        """Build the info dict exposed to the agent (same format in reset and step)."""
        return {
            'hp': health,
            'boss_hp': boss_health,
            'lives': lives,
            'x': info.get('x', 0),
            'y': info.get('y', 0),
            'screen': info.get('screen', 0),
            'weapon_energy': weapon_energy,
            'eye_state': eye_state,
        }

    def reset(self, **kwargs):
        """
        Resets environment tracking variables for health, lives, and boss health.

        Returns:
            tuple: (observation, custom_info)
        """
        obs, info = self.env.reset(**kwargs)
        self._prev_align_phi = self._align_phi(info) if self.align_bonus > 0.0 else 0.0
        self._post_kill_left = None  # reset the post-kill linger counter (recording)
        self._shots_fired = 0        # reset the shot counter (ammo curriculum)
        # Graduated HP curriculum: give the agent a starting life buffer larger than 28 (tested: the
        # game honors it and decrements normally), so it survives more eye windows while learning to
        # dodge. Still lethal (dies at HP 0).
        if self.bonus_hp > 0:
            try:
                self.unwrapped.data.set_value("health", self.bonus_hp)
            except Exception:
                pass
            info = dict(info); info['health'] = self.bonus_hp
        # IMPORTANT: retro's info on reset is usually EMPTY (no 'lives', 'boss_health', etc.). Trusting
        # info.get(name, default) with a wrong default (e.g. lives=9 when the save has 8) made the 1st
        # step read lives=8 < prev_lives=9 -> spurious "lost a life" termination -> 1-step episodes
        # (broke both the lethal ladder AND eval). We read the REAL values from RAM.
        def _ram(name, default):
            try:
                return int(self.unwrapped.data.lookup_value(name))
            except Exception:
                return int(info.get(name, default))
        self.prev_health = self.bonus_hp if self.bonus_hp > 0 else _ram('health', 28)
        self.prev_boss_health = _ram('boss_health', 28)
        self.prev_lives = _ram('lives', 9)
        self.prev_weapon_energy = _ram('weapon_energy', 28)
        self.prev_eye_state = _ram('eye_state', 128)

        custom_info = self._info_dict(self.prev_health, self.prev_boss_health, self.prev_lives,
                                      self.prev_weapon_energy, self.prev_eye_state, info)
        return obs, custom_info

    def step(self, action):
        """
        Computes custom flat reward based on damage events, and returns early episode
        termination when boss or player health drops to zero, lives decrease, or spikes are hit.

        Parameters:
            action (np.ndarray): Action to take.

        Returns:
            tuple: (observation, reward, terminated, truncated, custom_info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Use the previous value as default if a RAM read fails/is missing, avoiding spurious
        # terminations and win bonuses (default 0 would trigger health=0 / boss=0).
        cur_health = info.get('health', self.prev_health)
        cur_boss_health = info.get('boss_health', self.prev_boss_health)
        cur_lives = info.get('lives', self.prev_lives)
        touching_obj_top = info.get('touching_obj_top', 0)
        touching_obj_side = info.get('touching_obj_side', 0)
        cur_weapon_energy = info.get('weapon_energy', self.prev_weapon_energy)
        cur_eye_state = info.get('eye_state', self.prev_eye_state)
        eye_open = (cur_eye_state == 0)  # RAM addr 1395: 0 = eye open/vulnerable, 128 = closed

        # Reward function (every term below is parameterized; 0 = off):
        #   +d                  per hit event on the boss
        #   -d*damage_mult      per damage-taken event
        #   +aim_bonus          for shooting with the eye OPEN  (dense timing signal)
        #   -waste_penalty      for shooting with the eye CLOSED (suppresses spray)
        #   +win_bonus          terminal, on killing the boss
        #   +survival_bonus     per step alive (dense; default 0)
        reward_boss = 0.0

        boss_took_damage = self.prev_boss_health > cur_boss_health
        if boss_took_damage:
            reward_boss += self.d
        if self.prev_health > cur_health:
            reward_boss -= self.d * self.damage_penalty_mult

        # Dense aiming reward (eye state, RAM addr 1395). Uses the RAW weapon_energy (before any
        # unlimited-ammo refill) to detect the shot. Solves the timing credit-assignment: teaches
        # WHEN to shoot without relying only on the sparse hit (~1 every 184 steps).
        if self.fire_from_action:
            # Buster: weapon_energy doesn't drop -> detect the shot from the agent's intent (B = action[0]).
            try:
                fired = int(action[0]) == 1
            except (IndexError, TypeError, ValueError):
                fired = False
        else:
            fired = cur_weapon_energy < self.prev_weapon_energy
        if fired:
            self._shots_fired += 1                 # ammo curriculum: count the shot
            if eye_open:
                reward_boss += self.aim_bonus      # shot in the window = right timing
            else:
                reward_boss -= self.waste_penalty  # shot with eye closed = doomed to miss

        # Curriculum: unlimited ammo. Refills weapon_energy=28 in RAM (drops ~2/step, so it never
        # reaches 0) so the agent can train AIMING without the 28-shot cap.
        # Refill if: unlimited ammo, OR ammo curriculum still within the shot budget.
        if self.unlimited_ammo or (self.ammo_budget > 0 and self._shots_fired < self.ammo_budget):
            try:
                self.unwrapped.data.set_value("weapon_energy", 28)
            except Exception:
                pass
            cur_weapon_energy = 28

        # Survival curriculum: invincibility. The -d damage penalty was ALREADY applied above (the
        # agent is still encouraged to dodge), but HP is restored to 28 and death is disabled — the
        # episode runs to timeout, giving the agent all ~N eye windows to learn to land the hits.
        # Safe: YD damage ~4/hit << 28.
        if self.invincible:
            try:
                self.unwrapped.data.set_value("health", 28)
            except Exception:
                pass
            cur_health = 28

        # Determine termination
        custom_terminated = False
        if (not self.invincible) and cur_health == 0:
            custom_terminated = True
        elif (not self.invincible) and cur_lives < self.prev_lives:
            custom_terminated = True
        elif (not self.invincible) and (touching_obj_top == 3 or touching_obj_side == 3):
            custom_terminated = True
        elif cur_boss_health == 0:
            # Win. If post_kill_frames>0 (recording mode), do NOT end immediately: keep N steps so the
            # .bk2 captures the boss's explosion/death animation; only then terminate.
            if self.post_kill_frames > 0:
                if self._post_kill_left is None:
                    self._post_kill_left = self.post_kill_frames
                    reward_boss += self.win_bonus  # terminal bonus only on the death transition
                if self._post_kill_left > 0:
                    self._post_kill_left -= 1
                    custom_terminated = False
                else:
                    custom_terminated = True
            else:
                custom_terminated = True
                reward_boss += self.win_bonus  # terminal win bonus (sparse)
        elif (not self.unlimited_ammo) and cur_weapon_energy == 0 and cur_boss_health > 0:
            # Early termination if ammo runs out and the boss is still alive
            custom_terminated = True

        # Dense survival reward: + per step while Mega Man is alive. Gives a continuous positive
        # signal (not sparse like the damage penalty) rewarding every "didn't get hit" = dodged well.
        # Small enough not to dominate the kill.
        if cur_health > 0:
            reward_boss += self.survival_bonus

        # Dense POSITIONAL aiming reward via POTENTIAL-BASED SHAPING (PBRS): F = γ·φ(s') - φ(s), where
        # φ = align_bonus·exp(-(err/18)²) and err = |shot height - EYE height| (RAM 0x0601). PBRS rewards
        # MOVING toward the eye but is provably optimal-policy invariant (you can't farm it by floating
        # without killing). Breaks the bootstrap wall with a "rise to the eye" gradient before the first
        # hit. φ=0 when the boss dies (correct terminal handling).
        if self.align_bonus > 0.0:
            phi = self._align_phi(info) if cur_boss_health > 0 else 0.0
            reward_boss += self.align_gamma * phi - self._prev_align_phi
            self._prev_align_phi = phi

        # Save previous values
        self.prev_health = cur_health
        self.prev_boss_health = cur_boss_health
        self.prev_lives = cur_lives
        self.prev_weapon_energy = cur_weapon_energy
        self.prev_eye_state = cur_eye_state

        custom_info = self._info_dict(cur_health, cur_boss_health, cur_lives,
                                      cur_weapon_energy, cur_eye_state, info)

        return obs, reward_boss, custom_terminated, truncated, custom_info


class WarpFrame(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale and resizes them to 84x84.
    """
    def __init__(self, env):
        """
        Initializes the frame preprocessing wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
        """
        super().__init__(env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        """
        Preprocesses raw RGB frames to 84x84 grayscale.

        Parameters:
            obs (np.ndarray): Raw RGB observation.

        Returns:
            np.ndarray: Grayscale resized observation of shape (84, 84, 1).
        """
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)
