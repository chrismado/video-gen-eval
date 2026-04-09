"""
Standardized RL Action API for Closed-Loop World Model Evaluation.

Wraps a generative video model as an RL environment.
The agent acts, the world model predicts, we measure whether predictions
are physically plausible and causally consistent.

This is the key upgrade over open-loop evaluation:
instead of scoring finished videos, we score models
under continuous interactive pressure.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple


class WorldModelEnv(gym.Env):
    """
    Generative video model wrapped as RL environment.
    Compatible with WorldArena EWMScore protocol (CVPR 2026).

    The model must expose a callable interface:
        model(current_frame: np.ndarray, action: np.ndarray) -> np.ndarray
    returning the predicted next frame given the current frame and action.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, model, resolution: Tuple[int, int] = (480, 832), max_steps: int = 300):
        super().__init__()
        self.model = model
        self.resolution = resolution
        self.max_steps = max_steps

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(*resolution, 3),
            dtype=np.uint8,
        )
        # 6-D action: 3D force vector + 3D camera orientation delta
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32,
        )

        self._current_frame: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._episode_frames: list = []

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Initialize episode with a starting frame.

        If `options["initial_frame"]` is provided, use it directly.
        Otherwise, generate a blank scene through the model (or a zero frame
        if the model does not support prompt-less generation).
        """
        super().reset(seed=seed)

        if options and "initial_frame" in options:
            frame = np.asarray(options["initial_frame"], dtype=np.uint8)
            if frame.shape[:2] != self.resolution:
                # Resize to expected resolution via simple nearest-neighbor
                import cv2
                frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))
        else:
            # Ask the model for an initial frame with a null action
            null_action = np.zeros(self.action_space.shape, dtype=np.float32)
            blank = np.zeros((*self.resolution, 3), dtype=np.uint8)
            try:
                frame = np.asarray(self.model(blank, null_action), dtype=np.uint8)
            except Exception:
                frame = blank

        self._current_frame = frame
        self._prev_frame = None
        self._step_count = 0
        self._episode_frames = [frame.copy()]

        return frame, {"step": 0}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Apply action vector to world model, generate next frame.

        Returns:
            observation: predicted next frame (H, W, 3) uint8
            reward: physics-plausibility reward for this transition
            terminated: True if episode exceeds max_steps
            truncated: always False (no external truncation)
            info: dict with per-step diagnostics
        """
        action = np.asarray(action, dtype=np.float32)

        # Generate next frame through the world model
        next_frame = np.asarray(
            self.model(self._current_frame, action), dtype=np.uint8
        )

        # Compute physics-based reward
        reward = self.compute_physics_reward(self._current_frame, next_frame, action)

        self._prev_frame = self._current_frame
        self._current_frame = next_frame
        self._step_count += 1
        self._episode_frames.append(next_frame.copy())

        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "step": self._step_count,
            "reward": reward,
            "frame_diff_mean": float(np.mean(np.abs(
                next_frame.astype(np.float32) - self._prev_frame.astype(np.float32)
            ))),
        }

        return next_frame, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Physics reward
    # ------------------------------------------------------------------

    def compute_physics_reward(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Reward based on physical plausibility of a state transition.

        Penalizes:
          1. Object teleportation: large sudden pixel displacement inconsistent
             with applied force magnitude.
          2. Impossible acceleration: frame-to-frame change that exceeds a
             physically plausible threshold given the action magnitude.
          3. Gravity violations: upward motion in the lower half of the frame
             when no upward force is applied (simple heuristic proxy).

        Returns a reward in [-1.0, 1.0]. Higher is more physically plausible.
        """
        reward = 1.0  # start optimistic

        prev_f = prev_frame.astype(np.float32)
        curr_f = curr_frame.astype(np.float32)
        diff = np.abs(curr_f - prev_f)
        mean_diff = float(np.mean(diff))

        force_magnitude = float(np.linalg.norm(action[:3]))

        # --- Penalty 1: Object teleportation ---
        # Large frame delta with small applied force suggests teleportation.
        teleport_threshold = 30.0 + 80.0 * force_magnitude
        if mean_diff > teleport_threshold:
            overshoot = (mean_diff - teleport_threshold) / teleport_threshold
            reward -= min(0.5, overshoot * 0.3)

        # --- Penalty 2: Impossible acceleration ---
        # Maximum plausible per-pixel change given action strength.
        max_plausible_change = 50.0 + 120.0 * force_magnitude
        peak_change = float(np.max(diff))
        if peak_change > max_plausible_change:
            excess = (peak_change - max_plausible_change) / max_plausible_change
            reward -= min(0.3, excess * 0.2)

        # --- Penalty 3: Gravity violations ---
        # Check lower half: if objects move upward without upward force.
        h = prev_frame.shape[0]
        lower_half_diff = diff[h // 2:, :, :]
        upper_half_diff = diff[:h // 2, :, :]

        upward_force = action[1] if len(action) > 1 else 0.0  # y-component
        lower_activity = float(np.mean(lower_half_diff))
        upper_activity = float(np.mean(upper_half_diff))

        # If upper region changes more than lower and no upward force,
        # something may be floating upward without cause.
        if upper_activity > lower_activity * 1.5 and upward_force < 0.1:
            gravity_penalty = min(0.2, (upper_activity - lower_activity) / 100.0)
            reward -= gravity_penalty

        return float(np.clip(reward, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        """Return the current frame for visualization."""
        if self._current_frame is None:
            return np.zeros((*self.resolution, 3), dtype=np.uint8)
        return self._current_frame

    def get_episode_frames(self) -> list:
        """Return all frames collected in the current episode."""
        return list(self._episode_frames)
