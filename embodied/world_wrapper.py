"""
WorldModelWrapper: High-level Gymnasium environment wrapper that
combines the low-level WorldModelEnv with episode management,
metric collection, and optional video recording.

Usage:
    python -m embodied.world_wrapper
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from embodied.rl_action_api import WorldModelEnv


class WorldModelWrapper(gym.Wrapper):
    """Gymnasium wrapper adding episode tracking and metric collection.

    Wraps a WorldModelEnv to collect per-step rewards, frame diffs,
    and episode-level statistics for downstream evaluation.
    """

    def __init__(
        self,
        model,
        resolution: Tuple[int, int] = (480, 832),
        max_steps: int = 300,
        record_frames: bool = False,
    ):
        """
        Args:
            model: Callable world model with signature
                   model(frame: ndarray, action: ndarray) -> ndarray.
            resolution: (height, width) of the observation space.
            max_steps: Maximum steps per episode.
            record_frames: If True, store all frames for later playback.
        """
        env = WorldModelEnv(model=model, resolution=resolution, max_steps=max_steps)
        super().__init__(env)
        self.record_frames = record_frames
        self._episode_rewards: List[float] = []
        self._episode_diffs: List[float] = []
        self._recorded_frames: List[np.ndarray] = []
        self._episode_count: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and clear episode accumulators."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_rewards = []
        self._episode_diffs = []
        self._recorded_frames = []
        if self.record_frames:
            self._recorded_frames.append(obs.copy())
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment and record metrics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_rewards.append(reward)
        self._episode_diffs.append(info.get("frame_diff_mean", 0.0))
        if self.record_frames:
            self._recorded_frames.append(obs.copy())

        if terminated or truncated:
            self._episode_count += 1
            info["episode_stats"] = self.get_episode_stats()

        return obs, reward, terminated, truncated, info

    def get_episode_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the current episode."""
        rewards = np.array(self._episode_rewards) if self._episode_rewards else np.array([0.0])
        diffs = np.array(self._episode_diffs) if self._episode_diffs else np.array([0.0])
        return {
            "episode": self._episode_count,
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "min_reward": float(np.min(rewards)),
            "mean_frame_diff": float(np.mean(diffs)),
            "steps": len(self._episode_rewards),
        }

    def get_recorded_frames(self) -> List[np.ndarray]:
        """Return frames recorded during the episode."""
        return list(self._recorded_frames)


# ------------------------------------------------------------------
# CLI demo entry point
# ------------------------------------------------------------------

if __name__ == "__main__":

    class DummyModel:
        """Trivial model that returns a slightly noisy copy of the input."""

        def __call__(self, frame: np.ndarray, action: np.ndarray) -> np.ndarray:
            noise = np.random.randint(-5, 6, size=frame.shape, dtype=np.int16)
            return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    model = DummyModel()
    wrapper = WorldModelWrapper(model=model, resolution=(64, 64), max_steps=10)
    obs, info = wrapper.reset()
    print(f"Initial observation shape: {obs.shape}")

    done = False
    step = 0
    while not done:
        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        done = terminated or truncated
        step += 1
        print(f"Step {step}: reward={reward:.4f}")

    print(f"\nEpisode stats: {info.get('episode_stats', {})}")
