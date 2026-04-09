"""
TaskEvaluator: Evaluate embodied task performance using the WorldModelWrapper.

Runs a series of episodes with a given policy and world model, collecting
per-episode statistics and computing aggregate task success metrics.
"""
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

from embodied.world_wrapper import WorldModelWrapper


class TaskEvaluator:
    """Run and evaluate embodied tasks within a wrapped world model environment.

    Executes multiple episodes using a provided policy function and
    aggregates the results for benchmarking.
    """

    def __init__(
        self,
        model,
        resolution: Tuple[int, int] = (480, 832),
        max_steps: int = 300,
        n_episodes: int = 5,
    ):
        """
        Args:
            model: Callable world model for WorldModelWrapper.
            resolution: Frame resolution (height, width).
            max_steps: Maximum steps per episode.
            n_episodes: Number of evaluation episodes.
        """
        self.model = model
        self.resolution = resolution
        self.max_steps = max_steps
        self.n_episodes = n_episodes

    def evaluate(
        self,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        success_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Run evaluation episodes and return aggregate metrics.

        Args:
            policy: Callable that takes an observation (ndarray) and returns
                    an action (ndarray).  If None, a random policy is used.
            success_threshold: Minimum total episode reward to count as
                               a successful episode.

        Returns:
            Dict with per-episode stats and aggregate metrics including
            success rate, mean reward, and mean physics plausibility.
        """
        wrapper = WorldModelWrapper(
            model=self.model,
            resolution=self.resolution,
            max_steps=self.max_steps,
        )

        episode_results: List[Dict[str, Any]] = []

        for ep in range(self.n_episodes):
            obs, _ = wrapper.reset(seed=ep)
            done = False
            while not done:
                if policy is not None:
                    action = policy(obs)
                else:
                    action = wrapper.action_space.sample()
                obs, reward, terminated, truncated, info = wrapper.step(action)
                done = terminated or truncated

            stats = wrapper.get_episode_stats()
            stats["success"] = stats["total_reward"] >= success_threshold
            episode_results.append(stats)

        return self._aggregate(episode_results, success_threshold)

    def _aggregate(
        self,
        episode_results: List[Dict[str, Any]],
        success_threshold: float,
    ) -> Dict[str, Any]:
        """Compute aggregate metrics from per-episode results."""
        total_rewards = [ep["total_reward"] for ep in episode_results]
        mean_rewards = [ep["mean_reward"] for ep in episode_results]
        successes = [ep["success"] for ep in episode_results]

        return {
            "n_episodes": len(episode_results),
            "success_rate": float(np.mean(successes)),
            "mean_total_reward": float(np.mean(total_rewards)),
            "std_total_reward": float(np.std(total_rewards)),
            "mean_step_reward": float(np.mean(mean_rewards)),
            "success_threshold": success_threshold,
            "episodes": episode_results,
        }
