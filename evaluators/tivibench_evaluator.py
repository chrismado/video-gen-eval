"""
TiViBenchEvaluator: Wrapper around the TiViBench temporal and causal
reasoning benchmark for generated videos.

Assesses whether generated videos exhibit plausible temporal ordering,
causal consistency, and event sequencing.
The official benchmark runner is intentionally exposed as an adapter hook
instead of importing an assumed public Python API.
"""

from typing import Any, Callable, Dict, Optional

from pipeline.unified_pipeline import EvaluatorResult


class TiViBenchEvaluator:
    """Evaluate causal and temporal reasoning quality via TiViBench.

    This class normalizes TiViBench-shaped outputs. Pass an adapter callable
    when wiring an official runner.
    """

    name: str = "tivibench"

    def __init__(
        self,
        causal_weight: float = 0.5,
        temporal_weight: float = 0.5,
        adapter: Optional[Callable[[str], Any]] = None,
    ):
        """
        Args:
            causal_weight:  Weight for causal consistency sub-score.
            temporal_weight: Weight for temporal ordering sub-score.
        """
        self.causal_weight = causal_weight
        self.temporal_weight = temporal_weight
        self.adapter = adapter

    def evaluate(self, video_path: str) -> EvaluatorResult:
        """Run TiViBench evaluation on a single video.

        Returns:
            EvaluatorResult with causal / temporal reasoning scores.
        """
        if self.adapter is None:
            return EvaluatorResult(
                name=self.name,
                error="external TiViBench adapter not configured; wire an official TiViBench runner before enabling",
            )
        try:
            results = self.adapter(str(video_path))
            return EvaluatorResult(name=self.name, scores=self._parse_results(results))
        except Exception as e:
            return EvaluatorResult(name=self.name, error=str(e))

    def _parse_results(self, results: Any) -> Dict[str, float]:
        """Normalize TiViBench output into a flat score dict."""
        scores: Dict[str, float] = {}
        if isinstance(results, dict):
            for key, val in results.items():
                norm_key = key.lower().replace(" ", "_").replace("-", "_")
                if isinstance(val, (int, float)):
                    scores[norm_key] = float(val)
        return scores
