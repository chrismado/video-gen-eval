"""
TiViBenchEvaluator: Wrapper around the TiViBench temporal and causal
reasoning benchmark for generated videos.

Assesses whether generated videos exhibit plausible temporal ordering,
causal consistency, and event sequencing.
"""

from typing import Any, Dict

from pipeline.unified_pipeline import EvaluatorResult


class TiViBenchEvaluator:
    """Evaluate causal and temporal reasoning quality via TiViBench.

    Wraps the TiViBench library with graceful fallback when it is
    not installed.
    """

    name: str = "tivibench"

    def __init__(self, causal_weight: float = 0.5, temporal_weight: float = 0.5):
        """
        Args:
            causal_weight:  Weight for causal consistency sub-score.
            temporal_weight: Weight for temporal ordering sub-score.
        """
        self.causal_weight = causal_weight
        self.temporal_weight = temporal_weight

    def evaluate(self, video_path: str) -> EvaluatorResult:
        """Run TiViBench evaluation on a single video.

        Returns:
            EvaluatorResult with causal / temporal reasoning scores.
        """
        video_path = str(video_path)
        try:
            from tivibench import TiViBench

            bench = TiViBench()
            results = bench.evaluate(video_path)
            scores = self._parse_results(results)
            return EvaluatorResult(name=self.name, scores=scores)
        except ImportError:
            return EvaluatorResult(name=self.name, error="tivibench not installed")
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
