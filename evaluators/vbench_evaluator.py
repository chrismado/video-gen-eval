"""
VBenchEvaluator: Wrapper around the VBench 16-dimension video quality benchmark.

Provides a unified evaluate() interface returning EvaluatorResult,
consistent with the UnifiedPipeline contract.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.unified_pipeline import EvaluatorResult


class VBenchEvaluator:
    """Evaluate a generated video across VBench's 16 perceptual dimensions.

    Wraps the Vchitect/VBench library with graceful fallback when it is
    not installed.
    """

    name: str = "vbench"

    def __init__(self, dimensions: Optional[List[str]] = None):
        """
        Args:
            dimensions: Subset of VBench dimensions to evaluate.
                        If None, all 16 dimensions are used.
        """
        self.dimensions = dimensions

    def evaluate(self, video_path: str) -> EvaluatorResult:
        """Run VBench evaluation on a single video.

        Returns:
            EvaluatorResult with per-dimension scores in ``scores``.
        """
        video_path = str(video_path)
        try:
            from vbench import VBench

            bench = VBench()
            results = bench.evaluate(video_path)
            scores = self._parse_results(results)
            return EvaluatorResult(name=self.name, scores=scores)
        except ImportError:
            return EvaluatorResult(name=self.name, error="vbench not installed")
        except Exception as e:
            return EvaluatorResult(name=self.name, error=str(e))

    def _parse_results(self, results: Any) -> Dict[str, float]:
        """Normalize VBench output into a flat dict of dimension -> score."""
        scores: Dict[str, float] = {}
        if isinstance(results, list):
            for entry in results:
                dim = entry.get("dimension", "")
                key = dim.lower().replace(" ", "_").replace("-", "_")
                score = entry.get("score")
                if key and score is not None:
                    scores[key] = float(score)
        elif isinstance(results, dict):
            for key, val in results.items():
                norm_key = key.lower().replace(" ", "_").replace("-", "_")
                if isinstance(val, (int, float)):
                    scores[norm_key] = float(val)
        if self.dimensions:
            scores = {k: v for k, v in scores.items() if k in self.dimensions}
        return scores
