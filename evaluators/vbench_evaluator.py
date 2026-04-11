"""
VBenchEvaluator: Wrapper around the VBench 16-dimension video quality benchmark.

Provides a unified evaluate() interface returning EvaluatorResult.
The official VBench runner is intentionally exposed as an adapter hook instead
of importing an assumed public Python API.
"""

from typing import Any, Callable, Dict, List, Optional

from pipeline.unified_pipeline import EvaluatorResult


class VBenchEvaluator:
    """Evaluate a generated video across VBench's 16 perceptual dimensions.

    This class normalizes VBench-shaped outputs. It does not assume that an
    installed package exposes a package-root VBench class; pass an adapter
    callable when wiring an official runner.
    """

    name: str = "vbench"

    def __init__(self, dimensions: Optional[List[str]] = None, adapter: Optional[Callable[[str], Any]] = None):
        """
        Args:
            dimensions: Subset of VBench dimensions to evaluate.
                        If None, all 16 dimensions are used.
        """
        self.dimensions = dimensions
        self.adapter = adapter

    def evaluate(self, video_path: str) -> EvaluatorResult:
        """Run VBench evaluation on a single video.

        Returns:
            EvaluatorResult with per-dimension scores in ``scores``.
        """
        if self.adapter is None:
            return EvaluatorResult(
                name=self.name,
                error="external VBench adapter not configured; wire an official VBench runner before enabling",
            )
        try:
            results = self.adapter(str(video_path))
            return EvaluatorResult(name=self.name, scores=self._parse_results(results))
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
