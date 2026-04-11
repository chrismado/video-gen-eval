"""
IVEBenchEvaluator: Wrapper around the IVEBench instruction-following
video evaluation benchmark.

Measures how well a generated video follows the text instruction that
prompted it (compositional accuracy, attribute binding, action fidelity).
The official benchmark runner is intentionally exposed as an adapter hook
instead of importing an assumed public Python API.
"""

from typing import Any, Callable, Dict, Optional

from pipeline.unified_pipeline import EvaluatorResult


class IVEBenchEvaluator:
    """Evaluate instruction-compliance of a generated video via IVEBench.

    This class normalizes IVEBench-shaped outputs. Pass an adapter callable when
    wiring an official runner.
    """

    name: str = "ivebench"

    def __init__(self, instruction: Optional[str] = None, adapter: Optional[Callable[[str], Any]] = None):
        """
        Args:
            instruction: The text prompt / instruction the video was
                         generated from.  Required by IVEBench for
                         compliance scoring.
        """
        self.instruction = instruction
        self.adapter = adapter

    def evaluate(self, video_path: str) -> EvaluatorResult:
        """Run IVEBench evaluation on a single video.

        Returns:
            EvaluatorResult with instruction-compliance scores.
        """
        if self.adapter is None:
            return EvaluatorResult(
                name=self.name,
                error="external IVEBench adapter not configured; wire an official IVEBench runner before enabling",
            )
        try:
            results = self.adapter(str(video_path))
            scores = self._parse_results(results)
            metadata: Dict[str, Any] = {}
            if self.instruction:
                metadata["instruction"] = self.instruction
            return EvaluatorResult(name=self.name, scores=scores, metadata=metadata)
        except Exception as e:
            return EvaluatorResult(name=self.name, error=str(e))

    def _parse_results(self, results: Any) -> Dict[str, float]:
        """Normalize IVEBench output into a flat score dict."""
        scores: Dict[str, float] = {}
        if isinstance(results, dict):
            for key, val in results.items():
                norm_key = key.lower().replace(" ", "_").replace("-", "_")
                if isinstance(val, (int, float)):
                    scores[norm_key] = float(val)
        return scores
