"""
IVEBenchEvaluator: Wrapper around the IVEBench instruction-following
video evaluation benchmark.

Measures how well a generated video follows the text instruction that
prompted it (compositional accuracy, attribute binding, action fidelity).
"""

from typing import Any, Dict, Optional

from pipeline.unified_pipeline import EvaluatorResult


class IVEBenchEvaluator:
    """Evaluate instruction-compliance of a generated video via IVEBench.

    Wraps the IVEBench library with graceful fallback when it is
    not installed.
    """

    name: str = "ivebench"

    def __init__(self, instruction: Optional[str] = None):
        """
        Args:
            instruction: The text prompt / instruction the video was
                         generated from.  Required by IVEBench for
                         compliance scoring.
        """
        self.instruction = instruction

    def evaluate(self, video_path: str) -> EvaluatorResult:
        """Run IVEBench evaluation on a single video.

        Returns:
            EvaluatorResult with instruction-compliance scores.
        """
        video_path = str(video_path)
        try:
            from ivebench import IVEBench

            bench = IVEBench()
            results = bench.evaluate(video_path)
            scores = self._parse_results(results)
            metadata: Dict[str, Any] = {}
            if self.instruction:
                metadata["instruction"] = self.instruction
            return EvaluatorResult(name=self.name, scores=scores, metadata=metadata)
        except ImportError:
            return EvaluatorResult(name=self.name, error="ivebench not installed")
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
