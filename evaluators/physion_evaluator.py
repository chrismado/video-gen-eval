"""
PhysionEvaluator: Wrapper around Physion-Eval physical plausibility
assessment, integrated with the PhysicsJudge.

Provides a unified evaluate() interface returning EvaluatorResult,
delegating to the PhysicsJudge for violation detection and scoring.
"""
from typing import Dict, Optional

from judge.physics_judge import PhysicsJudge, JudgmentResult
from pipeline.unified_pipeline import EvaluatorResult


class PhysionEvaluator:
    """Evaluate physical plausibility of a generated video.

    Uses PhysicsJudge for heuristic + optional MLLM-based violation
    detection and returns results in the standard EvaluatorResult format.
    """

    name: str = "physion"

    def __init__(
        self,
        mllm_model: str = "qwen2.5-vl",
        mllm_endpoint: Optional[str] = None,
        sample_fps: int = 8,
    ):
        self.judge = PhysicsJudge(
            mllm_model=mllm_model,
            mllm_endpoint=mllm_endpoint,
            sample_fps=sample_fps,
        )

    def evaluate(self, video_path: str) -> EvaluatorResult:
        """Run Physion-Eval style analysis on a single video.

        Returns:
            EvaluatorResult with ``physics_compliance`` score and
            violation details in ``metadata``.
        """
        video_path = str(video_path)
        try:
            judgment: JudgmentResult = self.judge.judge(video_path)
            scores: Dict[str, float] = {
                "physics_compliance": judgment.overall_physics_score,
            }
            metadata = judgment.to_dict()
            return EvaluatorResult(name=self.name, scores=scores, metadata=metadata)
        except Exception as e:
            return EvaluatorResult(name=self.name, error=str(e))
