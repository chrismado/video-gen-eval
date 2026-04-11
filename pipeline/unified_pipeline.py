"""
UnifiedPipeline: Master evaluation orchestrator.

Runs local physics/EWM evaluation plus optional external benchmark adapters,
consolidating results into a single structured report.

Usage:
    python -m pipeline.unified_pipeline --video path/to/video.mp4
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluators.ewm_score import EWMScorer


@dataclass
class EvaluatorResult:
    """Result from a single evaluator."""

    name: str
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        d: Dict[str, Any] = {"name": self.name, "scores": self.scores}
        if self.metadata:
            d["metadata"] = self.metadata
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class PipelineReport:
    """Consolidated report from all evaluators."""

    video_path: str
    ewm_score: Optional[float] = None
    evaluator_results: List[EvaluatorResult] = field(default_factory=list)
    physics_judgment: Optional[Dict] = None
    raw_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "video_path": self.video_path,
            "ewm_score": self.ewm_score,
            "evaluators": [r.to_dict() for r in self.evaluator_results],
            "physics_judgment": self.physics_judgment,
            "raw_scores": self.raw_scores,
            "timestamp": self.timestamp,
        }


class UnifiedPipeline:
    """Orchestrates optional benchmark adapters, Physion-Eval, and EWMScore.

    VBench, IVEBench, and TiViBench are treated as adapter integration points
    because their official runners may not expose stable importable classes.
    """

    def __init__(
        self,
        enable_vbench: bool = True,
        enable_ivebench: bool = True,
        enable_tivibench: bool = True,
        enable_physion: bool = True,
        mllm_model: str = "qwen2.5-vl",
        mllm_endpoint: Optional[str] = None,
        vbench_report: Optional[str] = None,
        ivebench_report: Optional[str] = None,
        tivibench_report: Optional[str] = None,
    ):
        self.enable_vbench = enable_vbench
        self.enable_ivebench = enable_ivebench
        self.enable_tivibench = enable_tivibench
        self.enable_physion = enable_physion
        self.vbench_report = vbench_report
        self.ivebench_report = ivebench_report
        self.tivibench_report = tivibench_report
        self.ewm_scorer = EWMScorer()
        self.physics_judge: Any | None = None
        if enable_physion:
            from judge.physics_judge import PhysicsJudge

            self.physics_judge = PhysicsJudge(
                mllm_model=mllm_model,
                mllm_endpoint=mllm_endpoint,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, video_path: str) -> PipelineReport:
        """Run all enabled evaluators on a video and return consolidated report."""
        video_path = str(video_path)
        report = PipelineReport(
            video_path=video_path,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )

        # 1. VBench
        if self.enable_vbench:
            report.evaluator_results.append(self._run_vbench(video_path))

        # 2. IVEBench
        if self.enable_ivebench:
            report.evaluator_results.append(self._run_ivebench(video_path))

        # 3. TiViBench
        if self.enable_tivibench:
            report.evaluator_results.append(self._run_tivibench(video_path))

        # 4. Physion-Eval (physics judge)
        if self.enable_physion:
            assert self.physics_judge is not None
            try:
                judgment = self.physics_judge.judge(video_path)
            except Exception as exc:
                report.evaluator_results.append(
                    EvaluatorResult(
                        name="physion",
                        error=str(exc),
                        metadata={"error_type": type(exc).__name__},
                    )
                )
            else:
                report.physics_judgment = judgment.to_dict()
                report.evaluator_results.append(
                    EvaluatorResult(
                        name="physion",
                        scores={"physics_compliance": judgment.overall_physics_score},
                    )
                )

        # 5. Aggregate raw scores from all evaluators into a single dict
        for er in report.evaluator_results:
            for key, val in er.scores.items():
                report.raw_scores[key] = val

        # 6. Compute EWMScore from aggregated raw scores
        if report.raw_scores:
            try:
                report.ewm_score = self.ewm_scorer.compute(report.raw_scores)
            except ValueError:
                report.ewm_score = None

        return report

    def save_report(self, report: PipelineReport, output_path: str) -> None:
        """Write the full report as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Individual evaluator runners
    # ------------------------------------------------------------------

    def _run_vbench(self, video_path: str) -> EvaluatorResult:
        """Run VBench via a JSON report adapter when provided."""
        if self.vbench_report:
            from evaluators.vbench_evaluator import VBenchEvaluator

            evaluator = VBenchEvaluator(adapter=lambda _: self._load_report_json(self.vbench_report))
            result = evaluator.evaluate(video_path)
            result.metadata.update(
                {
                    "report_source": "json_file",
                    "report_path": self.vbench_report,
                }
            )
            return result

        _ = video_path
        return EvaluatorResult(
            name="vbench",
            error="external VBench adapter not configured; use --no-vbench or wire an official runner",
        )

    def _run_ivebench(self, video_path: str) -> EvaluatorResult:
        """Run IVEBench via a JSON report adapter when provided."""
        if self.ivebench_report:
            from evaluators.ivebench_evaluator import IVEBenchEvaluator

            evaluator = IVEBenchEvaluator(adapter=lambda _: self._load_report_json(self.ivebench_report))
            result = evaluator.evaluate(video_path)
            result.metadata.update(
                {
                    "report_source": "json_file",
                    "report_path": self.ivebench_report,
                }
            )
            return result

        _ = video_path
        return EvaluatorResult(
            name="ivebench",
            error="external IVEBench adapter not configured; use --no-ivebench or wire an official runner",
        )

    def _run_tivibench(self, video_path: str) -> EvaluatorResult:
        """Run TiViBench via a JSON report adapter when provided."""
        if self.tivibench_report:
            from evaluators.tivibench_evaluator import TiViBenchEvaluator

            evaluator = TiViBenchEvaluator(adapter=lambda _: self._load_report_json(self.tivibench_report))
            result = evaluator.evaluate(video_path)
            result.metadata.update(
                {
                    "report_source": "json_file",
                    "report_path": self.tivibench_report,
                }
            )
            return result

        _ = video_path
        return EvaluatorResult(
            name="tivibench",
            error="external TiViBench adapter not configured; use --no-tivibench or wire an official runner",
        )

    def _load_report_json(self, report_path: str) -> Any:
        """Load a benchmark report JSON produced by an external runner."""
        path = Path(report_path)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Physion-Judge evaluation pipeline")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <video>.report.json)")
    parser.add_argument("--no-vbench", action="store_true")
    parser.add_argument("--no-ivebench", action="store_true")
    parser.add_argument("--no-tivibench", action="store_true")
    parser.add_argument("--no-physion", action="store_true")
    parser.add_argument("--mllm-model", default="qwen2.5-vl")
    parser.add_argument("--mllm-endpoint", default=None)
    parser.add_argument("--vbench-report", default=None, help="Path to official VBench JSON output")
    parser.add_argument("--ivebench-report", default=None, help="Path to official IVEBench JSON output")
    parser.add_argument("--tivibench-report", default=None, help="Path to official TiViBench JSON output")
    args = parser.parse_args()

    pipeline = UnifiedPipeline(
        enable_vbench=not args.no_vbench,
        enable_ivebench=not args.no_ivebench,
        enable_tivibench=not args.no_tivibench,
        enable_physion=not args.no_physion,
        mllm_model=args.mllm_model,
        mllm_endpoint=args.mllm_endpoint,
        vbench_report=args.vbench_report,
        ivebench_report=args.ivebench_report,
        tivibench_report=args.tivibench_report,
    )

    report = pipeline.run(args.video)

    output_path = args.output or str(Path(args.video).with_suffix(".report.json"))
    pipeline.save_report(report, output_path)

    print(f"EWMScore: {report.ewm_score}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
