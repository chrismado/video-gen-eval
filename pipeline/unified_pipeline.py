"""
UnifiedPipeline: Master evaluation orchestrator.

Runs all four evaluators (VBench, IVEBench, TiViBench, Physion-Eval)
plus the EWMScore closed-loop assessment, consolidating results into
a single structured report.

Usage:
    python -m pipeline.unified_pipeline --video path/to/video.mp4 --all-dimensions
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from evaluators.ewm_score import EWMScorer, METRIC_BOUNDS
from judge.physics_judge import PhysicsJudge, JudgmentResult


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
    """Orchestrates VBench, IVEBench, TiViBench, Physion-Eval, and EWMScore.

    Each evaluator is optional. If the corresponding library is not installed
    or the evaluator is not enabled, it is skipped gracefully.
    """

    def __init__(
        self,
        enable_vbench: bool = True,
        enable_ivebench: bool = True,
        enable_tivibench: bool = True,
        enable_physion: bool = True,
        mllm_model: str = "qwen2.5-vl",
        mllm_endpoint: Optional[str] = None,
    ):
        self.enable_vbench = enable_vbench
        self.enable_ivebench = enable_ivebench
        self.enable_tivibench = enable_tivibench
        self.enable_physion = enable_physion
        self.ewm_scorer = EWMScorer()
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
            judgment = self.physics_judge.judge(video_path)
            report.physics_judgment = judgment.to_dict()
            report.evaluator_results.append(EvaluatorResult(
                name="physion",
                scores={"physics_compliance": judgment.overall_physics_score},
            ))

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
        """Run VBench 16-dimension assessment."""
        try:
            from vbench import VBench
            bench = VBench()
            results = bench.evaluate(video_path)
            scores = {}
            if isinstance(results, list):
                for entry in results:
                    dim = entry.get("dimension", "")
                    key = dim.lower().replace(" ", "_").replace("-", "_")
                    scores[key] = float(entry.get("score", 0.0))
            elif isinstance(results, dict):
                for key, val in results.items():
                    norm_key = key.lower().replace(" ", "_").replace("-", "_")
                    scores[norm_key] = float(val) if isinstance(val, (int, float)) else 0.0
            return EvaluatorResult(name="vbench", scores=scores)
        except ImportError:
            return EvaluatorResult(name="vbench", error="vbench not installed")
        except Exception as e:
            return EvaluatorResult(name="vbench", error=str(e))

    def _run_ivebench(self, video_path: str) -> EvaluatorResult:
        """Run IVEBench instruction compliance evaluation."""
        try:
            from ivebench import IVEBench
            bench = IVEBench()
            results = bench.evaluate(video_path)
            scores = {}
            if isinstance(results, dict):
                for key, val in results.items():
                    norm_key = key.lower().replace(" ", "_").replace("-", "_")
                    scores[norm_key] = float(val) if isinstance(val, (int, float)) else 0.0
            return EvaluatorResult(name="ivebench", scores=scores)
        except ImportError:
            return EvaluatorResult(name="ivebench", error="ivebench not installed")
        except Exception as e:
            return EvaluatorResult(name="ivebench", error=str(e))

    def _run_tivibench(self, video_path: str) -> EvaluatorResult:
        """Run TiViBench causal reasoning evaluation."""
        try:
            from tivibench import TiViBench
            bench = TiViBench()
            results = bench.evaluate(video_path)
            scores = {}
            if isinstance(results, dict):
                for key, val in results.items():
                    norm_key = key.lower().replace(" ", "_").replace("-", "_")
                    scores[norm_key] = float(val) if isinstance(val, (int, float)) else 0.0
            return EvaluatorResult(name="tivibench", scores=scores)
        except ImportError:
            return EvaluatorResult(name="tivibench", error="tivibench not installed")
        except Exception as e:
            return EvaluatorResult(name="tivibench", error=str(e))


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Physion-Judge evaluation pipeline")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <video>.report.json)")
    parser.add_argument("--all-dimensions", action="store_true", help="Enable all evaluators")
    parser.add_argument("--no-vbench", action="store_true")
    parser.add_argument("--no-ivebench", action="store_true")
    parser.add_argument("--no-tivibench", action="store_true")
    parser.add_argument("--no-physion", action="store_true")
    parser.add_argument("--mllm-model", default="qwen2.5-vl")
    parser.add_argument("--mllm-endpoint", default=None)
    args = parser.parse_args()

    pipeline = UnifiedPipeline(
        enable_vbench=not args.no_vbench,
        enable_ivebench=not args.no_ivebench,
        enable_tivibench=not args.no_tivibench,
        enable_physion=not args.no_physion,
        mllm_model=args.mllm_model,
        mllm_endpoint=args.mllm_endpoint,
    )

    report = pipeline.run(args.video)

    output_path = args.output or str(Path(args.video).with_suffix(".report.json"))
    pipeline.save_report(report, output_path)

    print(f"EWMScore: {report.ewm_score}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
