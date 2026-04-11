"""
Unit tests for the video-gen-eval evaluation framework.

These tests exercise core logic without requiring a GPU, video files,
or optional dependencies (vbench, ivebench, tivibench).
"""

import json
import unittest
from pathlib import Path

from evaluators.ewm_score import METRIC_BOUNDS, N_METRICS, EWMScorer
from judge.physics_judge import VIOLATION_TYPES, JudgmentResult, Violation
from judge.rationale_generator import RationaleGenerator
from pipeline.score_aggregator import ScoreAggregator
from pipeline.unified_pipeline import EvaluatorResult, PipelineReport, UnifiedPipeline


class TestEWMScorer(unittest.TestCase):
    """Tests for the EWMScorer normalization and scoring logic."""

    def setUp(self):
        self.scorer = EWMScorer()

    def test_metric_bounds_count(self):
        self.assertEqual(len(METRIC_BOUNDS), N_METRICS)

    def test_normalize_within_bounds(self):
        score = self.scorer.normalize(0.85, "motion_smoothness")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_normalize_below_lower_clamps_to_zero(self):
        score = self.scorer.normalize(0.0, "motion_smoothness")
        self.assertEqual(score, 0.0)

    def test_normalize_above_upper_clamps_to_one(self):
        score = self.scorer.normalize(1.0, "motion_smoothness")
        self.assertEqual(score, 1.0)

    def test_normalize_unknown_metric_raises(self):
        with self.assertRaises(KeyError):
            self.scorer.normalize(0.5, "nonexistent_metric")

    def test_compute_all_metrics(self):
        raw = {k: (b.lower + b.upper) / 2 for k, b in METRIC_BOUNDS.items()}
        score = self.scorer.compute(raw)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_compute_no_valid_metrics_raises(self):
        with self.assertRaises(ValueError):
            self.scorer.compute({"bogus_metric": 0.5})

    def test_compute_detailed_returns_breakdown(self):
        raw = {"motion_smoothness": 0.85, "aesthetic_quality": 0.5}
        result = self.scorer.compute_detailed(raw)
        self.assertIn("ewm_score", result)
        self.assertIn("per_metric", result)
        self.assertEqual(result["n_metrics"], 2)

    def test_from_vbench_output_dict(self):
        data = {"motion_smoothness": 0.9, "aesthetic_quality": 0.6}
        scorer = EWMScorer.from_vbench_output(data)
        self.assertEqual(len(scorer.raw_scores), 2)

    def test_from_vbench_output_list(self):
        data = [
            {"dimension": "motion_smoothness", "score": 0.9},
            {"dimension": "aesthetic_quality", "score": 0.6},
        ]
        scorer = EWMScorer.from_vbench_output(data)
        self.assertEqual(len(scorer.raw_scores), 2)

    def test_perfect_scores_give_100(self):
        raw = {k: b.upper for k, b in METRIC_BOUNDS.items()}
        score = self.scorer.compute(raw)
        self.assertAlmostEqual(score, 100.0, places=2)

    def test_worst_scores_give_0(self):
        raw = {k: b.lower for k, b in METRIC_BOUNDS.items()}
        score = self.scorer.compute(raw)
        self.assertAlmostEqual(score, 0.0, places=2)


class TestViolation(unittest.TestCase):
    """Tests for the Violation and JudgmentResult data classes."""

    def test_violation_to_dict(self):
        v = Violation(
            violation_type="gravity_violation",
            frame_range=(10, 15),
            severity=0.42,
            rationale="Object floats upward without force.",
        )
        d = v.to_dict()
        self.assertEqual(d["violation_type"], "gravity_violation")
        self.assertEqual(d["frame_range"], [10, 15])
        self.assertAlmostEqual(d["severity"], 0.42, places=4)

    def test_judgment_result_to_dict(self):
        v = Violation("gravity_violation", (10, 15), 0.5, "test")
        jr = JudgmentResult(
            video_path="test.mp4",
            violations=[v],
            overall_physics_score=0.85,
            frame_count=100,
        )
        d = jr.to_dict()
        self.assertEqual(d["violation_count"], 1)
        self.assertEqual(d["frame_count"], 100)

    def test_violation_types_not_empty(self):
        self.assertGreater(len(VIOLATION_TYPES), 0)


class TestRationaleGenerator(unittest.TestCase):
    """Tests for the RationaleGenerator."""

    def test_generate_no_violations(self):
        jr = JudgmentResult(video_path="test.mp4", frame_count=50)
        gen = RationaleGenerator()
        text = gen.generate(jr)
        self.assertIn("No physical violations detected", text)

    def test_generate_with_violations(self):
        v = Violation("gravity_violation", (5, 8), 0.6, "Floats up.")
        jr = JudgmentResult(video_path="test.mp4", violations=[v], frame_count=50)
        gen = RationaleGenerator(verbose=True)
        text = gen.generate(jr)
        self.assertIn("gravity_violation", text)
        self.assertIn("Severity", text)

    def test_violation_summary(self):
        v = Violation("object_permanence", (1, 2), 0.3, "Vanished.")
        gen = RationaleGenerator()
        summary = gen.generate_violation_summary(v)
        self.assertIn("object_permanence", summary)
        self.assertIn("Vanished", summary)


class TestEvaluatorResult(unittest.TestCase):
    """Tests for the EvaluatorResult dataclass."""

    def test_to_dict_basic(self):
        er = EvaluatorResult(name="test", scores={"metric_a": 0.8})
        d = er.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["scores"]["metric_a"], 0.8)

    def test_to_dict_with_error(self):
        er = EvaluatorResult(name="test", error="not installed")
        d = er.to_dict()
        self.assertIn("error", d)


class TestPipelineReport(unittest.TestCase):
    """Tests for the PipelineReport dataclass."""

    def test_to_dict(self):
        report = PipelineReport(
            video_path="test.mp4",
            ewm_score=72.5,
            raw_scores={"motion_smoothness": 0.9},
        )
        d = report.to_dict()
        self.assertEqual(d["video_path"], "test.mp4")
        self.assertEqual(d["ewm_score"], 72.5)


class TestUnifiedPipeline(unittest.TestCase):
    """Tests for the unified pipeline orchestration layer."""

    def test_physion_errors_are_reported_without_crashing(self):
        class BrokenJudge:
            def judge(self, video_path: str):
                raise RuntimeError(f"decode failed for {video_path}")

        pipeline = UnifiedPipeline(
            enable_vbench=False,
            enable_ivebench=False,
            enable_tivibench=False,
            enable_physion=True,
        )
        pipeline.physics_judge = BrokenJudge()

        report = pipeline.run("broken.mp4")

        self.assertIsNone(report.physics_judgment)
        self.assertEqual(len(report.evaluator_results), 1)
        self.assertEqual(report.evaluator_results[0].name, "physion")
        self.assertIn("decode failed for broken.mp4", report.evaluator_results[0].error)
        self.assertEqual(report.raw_scores, {})
        self.assertIsNone(report.ewm_score)


class TestScoreAggregator(unittest.TestCase):
    """Tests for the ScoreAggregator."""

    def test_empty_aggregation(self):
        agg = ScoreAggregator()
        result = agg.aggregate()
        self.assertEqual(result["n_reports"], 0)

    def test_single_report(self):
        agg = ScoreAggregator()
        report = PipelineReport(
            video_path="test.mp4",
            ewm_score=50.0,
            raw_scores={"motion_smoothness": 0.85},
        )
        agg.add_report(report)
        result = agg.aggregate()
        self.assertEqual(result["n_reports"], 1)
        self.assertAlmostEqual(result["ewm_scores"]["mean"], 50.0)

    def test_multiple_reports(self):
        agg = ScoreAggregator()
        for ewm in [40.0, 60.0, 80.0]:
            report = PipelineReport(
                video_path="test.mp4",
                ewm_score=ewm,
                raw_scores={"motion_smoothness": ewm / 100.0},
            )
            agg.add_report(report)
        result = agg.aggregate()
        self.assertEqual(result["n_reports"], 3)
        self.assertAlmostEqual(result["ewm_scores"]["mean"], 60.0)

    def test_clear(self):
        agg = ScoreAggregator()
        agg.add_report(PipelineReport(video_path="t.mp4", ewm_score=50.0))
        agg.clear()
        self.assertEqual(agg.aggregate()["n_reports"], 0)


class TestExampleResultsJson(unittest.TestCase):
    """Verify the example results JSON is valid."""

    def test_example_results_loads(self):
        path = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "example_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.assertIn("ewm_score", data)
            self.assertIn("evaluators", data)
            self.assertIsInstance(data["evaluators"], list)


if __name__ == "__main__":
    unittest.main()
