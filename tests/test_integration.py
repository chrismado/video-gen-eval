from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from judge.physics_judge import JudgmentResult, PhysicsJudge
from judge.rationale_generator import RationaleGenerator
from pipeline.batch_processor import BatchProcessor
from pipeline.score_aggregator import ScoreAggregator
from pipeline.unified_pipeline import PipelineReport, UnifiedPipeline

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.integration


def _write_synthetic_video(path: Path, seed: int, frame_count: int = 12) -> Path:
    rng = np.random.default_rng(seed)
    height, width = 96, 96
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create synthetic video at {path}")

    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        offset = (frame_idx * 5 + seed * 7) % 48
        frame[16 + offset : 40 + offset, 18:42] = (40 + seed * 10, 180, 220)
        noise = rng.integers(0, 15, size=frame.shape, dtype=np.uint8)
        writer.write(cv2.add(frame, noise))

    writer.release()
    return path


def _build_pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        enable_vbench=False,
        enable_ivebench=False,
        enable_tivibench=False,
        enable_physion=True,
    )


def test_pipeline_report_contains_expected_fields(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "sample.avi", seed=1)

    pipeline = _build_pipeline()
    report = pipeline.run(str(video_path))

    assert isinstance(report, PipelineReport)
    assert report.video_path == str(video_path)
    assert report.physics_judgment is not None
    assert "overall_physics_score" in report.physics_judgment
    assert "violation_count" in report.physics_judgment
    assert report.ewm_score is not None
    assert "physics_compliance" in report.raw_scores

    aggregator = ScoreAggregator()
    aggregator.add_report(report)
    summary = aggregator.aggregate()
    assert summary["n_reports"] == 1
    assert "physics_compliance" in summary["per_metric"]


def test_batch_processor_handles_directory_of_videos(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    reports_dir = tmp_path / "reports"
    videos_dir.mkdir()

    for idx in range(3):
        _write_synthetic_video(videos_dir / f"video_{idx}.avi", seed=idx)

    processor = BatchProcessor(pipeline=_build_pipeline())
    reports = processor.process_directory(str(videos_dir), extensions=[".avi"], output_dir=str(reports_dir))

    assert len(reports) == 3
    assert all(report.physics_judgment is not None for report in reports)
    assert len(list(reports_dir.glob("*.report.json"))) == 3


def test_rationale_generator_uses_real_judgment(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "rationale.avi", seed=9)
    judgment = PhysicsJudge(sample_fps=4).judge(str(video_path))

    assert isinstance(judgment, JudgmentResult)
    text = RationaleGenerator(verbose=True).generate(judgment)

    assert text.strip()
    assert "Overall physics score" in text
    assert Path(judgment.video_path).name in text
