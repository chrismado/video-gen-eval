from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline.batch_processor import BatchProcessor
from pipeline.score_aggregator import ScoreAggregator
from pipeline.unified_pipeline import UnifiedPipeline


def write_video(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (96, 96),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to write demo video to {path}")

    for frame_idx in range(12):
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        offset = (seed * 4 + frame_idx * 3) % 48
        frame[20:44, 12 + offset : 36 + offset] = (90, 90 + seed * 30, 220)
        frame = cv2.add(frame, rng.integers(0, 10, size=frame.shape, dtype=np.uint8))
        writer.write(frame)

    writer.release()
    return path


def main() -> None:
    output_dir = Path("examples") / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = [write_video(output_dir / f"model_{idx}.avi", idx) for idx in range(3)]

    pipeline = UnifiedPipeline(
        enable_vbench=False,
        enable_ivebench=False,
        enable_tivibench=False,
        enable_physion=True,
    )
    processor = BatchProcessor(pipeline=pipeline)
    reports = processor.process_batch([str(path) for path in video_paths], output_dir=str(output_dir))

    aggregator = ScoreAggregator()
    aggregator.add_reports(reports)
    summary = aggregator.aggregate()

    print("| Model | EWMScore | Physics Score | Violations |")
    print("|-------|----------|---------------|------------|")
    for report in reports:
        violations = report.physics_judgment["violation_count"] if report.physics_judgment else 0
        physics = report.raw_scores.get("physics_compliance", 0.0)
        ewm = report.ewm_score if report.ewm_score is not None else 0.0
        print(f"| {Path(report.video_path).stem} | {ewm:>8.2f} | {physics:>13.4f} | {violations:>10} |")

    print("\nAggregate summary:")
    print(summary)


if __name__ == "__main__":
    main()
