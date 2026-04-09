from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline.unified_pipeline import UnifiedPipeline


def write_demo_video(path: Path, frame_count: int = 10) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (96, 96),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to write demo video to {path}")

    for frame_idx in range(frame_count):
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        frame[12 + frame_idx : 36 + frame_idx, 22:46] = (30, 200, 180)
        writer.write(frame)

    writer.release()
    return path


def main() -> None:
    output_dir = Path("examples") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = write_demo_video(output_dir / "quick_start.avi")

    pipeline = UnifiedPipeline(
        enable_vbench=False,
        enable_ivebench=False,
        enable_tivibench=False,
        enable_physion=True,
    )
    report = pipeline.run(str(video_path))
    report_path = output_dir / "quick_start.report.json"
    pipeline.save_report(report, str(report_path))

    print("Video:", video_path)
    print("EWMScore:", f"{report.ewm_score:.2f}" if report.ewm_score is not None else "n/a")
    print("Physics judgment:", report.physics_judgment)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
