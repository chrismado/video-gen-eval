"""
BatchProcessor: Process multiple videos through the UnifiedPipeline.

Supports batch evaluation of video files from a list or a directory,
with progress tracking and result aggregation.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.unified_pipeline import UnifiedPipeline, PipelineReport


class BatchProcessor:
    """Run the UnifiedPipeline on multiple videos."""

    def __init__(self, pipeline: Optional[UnifiedPipeline] = None):
        """
        Args:
            pipeline: A configured UnifiedPipeline instance.
                      If None, a default pipeline is created.
        """
        self.pipeline = pipeline or UnifiedPipeline()

    def process_batch(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[PipelineReport]:
        """Evaluate a list of video files.

        Args:
            video_paths: List of paths to video files.
            output_dir: Optional directory to write per-video JSON reports.

        Returns:
            List of PipelineReport objects, one per video.
        """
        reports: List[PipelineReport] = []

        for video_path in video_paths:
            try:
                report = self.pipeline.run(video_path)
                reports.append(report)

                if output_dir:
                    out_path = Path(output_dir) / (Path(video_path).stem + ".report.json")
                    self.pipeline.save_report(report, str(out_path))
            except Exception as e:
                # Create an error report so the batch continues
                error_report = PipelineReport(video_path=video_path)
                error_report.ewm_score = None
                reports.append(error_report)

        return reports

    def process_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> List[PipelineReport]:
        """Evaluate all video files in a directory.

        Args:
            directory: Path to directory containing video files.
            extensions: File extensions to include (default: common video formats).
            output_dir: Optional directory to write per-video JSON reports.

        Returns:
            List of PipelineReport objects.
        """
        if extensions is None:
            extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        dir_path = Path(directory)
        video_paths: List[str] = []
        for ext in extensions:
            video_paths.extend(str(p) for p in sorted(dir_path.glob(f"*{ext}")))

        return self.process_batch(video_paths, output_dir=output_dir)

    def save_summary(self, reports: List[PipelineReport], output_path: str) -> None:
        """Write a summary JSON containing all report results.

        Args:
            reports: List of PipelineReport objects.
            output_path: Path to the output summary JSON file.
        """
        summary = {
            "total_videos": len(reports),
            "reports": [r.to_dict() for r in reports],
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
