"""
ScoreAggregator: Combine and summarize scores from multiple
PipelineReports for cross-model comparison.
"""

from typing import Any, Dict, List

import numpy as np

from pipeline.unified_pipeline import PipelineReport


class ScoreAggregator:
    """Aggregate evaluation scores across multiple pipeline reports.

    Provides summary statistics (mean, std, min, max) for each metric
    across a corpus of evaluated videos or models.
    """

    def __init__(self) -> None:
        self._reports: List[PipelineReport] = []

    def add_report(self, report: PipelineReport) -> None:
        """Add a PipelineReport to the aggregation pool."""
        self._reports.append(report)

    def add_reports(self, reports: List[PipelineReport]) -> None:
        """Add multiple PipelineReports."""
        self._reports.extend(reports)

    def clear(self) -> None:
        """Remove all stored reports."""
        self._reports = []

    def aggregate(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all stored reports.

        Returns:
            Dict with per-metric statistics and an overall EWMScore summary.
        """
        if not self._reports:
            return {"n_reports": 0, "ewm_scores": {}, "per_metric": {}}

        ewm_scores = [r.ewm_score for r in self._reports if r.ewm_score is not None]

        # Collect all raw score keys across reports
        all_keys: set = set()
        for r in self._reports:
            all_keys.update(r.raw_scores.keys())

        per_metric: Dict[str, Dict[str, float]] = {}
        for key in sorted(all_keys):
            values = [r.raw_scores[key] for r in self._reports if key in r.raw_scores]
            if values:
                arr = np.array(values)
                per_metric[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "count": len(values),
                }

        ewm_summary: Dict[str, Any] = {}
        if ewm_scores:
            arr = np.array(ewm_scores)
            ewm_summary = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(ewm_scores),
            }

        return {
            "n_reports": len(self._reports),
            "ewm_scores": ewm_summary,
            "per_metric": per_metric,
        }
