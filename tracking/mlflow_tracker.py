"""
MLflowTracker: Log evaluation results to MLflow for experiment tracking.

Gracefully handles missing mlflow installation.
"""
from typing import Any, Dict, Optional

from pipeline.unified_pipeline import PipelineReport


class MLflowTracker:
    """Log PipelineReport results to MLflow.

    Wraps the mlflow API with try/except so the tracker can be
    instantiated even when mlflow is not installed.
    """

    def __init__(
        self,
        experiment_name: str = "video-gen-eval",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow = None

        try:
            import mlflow
            self._mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        """Return True if mlflow is importable and configured."""
        return self._mlflow is not None

    def log_report(
        self,
        report: PipelineReport,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Log a PipelineReport as an MLflow run.

        Args:
            report: The evaluation report to log.
            run_name: Optional name for the MLflow run.
            tags: Optional tags dict.

        Returns:
            The MLflow run ID, or None if mlflow is not available.
        """
        if not self.available:
            return None

        mlflow = self._mlflow
        with mlflow.start_run(run_name=run_name) as run:
            # Log EWMScore
            if report.ewm_score is not None:
                mlflow.log_metric("ewm_score", report.ewm_score)

            # Log all raw scores
            for key, val in report.raw_scores.items():
                mlflow.log_metric(key, val)

            # Log violation count if available
            if report.physics_judgment:
                mlflow.log_metric(
                    "violation_count",
                    report.physics_judgment.get("violation_count", 0),
                )

            # Log parameters
            mlflow.log_param("video_path", report.video_path)
            mlflow.log_param("timestamp", report.timestamp)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            return run.info.run_id

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log arbitrary metrics to the active MLflow run.

        Args:
            metrics: Dict of metric name -> value.
            step: Optional step number.
        """
        if not self.available:
            return

        mlflow = self._mlflow
        for key, val in metrics.items():
            mlflow.log_metric(key, val, step=step)
