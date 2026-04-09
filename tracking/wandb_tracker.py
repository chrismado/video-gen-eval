"""
WandbTracker: Log evaluation results to Weights & Biases for experiment tracking.

Gracefully handles missing wandb installation.
"""
from typing import Any, Dict, Optional

from pipeline.unified_pipeline import PipelineReport


class WandbTracker:
    """Log PipelineReport results to Weights & Biases.

    Wraps the wandb API with try/except so the tracker can be
    instantiated even when wandb is not installed.
    """

    def __init__(
        self,
        project: str = "video-gen-eval",
        entity: Optional[str] = None,
    ):
        self.project = project
        self.entity = entity
        self._wandb = None
        self._run = None

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        """Return True if wandb is importable."""
        return self._wandb is not None

    def init_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Initialize a wandb run.

        Args:
            run_name: Optional display name for the run.
            config: Optional config dict.

        Returns:
            True if the run was initialized successfully.
        """
        if not self.available:
            return False

        wandb = self._wandb
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config or {},
        )
        return True

    def log_report(
        self,
        report: PipelineReport,
        run_name: Optional[str] = None,
    ) -> bool:
        """Log a PipelineReport to wandb.

        If no run is active, one is initialized automatically.

        Args:
            report: The evaluation report to log.
            run_name: Optional run name (used if creating a new run).

        Returns:
            True if logging succeeded, False otherwise.
        """
        if not self.available:
            return False

        wandb = self._wandb

        if self._run is None:
            self.init_run(run_name=run_name, config={"video_path": report.video_path})

        log_data: Dict[str, Any] = {}

        if report.ewm_score is not None:
            log_data["ewm_score"] = report.ewm_score

        for key, val in report.raw_scores.items():
            log_data[key] = val

        if report.physics_judgment:
            log_data["violation_count"] = report.physics_judgment.get("violation_count", 0)

        wandb.log(log_data)
        return True

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log arbitrary metrics to the active wandb run.

        Args:
            metrics: Dict of metric name -> value.
        """
        if not self.available or self._run is None:
            return

        self._wandb.log(metrics)

    def finish(self) -> None:
        """Finish the active wandb run."""
        if self._run is not None:
            self._run.finish()
            self._run = None
