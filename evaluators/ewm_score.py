"""
EWMScore: Embodied World Model Score
Based on WorldArena framework, CVPR 2026.

Closed-loop evaluation that wraps a generative model in an RL action API
and measures embodied task success — not just visual quality.

EWMScore = (1/N) * Sigma Normalize(m_i) * 100
Where N=16 and m_i = raw score of i-th perceptual/physical metric,
linearly normalized against empirical upper/lower bounds.

The Perception-Functionality Gap:
  Models with high VBench scores frequently fail closed-loop physical tasks.
  EWMScore reveals this gap that open-loop metrics cannot detect.
"""
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class MetricBounds:
    """Empirically defined normalization bounds per metric."""
    lower: float
    upper: float


# All 16 metric bounds derived from empirical VBench/WorldArena distributions.
# Lower bounds represent worst observed model performance; upper bounds represent
# best observed performance across the evaluation corpus.
METRIC_BOUNDS: Dict[str, MetricBounds] = {
    "motion_smoothness":     MetricBounds(0.706, 0.997),
    "temporal_flickering":   MetricBounds(0.800, 0.998),
    "background_stability":  MetricBounds(0.850, 0.995),
    "subject_consistency":   MetricBounds(0.710, 0.980),
    "aesthetic_quality":     MetricBounds(0.300, 0.750),
    "imaging_quality":       MetricBounds(0.400, 0.850),
    "object_class":          MetricBounds(0.200, 0.950),
    "multiple_objects":      MetricBounds(0.100, 0.850),
    "human_action":          MetricBounds(0.600, 0.990),
    "color":                 MetricBounds(0.500, 0.950),
    "spatial_relationship":  MetricBounds(0.200, 0.800),
    "scene":                 MetricBounds(0.300, 0.900),
    "temporal_style":        MetricBounds(0.150, 0.850),
    "overall_consistency":   MetricBounds(0.150, 0.800),
    "dynamic_degree":        MetricBounds(0.000, 1.000),
    "physics_compliance":    MetricBounds(0.000, 0.850),
}

N_METRICS = 16


class EWMScorer:
    """Compute the Embodied World Model Score from 16 raw metric values."""

    def __init__(self, metric_bounds: Optional[Dict[str, MetricBounds]] = None):
        self.bounds = metric_bounds if metric_bounds is not None else METRIC_BOUNDS
        self.raw_scores: Dict[str, float] = {}

    def normalize(self, score: float, metric_name: str) -> float:
        """Linearly normalize a raw score to [0, 1] using empirical bounds.

        Scores below the lower bound clamp to 0; scores above the upper bound
        clamp to 1.
        """
        if metric_name not in self.bounds:
            raise KeyError(f"Unknown metric: {metric_name}")
        bounds = self.bounds[metric_name]
        if bounds.upper == bounds.lower:
            return 1.0 if score >= bounds.upper else 0.0
        normalized = (score - bounds.lower) / (bounds.upper - bounds.lower)
        return float(np.clip(normalized, 0.0, 1.0))

    def compute(self, raw_scores: Dict[str, float]) -> float:
        """Compute EWMScore from up to 16 raw metric scores.

        EWMScore = (1/N) * sum(Normalize(m_i)) * 100

        Only metrics present in both `raw_scores` and `self.bounds` are included.
        Raises ValueError if no valid metrics are provided.
        """
        self.raw_scores = dict(raw_scores)
        valid_keys = [k for k in raw_scores if k in self.bounds]
        if not valid_keys:
            raise ValueError("No valid metrics found in raw_scores")

        normalized = [self.normalize(raw_scores[k], k) for k in valid_keys]
        n = len(valid_keys)
        return float(np.mean(normalized) * 100)

    def compute_detailed(self, raw_scores: Dict[str, float]) -> Dict:
        """Compute EWMScore and return per-metric breakdown."""
        self.raw_scores = dict(raw_scores)
        valid_keys = [k for k in raw_scores if k in self.bounds]
        if not valid_keys:
            raise ValueError("No valid metrics found in raw_scores")

        per_metric = {}
        for k in valid_keys:
            per_metric[k] = {
                "raw": raw_scores[k],
                "normalized": self.normalize(raw_scores[k], k),
                "bounds": {"lower": self.bounds[k].lower, "upper": self.bounds[k].upper},
            }

        score = float(np.mean([m["normalized"] for m in per_metric.values()]) * 100)
        return {
            "ewm_score": score,
            "n_metrics": len(valid_keys),
            "per_metric": per_metric,
        }

    @classmethod
    def from_vbench_output(cls, vbench_json: Union[str, Path, dict]) -> "EWMScorer":
        """Parse raw VBench JSON output into an EWMScorer with populated raw_scores.

        VBench JSON structure (from Vchitect/VBench):
        [
          {
            "video_list": [...],
            "dimension": "motion_smoothness",
            "score": 0.92,
            ...
          },
          ...
        ]

        Or a dict keyed by dimension name:
        { "motion_smoothness": 0.92, "temporal_flickering": 0.95, ... }
        """
        if isinstance(vbench_json, (str, Path)):
            path = Path(vbench_json)
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = vbench_json

        raw_scores: Dict[str, float] = {}

        if isinstance(data, list):
            # VBench outputs a list of per-dimension result dicts
            for entry in data:
                dim = entry.get("dimension", "")
                score = entry.get("score")
                if dim and score is not None:
                    key = dim.lower().replace(" ", "_").replace("-", "_")
                    raw_scores[key] = float(score)
        elif isinstance(data, dict):
            # Accept a flat dict mapping dimension -> score, or a nested structure
            for key, value in data.items():
                norm_key = key.lower().replace(" ", "_").replace("-", "_")
                if isinstance(value, (int, float)):
                    raw_scores[norm_key] = float(value)
                elif isinstance(value, dict) and "score" in value:
                    raw_scores[norm_key] = float(value["score"])
                elif isinstance(value, list) and len(value) > 0:
                    # Some VBench outputs nest scores in a list
                    raw_scores[norm_key] = float(value[0]) if isinstance(value[0], (int, float)) else 0.0

        scorer = cls()
        scorer.raw_scores = raw_scores
        return scorer
