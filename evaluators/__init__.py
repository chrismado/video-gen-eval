"""Lazy exports for evaluator wrappers and scoring helpers."""
from importlib import import_module


__all__ = [
    "EWMScorer",
    "MetricBounds",
    "METRIC_BOUNDS",
    "VBenchEvaluator",
    "IVEBenchEvaluator",
    "TiViBenchEvaluator",
    "PhysionEvaluator",
]

_EXPORTS = {
    "EWMScorer": ("evaluators.ewm_score", "EWMScorer"),
    "MetricBounds": ("evaluators.ewm_score", "MetricBounds"),
    "METRIC_BOUNDS": ("evaluators.ewm_score", "METRIC_BOUNDS"),
    "VBenchEvaluator": ("evaluators.vbench_evaluator", "VBenchEvaluator"),
    "IVEBenchEvaluator": ("evaluators.ivebench_evaluator", "IVEBenchEvaluator"),
    "TiViBenchEvaluator": ("evaluators.tivibench_evaluator", "TiViBenchEvaluator"),
    "PhysionEvaluator": ("evaluators.physion_evaluator", "PhysionEvaluator"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)

