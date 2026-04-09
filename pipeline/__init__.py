"""Lazy exports for the pipeline package.

Avoid eager imports here so ``python -m pipeline.unified_pipeline`` can execute
without partially initializing the module graph first.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "UnifiedPipeline",
    "EvaluatorResult",
    "PipelineReport",
    "BatchProcessor",
    "ScoreAggregator",
]

_EXPORTS = {
    "UnifiedPipeline": ("pipeline.unified_pipeline", "UnifiedPipeline"),
    "EvaluatorResult": ("pipeline.unified_pipeline", "EvaluatorResult"),
    "PipelineReport": ("pipeline.unified_pipeline", "PipelineReport"),
    "BatchProcessor": ("pipeline.batch_processor", "BatchProcessor"),
    "ScoreAggregator": ("pipeline.score_aggregator", "ScoreAggregator"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
