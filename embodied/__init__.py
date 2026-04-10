"""Lazy exports for embodied evaluation helpers."""

from importlib import import_module

__all__ = [
    "WorldModelEnv",
    "WorldModelWrapper",
    "TaskEvaluator",
]

_EXPORTS = {
    "WorldModelEnv": ("embodied.rl_action_api", "WorldModelEnv"),
    "WorldModelWrapper": ("embodied.world_wrapper", "WorldModelWrapper"),
    "TaskEvaluator": ("embodied.task_evaluator", "TaskEvaluator"),
}


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
