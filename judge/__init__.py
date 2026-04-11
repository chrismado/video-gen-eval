from judge.physics_judge import VIOLATION_TYPES, JudgmentResult, PhysicsJudge, Violation
from judge.rationale_generator import RationaleGenerator

__all__ = [
    "PhysicsJudge",
    "JudgmentResult",
    "Violation",
    "VIOLATION_TYPES",
    "RationaleGenerator",
    "AnomalyDetector",
]


def __getattr__(name: str) -> object:
    if name == "AnomalyDetector":
        from judge.anomaly_detector import AnomalyDetector

        return AnomalyDetector
    raise AttributeError(f"module 'judge' has no attribute {name!r}")
