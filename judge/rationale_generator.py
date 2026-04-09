"""
RationaleGenerator: Produce human-readable rationales from JudgmentResults.

Takes the structured violation data from PhysicsJudge and generates
formatted text explanations suitable for reports or logging.
"""

from typing import List

from judge.physics_judge import JudgmentResult, Violation


class RationaleGenerator:
    """Generate natural language rationale summaries from physics judgments."""

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: If True, include per-violation frame ranges and
                     severity details in the output.
        """
        self.verbose = verbose

    def generate(self, judgment: JudgmentResult) -> str:
        """Generate a full rationale summary for a judgment.

        Args:
            judgment: A JudgmentResult from PhysicsJudge.

        Returns:
            A multi-line string summarizing all violations and the
            overall physics compliance score.
        """
        lines: List[str] = []
        lines.append(f"Physics Judgment for: {judgment.video_path}")
        lines.append(f"Overall physics score: {judgment.overall_physics_score:.4f}")
        lines.append(f"Total frames analyzed: {judgment.frame_count}")
        lines.append(f"Violations detected: {len(judgment.violations)}")
        lines.append("")

        if not judgment.violations:
            lines.append("No physical violations detected.")
            return "\n".join(lines)

        for i, violation in enumerate(judgment.violations, 1):
            lines.append(f"Violation {i}: {violation.violation_type}")
            if self.verbose:
                lines.append(f"  Frame range: {violation.frame_range[0]}-{violation.frame_range[1]}")
                lines.append(f"  Severity: {violation.severity:.4f}")
            lines.append(f"  Rationale: {violation.rationale}")
            lines.append("")

        return "\n".join(lines)

    def generate_violation_summary(self, violation: Violation) -> str:
        """Generate a single-line summary for one violation.

        Args:
            violation: A single Violation instance.

        Returns:
            A concise one-line description.
        """
        return (
            f"[{violation.violation_type}] "
            f"frames {violation.frame_range[0]}-{violation.frame_range[1]}, "
            f"severity={violation.severity:.2f}: {violation.rationale}"
        )
