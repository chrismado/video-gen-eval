"""
AnomalyDetector: OpenCV-based heuristic detector for physical anomalies
in video frames.

Provides standalone anomaly detection that can be used independently
of the full PhysicsJudge pipeline.  Returns a list of Violation objects
for each detected anomaly.
"""
from typing import List, Optional

import cv2
import numpy as np

from judge.physics_judge import Violation


class AnomalyDetector:
    """Detect physical anomalies in video frames using OpenCV heuristics.

    Analyzes frame pairs for trajectory discontinuities, gravity violations,
    object permanence issues, and other physically implausible transitions.
    """

    def __init__(
        self,
        diff_threshold: float = 40.0,
        flow_threshold: float = 25.0,
        sample_fps: int = 8,
    ):
        """
        Args:
            diff_threshold: Mean pixel difference threshold for flagging
                            trajectory discontinuity.
            flow_threshold: Optical flow magnitude threshold.
            sample_fps: Target frames per second for sampling.
        """
        self.diff_threshold = diff_threshold
        self.flow_threshold = flow_threshold
        self.sample_fps = sample_fps

    def detect(self, video_path: str) -> List[Violation]:
        """Detect physical anomalies in a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of Violation objects, one per detected anomaly window.
        """
        frames = self._load_frames(video_path)
        if len(frames) < 2:
            return []

        violations: List[Violation] = []

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)

            diff = np.abs(curr_gray - prev_gray)
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))

            # Trajectory discontinuity
            if mean_diff > self.diff_threshold:
                violations.append(Violation(
                    violation_type="trajectory_discontinuity",
                    frame_range=(i - 1, i),
                    severity=min(1.0, mean_diff / 100.0),
                    rationale=(
                        f"Abrupt spatial discontinuity between frames {i - 1}-{i} "
                        f"(mean delta={mean_diff:.1f})."
                    ),
                ))

            # Gravity violation
            h = diff.shape[0]
            upper_activity = float(np.mean(diff[:h // 3, :]))
            lower_activity = float(np.mean(diff[2 * h // 3:, :]))
            if upper_activity > lower_activity * 2.0 and upper_activity > 20.0:
                violations.append(Violation(
                    violation_type="gravity_violation",
                    frame_range=(i - 1, i),
                    severity=min(1.0, upper_activity / 80.0),
                    rationale=(
                        f"Upward motion without applied force in frames {i - 1}-{i}."
                    ),
                ))

            # Object permanence
            if max_diff > 200 and mean_diff < 10:
                violations.append(Violation(
                    violation_type="object_permanence",
                    frame_range=(i - 1, i),
                    severity=0.6,
                    rationale=(
                        f"Object appeared or disappeared instantaneously in frames {i - 1}-{i}."
                    ),
                ))

        return violations

    def detect_from_frames(self, frames: List[np.ndarray]) -> List[Violation]:
        """Detect anomalies from pre-loaded frames (RGB uint8 arrays).

        Args:
            frames: List of RGB numpy arrays (H, W, 3).

        Returns:
            List of Violation objects.
        """
        if len(frames) < 2:
            return []

        violations: List[Violation] = []
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)

            diff = np.abs(curr_gray - prev_gray)
            mean_diff = float(np.mean(diff))

            if mean_diff > self.diff_threshold:
                violations.append(Violation(
                    violation_type="trajectory_discontinuity",
                    frame_range=(i - 1, i),
                    severity=min(1.0, mean_diff / 100.0),
                    rationale=f"Discontinuity detected between frames {i - 1}-{i}.",
                ))

        return violations

    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        """Load video frames at the configured sample FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / self.sample_fps)))
        frames: List[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        return frames
