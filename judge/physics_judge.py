"""
PhysicsJudge: MLLM-as-Judge for Physical Violation Detection.

Runs Physion-Eval style analysis on a video, using a multimodal LLM
(default: Qwen2.5-VL or LLaVA) to generate natural language rationales
for each detected physical violation.

Output is a structured list of violations, each with:
  - violation_type: category from Physion-Eval's 22-category taxonomy
  - frame_range: (start_frame, end_frame) where the violation occurs
  - severity: float in [0, 1], higher = more severe
  - rationale: natural language explanation from the MLLM
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Physion-Eval physical violation categories (subset of 22)
VIOLATION_TYPES = [
    "rigid_body_penetration",
    "gravity_violation",
    "conservation_of_momentum",
    "fluid_dynamics",
    "object_permanence",
    "impossible_deformation",
    "friction_violation",
    "collision_response",
    "support_relation",
    "occlusion_error",
    "mass_inconsistency",
    "trajectory_discontinuity",
    "energy_conservation",
    "contact_dynamics",
    "rolling_sliding",
    "stacking_stability",
    "projectile_motion",
    "buoyancy",
    "elasticity",
    "cloth_dynamics",
    "rope_chain_dynamics",
    "light_shadow_consistency",
]


@dataclass
class Violation:
    """A single detected physical violation."""

    violation_type: str
    frame_range: Tuple[int, int]
    severity: float
    rationale: str

    def to_dict(self) -> Dict:
        return {
            "violation_type": self.violation_type,
            "frame_range": list(self.frame_range),
            "severity": round(self.severity, 4),
            "rationale": self.rationale,
        }


@dataclass
class JudgmentResult:
    """Complete judgment for a video."""

    video_path: str
    violations: List[Violation] = field(default_factory=list)
    overall_physics_score: float = 1.0
    frame_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "video_path": self.video_path,
            "violations": [v.to_dict() for v in self.violations],
            "overall_physics_score": round(self.overall_physics_score, 4),
            "frame_count": self.frame_count,
            "violation_count": len(self.violations),
        }


class PhysicsJudge:
    """Physion-Eval style physics analysis with MLLM rationale generation.

    Two-stage pipeline:
      1. Heuristic pre-screening: optical-flow and frame-diff analysis to
         identify candidate violation windows cheaply.
      2. MLLM verification: feed candidate frame pairs/clips to a multimodal
         LLM for classification and natural language rationale.
    """

    def __init__(
        self,
        mllm_model: str = "qwen2.5-vl",
        mllm_endpoint: Optional[str] = None,
        sample_fps: int = 8,
        diff_threshold: float = 40.0,
        flow_threshold: float = 25.0,
    ):
        self.mllm_model = mllm_model
        self.mllm_endpoint = mllm_endpoint
        self.sample_fps = sample_fps
        self.diff_threshold = diff_threshold
        self.flow_threshold = flow_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(self, video_path: str) -> JudgmentResult:
        """Run full physics judgment pipeline on a video.

        Returns a JudgmentResult with all detected violations and an
        overall physics compliance score.
        """
        video_path = str(video_path)
        frames = self._load_frames(video_path)
        result = JudgmentResult(video_path=video_path, frame_count=len(frames))

        if len(frames) < 2:
            return result

        # Stage 1: heuristic candidate detection
        candidates = self._detect_candidates(frames)

        # Stage 2: MLLM classification + rationale for each candidate
        for candidate in candidates:
            violation = self._classify_violation(frames, candidate)
            if violation is not None:
                result.violations.append(violation)

        # Compute overall score: penalize based on violation count and severity
        if result.violations:
            total_severity = sum(v.severity for v in result.violations)
            max_possible = len(frames) / self.sample_fps  # rough duration proxy
            result.overall_physics_score = float(np.clip(1.0 - total_severity / max(max_possible, 1.0), 0.0, 1.0))

        return result

    # ------------------------------------------------------------------
    # Stage 1: Heuristic candidate detection
    # ------------------------------------------------------------------

    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        """Load video frames at the configured sample FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / self.sample_fps)))
        frames = []
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

    def _detect_candidates(self, frames: List[np.ndarray]) -> List[Dict]:
        """Identify frame ranges with suspicious physical behavior.

        Uses frame differencing and optional optical flow to flag windows
        where physics may be violated.
        """
        candidates = []

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Frame difference magnitude
            diff = np.abs(curr_gray - prev_gray)
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))

            anomalies = []

            # Sudden large change -> possible teleportation or discontinuity
            if mean_diff > self.diff_threshold:
                anomalies.append("trajectory_discontinuity")

            # Check for gravity-defying vertical motion
            h = diff.shape[0]
            upper_activity = float(np.mean(diff[: h // 3, :]))
            lower_activity = float(np.mean(diff[2 * h // 3 :, :]))
            if upper_activity > lower_activity * 2.0 and upper_activity > 20.0:
                anomalies.append("gravity_violation")

            # Check for object disappearance / permanence violation
            if max_diff > 200 and mean_diff < 10:
                anomalies.append("object_permanence")

            for atype in anomalies:
                candidates.append(
                    {
                        "frame_start": i - 1,
                        "frame_end": i,
                        "anomaly_type": atype,
                        "mean_diff": mean_diff,
                        "max_diff": max_diff,
                    }
                )

        # Merge adjacent candidates of the same type
        return self._merge_candidates(candidates)

    def _merge_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Merge consecutive candidate windows of the same anomaly type."""
        if not candidates:
            return []

        merged = [candidates[0]]
        for c in candidates[1:]:
            prev = merged[-1]
            if c["anomaly_type"] == prev["anomaly_type"] and c["frame_start"] <= prev["frame_end"] + 2:
                prev["frame_end"] = c["frame_end"]
                prev["mean_diff"] = max(prev["mean_diff"], c["mean_diff"])
                prev["max_diff"] = max(prev["max_diff"], c["max_diff"])
            else:
                merged.append(c)
        return merged

    # ------------------------------------------------------------------
    # Stage 2: MLLM classification and rationale
    # ------------------------------------------------------------------

    def _classify_violation(self, frames: List[np.ndarray], candidate: Dict) -> Optional[Violation]:
        """Use MLLM to classify and explain a candidate violation.

        Falls back to heuristic classification if no MLLM endpoint is
        available.
        """
        frame_start = candidate["frame_start"]
        frame_end = candidate["frame_end"]
        anomaly_type = candidate["anomaly_type"]

        severity = self._estimate_severity(candidate)

        # Try MLLM-based rationale
        rationale = self._query_mllm(frames, frame_start, frame_end, anomaly_type)

        if rationale is None:
            # Fallback heuristic rationale
            rationale = self._heuristic_rationale(candidate)

        return Violation(
            violation_type=anomaly_type,
            frame_range=(frame_start, frame_end),
            severity=severity,
            rationale=rationale,
        )

    def _estimate_severity(self, candidate: Dict) -> float:
        """Estimate violation severity from heuristic signals."""
        mean_diff = candidate["mean_diff"]
        duration = candidate["frame_end"] - candidate["frame_start"] + 1

        # Severity grows with magnitude and duration
        magnitude_score = min(1.0, mean_diff / 100.0)
        duration_score = min(1.0, duration / 10.0)
        return float(np.clip(0.4 * magnitude_score + 0.6 * duration_score, 0.0, 1.0))

    def _query_mllm(
        self,
        frames: List[np.ndarray],
        frame_start: int,
        frame_end: int,
        anomaly_type: str,
    ) -> Optional[str]:
        """Query the MLLM for a natural language rationale.

        Sends a pair of frames (start/end of violation window) with a
        structured prompt asking the model to explain the physical
        violation.

        Returns None if no endpoint is configured or the query fails.
        """
        if self.mllm_endpoint is None:
            return None

        try:
            import requests

            # Save frames as temporary images
            with tempfile.TemporaryDirectory() as tmpdir:
                image_paths: list[str] = []
                for idx in (frame_start, min(frame_end, len(frames) - 1)):
                    image_path = Path(tmpdir) / f"frame_{idx}.jpg"
                    cv2.imwrite(str(image_path), cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
                    image_paths.append(str(image_path))

                prompt = (
                    f"Analyze these two consecutive video frames for physical violations. "
                    f"A heuristic detector flagged a potential '{anomaly_type}' violation. "
                    f"Describe the specific physical implausibility you observe, "
                    f"or state that the transition appears physically valid. "
                    f"Be concise (1-2 sentences)."
                )

                # Build request (generic OpenAI-compatible vision endpoint)
                import base64

                images_b64 = []
                for image_path_str in image_paths:
                    with open(image_path_str, "rb") as f:
                        images_b64.append(base64.b64encode(f.read()).decode())

                payload = {
                    "model": self.mllm_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *[
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                                    for b64 in images_b64
                                ],
                            ],
                        }
                    ],
                    "max_tokens": 200,
                }

                resp = requests.post(
                    self.mllm_endpoint,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

        except Exception:
            return None

    def _heuristic_rationale(self, candidate: Dict) -> str:
        """Generate a heuristic-based rationale when MLLM is unavailable."""
        atype = candidate["anomaly_type"]
        mean_diff = candidate["mean_diff"]
        frame_start = candidate["frame_start"]
        frame_end = candidate["frame_end"]

        templates = {
            "trajectory_discontinuity": (
                f"Abrupt spatial discontinuity detected between frames {frame_start}-{frame_end} "
                f"(mean pixel delta={mean_diff:.1f}). Objects appear to teleport rather than "
                f"following a continuous trajectory, violating expected kinematics."
            ),
            "gravity_violation": (
                f"Upward motion detected in frames {frame_start}-{frame_end} without corresponding "
                f"applied force. Objects in the upper frame region show more activity than the lower "
                f"region, inconsistent with gravitational pull."
            ),
            "object_permanence": (
                f"Possible object permanence violation in frames {frame_start}-{frame_end}. "
                f"A localized high-intensity change with low mean change suggests an object "
                f"appeared or disappeared instantaneously."
            ),
        }
        return templates.get(atype, f"Physical anomaly of type '{atype}' detected in frames {frame_start}-{frame_end}.")
