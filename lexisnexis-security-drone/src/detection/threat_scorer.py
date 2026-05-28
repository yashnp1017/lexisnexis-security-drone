"""
Threat Scoring Engine
======================

Combines multiple sensor signals into a weighted threat score (0.0–1.0):

  score = w_face   * face_component
        + w_motion * motion_component
        + w_zone   * zone_component
        + w_time   * time_component

Threat levels:
  CLEAR    (0.0 – 0.35)  — normal activity, no action
  LOW      (0.35 – 0.50) — monitor
  MEDIUM   (0.50 – 0.65) — increased surveillance
  HIGH     (0.65 – 0.85) — dispatch alert
  CRITICAL (0.85 – 1.0)  — immediate response required
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from src.detection.face_pipeline import DetectedFace
from config.settings import ThreatSettings

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    CLEAR    = "CLEAR"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_score(cls, score: float) -> "ThreatLevel":
        if score < 0.35:   return cls.CLEAR
        if score < 0.50:   return cls.LOW
        if score < 0.65:   return cls.MEDIUM
        if score < 0.85:   return cls.HIGH
        return cls.CRITICAL


@dataclass
class ThreatAssessment:
    score: float                      # 0.0 – 1.0 composite threat score
    level: ThreatLevel
    timestamp: datetime
    face_component: float             # contribution from face recognition
    motion_component: float           # contribution from motion sensors
    zone_component: float             # contribution from zone violation
    time_component: float             # contribution from time-of-day risk
    unknown_faces: int
    total_faces: int
    motion_intensity: float
    in_restricted_zone: bool
    alert_required: bool
    critical: bool
    details: str                      # human-readable summary


@dataclass
class SensorReading:
    """Fused sensor data for one processing cycle."""
    frame_id: int
    timestamp: datetime
    faces: List[DetectedFace] = field(default_factory=list)
    motion_intensity: float = 0.0      # 0–1: fraction of changed pixels
    motion_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    depth_map: Optional[object] = None


class ThreatScorer:
    """
    Multi-factor threat scoring engine.

    Design:
      - Each component is normalized to [0, 1] independently
      - Components are linearly combined with configurable weights
      - Time-of-day risk follows a sinusoidal model (peak risk at 2am)
      - Unknown faces in restricted zones multiply the zone component
    """

    def __init__(self, settings: ThreatSettings):
        self.s = settings
        self._validate_weights()

    def _validate_weights(self):
        total = (
            self.s.weight_unknown_face +
            self.s.weight_motion_intensity +
            self.s.weight_restricted_zone +
            self.s.weight_time_of_day
        )
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Threat weights sum to {total:.3f}, not 1.0. Normalizing.")
            self.s.weight_unknown_face      /= total
            self.s.weight_motion_intensity  /= total
            self.s.weight_restricted_zone   /= total
            self.s.weight_time_of_day       /= total

    def assess(self, reading: SensorReading) -> ThreatAssessment:
        """Compute a full threat assessment from one sensor reading."""
        now = reading.timestamp

        # ── Component 1: Face recognition ──────────────────────────────────
        face_component = self._face_component(reading.faces)

        # ── Component 2: Motion intensity ──────────────────────────────────
        motion_component = min(reading.motion_intensity / 0.8, 1.0)

        # ── Component 3: Restricted zone violation ─────────────────────────
        in_zone, zone_component = self._zone_component(reading.faces)

        # ── Component 4: Time-of-day risk ──────────────────────────────────
        time_component = self._time_of_day_risk(now)

        # ── Composite score ────────────────────────────────────────────────
        score = (
            self.s.weight_unknown_face     * face_component  +
            self.s.weight_motion_intensity * motion_component +
            self.s.weight_restricted_zone  * zone_component  +
            self.s.weight_time_of_day      * time_component
        )
        score = max(0.0, min(1.0, score))
        level = ThreatLevel.from_score(score)

        unknown_count = sum(1 for f in reading.faces if not f.is_known)
        total_count   = len(reading.faces)

        details = self._build_summary(
            score, level, unknown_count, total_count,
            reading.motion_intensity, in_zone, now
        )

        assessment = ThreatAssessment(
            score=round(score, 4),
            level=level,
            timestamp=now,
            face_component=round(face_component, 4),
            motion_component=round(motion_component, 4),
            zone_component=round(zone_component, 4),
            time_component=round(time_component, 4),
            unknown_faces=unknown_count,
            total_faces=total_count,
            motion_intensity=round(reading.motion_intensity, 4),
            in_restricted_zone=in_zone,
            alert_required=score >= self.s.alert_threshold,
            critical=score >= self.s.critical_threshold,
            details=details,
        )

        if assessment.critical:
            logger.warning(f"CRITICAL THREAT — score={score:.3f} | {details}")
        elif assessment.alert_required:
            logger.info(f"ALERT — score={score:.3f} | {details}")

        return assessment

    def _face_component(self, faces: List[DetectedFace]) -> float:
        """
        Score based on unknown faces.
        - No faces detected → 0.0 (no information)
        - All known → 0.05 (slight residual)
        - Any unknown → scales with fraction of unknown faces
        - Unknown with low similarity → higher score (more concerning)
        """
        if not faces:
            return 0.0

        unknown_faces = [f for f in faces if not f.is_known]
        if not unknown_faces:
            return 0.05

        # Scale by fraction of unknown faces, boosted by how dissimilar they are
        fraction_unknown = len(unknown_faces) / len(faces)
        avg_dissimilarity = sum(1.0 - f.similarity for f in unknown_faces) / len(unknown_faces)
        return min(fraction_unknown * (0.5 + 0.5 * avg_dissimilarity), 1.0)

    def _zone_component(self, faces: List[DetectedFace]) -> Tuple[bool, float]:
        """
        Score based on whether unknown faces appear in restricted zones.
        Returns (in_zone: bool, component_score: float).
        """
        if not self.s.restricted_zones or not faces:
            return False, 0.0

        unknown_faces = [f for f in faces if not f.is_known]
        if not unknown_faces:
            return False, 0.0

        violations = 0
        for face in unknown_faces:
            fx, fy, fw, fh = face.bbox
            cx, cy = fx + fw // 2, fy + fh // 2
            for zone in self.s.restricted_zones:
                x1, y1, x2, y2 = zone
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    violations += 1
                    break

        if violations == 0:
            return False, 0.0

        score = min(violations / max(len(unknown_faces), 1), 1.0)
        return True, score

    def _time_of_day_risk(self, dt: datetime) -> float:
        """
        Sinusoidal risk model: peak risk at 2pm (hour=14, normal business hours with most activity),
        trough at 2am. Higher score during active hours when more people are present.
        Risk is always > 0 even during business hours.
        """
        hour = dt.hour + dt.minute / 60.0
        # Map to radians: hour 2 = peak (π), hour 14 = trough (0)
        radians = math.pi * (hour - 14) / 12
        raw = 0.5 + 0.5 * math.cos(radians)   # 0–1
        # Compress to [0.1, 0.9] so no time is completely safe or maximally risky
        return 0.1 + 0.8 * raw

    def _build_summary(self, score, level, unknown, total,
                       motion, in_zone, dt) -> str:
        parts = [
            f"score={score:.3f}",
            f"level={level.value}",
            f"faces={total}({unknown} unknown)",
            f"motion={motion:.2f}",
            f"zone={'YES' if in_zone else 'no'}",
            f"time={dt.strftime('%H:%M')}",
        ]
        return " | ".join(parts)
