"""
Mission Controller
==================

The central orchestrator. Runs the main processing loop:

  Camera → MotionDetector → DetectionPipeline → ThreatScorer → AlertDispatcher

State machine:
  IDLE → PATROLLING → INVESTIGATING → ALERTING → RETURNING
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from src.detection.face_pipeline import DetectionPipeline
from src.detection.identity_db import IdentityDatabase
from src.detection.threat_scorer import ThreatScorer, SensorReading, ThreatAssessment
from src.sensors.camera import Camera
from src.sensors.motion_detector import MotionDetector
from src.utils.alert_dispatcher import AlertDispatcher
from config.settings import Settings

logger = logging.getLogger(__name__)


class MissionState(Enum):
    IDLE         = "IDLE"
    PATROLLING   = "PATROLLING"
    INVESTIGATING = "INVESTIGATING"
    ALERTING     = "ALERTING"
    RETURNING    = "RETURNING"
    SHUTDOWN     = "SHUTDOWN"


@dataclass
class MissionStatus:
    state: MissionState
    frame_count: int
    alerts_dispatched: int
    last_assessment: Optional[ThreatAssessment]
    uptime_seconds: float
    enrolled_identities: int


class MissionController:
    """
    Manages the complete drone mission lifecycle.
    Designed to run as an async task; exposes status via .get_status().
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = MissionState.IDLE
        self._frame_count = 0
        self._alerts_dispatched = 0
        self._start_time: Optional[datetime] = None
        self._last_assessment: Optional[ThreatAssessment] = None
        self._running = False

        synthetic = settings.sensor.synthetic_mode

        self.camera = Camera(
            index=settings.sensor.camera_index,
            width=settings.detection.frame_width,
            height=settings.detection.frame_height,
            fps=settings.detection.fps,
            synthetic=synthetic,
        )
        self.motion_detector = MotionDetector(
            sensitivity=settings.sensor.motion_sensitivity,
            synthetic=synthetic,
        )
        self.db = IdentityDatabase(settings.database_path)
        self.detection_pipeline = DetectionPipeline(
            settings.detection, self.db, synthetic=synthetic
        )
        self.threat_scorer = ThreatScorer(settings.threat)
        self.alert_dispatcher = AlertDispatcher(settings.api)

    async def run(self):
        """Main async mission loop."""
        logger.info("Mission controller starting.")
        self._start_time = datetime.now()
        self._running = True

        if not self.camera.open():
            logger.error("Failed to open camera. Aborting.")
            return

        self.state = MissionState.PATROLLING
        logger.info(f"State → {self.state.value}")

        try:
            for frame in self.camera.stream():
                if not self._running:
                    break

                await self._process_frame(frame)
                await asyncio.sleep(0)  # yield to event loop
        finally:
            self.state = MissionState.SHUTDOWN

    async def _process_frame(self, frame):
        """Process one camera frame through the full pipeline."""
        self._frame_count += 1
        now = datetime.now()

        # Sensor fusion
        motion = self.motion_detector.process(frame)
        faces  = self.detection_pipeline.process_frame(frame)

        reading = SensorReading(
            frame_id=self._frame_count,
            timestamp=now,
            faces=faces,
            motion_intensity=motion.intensity,
            motion_regions=motion.regions,
        )

        # Threat assessment
        assessment = self.threat_scorer.assess(reading)
        self._last_assessment = assessment

        # Update state machine
        self._update_state(assessment)

        # Dispatch alert if needed
        if assessment.alert_required:
            self._alerts_dispatched += 1
            await self.alert_dispatcher.dispatch(assessment)

        # Log periodic status
        if self._frame_count % 30 == 0:
            logger.info(
                f"Frame {self._frame_count} | state={self.state.value} | "
                f"threat={assessment.level.value} ({assessment.score:.3f}) | "
                f"faces={assessment.total_faces} | alerts={self._alerts_dispatched}"
            )

    def _update_state(self, assessment: ThreatAssessment):
        """Transition state machine based on threat level."""
        from src.detection.threat_scorer import ThreatLevel
        if assessment.level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            if self.state != MissionState.ALERTING:
                self.state = MissionState.ALERTING
                logger.info(f"State → ALERTING (threat={assessment.score:.3f})")
        elif assessment.level == ThreatLevel.MEDIUM:
            if self.state not in (MissionState.INVESTIGATING, MissionState.ALERTING):
                self.state = MissionState.INVESTIGATING
                logger.info(f"State → INVESTIGATING (threat={assessment.score:.3f})")
        else:
            if self.state in (MissionState.INVESTIGATING, MissionState.ALERTING):
                self.state = MissionState.PATROLLING
                logger.info("State → PATROLLING (threat cleared)")

    def get_status(self) -> MissionStatus:
        uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0.0
        return MissionStatus(
            state=self.state,
            frame_count=self._frame_count,
            alerts_dispatched=self._alerts_dispatched,
            last_assessment=self._last_assessment,
            uptime_seconds=round(uptime, 1),
            enrolled_identities=self.db.count(),
        )

    async def shutdown(self):
        logger.info("Shutting down mission controller.")
        self._running = False
        self.camera.close()
        self.db.close()
        self.state = MissionState.SHUTDOWN
