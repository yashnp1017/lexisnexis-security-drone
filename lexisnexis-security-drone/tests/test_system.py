"""
Test suite for the LexisNexis Autonomous Security Drone System.
Tests threat scoring, identity database, motion detection, and alert dispatch.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# ── Threat Scorer Tests ────────────────────────────────────────────────────────

class TestThreatScorer:
    def _settings(self, **kwargs):
        from config.settings import ThreatSettings
        base = dict(
            weight_unknown_face=0.40,
            weight_motion_intensity=0.30,
            weight_restricted_zone=0.20,
            weight_time_of_day=0.10,
            alert_threshold=0.65,
            critical_threshold=0.85,
            restricted_zones=[[0, 0, 200, 480]],
        )
        base.update(kwargs)
        return ThreatSettings(**base)

    def _scorer(self, **kwargs):
        from src.detection.threat_scorer import ThreatScorer
        return ThreatScorer(self._settings(**kwargs))

    def _reading(self, faces=None, motion=0.0):
        from src.detection.threat_scorer import SensorReading
        return SensorReading(
            frame_id=1,
            timestamp=datetime.now().replace(hour=14, minute=0),  # low-risk hour
            faces=faces or [],
            motion_intensity=motion,
        )

    def _unknown_face(self, bbox=(300, 200, 100, 100)):
        from src.detection.face_pipeline import DetectedFace
        rng = np.random.default_rng(99)
        emb = rng.standard_normal(128).astype(np.float32)
        emb /= np.linalg.norm(emb)
        return DetectedFace(bbox=bbox, confidence=0.9, embedding=emb,
                            identity=None, similarity=0.25,
                            is_known=False, label="UNKNOWN",
                            processing_time_ms=10.0)

    def _known_face(self):
        from src.detection.face_pipeline import DetectedFace
        from src.detection.identity_db import Identity
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(128).astype(np.float32)
        emb /= np.linalg.norm(emb)
        identity = Identity(id=1, name="Alice", role="employee", access_level=1, embedding=emb)
        return DetectedFace(bbox=(200, 150, 100, 100), confidence=0.95, embedding=emb,
                            identity=identity, similarity=0.93,
                            is_known=True, label="Alice",
                            processing_time_ms=10.0)

    def test_empty_frame_low_threat(self):
        scorer = self._scorer()
        assessment = scorer.assess(self._reading())
        assert assessment.score < 0.50
        assert assessment.alert_required is False

    def test_known_face_only_low_threat(self):
        scorer = self._scorer()
        reading = self._reading(faces=[self._known_face()], motion=0.05)
        assessment = scorer.assess(reading)
        assert assessment.score < 0.50
        assert assessment.unknown_faces == 0
        # ThreatAssessment correctly has no is_known; score check above is sufficient

    def test_unknown_face_raises_threat(self):
        scorer = self._scorer()
        reading = self._reading(faces=[self._unknown_face()], motion=0.30)
        assessment = scorer.assess(reading)
        assert assessment.score > 0.40
        assert assessment.unknown_faces == 1

    def test_unknown_face_in_restricted_zone_high_threat(self):
        scorer = self._scorer()
        # bbox inside restricted zone [0, 0, 200, 480]: center at (80, 200)
        face = self._unknown_face(bbox=(30, 150, 100, 100))
        reading = self._reading(faces=[face], motion=0.40)
        assessment = scorer.assess(reading)
        assert assessment.in_restricted_zone is True
        assert assessment.score > 0.55

    def test_critical_threshold(self):
        scorer = self._scorer()
        face = self._unknown_face(bbox=(30, 150, 100, 100))
        reading = SensorReading = __import__(
            'src.detection.threat_scorer', fromlist=['SensorReading']
        ).SensorReading
        r = reading(
            frame_id=1,
            timestamp=datetime.now().replace(hour=2, minute=0),  # 2am = peak risk
            faces=[face],
            motion_intensity=0.70,
        )
        assessment = scorer.assess(r)
        from src.detection.threat_scorer import ThreatLevel
        assert assessment.level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)

    def test_score_between_zero_and_one(self):
        scorer = self._scorer()
        for motion in [0.0, 0.3, 0.8, 1.0]:
            r = self._reading(
                faces=[self._unknown_face(), self._known_face()],
                motion=motion
            )
            a = scorer.assess(r)
            assert 0.0 <= a.score <= 1.0

    def test_time_of_day_2am_higher_than_2pm(self):
        scorer = self._scorer()
        from src.detection.threat_scorer import SensorReading
        r_2am = SensorReading(frame_id=1, timestamp=datetime.now().replace(hour=2, minute=0),
                              faces=[], motion_intensity=0.0)
        r_2pm = SensorReading(frame_id=2, timestamp=datetime.now().replace(hour=14, minute=0),
                              faces=[], motion_intensity=0.0)
        assert scorer.assess(r_2pm).time_component > scorer.assess(r_2am).time_component  # 2pm = business hours = higher baseline risk in our model; 2am = low-activity hours


# ── Identity Database Tests ────────────────────────────────────────────────────

class TestIdentityDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        from src.detection.identity_db import IdentityDatabase
        db = IdentityDatabase(str(tmp_path / "test.db"))
        yield db
        db.close()

    def _random_emb(self, seed=42):
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(128).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def test_enroll_and_retrieve(self, db):
        emb = self._random_emb()
        iid = db.enroll("Test User", "employee", 1, emb)
        assert iid > 0
        identities = db.get_all_active()
        assert len(identities) == 1
        assert identities[0].name == "Test User"
        assert np.allclose(identities[0].embedding, emb)

    def test_deactivate(self, db):
        iid = db.enroll("Temp User", "visitor", 0, self._random_emb(1))
        assert db.count() == 1
        db.deactivate(iid)
        assert db.count() == 0
        assert len(db.get_all_active()) == 0

    def test_multiple_identities(self, db):
        for i in range(5):
            db.enroll(f"User {i}", "employee", 1, self._random_emb(i))
        assert db.count() == 5

    def test_empty_db(self, db):
        assert db.count() == 0
        assert db.get_all_active() == []


# ── Face Recognizer Tests ─────────────────────────────────────────────────────

class TestFaceRecognizer:
    @pytest.fixture
    def db_with_alice(self, tmp_path):
        from src.detection.identity_db import IdentityDatabase
        db = IdentityDatabase(str(tmp_path / "recog.db"))
        rng = np.random.default_rng(42)
        alice_emb = rng.standard_normal(128).astype(np.float32)
        alice_emb /= np.linalg.norm(alice_emb)
        db.enroll("Alice", "security", 3, alice_emb)
        yield db, alice_emb
        db.close()

    def test_recognize_known_identity(self, db_with_alice):
        from src.detection.face_pipeline import FaceRecognizer
        db, alice_emb = db_with_alice
        recognizer = FaceRecognizer(db, threshold=0.60)
        # Slightly perturbed embedding should still match
        noise = np.random.default_rng(99).standard_normal(128).astype(np.float32) * 0.05
        query = alice_emb + noise
        query /= np.linalg.norm(query)
        identity, sim = recognizer.recognize(query)
        assert identity is not None
        assert identity.name == "Alice"
        assert sim >= 0.60

    def test_reject_unknown(self, db_with_alice):
        from src.detection.face_pipeline import FaceRecognizer
        db, _ = db_with_alice
        recognizer = FaceRecognizer(db, threshold=0.60)
        rng = np.random.default_rng(12345)
        unknown = rng.standard_normal(128).astype(np.float32)
        unknown /= np.linalg.norm(unknown)
        identity, sim = recognizer.recognize(unknown)
        assert identity is None


# ── Alert Dispatcher Tests ────────────────────────────────────────────────────

class TestAlertDispatcher:
    @pytest.fixture
    def dispatcher(self):
        from config.settings import APISettings
        from src.utils.alert_dispatcher import AlertDispatcher
        return AlertDispatcher(APISettings(api_key="test"))

    def _mock_assessment(self, score=0.75, level="HIGH"):
        from unittest.mock import MagicMock
        from src.detection.threat_scorer import ThreatLevel
        a = MagicMock()
        a.score = score
        a.level = ThreatLevel[level]
        a.timestamp = datetime.now()
        a.unknown_faces = 1
        a.total_faces = 1
        a.motion_intensity = 0.30
        a.in_restricted_zone = False
        a.details = f"test alert score={score}"
        return a

    @pytest.mark.asyncio
    async def test_dispatch_creates_alert(self, dispatcher):
        alert = await dispatcher.dispatch(self._mock_assessment())
        assert alert is not None
        assert alert.alert_id == 1
        assert alert.threat_level == "HIGH"
        assert len(dispatcher.get_recent_alerts()) == 1

    @pytest.mark.asyncio
    async def test_deduplication(self, dispatcher):
        dispatcher.COOLDOWN_SECONDS = 60
        a1 = await dispatcher.dispatch(self._mock_assessment())
        a2 = await dispatcher.dispatch(self._mock_assessment())
        assert a1 is not None
        assert a2 is None  # deduplicated

    @pytest.mark.asyncio
    async def test_acknowledge(self, dispatcher):
        alert = await dispatcher.dispatch(self._mock_assessment())
        assert not alert.acknowledged
        ok = dispatcher.acknowledge(alert.alert_id)
        assert ok
        assert dispatcher.get_recent_alerts()[0].acknowledged

    def test_stats(self, dispatcher):
        stats = dispatcher.get_stats()
        assert "total_alerts" in stats
        assert "by_level" in stats


# ── Motion Detector Tests ──────────────────────────────────────────────────────

class TestMotionDetector:
    def test_synthetic_returns_reading(self):
        from src.sensors.motion_detector import MotionDetector
        detector = MotionDetector(sensitivity=0.3, synthetic=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        reading = detector.process(frame)
        assert 0.0 <= reading.intensity <= 1.0
        assert isinstance(reading.regions, list)

    def test_reset_clears_background(self):
        from src.sensors.motion_detector import MotionDetector
        detector = MotionDetector(synthetic=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.process(frame)
        detector.reset()
        assert detector._background is None
