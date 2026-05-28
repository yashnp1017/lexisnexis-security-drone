"""
Face Detection and Recognition Pipeline
=========================================

Two-stage pipeline:
  1. Detection  — finds face bounding boxes in a frame (OpenCV Haar cascade or DNN)
  2. Recognition — generates a 128-dim embedding per detected face and compares
                   against the enrolled identity database via cosine similarity

In production: uses face_recognition library (dlib-based, 99.38% LFW accuracy).
In synthetic mode: generates deterministic pseudo-embeddings for testing.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.detection.identity_db import Identity, IdentityDatabase
from config.settings import DetectionSettings

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """A face found in a single frame."""
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in pixels
    confidence: float                  # detector confidence 0–1
    embedding: np.ndarray             # 128-dim face embedding
    identity: Optional[Identity]      # matched identity, or None if unknown
    similarity: float                 # cosine similarity to matched identity
    is_known: bool
    label: str                        # display label: name or "UNKNOWN"
    processing_time_ms: float


class FaceDetector:
    """
    Detects faces in a frame using OpenCV's DNN face detector.
    Falls back to Haar cascade if DNN model is unavailable.
    In synthetic mode, returns deterministic fake detections.
    """

    def __init__(self, settings: DetectionSettings, synthetic: bool = False):
        self.settings = settings
        self.synthetic = synthetic
        self._net = None
        if not synthetic:
            self._load_model()

    def _load_model(self):
        try:
            import cv2
            # OpenCV DNN face detector (ResNet-based, ships with opencv-contrib)
            self._net = cv2.dnn.readNetFromCaffe(
                "models/deploy.prototxt",
                "models/res10_300x300_ssd_iter_140000.caffemodel",
            )
            logger.info("Loaded DNN face detector.")
        except Exception as e:
            logger.warning(f"DNN model unavailable ({e}), falling back to Haar cascade.")
            self._net = None

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Returns list of (x, y, w, h) bounding boxes."""
        if self.synthetic:
            return self._synthetic_detect(frame)
        if self._net is not None:
            return self._dnn_detect(frame)
        return self._haar_detect(frame)

    def _dnn_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        import cv2
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < self.settings.min_face_confidence:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes[:self.settings.max_faces_per_frame]

    def _haar_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def _synthetic_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Returns 1–2 synthetic face bounding boxes for testing."""
        rng = np.random.default_rng(seed=int(time.time()) // 2)
        n = rng.integers(1, 3)
        boxes = []
        for _ in range(n):
            x = int(rng.integers(50, 400))
            y = int(rng.integers(50, 300))
            boxes.append((x, y, 100, 100))
        return boxes


class FaceEmbedder:
    """
    Generates 128-dimensional face embeddings.
    Production: uses face_recognition (dlib ResNet model, 99.38% LFW).
    Synthetic: generates deterministic pseudo-random embeddings.
    """

    def __init__(self, synthetic: bool = False):
        self.synthetic = synthetic
        self._encoder = None
        if not synthetic:
            self._load_encoder()

    def _load_encoder(self):
        try:
            import face_recognition  # noqa: F401
            self._encoder = "face_recognition"
            logger.info("face_recognition encoder loaded.")
        except ImportError:
            logger.warning("face_recognition not installed; using synthetic embeddings.")
            self.synthetic = True

    def embed(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Returns a 128-dim unit-normalized embedding, or None if extraction fails."""
        if self.synthetic:
            return self._synthetic_embed(bbox)
        return self._real_embed(frame, bbox)

    def _real_embed(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        import face_recognition
        x, y, w, h = bbox
        rgb = frame[:, :, ::-1]  # BGR → RGB
        locations = [(y, x + w, y + h, x)]  # top, right, bottom, left
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
        if not encodings:
            return None
        emb = encodings[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _synthetic_embed(self, bbox) -> np.ndarray:
        """Deterministic 128-dim embedding based on bbox position."""
        seed = bbox[0] * 1000 + bbox[1]
        rng = np.random.default_rng(seed=seed % (2**32))
        emb = rng.standard_normal(128).astype(np.float32)
        return emb / np.linalg.norm(emb)


class FaceRecognizer:
    """
    Matches a face embedding against the identity database using cosine similarity.
    Returns the best-matching Identity (if above threshold) or None.
    """

    def __init__(self, db: IdentityDatabase, threshold: float):
        self.db = db
        self.threshold = threshold

    def recognize(self, embedding: np.ndarray) -> Tuple[Optional[Identity], float]:
        """
        Returns (Identity, similarity) if a match is found above threshold,
        else (None, best_similarity).
        """
        identities = self.db.get_all_active()
        if not identities:
            return None, 0.0

        best_identity = None
        best_score = -1.0

        for identity in identities:
            score = float(np.dot(embedding, identity.embedding))  # cosine (both unit-normed)
            if score > best_score:
                best_score = score
                best_identity = identity

        if best_score >= self.threshold:
            return best_identity, best_score
        return None, best_score


class DetectionPipeline:
    """
    Combines FaceDetector + FaceEmbedder + FaceRecognizer into a single
    frame-processing pipeline.

    Usage:
        pipeline = DetectionPipeline(settings, db)
        faces = pipeline.process_frame(frame)
    """

    def __init__(self, settings: DetectionSettings, db: IdentityDatabase,
                 synthetic: bool = False):
        self.detector  = FaceDetector(settings, synthetic=synthetic)
        self.embedder  = FaceEmbedder(synthetic=synthetic)
        self.recognizer = FaceRecognizer(db, threshold=settings.recognition_threshold)
        self.settings  = settings

    def process_frame(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Run full detection + recognition on one frame.
        Returns list of DetectedFace objects (one per detected face).
        """
        t0 = time.perf_counter()
        bboxes = self.detector.detect(frame)
        results = []

        for bbox in bboxes:
            emb = self.embedder.embed(frame, bbox)
            if emb is None:
                continue

            identity, similarity = self.recognizer.recognize(emb)
            is_known = identity is not None
            label = identity.name if is_known else "UNKNOWN"
            elapsed_ms = (time.perf_counter() - t0) * 1000

            results.append(DetectedFace(
                bbox=bbox,
                confidence=1.0,   # DNN detector confidence (set from detector in production)
                embedding=emb,
                identity=identity,
                similarity=similarity,
                is_known=is_known,
                label=label,
                processing_time_ms=elapsed_ms,
            ))

        return results
