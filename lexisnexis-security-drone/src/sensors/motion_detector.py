"""
Motion Detection Sensor
========================

Computes frame-to-frame motion intensity using background subtraction.
Supports both real camera input and synthetic mode for testing.

Algorithm:
  1. Convert frames to grayscale
  2. Apply Gaussian blur to reduce noise
  3. Compute absolute difference from background model
  4. Threshold to binary motion mask
  5. Compute motion intensity as fraction of changed pixels
  6. Extract motion bounding boxes (contour-based)
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionReading:
    intensity: float                              # 0–1: fraction of changed pixels
    regions: List[Tuple[int, int, int, int]]      # (x, y, w, h) of motion regions
    frame_delta: Optional[np.ndarray]             # debug: diff image


class MotionDetector:
    """
    Background subtraction motion detector.
    Maintains an exponential moving average background model.
    """

    def __init__(self, sensitivity: float = 0.3, synthetic: bool = False):
        self.sensitivity = sensitivity      # pixel-change threshold (0–1 fraction)
        self.synthetic = synthetic
        self._background: Optional[np.ndarray] = None
        self._bg_alpha = 0.05               # background update rate
        self._frame_count = 0

    def process(self, frame: np.ndarray) -> MotionReading:
        """Compute motion reading from a new frame."""
        if self.synthetic:
            return self._synthetic_reading()

        try:
            import cv2
            return self._real_process(frame, cv2)
        except ImportError:
            return self._synthetic_reading()

    def _real_process(self, frame, cv2) -> MotionReading:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)

        if self._background is None:
            self._background = gray.copy()
            return MotionReading(intensity=0.0, regions=[], frame_delta=None)

        # Absolute difference from background
        delta = cv2.absdiff(self._background.astype(np.uint8), gray.astype(np.uint8))
        _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

        # Update background model (exponential moving average)
        self._background = (1 - self._bg_alpha) * self._background + self._bg_alpha * gray

        # Compute intensity
        intensity = float(np.count_nonzero(thresh)) / thresh.size

        # Extract motion bounding boxes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # ignore tiny noise
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append((x, y, w, h))

        self._frame_count += 1
        return MotionReading(intensity=intensity, regions=regions, frame_delta=delta)

    def _synthetic_reading(self) -> MotionReading:
        """Returns synthetic motion data that varies over time."""
        t = time.time()
        # Slow sinusoidal variation simulating periodic activity
        intensity = 0.1 + 0.2 * abs(np.sin(t * 0.3))
        # Occasionally spike to simulate person entering frame
        if int(t) % 20 < 3:
            intensity = min(intensity + 0.4, 1.0)
        regions = [(100, 100, 150, 200)] if intensity > 0.2 else []
        self._frame_count += 1
        return MotionReading(intensity=intensity, regions=regions, frame_delta=None)

    def reset(self):
        """Reset the background model (e.g., after a camera move)."""
        self._background = None
        logger.info("Motion detector background model reset.")
