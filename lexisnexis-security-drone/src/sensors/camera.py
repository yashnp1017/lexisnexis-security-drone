"""
Camera Capture
==============

Wraps OpenCV VideoCapture for real hardware and provides
a synthetic frame generator for testing without a camera.
"""

import logging
import time
from typing import Optional, Generator

import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """
    Camera abstraction supporting real hardware and synthetic mode.

    Usage:
        cam = Camera(index=0, synthetic=False)
        for frame in cam.stream():
            process(frame)
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: int = 15, synthetic: bool = False):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.synthetic = synthetic
        self._cap = None

    def open(self) -> bool:
        """Open the camera. Returns True on success."""
        if self.synthetic:
            logger.info("Camera opened in synthetic mode.")
            return True
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera index {self.index}")
                return False
            logger.info(f"Camera {self.index} opened ({self.width}x{self.height} @ {self.fps}fps)")
            return True
        except Exception as e:
            logger.error(f"Camera open error: {e}")
            return False

    def read(self) -> Optional[np.ndarray]:
        """Read one frame. Returns None on failure."""
        if self.synthetic:
            return self._synthetic_frame()
        if self._cap is None:
            return None
        import cv2
        ret, frame = self._cap.read()
        return frame if ret else None

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames at the configured fps."""
        interval = 1.0 / self.fps
        while True:
            t0 = time.perf_counter()
            frame = self.read()
            if frame is None:
                logger.warning("Empty frame — skipping.")
                time.sleep(interval)
                continue
            yield frame
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def _synthetic_frame(self) -> np.ndarray:
        """Generate a realistic synthetic frame (gray gradient + noise)."""
        rng = np.random.default_rng(seed=int(time.time() * self.fps) % 10000)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        for i in range(self.height):
            frame[i, :] = int(40 + 80 * i / self.height)
        # Add noise
        noise = rng.integers(0, 15, (self.height, self.width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        # Draw a synthetic "person silhouette" box
        cx = int(self.width * 0.4 + 50 * np.sin(time.time() * 0.5))
        cy = self.height // 2
        frame[cy-80:cy+80, cx-40:cx+40] = [180, 160, 140]  # skin-tone rectangle
        return frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            logger.info("Camera released.")
