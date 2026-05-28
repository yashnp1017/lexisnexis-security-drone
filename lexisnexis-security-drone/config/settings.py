"""Application settings loaded from YAML or environment variables."""

import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class DetectionSettings:
    face_model: str = "models/face_recognition_model.pkl"
    min_face_confidence: float = 0.75
    recognition_threshold: float = 0.60   # cosine similarity threshold
    max_faces_per_frame: int = 10
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 15


@dataclass
class ThreatSettings:
    # Weighted threat score components (must sum to 1.0)
    weight_unknown_face: float = 0.40
    weight_motion_intensity: float = 0.30
    weight_restricted_zone: float = 0.20
    weight_time_of_day: float = 0.10
    # Thresholds
    alert_threshold: float = 0.65
    critical_threshold: float = 0.85
    # Restricted zones: list of (x1, y1, x2, y2) in frame coordinates
    restricted_zones: List[List[int]] = field(default_factory=list)


@dataclass
class APISettings:
    host: str = "0.0.0.0"
    port: int = 8000
    alert_webhook_url: str = ""
    webhook_timeout_seconds: int = 5
    api_key: str = ""


@dataclass
class SensorSettings:
    camera_index: int = 0
    motion_sensitivity: float = 0.3   # 0–1: fraction of pixels changed
    depth_enabled: bool = False
    synthetic_mode: bool = True        # use synthetic data (no hardware)


@dataclass
class Settings:
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    threat: ThreatSettings = field(default_factory=ThreatSettings)
    api: APISettings = field(default_factory=APISettings)
    sensor: SensorSettings = field(default_factory=SensorSettings)
    log_level: str = "INFO"
    database_path: str = "data/identities.db"

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        settings = cls()
        for section_name, section_cls in [
            ("detection", DetectionSettings),
            ("threat", ThreatSettings),
            ("api", APISettings),
            ("sensor", SensorSettings),
        ]:
            if section_name in raw:
                section = section_cls(**{
                    k: v for k, v in raw[section_name].items()
                    if hasattr(section_cls(), k)
                })
                setattr(settings, section_name, section)
        if "log_level" in raw:
            settings.log_level = raw["log_level"]
        if "database_path" in raw:
            settings.database_path = raw["database_path"]
        return settings
