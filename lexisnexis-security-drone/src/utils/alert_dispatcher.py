"""
Alert Dispatcher
================

Dispatches threat alerts via:
  1. Webhook (configurable URL, JSON POST)
  2. In-memory alert log (always active, accessible via REST API)

Implements exponential backoff for failed webhook calls and
deduplication to prevent alert flooding.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

from config.settings import APISettings

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    alert_id: int
    timestamp: str
    threat_level: str
    threat_score: float
    unknown_faces: int
    total_faces: int
    motion_intensity: float
    in_restricted_zone: bool
    details: str
    acknowledged: bool = False


class AlertDispatcher:
    """
    Manages alert creation, deduplication, and dispatch.

    Deduplication: won't fire a new alert of the same or lower level
    within the cooldown window (30 seconds by default).
    """

    MAX_ALERTS = 1000
    COOLDOWN_SECONDS = 30

    def __init__(self, settings: APISettings):
        self.settings = settings
        self._alerts: deque = deque(maxlen=self.MAX_ALERTS)
        self._alert_counter = 0
        self._last_alert_time: Optional[float] = None
        self._last_alert_level: Optional[str] = None

    async def dispatch(self, assessment) -> Optional[Alert]:
        """
        Create and dispatch an alert for this threat assessment.
        Returns the Alert object, or None if deduplicated.
        """
        now = time.time()

        # Deduplication: skip if same level within cooldown
        if (self._last_alert_time and
                now - self._last_alert_time < self.COOLDOWN_SECONDS and
                self._last_alert_level == assessment.level.value):
            logger.debug(f"Alert deduplicated (cooldown): {assessment.level.value}")
            return None

        self._alert_counter += 1
        alert = Alert(
            alert_id=self._alert_counter,
            timestamp=assessment.timestamp.isoformat(),
            threat_level=assessment.level.value,
            threat_score=assessment.score,
            unknown_faces=assessment.unknown_faces,
            total_faces=assessment.total_faces,
            motion_intensity=assessment.motion_intensity,
            in_restricted_zone=assessment.in_restricted_zone,
            details=assessment.details,
        )

        self._alerts.appendleft(alert)
        self._last_alert_time = now
        self._last_alert_level = assessment.level.value

        logger.info(
            f"Alert #{alert.alert_id} — {alert.threat_level} "
            f"(score={alert.threat_score:.3f})"
        )

        # Send webhook if configured
        if self.settings.alert_webhook_url:
            asyncio.create_task(self._send_webhook(alert))

        return alert

    async def _send_webhook(self, alert: Alert, max_retries: int = 3):
        """POST alert to the configured webhook URL with exponential backoff."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed; webhook dispatch skipped.")
            return

        payload = json.dumps(asdict(alert))
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["X-API-Key"] = self.settings.api_key

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.settings.webhook_timeout_seconds)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.settings.alert_webhook_url,
                        data=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status < 300:
                            logger.info(f"Webhook delivered: {resp.status}")
                            return
                        logger.warning(f"Webhook non-2xx: {resp.status}")
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Webhook attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
                await asyncio.sleep(wait)

        logger.error(f"Webhook failed after {max_retries} attempts for alert #{alert.alert_id}")

    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        return list(self._alerts)[:limit]

    def acknowledge(self, alert_id: int) -> bool:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_stats(self) -> dict:
        total = len(self._alerts)
        unacked = sum(1 for a in self._alerts if not a.acknowledged)
        by_level = {}
        for a in self._alerts:
            by_level[a.threat_level] = by_level.get(a.threat_level, 0) + 1
        return {
            "total_alerts": total,
            "unacknowledged": unacked,
            "by_level": by_level,
            "cooldown_seconds": self.COOLDOWN_SECONDS,
        }
