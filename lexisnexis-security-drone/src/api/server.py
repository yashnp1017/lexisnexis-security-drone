"""
REST API Server
================

Exposes the drone system via FastAPI:

  GET  /health           — system health
  GET  /status           — mission status and last threat assessment
  GET  /alerts           — recent alert log
  POST /alerts/{id}/ack  — acknowledge an alert
  POST /enroll           — enroll a new identity
  GET  /identities       — list enrolled identities
  POST /identities/{id}/deactivate — deactivate an identity
"""

import logging
import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import APISettings, Settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(controller, settings: APISettings) -> FastAPI:
    app = FastAPI(
        title="LexisNexis Autonomous Security Drone API",
        description="REST interface for drone surveillance, threat alerts, and identity management.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_key(api_key: str = Security(api_key_header)):
        if settings.api_key and api_key != settings.api_key:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    # ── Health ──────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "lexisnexis-security-drone"}

    # ── Mission status ───────────────────────────────────────────────────────

    @app.get("/status")
    async def get_status(_=Depends(verify_key)):
        if controller is None:
            return {"state": "API_ONLY", "message": "No mission controller attached"}
        status = controller.get_status()
        result = {
            "state": status.state.value,
            "frame_count": status.frame_count,
            "alerts_dispatched": status.alerts_dispatched,
            "uptime_seconds": status.uptime_seconds,
            "enrolled_identities": status.enrolled_identities,
        }
        if status.last_assessment:
            a = status.last_assessment
            result["last_assessment"] = {
                "score": a.score,
                "level": a.level.value,
                "unknown_faces": a.unknown_faces,
                "total_faces": a.total_faces,
                "alert_required": a.alert_required,
                "critical": a.critical,
                "details": a.details,
            }
        return result

    # ── Alerts ───────────────────────────────────────────────────────────────

    @app.get("/alerts")
    async def get_alerts(limit: int = 50, _=Depends(verify_key)):
        if controller is None:
            return {"alerts": []}
        alerts = controller.alert_dispatcher.get_recent_alerts(limit=limit)
        return {
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "timestamp": a.timestamp,
                    "threat_level": a.threat_level,
                    "threat_score": a.threat_score,
                    "unknown_faces": a.unknown_faces,
                    "in_restricted_zone": a.in_restricted_zone,
                    "acknowledged": a.acknowledged,
                    "details": a.details,
                }
                for a in alerts
            ],
            "stats": controller.alert_dispatcher.get_stats(),
        }

    @app.post("/alerts/{alert_id}/ack")
    async def acknowledge_alert(alert_id: int, _=Depends(verify_key)):
        if controller is None:
            raise HTTPException(404, "No mission controller")
        ok = controller.alert_dispatcher.acknowledge(alert_id)
        if not ok:
            raise HTTPException(404, f"Alert {alert_id} not found")
        return {"acknowledged": True, "alert_id": alert_id}

    # ── Identity management ──────────────────────────────────────────────────

    class EnrollRequest(BaseModel):
        name: str
        role: str = "employee"
        access_level: int = 1
        embedding: list[float]   # 128-dim face embedding

    @app.post("/enroll", status_code=201)
    async def enroll_identity(req: EnrollRequest, _=Depends(verify_key)):
        if controller is None:
            raise HTTPException(503, "No mission controller")
        if len(req.embedding) != 128:
            raise HTTPException(400, "Embedding must be 128-dimensional")
        import numpy as np
        emb = np.array(req.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        identity_id = controller.db.enroll(req.name, req.role, req.access_level, emb)
        return {"identity_id": identity_id, "name": req.name, "enrolled": True}

    @app.get("/identities")
    async def list_identities(_=Depends(verify_key)):
        if controller is None:
            return {"identities": []}
        identities = controller.db.get_all_active()
        return {
            "identities": [
                {"id": i.id, "name": i.name, "role": i.role, "access_level": i.access_level}
                for i in identities
            ],
            "count": len(identities),
        }

    @app.post("/identities/{identity_id}/deactivate")
    async def deactivate_identity(identity_id: int, _=Depends(verify_key)):
        if controller is None:
            raise HTTPException(503, "No mission controller")
        controller.db.deactivate(identity_id)
        return {"deactivated": True, "identity_id": identity_id}

    return app


async def start_api_server(controller, settings: Settings):
    app = create_app(controller, settings.api)
    config = uvicorn.Config(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    logger.info(f"REST API starting on {settings.api.host}:{settings.api.port}")
    await server.serve()
