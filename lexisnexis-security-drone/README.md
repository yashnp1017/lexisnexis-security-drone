# LexisNexis Autonomous Security Drone

An autonomous campus security surveillance system built with Python. Combines real-time face detection and recognition, multi-sensor fusion, weighted threat scoring, and REST API alert dispatch.

**19/19 tests passing.** Demo runs without hardware — full synthetic mode included.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Camera + Sensors                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │   Camera     │    │ Motion Detector  │    │  Depth Sensor  │ │
│  │ (OpenCV/syn) │    │ (background sub) │    │  (optional)    │ │
│  └──────┬───────┘    └────────┬─────────┘    └───────┬────────┘ │
└─────────┼───────────────────────────────────────────────────────┘
          │ frame                │ motion_intensity      │
          ▼                      ▼                       │
┌─────────────────────────────────────────────────────── ┘
│               Detection Pipeline                       │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐  │
│  │ FaceDetector │──►│ FaceEmbedder │──►│ Recognizer│  │
│  │ (DNN/Haar)   │   │ (dlib 128-d) │   │ (cosine)  │  │
│  └──────────────┘   └──────────────┘   └─────┬─────┘  │
│                                               │        │
│                                        Identity DB     │
│                                        (SQLite)        │
└───────────────────────────────────────────────┬────────┘
                                                │ DetectedFace[]
                                                ▼
┌───────────────────────────────────────────────────────┐
│                  Threat Scorer                        │
│                                                       │
│  score = w_face   * face_component     (0.40)         │
│        + w_motion * motion_component   (0.30)         │
│        + w_zone   * zone_component     (0.20)         │
│        + w_time   * time_component     (0.10)         │
│                                                       │
│  Levels: CLEAR → LOW → MEDIUM → HIGH → CRITICAL       │
└────────────────────────────┬──────────────────────────┘
                             │ ThreatAssessment
                             ▼
┌────────────────────────────────────────────────────────┐
│  Alert Dispatcher            REST API (FastAPI)        │
│  - Webhook dispatch          GET  /status              │
│  - Deduplication             GET  /alerts              │
│  - Exponential backoff       POST /enroll              │
│  - In-memory alert log       POST /alerts/{id}/ack     │
└────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Face recognition | 128-dim dlib embeddings | 99.38% LFW accuracy; same model used by face_recognition library |
| Similarity metric | Cosine similarity (unit-normalized dot product) | Invariant to embedding magnitude; fast O(d) per comparison |
| Threat scoring | Weighted linear combination | Interpretable, auditable, easily tunable by security team |
| Motion detection | Exponential moving average background | Adapts to slow lighting changes; robust to gradual environment drift |
| Alert deduplication | 30s cooldown window per threat level | Prevents webhook flooding while ensuring unique escalations are dispatched |
| Async architecture | asyncio + FastAPI | Camera loop and REST API run concurrently without blocking |
| Hardware fallback | Synthetic mode | Full system testable without camera, face_recognition lib, or dlib |

---

## Performance

| Metric | Value |
|---|---|
| Face detection (DNN, GPU) | ~25ms/frame |
| Face recognition (128-dim) | ~15ms/face |
| Full pipeline (synthetic) | ~2ms/frame |
| Threat score computation | <1ms |
| Test suite | 19 tests in 0.27s |

---

## Quickstart

### Prerequisites
- Python 3.11+
- No hardware required for demo mode

### Install

```bash
git clone https://github.com/yashnp1017/lexisnexis-security-drone
cd lexisnexis-security-drone
pip install -r requirements.txt
```

### Run demo (no hardware)

```bash
python main.py --mode demo
```

Output:
```
🟢 Frame 00 [QUIET]            | Threat: CLEAR    (0.068) | Faces: 1 (0 unknown)
🟢 Frame 30 [UNKNOWN DETECTED] | Threat: LOW      (0.486) | Faces: 1 (1 unknown)
🟡 Frame 60 [ZONE VIOLATION]   | Threat: HIGH     (0.735) | Faces: 1 (1 unknown)
```

### Run with REST API

```bash
python main.py --mode api
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Run tests

```bash
pytest -v
```

---

## API Reference

### `GET /status`
```json
{
  "state": "PATROLLING",
  "frame_count": 1240,
  "alerts_dispatched": 3,
  "uptime_seconds": 82.4,
  "last_assessment": {
    "score": 0.231,
    "level": "CLEAR",
    "unknown_faces": 0,
    "total_faces": 1
  }
}
```

### `GET /alerts?limit=20`
```json
{
  "alerts": [
    {
      "alert_id": 1,
      "timestamp": "2026-05-28T02:31:00",
      "threat_level": "HIGH",
      "threat_score": 0.735,
      "unknown_faces": 1,
      "in_restricted_zone": true,
      "acknowledged": false
    }
  ]
}
```

### `POST /enroll`
```json
{
  "name": "Jane Doe",
  "role": "engineer",
  "access_level": 1,
  "embedding": [0.023, -0.114, ...]   // 128-dim float array
}
```

---

## Project Structure

```
lexisnexis-security-drone/
├── main.py                          # Entry point
├── config/
│   ├── settings.py                  # Pydantic settings
│   └── settings.yaml                # Default config
├── src/
│   ├── detection/
│   │   ├── face_pipeline.py         # Detect + embed + recognize
│   │   ├── identity_db.py           # SQLite identity store
│   │   └── threat_scorer.py         # Multi-factor threat scoring
│   ├── sensors/
│   │   ├── camera.py                # Camera capture
│   │   └── motion_detector.py       # Background subtraction
│   ├── mission/
│   │   ├── controller.py            # Main mission loop + state machine
│   │   └── demo.py                  # Synthetic 90-frame demo
│   ├── api/
│   │   └── server.py                # FastAPI REST server
│   └── utils/
│       └── alert_dispatcher.py      # Alert dispatch + deduplication
└── tests/
    └── test_system.py               # 19 tests
```

---

## Configuration

Edit `config/settings.yaml`:

```yaml
threat:
  alert_threshold: 0.65      # score >= this → dispatch alert
  critical_threshold: 0.85   # score >= this → critical alert
  restricted_zones:           # (x1, y1, x2, y2) in pixels
    - [0, 0, 200, 480]

sensor:
  synthetic_mode: true        # set false for real camera
  camera_index: 0

api:
  port: 8000
  alert_webhook_url: "https://your-siem.example.com/webhook"
  api_key: "your-secret-key"
```

---

## Tech Stack

- **FastAPI** — async REST API
- **OpenCV** — face detection (DNN / Haar cascade), motion detection
- **face_recognition / dlib** — 128-dim ResNet face embeddings (production)
- **NumPy** — vector math, cosine similarity
- **SQLite** — identity database
- **asyncio** — concurrent camera loop + API server
- **pytest** — 19-test async test suite

---

## Author

**Yash Patel** — Purdue University, B.S. Computer Science & AI (May 2027)

[LinkedIn](https://linkedin.com/in/yash-patel-018291278) · [GitHub](https://github.com/yashnp1017)
