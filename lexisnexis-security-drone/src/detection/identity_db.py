"""
Identity database backed by SQLite.

Stores enrolled persons with their name, role, access level,
and a serialized face embedding vector for recognition.
"""

import sqlite3
import pickle
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Identity:
    id: int
    name: str
    role: str
    access_level: int      # 0 = no access, 1 = standard, 2 = restricted, 3 = all-access
    embedding: np.ndarray  # 128-dim face embedding
    active: bool = True


class IdentityDatabase:
    """SQLite-backed store for enrolled identity embeddings."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        logger.info(f"Identity database opened: {db_path}")

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS identities (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL,
                role         TEXT NOT NULL DEFAULT 'employee',
                access_level INTEGER NOT NULL DEFAULT 1,
                embedding    BLOB NOT NULL,
                active       INTEGER NOT NULL DEFAULT 1,
                enrolled_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON identities(active)")
        self.conn.commit()

    def enroll(self, name: str, role: str, access_level: int,
               embedding: np.ndarray) -> int:
        """Add a new identity. Returns the row id."""
        blob = pickle.dumps(embedding)
        cursor = self.conn.execute(
            "INSERT INTO identities (name, role, access_level, embedding) VALUES (?,?,?,?)",
            (name, role, access_level, blob),
        )
        self.conn.commit()
        logger.info(f"Enrolled identity: {name} (role={role}, access={access_level})")
        return cursor.lastrowid

    def get_all_active(self) -> List[Identity]:
        rows = self.conn.execute(
            "SELECT id, name, role, access_level, embedding, active FROM identities WHERE active=1"
        ).fetchall()
        return [
            Identity(
                id=r[0], name=r[1], role=r[2],
                access_level=r[3], embedding=pickle.loads(r[4]), active=bool(r[5])
            )
            for r in rows
        ]

    def deactivate(self, identity_id: int):
        self.conn.execute("UPDATE identities SET active=0 WHERE id=?", (identity_id,))
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM identities WHERE active=1").fetchone()[0]

    def close(self):
        self.conn.close()
