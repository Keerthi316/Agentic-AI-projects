"""
SQLite database layer.
Schema includes final_status for post-approval states.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional

from config import DB_PATH


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and migrate schema if needed."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id          TEXT PRIMARY KEY,
            filename        TEXT,
            candidate_name  TEXT,
            overall_score   REAL,
            recommendation  TEXT,
            explanation     TEXT,
            skills          TEXT,
            experience      TEXT,
            education       TEXT,
            timestamp       TEXT,
            approved        INTEGER,
            score_breakdown TEXT,
            final_status    TEXT,
            finalized_at    TEXT,
            workflow_completed INTEGER DEFAULT 0
        )
        """
    )
    # Migrate: add columns if they don't exist yet (for existing DBs)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    migrations = {
        "final_status":         "ALTER TABLE runs ADD COLUMN final_status TEXT",
        "finalized_at":         "ALTER TABLE runs ADD COLUMN finalized_at TEXT",
        "workflow_completed":   "ALTER TABLE runs ADD COLUMN workflow_completed INTEGER DEFAULT 0",
    }
    for col, sql in migrations.items():
        if col not in existing:
            conn.execute(sql)
    conn.commit()
    conn.close()


def save_run(
    filename: str,
    candidate_name: str,
    overall_score: float,
    recommendation: str,
    explanation: str,
    skills: List[str],
    experience: List[str],
    education: List[str],
    score_breakdown: dict | None = None,
    approved: Optional[bool] = None,
) -> str:
    run_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO runs
          (run_id, filename, candidate_name, overall_score, recommendation,
           explanation, skills, experience, education, timestamp, approved,
           score_breakdown, final_status, finalized_at, workflow_completed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, filename, candidate_name, overall_score, recommendation,
            explanation,
            json.dumps(skills), json.dumps(experience), json.dumps(education),
            timestamp,
            int(approved) if approved is not None else None,
            json.dumps(score_breakdown or {}),
            None, None, 0,
        ),
    )
    conn.commit()
    conn.close()
    return run_id


def finalize_approval(run_id: str, approved: bool) -> None:
    """
    Complete the human approval step.
    approved=True  → final_status = 'Interview Finalized'
    approved=False → final_status = 'Rejected by Human Reviewer'
    """
    final_status = "Interview Finalized" if approved else "Rejected by Human Reviewer"
    finalized_at = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        """
        UPDATE runs
           SET approved=?, final_status=?, finalized_at=?, workflow_completed=1
         WHERE run_id=?
        """,
        (int(approved), final_status, finalized_at, run_id),
    )
    conn.commit()
    conn.close()


# Keep legacy name for backward compat
def update_approval(run_id: str, approved: bool) -> None:
    finalize_approval(run_id, approved)


def get_all_runs() -> List[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["skills"]         = json.loads(d["skills"] or "[]")
        d["experience"]     = json.loads(d["experience"] or "[]")
        d["education"]      = json.loads(d["education"] or "[]")
        d["score_breakdown"]= json.loads(d["score_breakdown"] or "{}")
        result.append(d)
    return result


def get_run_by_id(run_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["skills"]         = json.loads(d["skills"] or "[]")
    d["experience"]     = json.loads(d["experience"] or "[]")
    d["education"]      = json.loads(d["education"] or "[]")
    d["score_breakdown"]= json.loads(d["score_breakdown"] or "{}")
    return d


def delete_run(run_id: str) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
    conn.commit()
    conn.close()


def get_stats() -> dict:
    conn = _get_conn()
    total     = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    interview = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE recommendation='Interview' OR final_status='Interview Finalized'"
    ).fetchone()[0]
    rejected  = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE recommendation='Reject' OR final_status='Rejected by Human Reviewer'"
    ).fetchone()[0]
    pending   = conn.execute(
        """SELECT COUNT(*) FROM runs
           WHERE (recommendation IN ('Need Human Review','Hold')
                  OR (recommendation IN ('Interview','Reject') AND approved IS NULL))
             AND workflow_completed=0"""
    ).fetchone()[0]
    avg_score = conn.execute("SELECT AVG(overall_score) FROM runs").fetchone()[0] or 0
    conn.close()
    return {
        "total":     total,
        "interview": interview,
        "rejected":  rejected,
        "pending":   pending,
        "avg_score": round(avg_score, 1),
    }
