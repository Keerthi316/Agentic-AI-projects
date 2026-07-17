"""Structured logging for agents."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import List


# ── module-level Python logger ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ── in-memory run log (appended by agents, displayed in UI) ─────────────────
_run_log: List[dict] = []


def clear_run_log() -> None:
    _run_log.clear()


def log_event(agent: str, message: str, level: str = "info") -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent,
        "message": message,
        "level": level,
    }
    _run_log.append(entry)
    logger = get_logger(agent)
    getattr(logger, level, logger.info)(message)


def get_run_log() -> List[dict]:
    return list(_run_log)
