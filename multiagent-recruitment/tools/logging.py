"""
Logging configuration for the recruitment system.

Design decisions:
1. Centralized logging setup with consistent formatting across all agents.
2. Each agent gets its own logger (via __name__) for traceability.
3. Supports structured logging with correlation IDs for debugging workflows.
"""

import logging
import sys
from datetime import datetime, timezone


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging for the recruitment system.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get a logger for a specific agent.

    Args:
        agent_name: The name of the agent (e.g., 'ResumeAnalyst').

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"agent.{agent_name}")
    logger.setLevel(logging.INFO)
    return logger


def log_agent_action(
    logger: logging.Logger,
    action: str,
    details: dict | None = None,
    level: str = "INFO",
) -> None:
    """Log an agent action with optional structured details.

    Args:
        logger: The agent's logger instance.
        action: Description of the action (e.g., "Parsing resume").
        details: Optional dict with structured data (e.g., candidate_id, score).
        level: Log level string ("INFO", "WARNING", "ERROR", "DEBUG").
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    message = f"[{action}]"
    if details:
        # Format details as key=value pairs for readability
        detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
        message += f" {detail_str}"

    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)