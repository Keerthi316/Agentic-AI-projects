"""
observability/llm_logger.py — LLM Call Logger

Records every LLM invocation to a JSONL log file with:
  - timestamp, model, prompt_version
  - input/output token counts (estimated via tiktoken)
  - latency (seconds)
  - estimated cost (USD)
  - success / failure status + error message
  - question, answer snippet, routing, confidence

Thread-safe: uses a file lock so concurrent Streamlit reruns
don't corrupt the log file.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Cost table (USD per 1 000 tokens) ────────────────────────────────────────
# Source: OpenRouter pricing as of 2026-07.
# Extend this dict when you add new models.
_COST_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o":       {"input": 0.002500, "output": 0.010000},
    "gpt-3.5-turbo": {"input": 0.000050, "output": 0.000150},
}

# Fallback cost when model is not in the table
_DEFAULT_COST = {"input": 0.000200, "output": 0.000800}

# Where logs are stored (relative to project root)
LOG_DIR = Path("observability_logs")
LOG_FILE = LOG_DIR / "llm_calls.jsonl"

# Thread lock for safe multi-threaded file writes
_write_lock = threading.Lock()


def _estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate the number of tokens in *text* using tiktoken.

    Falls back to a character-based heuristic (chars / 4) if tiktoken
    is not installed or the model encoding is unknown.

    Args:
        text:  The text to tokenise.
        model: The model name (used to pick the right encoding).

    Returns:
        Estimated token count.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough heuristic: ~4 chars per token
        return max(1, len(text) // 4)


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Estimate the USD cost for one LLM call.

    Args:
        input_tokens:  Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        model:         Model name string.

    Returns:
        Estimated cost in USD (rounded to 8 decimal places).
    """
    # Normalise model name: strip provider prefix like "openai/"
    short_model = model.split("/")[-1] if "/" in model else model
    rates = _COST_TABLE.get(short_model, _DEFAULT_COST)
    cost = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
    return round(cost, 8)


class LLMLogger:
    """
    Singleton-style logger for all LLM calls.

    Usage::

        logger = LLMLogger()

        call_id = logger.start_call(
            question="...",
            model="gpt-4o-mini",
            prompt_version="v1",
            input_text="full prompt text",
        )
        # ... invoke the LLM ...
        logger.end_call(
            call_id=call_id,
            output_text="response text",
            success=True,
        )

    Or use the convenience wrapper::

        with logger.log_call(question, model, prompt_version, input_text) as call_id:
            # do LLM work
            logger.set_output(call_id, output_text, success=True)
    """

    def __init__(self, log_file: Optional[Path] = None) -> None:
        self._log_file = Path(log_file) if log_file else LOG_FILE
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        # In-memory store: call_id → metadata (lives until end_call)
        self._pending: Dict[str, Dict[str, Any]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def start_call(
        self,
        question: str,
        model: str,
        prompt_version: str = "v1",
        input_text: str = "",
    ) -> str:
        """
        Record the start of an LLM call.

        Args:
            question:       The user's original question.
            model:          Model identifier string.
            prompt_version: A/B prompt variant being used ("v1" or "v2").
            input_text:     The full prompt text sent to the LLM.

        Returns:
            call_id: A unique string you pass to end_call().
        """
        call_id = f"{time.time_ns()}"
        self._pending[call_id] = {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "prompt_version": prompt_version,
            "question": question[:500],           # cap at 500 chars for privacy
            "input_tokens": _estimate_tokens(input_text, model),
            "input_text_len": len(input_text),
            "start_time": time.time(),
        }
        return call_id

    def end_call(
        self,
        call_id: str,
        output_text: str = "",
        success: bool = True,
        error_message: str = "",
        routing: str = "RAG",
        confidence: float = 0.0,
        citations: Optional[list] = None,
        refusal: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Finalise and persist an LLM call record.

        Args:
            call_id:       The ID returned by start_call().
            output_text:   The LLM's response text.
            success:       True if the call succeeded.
            error_message: Error description on failure.
            routing:       Routing path used (e.g. "RAG", "Tool only").
            confidence:    Retrieval confidence score (0–1).
            citations:     List of citation strings extracted from the answer.
            refusal:       True if the LLM refused to answer (safety boundary).
            extra:         Any additional key/value pairs to attach to the record.

        Returns:
            The completed log record as a dict.
        """
        record = self._pending.pop(call_id, {})
        if not record:
            logger.warning(f"LLMLogger.end_call: unknown call_id {call_id}")
            record = {"call_id": call_id}

        end_time = time.time()
        start_time = record.get("start_time", end_time)
        latency = round(end_time - start_time, 4)

        input_tokens = record.get("input_tokens", 0)
        output_tokens = _estimate_tokens(output_text, record.get("model", "gpt-4o-mini"))
        cost = _estimate_cost(input_tokens, output_tokens, record.get("model", "gpt-4o-mini"))

        record.update(
            {
                "latency_seconds": latency,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "estimated_cost_usd": cost,
                "success": success,
                "error_message": error_message,
                "routing": routing,
                "confidence": round(confidence, 4),
                "citations": citations or [],
                "refusal": refusal,
                "answer_snippet": output_text[:300],   # first 300 chars only
            }
        )

        # Remove internal tracking fields before writing
        record.pop("start_time", None)
        record.pop("input_text_len", None)

        if extra:
            record.update(extra)

        self._write(record)
        return record

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write(self, record: Dict[str, Any]) -> None:
        """Append a JSON record to the JSONL log file (thread-safe)."""
        with _write_lock:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.error(f"LLMLogger: could not write to {self._log_file}: {exc}")

    def read_all(self) -> list[Dict[str, Any]]:
        """
        Read all persisted log records.

        Returns:
            List of log record dicts (most recent last).
        """
        records: list[Dict[str, Any]] = []
        if not self._log_file.exists():
            return records
        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records


# Module-level singleton — import and reuse this everywhere
llm_logger = LLMLogger()
