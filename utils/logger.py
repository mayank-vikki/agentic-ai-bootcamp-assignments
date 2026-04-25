"""
Structured JSON logger for LLM calls.

Adapted from Manifold AI Week-1 d06_structured_logging.py — made model-agnostic
and writes to both console and a per-session JSONL file in logs/.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from utils.models import MODEL_REGISTRY, DEFAULT_MODEL

# ── Setup ────────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bootcamp")


def _get_cost(model_name: str, input_tokens: int, output_tokens: int) -> dict:
    """Calculate cost from the model registry."""
    entry = MODEL_REGISTRY.get(model_name, {})
    costs = entry.get("cost_per_token")
    if not costs:
        return {"input_cost": 0, "output_cost": 0, "total_cost": 0}
    ic = input_tokens * costs["input"]
    oc = output_tokens * costs["output"]
    return {"input_cost": round(ic, 8), "output_cost": round(oc, 8), "total_cost": round(ic + oc, 8)}


def _write_log(session_id: str, record: dict):
    """Append a JSON record to the session log file."""
    log_file = LOGS_DIR / f"{session_id}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Main function ────────────────────────────────────────────────────────────

def logged_invoke(
    llm,
    messages,
    model_name: str | None = None,
    session_id: str | None = None,
):
    """Invoke an LLM with structured logging (console + file).

    Args:
        llm: A LangChain ChatModel instance (from get_model()).
        messages: List of LangChain message objects or a single string.
        model_name: Registry key (for cost lookup). Auto-detected if None.
        session_id: UUID for grouping calls. Auto-generated if None.

    Returns:
        The AIMessage response object.
    """
    from langchain_core.messages import HumanMessage

    session_id = session_id or str(uuid.uuid4())

    # Allow passing a plain string as a convenience
    if isinstance(messages, str):
        messages = [HumanMessage(content=messages)]

    # Try to detect model name from registry
    if not model_name:
        model_name = DEFAULT_MODEL

    start = datetime.now()
    log_start = {
        "event": "llm_call_start",
        "session_id": session_id,
        "timestamp": start.isoformat(),
        "model": model_name,
        "input_preview": str(messages[-1].content)[:200],
    }
    logger.info(json.dumps(log_start))
    _write_log(session_id, log_start)

    try:
        response = llm.invoke(messages)
        latency_s = (datetime.now() - start).total_seconds()

        # Extract token usage if available
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
        output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
        cost = _get_cost(model_name, input_tokens, output_tokens)

        log_success = {
            "event": "llm_call_success",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "latency_seconds": round(latency_s, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **cost,
            "response_preview": str(response.content)[:300],
        }
        logger.info(json.dumps(log_success))
        _write_log(session_id, log_success)

        return response

    except Exception as e:
        latency_s = (datetime.now() - start).total_seconds()
        log_error = {
            "event": "llm_call_error",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "latency_seconds": round(latency_s, 3),
            "error": str(e),
        }
        logger.error(json.dumps(log_error))
        _write_log(session_id, log_error)
        raise
