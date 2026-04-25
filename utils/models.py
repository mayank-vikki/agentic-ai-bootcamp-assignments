"""
Model registry — single source of truth for all LLM providers.

Add new models here. The /bootcamp skill and logged_invoke() both read from this.
Env vars are loaded from .env automatically.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model Registry ──────────────────────────────────────────────────────────
# Each entry: provider, model_id, env_var for API key, base_url (if non-default),
# and cost per token (USD) for tracking spend.

MODEL_REGISTRY = {
    "deepseek": {
        "provider": "openai_compat",  # DeepSeek uses OpenAI-compatible API
        "model_id": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "cost_per_token": {
            "input": 0.00000014,   # $0.14 / 1M tokens
            "output": 0.00000028,  # $0.28 / 1M tokens
        },
    },
    "deepseek-reasoner": {
        "provider": "openai_compat",
        "model_id": "deepseek-reasoner",
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "cost_per_token": {
            "input": 0.00000055,
            "output": 0.00000219,
        },
    },
    "openai": {
        "provider": "openai",
        "model_id": "gpt-4.1-nano",
        "env_var": "OPENAI_API_KEY",
        "base_url": None,
        "cost_per_token": {
            "input": 0.00000015,
            "output": 0.0000006,
        },
    },
    "openai-mini": {
        "provider": "openai",
        "model_id": "gpt-4.1-mini",
        "env_var": "OPENAI_API_KEY",
        "base_url": None,
        "cost_per_token": {
            "input": 0.0000004,
            "output": 0.0000016,
        },
    },
    "tavily": {
        "provider": "tavily",
        "env_var": "TAVILY_API_KEY",
        "base_url": None,
        "cost_per_token": None,  # search tool, not LLM
    },
}

DEFAULT_MODEL = "deepseek"
FALLBACK_CHAIN = ["deepseek", "openai"]  # Try in order; first that works wins


def _build_chat_model(name: str, **kwargs):
    """Build a LangChain ChatModel for a single registry entry (no fallback)."""
    from langchain_openai import ChatOpenAI

    entry = MODEL_REGISTRY.get(name)
    if not entry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    api_key = os.getenv(entry["env_var"])
    if not api_key:
        raise EnvironmentError(f"Missing env var {entry['env_var']} for model '{name}'.")

    if entry["provider"] in ("openai", "openai_compat"):
        init_args = {"model": entry["model_id"], "api_key": api_key, **kwargs}
        if entry.get("base_url"):
            init_args["base_url"] = entry["base_url"]
        return ChatOpenAI(**init_args)

    raise ValueError(f"Unsupported provider '{entry['provider']}' for model '{name}'")


def get_model(name: str | None = None, fallback: bool = True, **kwargs):
    """Return a LangChain ChatModel instance with automatic fallback.

    Args:
        name: Key from MODEL_REGISTRY. Defaults to DEFAULT_MODEL.
        fallback: If True (default), on failure try the next model in FALLBACK_CHAIN.
        **kwargs: Extra args forwarded to the ChatModel constructor.

    Returns:
        Tuple-style: the ChatModel instance. The resolved model name is stored
        as model._bootcamp_model_name for the logger to pick up.
    """
    name = name or DEFAULT_MODEL

    # Build the ordered list of models to try
    if fallback:
        chain = [name] + [m for m in FALLBACK_CHAIN if m != name]
    else:
        chain = [name]

    errors = []
    for candidate in chain:
        try:
            llm = _build_chat_model(candidate, **kwargs)
            # Quick connectivity check — a tiny call to verify the key works
            llm.invoke("hi")
            llm._bootcamp_model_name = candidate
            if candidate != name:
                print(f"[bootcamp] {name} unavailable, fell back to {candidate}")
            else:
                print(f"[bootcamp] Using model: {candidate}")
            return llm
        except Exception as e:
            errors.append((candidate, str(e)))
            continue

    # All failed
    error_summary = "\n".join(f"  - {m}: {err[:120]}" for m, err in errors)
    raise RuntimeError(f"All models failed:\n{error_summary}")


def list_models() -> dict:
    """Return the full model registry for inspection."""
    return MODEL_REGISTRY
