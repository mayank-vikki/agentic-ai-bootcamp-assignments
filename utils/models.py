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


def get_model(name: str | None = None, **kwargs):
    """Return a LangChain ChatModel instance for the given registry name.

    Args:
        name: Key from MODEL_REGISTRY (e.g. "deepseek", "openai"). Defaults to DEFAULT_MODEL.
        **kwargs: Extra args forwarded to the ChatModel constructor (temperature, seed, etc.)
    """
    from langchain_openai import ChatOpenAI

    name = name or DEFAULT_MODEL
    entry = MODEL_REGISTRY.get(name)
    if not entry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    api_key = os.getenv(entry["env_var"])
    if not api_key:
        raise EnvironmentError(
            f"Missing env var {entry['env_var']} for model '{name}'. "
            f"Set it in your .env file or system environment."
        )

    if entry["provider"] in ("openai", "openai_compat"):
        init_args = {
            "model": entry["model_id"],
            "api_key": api_key,
            **kwargs,
        }
        if entry.get("base_url"):
            init_args["base_url"] = entry["base_url"]
        return ChatOpenAI(**init_args)

    raise ValueError(f"Unsupported provider '{entry['provider']}' for model '{name}'")


def list_models() -> dict:
    """Return the full model registry for inspection."""
    return MODEL_REGISTRY
