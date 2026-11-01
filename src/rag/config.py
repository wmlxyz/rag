"""Load explicit configuration; secrets are read only from environment variables."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ConfigurationError


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    base_url: str
    api_key: str = field(repr=False)
    model: str = ""
    timeout_seconds: float = 60
    max_retries: int = 2


def load_model_config(path: str | Path, section: str) -> ModelConfig:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)[section]
        if not isinstance(raw, dict):
            raise ValueError("section must be an object")
        provider = raw["provider"]
        expected = "dashscope_rerank" if section == "reranker" else "openai_compatible"
        if provider != expected:
            raise ConfigurationError(f"Unsupported provider in {section}")
        values = {}
        for key in ("base_url", "api_key", "model"):
            env_name = raw[f"{key}_env"]
            if not isinstance(env_name, str) or not env_name:
                raise ValueError("invalid environment variable name")
            value = os.environ.get(env_name, "").strip()
            if not value:
                raise ConfigurationError(f"Missing environment variable: {env_name}")
            values[key] = value
        parsed = urlsplit(values["base_url"])
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigurationError(f"Invalid base URL in {section}")
        timeout = float(raw.get("timeout_seconds", 60))
        retries = raw.get("max_retries", 2)
        if not 0 < timeout <= 600 or type(retries) is not int or not 0 <= retries <= 10:
            raise ValueError("invalid timeout or retries")
        return ModelConfig(provider=provider, timeout_seconds=timeout, max_retries=retries, **values)
    except ConfigurationError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration for {section}: {type(exc).__name__}") from exc
