from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Environment variable {name} is not set")
            return os.environ[name]
        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = _expand(json.load(handle))
    dataset_path = Path(config["dataset"]["path"])
    if not dataset_path.is_absolute():
        config["dataset"]["path"] = str((config_path.parent / dataset_path).resolve())
    output_root = Path(config.get("output_root", "results"))
    if not output_root.is_absolute():
        config["output_root"] = str((config_path.parent / output_root).resolve())
    config["_config_path"] = str(config_path)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted copy safe to persist with results."""
    secret_fragments = ("api_key", "authorization", "token", "secret", "password")

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("[REDACTED]" if any(part in key.lower() for part in secret_fragments) else clean(item))
                for key, item in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(config)
