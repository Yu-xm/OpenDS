from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, field: str = "config") -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"{field} requires environment variable {name}, which is not set")
            return os.environ[name]
        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _expand(item, f"{field}.{key}") for key, item in value.items()}
    return value


def _load_document(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            document = yaml.safe_load(handle)
        elif config_path.suffix.lower() == ".json":
            document = json.load(handle)
        else:
            raise ValueError("Config file must use .yaml, .yml, or .json")
    if not isinstance(document, dict):
        raise ValueError("Config root must be a mapping")
    return document


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    document = _load_document(config_path)
    for field in ("dataset", "model", "search"):
        if field in document and not isinstance(document[field], dict):
            raise ValueError(f"{field} must be a mapping ({config_path})")
    for field in ("agent", "toolbox"):
        if field in document and not isinstance(document[field], dict):
            raise ValueError(f"{field} must be a mapping ({config_path})")
    if "dataset" not in document or not isinstance(document["dataset"].get("path"), str) or not document["dataset"]["path"].strip():
        raise ValueError("dataset.path must be a non-empty path")
    for field in ("finalization_grace_seconds", "final_answer_reserve_steps"):
        if field in (document.get("agent") or {}):
            raise ValueError(f"agent.{field} is obsolete; use finalization_max_tokens and finalization_answer_tokens")
    # Optional converters must never make the whole evaluation config unusable.
    readers = (document.get("toolbox") or {}).get("file_readers")
    if isinstance(readers, dict):
        import warnings
        for kind in ("asr", "mineru"):
            try:
                section = readers.get(kind) or {}
                if not isinstance(section, dict):
                    raise ValueError("reader config must be a mapping")
                for key in ("headers", "parameters", *(["docker"] if kind == "mineru" else [])):
                    if section.get(key) is not None and not isinstance(section[key], dict):
                        raise ValueError(f"reader {key} must be a mapping")
                readers[kind] = _expand(section, f"toolbox.file_readers.{kind}")
            except (TypeError, ValueError) as exc:
                if (readers.get(kind) or {}).get("enabled"):
                    raise ValueError(f"toolbox.file_readers.{kind}: {exc}") from exc
                warnings.warn(f"{kind} reader has invalid configuration; disabled: {exc}", RuntimeWarning)
                readers[kind] = {"enabled": False}
    config = _expand(document)
    dataset_path = Path(config["dataset"]["path"])
    if not dataset_path.is_absolute():
        config["dataset"]["path"] = str((config_path.parent / dataset_path).resolve())
    input_video_dir = config["dataset"].get("input_video_dir")
    if (
        config["dataset"].get("input_video") is not False
        and input_video_dir
        and not Path(input_video_dir).is_absolute()
    ):
        config["dataset"]["input_video_dir"] = str(
            (config_path.parent / input_video_dir).resolve()
        )
    input_video_subtitles = config["dataset"].get("input_video_subtitles")
    if input_video_subtitles and not Path(str(input_video_subtitles)).is_absolute():
        config["dataset"]["input_video_subtitles"] = str(
            (config_path.parent / str(input_video_subtitles)).resolve()
        )
    output_root = Path(config.get("output_root", "results"))
    if not output_root.is_absolute():
        config["output_root"] = str((config_path.parent / output_root).resolve())
    toolbox = config.get("toolbox")
    if isinstance(toolbox, dict):
        for key in ("youtube_cache_dir", "youtube_cookie_file"):
            value = toolbox.get(key)
            if value and not Path(value).is_absolute():
                toolbox[key] = str((config_path.parent / value).resolve())
        readers = toolbox.setdefault("file_readers", {})
        if not isinstance(readers, dict):
            raise ValueError("toolbox.file_readers must be a mapping")
        readers.setdefault("cache_dir", ".cache/toolbox/files")
        if not Path(readers["cache_dir"]).is_absolute():
            readers["cache_dir"] = str((config_path.parent / readers["cache_dir"]).resolve())
        docker = (readers.get("mineru") or {}).get("docker") or {}
        try:
            for key in ("data_dir", "pipeline_models_dir", "vlm_models_dir"):
                if docker.get(key) is not None:
                    value = Path(docker[key])
                    if not value.is_absolute():
                        docker[key] = str((config_path.parent / value).resolve())
        except (TypeError, ValueError):
            import warnings
            warnings.warn("mineru reader has invalid model/data paths; disabled", RuntimeWarning)
            readers["mineru"] = {"enabled": False}
    config["_config_path"] = str(config_path)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted copy safe to persist with results."""
    secret_fragments = (
        "api_key", "authorization", "token", "secret", "password", "cookie"
    )
    public_budget_keys = {
        "max_tokens", "max_completion_tokens", "max_output_tokens",
        "finalization_max_tokens", "finalization_answer_tokens", "thinking_budget", "min_tokens", "token_budget",
    }

    def is_secret(key: str) -> bool:
        name = key.lower()
        return name == "headers" or (
            name not in public_budget_keys and any(part in name for part in secret_fragments)
        )

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("[REDACTED]" if is_secret(key) else clean(item))
                for key, item in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(config)
