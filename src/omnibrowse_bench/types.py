from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    sample_id: str
    question: str
    row: dict[str, str]
    csv_dir: Path
    video_path: Path | None = None
    audio_path: Path | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    text: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ToolResult:
    observation: dict[str, Any]
    raw: Any
    media: list[dict[str, str]] = field(default_factory=list)
