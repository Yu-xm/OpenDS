from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .trace import save_artifact
from .types import ToolArgumentError, ToolInfrastructureError, ToolResult


_BROWSER_FAILURE_PAGE = re.compile(
    r"(?im)^(?:#{1,3}\s*)?(?:401\s+unauthorized|403\s+forbidden|"
    r"407\s+proxy authentication required|429\s+too many requests|"
    r"50[0-9]\s+(?:bad gateway|service unavailable|gateway timeout|internal server error)|"
    r"http error (?:401|403|407|429|50[0-9])|access denied|request blocked|"
    r"proxy error|this site can.t be reached|err_[a-z0-9_]+)\b|"
    r"(?:captcha|verify you are human|checking your browser before accessing|"
    r"sorry, you have been blocked)"
)


def _checked_browser_text(value: Any, operation: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ToolInfrastructureError(f"{operation} returned an empty browser response")
    match = _BROWSER_FAILURE_PAGE.search(text[:4000])
    if match:
        raise ToolInfrastructureError(
            f"{operation} returned a blocked or network error page: {match.group(0)}"
        )
    return text


def _checked_navigation(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        searchable = "\n".join(
            str(value.get(key) or "") for key in ("content", "message", "title")
        ).strip()
        if searchable:
            _checked_browser_text(searchable, "browser_navigate")
        elif value.get("action") != "download":
            raise ToolInfrastructureError("browser_navigate returned no page or download result")
        return value
    return {"content": _checked_browser_text(value, "browser_navigate")}


def _definition(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
    }
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLBOX_TOOL_DEFINITIONS = [
    _definition(
        "browser_navigate",
        "Open a public URL in the browser. Returns the page snapshot, or a completed download's relative path for browser_get_download_content.",
        {"url": {"type": "string", "description": "Full public HTTP(S) URL."}},
        ["url"],
    ),
    _definition(
        "browser_get_markdown",
        "Return readable Markdown for the current browser page.",
    ),
    _definition(
        "browser_get_clickable_elements",
        "Return current interactive elements with numeric indexes for clicking or filling.",
    ),
    _definition(
        "browser_click",
        "Click an element from the current page snapshot. Returns the result, or a completed download's relative path for browser_get_download_content.",
        {"index": {"type": "integer", "description": "Index from the current page snapshot."}},
        ["index"],
    ),
    _definition(
        "browser_form_input_fill",
        "Fill one indexed input or textarea on the current page.",
        {
            "index": {"type": "integer", "description": "Current input element index."},
            "value": {"type": "string", "description": "Text to enter."},
            "clear": {"type": "boolean", "description": "Clear existing text first."},
        },
        ["index", "value"],
    ),
    _definition(
        "browser_press_key",
        "Press a keyboard key such as Enter, Escape, or ArrowDown on the current page.",
        {"key": {"type": "string", "description": "Playwright keyboard key name."}},
        ["key"],
    ),
    _definition(
        "browser_scroll",
        "Scroll the current page; positive is down, negative is up, and omission means bottom.",
        {"amount": {"type": "integer", "description": "Optional signed pixel amount."}},
    ),
    _definition(
        "browser_screenshot",
        "Capture the current browser page and return it as a visual observation.",
        {"name": {"type": "string", "description": "Optional artifact label."}},
    ),
    _definition(
        "browser_get_download_list",
        "List one Downloads directory. Paths are task-local and relative; omission means the root.",
        {"path": {"type": "string", "description": "Optional relative directory path."}},
    ),
    _definition(
        "browser_get_download_content",
        "Read a downloaded file when you need its contents as readable text or images. Provide query to find matching text, or line coordinates to read a specific portion.",
        {
            "path": {"type": "string", "description": "Relative file path returned by the download list."},
            "query": {"type": "string", "minLength": 1, "description": "Ripgrep-compatible regex pattern. Searches only the specified file. Cannot be combined with line coordinates."},
            "start_line": {"type": "integer", "minimum": 1, "description": "First line to read, inclusive. Line numbers start at 1."},
            "end_line": {"type": "integer", "minimum": 1, "description": "Last line to read, inclusive. Requires start_line and must be at least start_line."},
            "start_column": {"type": "integer", "minimum": 1, "description": "Character position in the first requested line, starting at 1. Requires start_line. Defaults to 1."},
        },
        ["path"],
    ),
    _definition(
        "youtube_search",
        "Search YouTube and return video IDs, titles, URLs, and durations.",
        {
            "query": {"type": "string", "description": "YouTube search query."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        ["query"],
    ),
    _definition(
        "youtube_read_video",
        "Sample video frames without reading subtitles. Use the reserved ID input_video only for the task-provided local video announced in the initial prompt; use 11-character IDs returned by youtube_search for online videos. Local input_video reads the complete read-only file without YouTube or network access. Omit both start_seconds and end_seconds for a global overview; provide both for a local range. The end is clamped to video duration.",
        {
            "url_or_id": {"type": "string", "description": "The reserved input_video ID, one YouTube URL, or one 11-character search-result video ID."},
            "num_frames": {"type": "integer", "minimum": 1, "maximum": 100},
            "start_seconds": {"type": "number", "minimum": 0},
            "end_seconds": {"type": "number", "exclusiveMinimum": 0},
        },
        ["url_or_id"],
    ),
    _definition(
        "youtube_read_subtitle",
        "Read timestamped subtitles when you need to know what is said in a video or during a specific time range. Omit both time parameters to read from the beginning, or provide both to select a range.",
        {
            "url_or_id": {"type": "string", "description": "The reserved input_video ID, one YouTube URL, or one 11-character search-result video ID."},
            "start_seconds": {"type": "number", "minimum": 0, "description": "Start of the requested time range, in seconds. Provide together with end_seconds."},
            "end_seconds": {"type": "number", "exclusiveMinimum": 0, "description": "End of the requested time range, in seconds. Must exceed start_seconds; values beyond the video duration are clamped."},
        },
        ["url_or_id"],
    ),
]

TOOLBOX_TOOL_NAMES = frozenset(
    definition["function"]["name"] for definition in TOOLBOX_TOOL_DEFINITIONS
)


def toolbox_task_id(
    run_id: str, benchmark: str, model_name: str, sample_id: str
) -> str:
    identity = "\0".join((run_id, benchmark, model_name, sample_id))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    prefix = "".join(
        character if character.isalnum() or character in "_.-" else "-"
        for character in sample_id
    ).strip(".-")[:48]
    return f"{prefix or 'sample'}-{digest}"


def create_toolbox(
    task_id: str,
    config: dict[str, Any],
    artifact_dir: Path,
    *,
    input_video_id: str | None = None,
    input_video_path: Path | None = None,
    input_video_sha256: str | None = None,
    input_video_subtitle: str | None = None,
) -> Any:
    """Import optional container dependencies only when Toolbox is enabled."""
    from toolbox import Toolbox

    arguments: dict[str, Any] = {
        "data_dir": artifact_dir / "toolbox",
        "browser_image": str(config.get("browser_image", "toolbox-browser:pdf-v1")),
        "youtube_image": str(config.get("youtube_image", "toolbox-youtube:latest")),
        "docker_command": str(config.get("docker_command", "docker")),
        "input_video_id": input_video_id,
        "input_video_path": input_video_path,
        "input_video_sha256": input_video_sha256,
        "input_video_subtitle": input_video_subtitle,
    }
    for name in (
        "youtube_cache_dir",
        "youtube_cookie_file",
        "proxy_server",
        "container_network",
        "docker_host",
        "browser_navigation_attempts",
        "file_readers",
        "subtitle_max_entries",
    ):
        if config.get(name) is not None:
            arguments[name] = config[name]
    return Toolbox(task_id, **arguments)


class ToolboxToolAdapter:
    def __init__(self, toolbox: Any, artifact_dir: Path) -> None:
        self.toolbox = toolbox
        self.artifact_dir = artifact_dir

    def execute(
        self, name: str, arguments: dict[str, Any], call_id: str
    ) -> ToolResult:
        if name == "browser_navigate":
            url = self._string_argument(arguments, "url")
            if not url.startswith(("http://", "https://")):
                raise ToolArgumentError(
                    "browser_navigate requires a public HTTP(S) URL"
                )
            value = self.toolbox.browser.navigate(url)
            observation = _checked_navigation(value)
            return ToolResult(observation, observation)
        if name == "browser_get_markdown":
            observation = {
                "content": _checked_browser_text(
                    self.toolbox.browser.get_markdown(), "browser_get_markdown"
                )
            }
            return ToolResult(observation, observation)
        if name == "browser_get_clickable_elements":
            elements = self.toolbox.browser.get_clickable_elements()
            _checked_browser_text(
                self.toolbox.browser.get_markdown(),
                "browser_get_clickable_elements",
            )
            observation = {"elements": elements}
            return ToolResult(observation, observation)
        if name == "browser_click":
            try:
                observation = self.toolbox.browser.click(
                    self._integer_argument(arguments, "index")
                )
            except ValueError as exc:
                raise ToolArgumentError(str(exc)) from exc
            if observation.get("action") != "download":
                _checked_browser_text(
                    self.toolbox.browser.get_markdown(), "browser_click"
                )
            return ToolResult(observation, observation)
        if name == "browser_form_input_fill":
            try:
                content = self.toolbox.browser.form_input_fill(
                    self._integer_argument(arguments, "index"),
                    self._string_argument(arguments, "value", allow_empty=True),
                    clear=bool(arguments.get("clear", False)),
                )
            except ValueError as exc:
                raise ToolArgumentError(str(exc)) from exc
            observation = {"content": content}
            return ToolResult(observation, observation)
        if name == "browser_press_key":
            observation = {
                "content": self.toolbox.browser.press_key(
                    self._string_argument(arguments, "key")
                )
            }
            _checked_browser_text(
                self.toolbox.browser.get_markdown(), "browser_press_key"
            )
            return ToolResult(observation, observation)
        if name == "browser_scroll":
            amount = arguments.get("amount")
            observation = {
                "content": self.toolbox.browser.scroll(
                    None
                    if amount is None
                    else self._integer_argument(arguments, "amount")
                )
            }
            return ToolResult(observation, observation)
        if name == "browser_screenshot":
            return self._screenshot(str(arguments.get("name", "agent-page")), call_id)
        if name == "browser_get_download_list":
            path = arguments.get("path")
            try:
                items = self.toolbox.browser.get_download_list(
                    None if path is None else str(path)
                )
            except ValueError as exc:
                raise ToolArgumentError(str(exc)) from exc
            observation = {"path": path, "items": items}
            return ToolResult(observation, observation)
        if name == "browser_get_download_content":
            path = self._string_argument(arguments, "path")
            query = self._string_argument(arguments, "query") if "query" in arguments else None
            coordinates = {key: self._positive_integer_argument(arguments, key)
                           for key in ("start_line", "end_line", "start_column") if key in arguments}
            if query is not None and coordinates:
                raise ToolArgumentError("query cannot be combined with line coordinates")
            if coordinates and "start_line" not in coordinates:
                raise ToolArgumentError("line coordinates require start_line")
            return self._download_content(path, call_id, query, **coordinates)
        if name == "youtube_search":
            query = self._string_argument(arguments, "query")
            try:
                results = self.toolbox.youtube.search(
                    query,
                    max_results=self._integer_argument(
                        arguments, "max_results", default=5
                    ),
                )
            except ValueError as exc:
                raise ToolArgumentError(str(exc)) from exc
            observation = {"query": query, "results": results}
            return ToolResult(observation, observation)
        if name == "youtube_read_video":
            try:
                return self._read_video(arguments, call_id)
            except ValueError as exc:
                raise ToolArgumentError(str(exc)) from exc
        if name == "youtube_read_subtitle":
            try:
                return self._read_subtitle(arguments)
            except ValueError as exc:
                raise ToolArgumentError(str(exc)) from exc
        raise ToolArgumentError(f"Unknown Toolbox tool: {name}")

    @staticmethod
    def _string_argument(
        arguments: dict[str, Any], name: str, *, allow_empty: bool = False
    ) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            requirement = "a string" if allow_empty else "a non-empty string"
            raise ToolArgumentError(f"{name} must be {requirement}")
        return value

    @staticmethod
    def _positive_integer_argument(
        arguments: dict[str, Any], name: str, *, default: int | None = None
    ) -> int:
        # File coordinates follow their schema strictly; never truncate or coerce.
        value = arguments.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ToolArgumentError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _integer_argument(
        arguments: dict[str, Any], name: str, *, default: int | None = None
    ) -> int:
        value = arguments.get(name, default)
        if isinstance(value, bool):
            raise ToolArgumentError(f"{name} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ToolArgumentError(f"{name} must be an integer") from exc

    def _screenshot(self, name: str, call_id: str) -> ToolResult:
        image = self.toolbox.browser.screenshot(name)
        data = base64.b64decode(str(image["data_base64"]), validate=True)
        if not data:
            raise ToolInfrastructureError("browser_screenshot returned an empty image")
        media = [self._save_media(call_id, "screenshot", data, str(image["mime_type"]))]
        observation = {
            "name": name,
            "type": "image",
            "mime_type": image["mime_type"],
            "size_bytes": len(data),
        }
        return ToolResult(observation, observation, media)

    def _download_content(
        self, path: str, call_id: str, query: str | None = None, **coordinates: int
    ) -> ToolResult:
        text_parts: list[str] = []
        summaries: list[dict[str, Any]] = []
        media: list[dict[str, str]] = []
        try:
            parts = self.toolbox.browser.get_download_content(path, query=query, **coordinates)
        except ValueError as exc:
            message = str(exc)
            if any(
                marker in message
                for marker in ("download path", "download file not found", "query", "line", "text selection")
            ):
                raise ToolArgumentError(message) from exc
            raise ToolInfrastructureError(
                f"downloaded file could not be converted: {message}"
            ) from exc
        for index, part in enumerate(parts):
            if part.get("type") == "text":
                text = str(part.get("text", ""))
                text_parts.append(text)
                summaries.append({"type": "text", "characters": len(text)})
            elif part.get("type") == "image":
                data = base64.b64decode(str(part.get("data_base64", "")), validate=True)
                mime_type = str(part.get("mime_type", "image/png"))
                media.append(
                    self._save_media(call_id, f"download-{index}", data, mime_type)
                )
                summaries.append({"type": "image", "size_bytes": len(data)})
        observation = {
            "path": path,
            "query": query,
            "parts": summaries,
            "content": "\n\n".join(text_parts),
        }
        view = getattr(self.toolbox.browser, "download_view", None)
        if view is not None:
            observation.update({k: v for k, v in view.metadata.items() if k != "content"})
        return ToolResult(observation, observation, media)

    def _read_video(
        self, arguments: dict[str, Any], call_id: str
    ) -> ToolResult:
        video = self.toolbox.youtube.read_video(
            str(arguments.get("url_or_id", "")),
            num_frames=int(arguments.get("num_frames", 10)),
            start_seconds=arguments.get("start_seconds"),
            end_seconds=arguments.get("end_seconds"),
        )
        media = []
        timestamps = []
        frames = video.get("frames") or []
        if not frames:
            raise ToolInfrastructureError("youtube_read_video returned no video frames")
        for index, frame in enumerate(frames):
            data = base64.b64decode(str(frame["data_base64"]), validate=True)
            mime_type = str(frame.get("mime_type", "image/jpeg"))
            media.append(self._save_media(call_id, f"frame-{index:03d}", data, mime_type))
            timestamps.append(frame["timestamp"])
        observation = {
            "video_id": video["video_id"],
            "source_type": video["source_type"],
            "mode": video["mode"],
            "duration_seconds": video["duration_seconds"],
            "start_seconds": video["start_seconds"],
            "end_seconds": video["end_seconds"],
            "end_was_clamped": video["end_was_clamped"],
            "frame_timestamps": timestamps,
        }
        return ToolResult(observation, observation, media)

    def _read_subtitle(self, arguments: dict[str, Any]) -> ToolResult:
        import math
        value = self._string_argument(arguments, "url_or_id")
        times = {}
        for key in ("start_seconds", "end_seconds"):
            if key in arguments:
                number = arguments[key]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                    raise ToolArgumentError(f"{key} must be a finite number")
                times[key] = number
        if times and (len(times) != 2 or not 0 <= times["start_seconds"] < times["end_seconds"]):
            raise ToolArgumentError("provide both times with 0 <= start_seconds < end_seconds")
        try:
            subtitle = self.toolbox.youtube.read_subtitle(value, **times)
        except ValueError as exc:
            raise ToolArgumentError(str(exc)) from exc
        observation = {
            key: value for key, value in subtitle.items() if key != "data_base64"
        }
        observation["content"] = subtitle.get("content", "") if subtitle.get("available") else f"Subtitle unavailable: {subtitle.get('reason', 'unknown reason')}"
        return ToolResult(observation, observation)

    def _save_media(
        self, call_id: str, label: str, data: bytes, mime_type: str
    ) -> dict[str, str]:
        safe_call = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in call_id
        )[:80]
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }.get(mime_type, ".bin")
        artifact = save_artifact(
            self.artifact_dir / f"toolbox_{safe_call}_{label}{suffix}", data
        )
        return {
            "path": artifact["path"],
            "mime_type": mime_type,
            "sha256": artifact["sha256"],
        }
