from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from .types import ToolInfrastructureError


def _compact_history(
    messages: list[dict[str, Any]], keep_recent_tools: int, history_search_results: int,
    history_page_chars: int, keep_recent_reasoning: int,
) -> tuple[int, int, int]:
    """Deterministically shrink old observations without adding another model call."""
    tool_indexes = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    reasoning_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("reasoning_content")
    ]
    compact_tool_indexes = set(tool_indexes[:-max(0, keep_recent_tools)] if keep_recent_tools else tool_indexes)
    compact_reasoning_indexes = set(
        reasoning_indexes[:-max(0, keep_recent_reasoning)] if keep_recent_reasoning else reasoning_indexes
    )
    changed = before = after = 0
    for index in compact_reasoning_indexes:
        reasoning = str(messages[index].pop("reasoning_content", ""))
        if reasoning:
            changed += 1
            before += len(reasoning)
    for index in compact_tool_indexes:
        content = messages[index].get("content")
        if not isinstance(content, str):
            continue
        try:
            observation = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(observation, dict):
            continue
        if observation.get("_history_compacted"):
            continue
        compacted = dict(observation)
        if isinstance(compacted.get("results"), list):
            compacted["results"] = compacted["results"][:history_search_results]
        if isinstance(compacted.get("content"), str):
            compacted["content"] = compacted["content"][:history_page_chars]
            compacted["history_content_truncated"] = len(observation["content"]) > history_page_chars
        compacted["_history_compacted"] = True
        replacement = json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
        if len(replacement) < len(content):
            messages[index]["content"] = replacement
            changed += 1
            before += len(content)
            after += len(replacement)
    return changed, before, after



def _compact_images(messages: list[dict[str, Any]], max_images: int) -> tuple[int, int]:
    """Keep the newest images across all messages; leave text and artifacts intact."""
    images = [
        (message, index)
        for message in messages if isinstance(message.get("content"), list)
        for index, part in enumerate(message["content"])
        if part.get("type") in {"image", "image_url"}
    ]
    for message, index in images[:max(0, len(images) - max_images)]:
        message["content"][index] = {
            "type": "text",
            "text": "[Older image omitted from model context to stay within the image limit; "
                    "the original image remains in the trajectory artifacts.]",
        }
    return len(images), min(len(images), max_images)


def _assistant_message(text: str, reasoning_content: str, calls: list[Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text or ""}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
            for call in calls
        ]
    return message


def _visual_message(media: list[dict[str, str]]) -> dict[str, Any] | None:
    # This caption is part of the historical model input; preserve it for replay.
    content: list[dict[str, str]] = [{"type": "text", "text": "Visual observations downloaded from the preceding image_search results:"}]
    for item in media:
        path = Path(item["path"])
        if not item.get("mime_type", "").startswith("image/"):
            raise ToolInfrastructureError(
                f"Tool returned unsupported visual media type: {item.get('mime_type')}"
            )
        if not path.is_file():
            raise ToolInfrastructureError(f"Tool visual artifact is missing: {path}")
        content.append({"type": "image", "mime_type": item["mime_type"], "data_base64": base64.b64encode(path.read_bytes()).decode("ascii")})
    return {"role": "user", "content": content} if len(content) > 1 else None
