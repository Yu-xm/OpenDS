from __future__ import annotations

import base64
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Protocol

from .http import post_json, post_sse_json
from .types import ModelResponse, ToolCall


class ModelBackend(Protocol):
    display_name: str
    supports_video: bool
    supports_audio: bool

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse: ...


def _file_segment(path: Path, kind: str, max_bytes: int) -> dict[str, str]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Media file exceeds max_media_bytes ({size} > {max_bytes}): {path}")
    mime = mimetypes.guess_type(path.name)[0] or f"{kind}/octet-stream"
    return {"type": kind, "mime_type": mime, "data_base64": base64.b64encode(path.read_bytes()).decode("ascii"), "filename": path.name}


def initial_messages(question: str, video: Path | None, audio: Path | None, system_prompt: str, max_bytes: int) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "text", "text": question}]
    if video:
        content.append(_file_segment(video, "video", max_bytes))
    if audio:
        content.append(_file_segment(audio, "audio", max_bytes))
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


class BridgeBackend:
    """Provider-neutral HTTP contract documented in README.md."""

    def __init__(self, config: dict[str, Any]):
        self.display_name = config["display_name"]
        self.supports_video = bool(config.get("supports_video", True))
        self.supports_audio = bool(config.get("supports_audio", True))
        self.model_id = config["model_id"]
        self.endpoint = config["endpoint"]
        self.timeout = float(config.get("timeout_seconds", 300))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_base_seconds = float(config.get("retry_base_seconds", 2))
        self.retry_max_seconds = float(config.get("retry_max_seconds", 60))
        self.headers = dict(config.get("headers", {}))
        self.parameters = dict(config.get("parameters", {}))

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        raw = post_json(self.endpoint, {
            "model": self.model_id, "messages": messages, "tools": tools,
            "tool_choice": "auto", "parameters": self.parameters,
        }, self.headers, self.timeout, self.max_retries, self.retry_base_seconds, self.retry_max_seconds)
        calls = []
        for item in raw.get("tool_calls", []):
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCall(str(item.get("id") or uuid.uuid4()), item["name"], dict(arguments)))
        return ModelResponse(
            text=str(raw.get("text", "")),
            reasoning_content=str(raw.get("reasoning_content") or ""), tool_calls=calls,
            usage=dict(raw.get("usage", {})), finish_reason=raw.get("finish_reason"), raw=raw,
        )


class OpenAICompatibleBackend:
    def __init__(self, config: dict[str, Any]):
        self.display_name = config["display_name"]
        self.supports_video = bool(config.get("supports_video", False))
        self.supports_audio = bool(config.get("supports_audio", False))
        self.model_id = config["model_id"]
        self.base_url = config["base_url"].rstrip("/")
        self.api_key = config.get("api_key", "")
        self.timeout = float(config.get("timeout_seconds", 300))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_base_seconds = float(config.get("retry_base_seconds", 2))
        self.retry_max_seconds = float(config.get("retry_max_seconds", 60))
        self.parameters = dict(config.get("parameters", {}))

    def _content(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content
        converted: list[dict[str, Any]] = []
        for part in content:
            if part["type"] == "text":
                converted.append(part)
            elif part["type"] == "image":
                data_url = f"data:{part['mime_type']};base64,{part['data_base64']}"
                converted.append({"type": "image_url", "image_url": {"url": data_url}})
            elif part["type"] == "audio":
                subtype = part["mime_type"].split("/", 1)[-1].replace("mpeg", "mp3")
                converted.append({"type": "input_audio", "input_audio": {"data": part["data_base64"], "format": subtype}})
            elif part["type"] == "video":
                data_url = f"data:{part['mime_type']};base64,{part['data_base64']}"
                converted.append({"type": "video_url", "video_url": {"url": data_url}})
        return converted

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for message in messages:
            item = dict(message)
            item["content"] = self._content(item.get("content"))
            output.append(item)
        return output

    @staticmethod
    def _assemble_stream(chunks: list[dict[str, Any]]) -> dict[str, Any]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish_reason = None
        response_id = model_name = None

        for chunk in chunks:
            response_id = chunk.get("id") or response_id
            model_name = chunk.get("model") or model_name
            if chunk.get("usage"):
                usage = dict(chunk["usage"])
            for choice in chunk.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or choice.get("message") or {}
                if delta.get("content"):
                    text_parts.append(str(delta["content"]))
                if delta.get("reasoning_content"):
                    reasoning_parts.append(str(delta["reasoning_content"]))
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index", len(tool_parts)))
                    target = tool_parts.setdefault(index, {
                        "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                    })
                    if item.get("id"):
                        target["id"] = str(item["id"])
                    function = item.get("function") or {}
                    if function.get("name"):
                        target["function"]["name"] += str(function["name"])
                    arguments = function.get("arguments")
                    if arguments:
                        if not isinstance(arguments, str):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        target["function"]["arguments"] += arguments

        message = {
            "role": "assistant", "content": "".join(text_parts),
            "reasoning_content": "".join(reasoning_parts),
            "tool_calls": [tool_parts[index] for index in sorted(tool_parts)],
        }
        return {
            "id": response_id, "object": "chat.completion", "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage, "streamed": True, "stream_event_count": len(chunks),
        }

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model_id, "messages": self._messages(messages),
            **self.parameters,
        }
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto"})
        endpoint = f"{self.base_url}/chat/completions"
        if payload.get("stream"):
            raw = self._assemble_stream(post_sse_json(
                endpoint, payload, headers, self.timeout, self.max_retries,
                self.retry_base_seconds, self.retry_max_seconds,
            ))
        else:
            raw = post_json(
                endpoint, payload, headers, self.timeout, self.max_retries,
                self.retry_base_seconds, self.retry_max_seconds,
            )
        choice = raw["choices"][0]
        message = choice.get("message", {})
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCall(str(item.get("id") or uuid.uuid4()), str(function.get("name", "")), dict(arguments)))
        usage = raw.get("usage") or {}
        return ModelResponse(
            text=message.get("content") or "",
            reasoning_content=str(message.get("reasoning_content") or ""), tool_calls=calls,
            usage={"input_tokens": int(usage.get("prompt_tokens", 0)), "output_tokens": int(usage.get("completion_tokens", 0))},
            finish_reason=choice.get("finish_reason"), raw=raw,
        )


class MockBackend:
    def __init__(self, config: dict[str, Any]):
        self.display_name = config.get("display_name", "mock-model")
        self.supports_video = True
        self.supports_audio = True

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        return ModelResponse(text="Mock answer for pipeline validation.", usage={"input_tokens": 1, "output_tokens": 6}, finish_reason="stop")


def create_backend(config: dict[str, Any]) -> ModelBackend:
    adapter = config.get("adapter")
    if adapter == "bridge":
        return BridgeBackend(config)
    if adapter == "openai_compatible":
        return OpenAICompatibleBackend(config)
    if adapter == "mock":
        return MockBackend(config)
    raise ValueError(f"Unknown model adapter: {adapter!r}")
