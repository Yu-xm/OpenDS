from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .http import get_bytes, post_json
from .trace import save_artifact
from .types import ToolResult


TOOL_DEFINITIONS = [
    {
        "type": "function", "function": {
            "name": "text_search",
            "description": "Search the live web. Returns ranked titles, URLs and snippets.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function", "function": {
            "name": "image_search",
            "description": "Search live web images. Returns image and source URLs; selected images are supplied visually when downloadable.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function", "function": {
            "name": "open_page",
            "description": "Fetch and extract the readable content of a public web page.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "An http or https URL"}}, "required": ["url"]},
        },
    },
]


class SerperTools:
    def __init__(self, config: dict[str, Any], artifact_dir: Path):
        self.api_key = config["api_key"]
        self.gl = config.get("gl", "us")
        self.hl = config.get("hl", "en")
        self.num_results = int(config.get("num_results", 10))
        self.timeout = float(config.get("timeout_seconds", 30))
        self.page_chars = int(config.get("max_page_chars", 20000))
        self.download_images = bool(config.get("download_images", True))
        self.image_download_count = int(config.get("image_download_count", 3))
        self.max_image_bytes = int(config.get("max_image_bytes", 8_000_000))
        self.artifact_dir = artifact_dir

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-KEY": self.api_key}

    def execute(self, name: str, arguments: dict[str, Any], call_id: str) -> ToolResult:
        if name == "text_search":
            return self.text_search(str(arguments.get("query", "")), call_id)
        if name == "image_search":
            return self.image_search(str(arguments.get("query", "")), call_id)
        if name == "open_page":
            return self.open_page(str(arguments.get("url", "")), call_id)
        raise ValueError(f"Unknown tool: {name}")

    def text_search(self, query: str, call_id: str) -> ToolResult:
        if not query.strip():
            raise ValueError("text_search query cannot be empty")
        raw = post_json(
            "https://google.serper.dev/search",
            {"q": query, "gl": self.gl, "hl": self.hl, "num": self.num_results},
            self.headers, self.timeout,
        )
        organic = raw.get("organic", [])[:self.num_results]
        results = [
            {"rank": i, "title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", "")}
            for i, item in enumerate(organic, 1)
        ]
        return ToolResult({"query": query, "results": results}, raw)

    def image_search(self, query: str, call_id: str) -> ToolResult:
        if not query.strip():
            raise ValueError("image_search query cannot be empty")
        raw = post_json(
            "https://google.serper.dev/images",
            {"q": query, "gl": self.gl, "hl": self.hl, "num": self.num_results},
            self.headers, self.timeout,
        )
        images = raw.get("images", [])[:self.num_results]
        results = [
            {
                "rank": i, "title": item.get("title", ""),
                "image_url": item.get("imageUrl", ""), "source_url": item.get("link", ""),
                "width": item.get("imageWidth"), "height": item.get("imageHeight"),
            }
            for i, item in enumerate(images, 1)
        ]
        media: list[dict[str, str]] = []
        if self.download_images:
            for item in results[:self.image_download_count]:
                url = item["image_url"]
                if not url.startswith(("http://", "https://")):
                    continue
                try:
                    data, mime = get_bytes(url, self.timeout, self.max_image_bytes)
                    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".bin")
                    safe_call_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in call_id)[:120]
                    path = self.artifact_dir / f"image_{safe_call_id}_{item['rank']:02d}{suffix}"
                    artifact = save_artifact(path, data)
                    media.append({"path": artifact["path"], "mime_type": mime, "sha256": artifact["sha256"], "source_url": url})
                except Exception as exc:
                    item["download_error"] = str(exc)
        return ToolResult({"query": query, "results": results}, raw, media)

    def open_page(self, url: str, call_id: str) -> ToolResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("open_page requires a public http(s) URL")
        raw = post_json("https://scrape.serper.dev", {"url": url, "includeMarkdown": True}, self.headers, self.timeout)
        content = raw.get("markdown") or raw.get("text") or raw.get("content") or ""
        content = str(content)
        observation = {
            "url": url,
            "title": raw.get("title", ""),
            "content": content[:self.page_chars],
            "content_chars": len(content),
            "content_truncated": len(content) > self.page_chars,
        }
        return ToolResult(observation, raw)
