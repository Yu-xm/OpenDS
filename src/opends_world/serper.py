from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .http import HttpError, get_bytes, post_json
from .trace import save_artifact
from .types import ToolArgumentError, ToolInfrastructureError, ToolResult


_BLOCKED_PAGE = re.compile(
    r"(?im)(?:^|\n)(?:#{1,3}\s*)?(?:401\s+unauthorized|403\s+forbidden|"
    r"407\s+proxy authentication required|429\s+too many requests|"
    r"50[0-9]\s+(?:bad gateway|service unavailable|gateway timeout|internal server error)|"
    r"http error (?:401|403|407|429|50[0-9])|access denied|request blocked|"
    r"proxy error|this site can.t be reached|err_[a-z0-9_]+)\b|"
    r"(?:captcha|verify you are human|checking your browser before accessing|"
    r"sorry, you have been blocked)"
)

_UNSUPPORTED_CONTENT_TYPE = re.compile(
    r'URLs with Content-Type "([^"\r\n]+)" currently not supported\.'
)


TOOL_DEFINITIONS = [
    {
        "type": "function", "function": {
            "name": "text_search",
            "description": (
                "Search the live public web with a text query. Returns ranked results containing "
                "page titles, URLs, and search-engine snippets. Results reflect the current web and "
                "may include webpages, documents, videos, and other publicly indexed resources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The complete text query to submit to the web search engine.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function", "function": {
            "name": "image_search",
            "description": (
                "Search images indexed on the live public web. Returns ranked image metadata, direct "
                "image URLs, and the webpages where the images were found. When downloads succeed, a "
                "limited number of top-ranked images are also supplied as visual observations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The complete text query describing the images to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function", "function": {
            "name": "open_page",
            "description": (
                "Extract readable text from a public webpage, primarily HTML articles and documentation. "
                "For JSON/YAML API endpoints or an unsupported Content-Type response, try browser_navigate "
                "instead, if available. For downloadable files, use the browser to obtain the file, then "
                "browser_get_download_list and browser_get_download_content. Browser access and file "
                "reading are not guaranteed. Returned text may be truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full public HTTP or HTTPS URL of the page to retrieve.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


class SerperTools:
    def __init__(self, config: dict[str, Any], artifact_dir: Path):
        self.api_key = config["api_key"]
        provider = str(config.get("provider", "serper"))
        if provider != "serper":
            raise ValueError("search.provider must be 'serper'")
        self.text_endpoint = str(
            config.get("text_endpoint", "https://google.serper.dev/search")
        )
        self.image_endpoint = str(
            config.get("image_endpoint", "https://google.serper.dev/images")
        )
        self.scrape_endpoint = str(
            config.get("scrape_endpoint", "https://scrape.serper.dev")
        )
        self.gl = config.get("gl", "us")
        self.hl = config.get("hl", "en")
        self.num_results = int(config.get("num_results", 10))
        self.timeout = float(config.get("timeout_seconds", 30))
        self.max_retries = int(config.get("max_retries", 0))
        self.retry_base_seconds = float(config.get("retry_base_seconds", 2))
        self.retry_max_seconds = float(config.get("retry_max_seconds", 30))
        self.attempt_timeout_seconds = float(
            config.get("attempt_timeout_seconds", self.timeout)
        )
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
            return self.text_search(self._string_argument(arguments, "query"), call_id)
        if name == "image_search":
            return self.image_search(self._string_argument(arguments, "query"), call_id)
        if name == "open_page":
            return self.open_page(self._string_argument(arguments, "url"), call_id)
        raise ToolArgumentError(f"Unknown tool: {name}")

    @staticmethod
    def _string_argument(arguments: dict[str, Any], name: str) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ToolArgumentError(f"{name} must be a non-empty string")
        return value

    def text_search(self, query: str, call_id: str) -> ToolResult:
        if not query.strip():
            raise ToolArgumentError("text_search query cannot be empty")
        raw = post_json(
            self.text_endpoint,
            {"q": query, "gl": self.gl, "hl": self.hl, "num": self.num_results},
            self.headers, self.timeout, self.max_retries,
            self.retry_base_seconds, self.retry_max_seconds,
            self.attempt_timeout_seconds,
        )
        organic = raw.get("organic", [])[:self.num_results]
        results = [
            {"rank": i, "title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", "")}
            for i, item in enumerate(organic, 1)
        ]
        return ToolResult({"query": query, "results": results}, raw)

    def image_search(self, query: str, call_id: str) -> ToolResult:
        if not query.strip():
            raise ToolArgumentError("image_search query cannot be empty")
        raw = post_json(
            self.image_endpoint,
            {"q": query, "gl": self.gl, "hl": self.hl, "num": self.num_results},
            self.headers, self.timeout, self.max_retries,
            self.retry_base_seconds, self.retry_max_seconds,
            self.attempt_timeout_seconds,
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
        download_failures = 0
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
                    download_failures += 1
        if (
            self.download_images
            and results[:self.image_download_count]
            and download_failures == len(results[:self.image_download_count])
        ):
            raise ToolInfrastructureError(
                "image_search returned metadata, but every requested visual download failed"
            )
        return ToolResult({"query": query, "results": results}, raw, media)

    def open_page(self, url: str, call_id: str) -> ToolResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ToolArgumentError("open_page requires a public http(s) URL")
        try:
            raw = post_json(
                self.scrape_endpoint,
                {"url": url, "includeMarkdown": True},
                self.headers,
                self.timeout,
                self.max_retries,
                self.retry_base_seconds,
                self.retry_max_seconds,
                self.attempt_timeout_seconds,
            )
        except HttpError as exc:
            body = exc.response_body
            message = body.get("message") if isinstance(body, dict) else None
            match = (_UNSUPPORTED_CONTENT_TYPE.fullmatch(message.strip())
                     if exc.status_code == 400 and isinstance(message, str) else None)
            if match is None:
                raise
            return ToolResult({
                "url": url,
                "available": False,
                "reason": "unsupported_content_type",
                "content_type": match.group(1).split(";", 1)[0].strip().lower(),
                "retryable": False,
                "suggested_tool": "browser_navigate",
                "message": (
                    "Serper cannot read this content type. Try browser_navigate for this URL "
                    "if available; browser access and file reading are not guaranteed."
                ),
            }, body)
        content = raw.get("markdown") or raw.get("text") or raw.get("content") or ""
        content = str(content)
        status = raw.get("statusCode", raw.get("status"))
        try:
            status_code = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_code = None
        if status_code is not None and status_code >= 400:
            raise ToolInfrastructureError(
                f"open_page could not retrieve {url}: upstream HTTP {status_code}"
            )
        if not content.strip():
            raise ToolInfrastructureError(
                f"open_page returned no readable content for {url}"
            )
        blocked = _BLOCKED_PAGE.search(content[:4000])
        if blocked:
            raise ToolInfrastructureError(
                f"open_page returned a blocked or network error page for {url}: "
                f"{blocked.group(0).strip()}"
            )
        observation = {
            "url": url,
            "title": raw.get("title", ""),
            "content": content[:self.page_chars],
            "content_chars": len(content),
            "content_truncated": len(content) > self.page_chars,
        }
        return ToolResult(observation, raw)
