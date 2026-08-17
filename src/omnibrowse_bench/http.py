from __future__ import annotations

import json
import http.client
import urllib.error
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    pass


def _request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> urllib.request.Request:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **headers},
    )


def _connection_error(url: str, exc: BaseException) -> HttpError:
    detail = getattr(exc, "reason", None) or str(exc) or type(exc).__name__
    return HttpError(f"Request to {url} failed: {detail}")


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = _request(url, payload, headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise HttpError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise _connection_error(url, exc) from exc
    except (TimeoutError, http.client.RemoteDisconnected, ConnectionError) as exc:
        raise _connection_error(url, exc) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise HttpError(f"Non-JSON response from {url}: {body[:500]!r}") from exc


def post_sse_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float,
) -> list[dict[str, Any]]:
    """POST an OpenAI-compatible request and decode its SSE data events."""
    request = _request(url, payload, headers)
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []

    def flush_event() -> bool:
        if not data_lines:
            return False
        data = "\n".join(data_lines)
        data_lines.clear()
        if data == "[DONE]":
            return True
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError as exc:
            raise HttpError(f"Invalid SSE JSON from {url}: {data[:500]!r}") from exc
        return False

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if flush_event():
                        break
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            else:
                flush_event()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise HttpError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise _connection_error(url, exc) from exc
    except (TimeoutError, http.client.RemoteDisconnected, ConnectionError) as exc:
        raise _connection_error(url, exc) from exc
    if not events:
        raise HttpError(f"Empty SSE response from {url}")
    return events


def get_bytes(url: str, timeout: float, max_bytes: int) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "OmniBrowseBench/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HttpError(f"Downloaded object exceeds {max_bytes} bytes")
    return data, content_type
