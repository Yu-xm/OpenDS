from __future__ import annotations

import json
import http.client
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from email.utils import parsedate_to_datetime
from typing import Any


class HttpError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, response_text: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        try:
            self.response_body = json.loads(response_text) if response_text is not None else None
        except json.JSONDecodeError:
            self.response_body = None


def _public_url(url: str) -> str:
    """Keep endpoint location in diagnostics without URL credentials or queries."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> urllib.request.Request:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **headers},
    )


def _connection_error(url: str, exc: BaseException) -> HttpError:
    return HttpError(f"Connection to {_public_url(url)} failed ({type(exc).__name__}); check service availability and network routing")


def _retry_delay(exc: urllib.error.HTTPError, attempt: int, base_seconds: float, max_seconds: float) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return min(max_seconds, max(0.0, float(retry_after)))
        except ValueError:
            try:
                delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
                return min(max_seconds, max(0.0, delay))
            except (TypeError, ValueError, OverflowError):
                pass
    delay = min(max_seconds, base_seconds * (2 ** attempt))
    if exc.code == 429:
        delay = min(max_seconds, max(60.0, delay))
    return delay


def _should_retry_http(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _remaining_timeout(deadline: float | None, configured_timeout: float | None, url: str) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HttpError(f"Request to {url} exceeded total timeout of {configured_timeout:g} seconds")
    return max(0.1, remaining)


def _wait_before_retry(delay: float, deadline: float | None, configured_timeout: float | None, url: str) -> None:
    remaining = _remaining_timeout(deadline, configured_timeout, url)
    if remaining is not None and delay >= remaining:
        raise HttpError(f"Request to {url} exceeded total timeout of {configured_timeout:g} seconds")
    time.sleep(delay)


def post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float | None,
    max_retries: int = 0, retry_base_seconds: float = 2, retry_max_seconds: float = 30,
    attempt_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    deadline = None if timeout is None else time.monotonic() + timeout
    for attempt in range(max_retries + 1):
        request = _request(url, payload, headers)
        try:
            request_timeout = _remaining_timeout(deadline, timeout, url)
            if attempt_timeout_seconds is not None:
                request_timeout = min(request_timeout, attempt_timeout_seconds) if request_timeout is not None else attempt_timeout_seconds
            with urllib.request.urlopen(
                request, timeout=request_timeout
            ) as response:
                body = response.read()
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if attempt < max_retries and _should_retry_http(exc.code):
                _wait_before_retry(
                    _retry_delay(exc, attempt, retry_base_seconds, retry_max_seconds),
                    deadline, timeout, url,
                )
                continue
            raise HttpError(f"HTTP {exc.code} from {_public_url(url)} after {attempt + 1}/{max_retries + 1} attempts", exc.code, detail) from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                _wait_before_retry(
                    _retry_delay(urllib.error.HTTPError(url, 503, "retry", {}, None), attempt, retry_base_seconds, retry_max_seconds),
                    deadline, timeout, url,
                )
                continue
            raise _connection_error(url, exc) from exc
        except (TimeoutError, http.client.RemoteDisconnected, ConnectionError) as exc:
            if attempt < max_retries:
                _wait_before_retry(
                    _retry_delay(urllib.error.HTTPError(url, 503, "retry", {}, None), attempt, retry_base_seconds, retry_max_seconds),
                    deadline, timeout, url,
                )
                continue
            raise _connection_error(url, exc) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise HttpError(f"Non-JSON response from {url}: {body[:500]!r}") from exc


def post_sse_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float | None,
    max_retries: int = 0, retry_base_seconds: float = 2, retry_max_seconds: float = 30,
) -> list[dict[str, Any]]:
    """POST an OpenAI-compatible request and decode its SSE data events."""
    deadline = None if timeout is None else time.monotonic() + timeout
    for attempt in range(max_retries + 1):
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
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise HttpError(f"Invalid SSE JSON from {url}: {data[:500]!r}") from exc
            if event.get("error"):
                detail = event["error"]
                code = detail.get("status_code", detail.get("status", 400)) if isinstance(detail, dict) else 400
                error = HttpError(f"SSE error from {url}: {detail}",
                                  code if isinstance(code, int) else 400, json.dumps(detail))
                error.response_started = bool(events)
                # Only the finalizer's repeated-context-overflow branch may use this.
                # Never retain reasoning or partially assembled tool arguments as an answer.
                error.partial_text = "".join(
                    str((choice.get("delta") or {}).get("content") or "")
                    for previous in events for choice in previous.get("choices", [])
                    if choice.get("index", 0) == 0
                )
                raise error
            events.append(event)
            return False

        try:
            with urllib.request.urlopen(
                request, timeout=_remaining_timeout(deadline, timeout, url)
            ) as response:
                for raw_line in response:
                    _remaining_timeout(deadline, timeout, url)
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
            if not events:
                raise HttpError(f"Empty SSE response from {url}")
            terminal = any(
                choice.get("index", 0) == 0 and choice.get("finish_reason") is not None
                for event in events for choice in event.get("choices", [])
            )
            if not terminal:
                raise HttpError("Incomplete SSE response: no terminal finish_reason")
            return events
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if attempt < max_retries and _should_retry_http(exc.code):
                _wait_before_retry(
                    _retry_delay(exc, attempt, retry_base_seconds, retry_max_seconds),
                    deadline, timeout, url,
                )
                continue
            raise HttpError(f"HTTP {exc.code} from {_public_url(url)} after {attempt + 1}/{max_retries + 1} attempts", exc.code, detail) from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                _wait_before_retry(
                    _retry_delay(urllib.error.HTTPError(url, 503, "retry", {}, None), attempt, retry_base_seconds, retry_max_seconds),
                    deadline, timeout, url,
                )
                continue
            raise _connection_error(url, exc) from exc
        except (TimeoutError, http.client.RemoteDisconnected, ConnectionError) as exc:
            if attempt < max_retries:
                _wait_before_retry(
                    _retry_delay(urllib.error.HTTPError(url, 503, "retry", {}, None), attempt, retry_base_seconds, retry_max_seconds),
                    deadline, timeout, url,
                )
                continue
            raise _connection_error(url, exc) from exc


def get_bytes(url: str, timeout: float, max_bytes: int) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenDS-World/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HttpError(f"Downloaded object exceeds {max_bytes} bytes")
    return data, content_type
