"""Minimal Toolbox for one Browser and one YouTube container per task."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import enum
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlparse

import docker
import httpx
from agent_sandbox import Sandbox
from docker.errors import APIError, DockerException, NotFound

if __package__:
    from .plugins import FileReaderRegistry, create_file_reader_registry
else:
    from plugins import FileReaderRegistry, create_file_reader_registry

BROWSER_IMAGE = "toolbox-browser:pdf-v1"
YOUTUBE_IMAGE = "toolbox-youtube:latest"
VIDEO_FORMAT = (
    "bestvideo[ext=mp4][vcodec^=avc1][height<=360]/"
    "bestvideo[ext=mp4][height<=360]/bestvideo[height<=360]"
)
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
INTERNAL_COOKIE = "/tmp/toolbox-youtube-cookies.txt"
INTERNAL_YOUTUBE_CACHE = "/home/gem/.cache/toolbox/youtube"
INTERNAL_INPUT_VIDEO_ROOT = "/home/gem/input-video"
INPUT_VIDEO_ID = "input_video"
DOWNLOAD_DETECTION_SECONDS = 2.0
DOWNLOAD_LISTENER_SECONDS = 15.0
DOWNLOAD_TIMEOUT_SECONDS = 180.0
BROWSER_NAVIGATION_ATTEMPTS = 6
BROWSER_NAVIGATION_RETRY_BASE_SECONDS = 0.5
TRANSIENT_BROWSER_FAILURE = re.compile(
    r"(?:\b502\s+bad\s+gateway\b|\b503\s+service\s+unavailable\b|"
    r"\b504\s+gateway\s+timeout\b|\bgateway\s+timeout\b|"
    r"\bproxy\s+error\b|net::ERR_[A-Z_]+|\bECONN[A-Z]+\b|"
    r"\bETIMEDOUT\b|\bconnection\s+(?:reset|closed|refused)\b|"
    r"\bserver\s+disconnected\b|\btemporary\s+failure\b|"
    r"\bnetwork\s+error\b|\btimed?\s*out\b|\btimeout\b)",
    re.IGNORECASE,
)


def _range_frame_timestamps(
    start: float, end: float, num_frames: int
) -> list[float]:
    width = (end - start) / num_frames
    return [start + (index + 0.5) * width for index in range(num_frames)]


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if hasattr(value, "dict"):
        return _plain(value.dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def _data(response: Any) -> Any:
    payload = _plain(response)
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(str(payload.get("message") or "sandbox call failed"))
    return payload.get("data") if isinstance(payload, dict) else payload


def _json_output(output: str) -> dict[str, Any]:
    for candidate in [output, *reversed(output.splitlines())]:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("command returned no JSON object")


def _youtube_url(value: str) -> tuple[str, str]:
    value = value.strip()
    if VIDEO_ID.fullmatch(value):
        return f"https://www.youtube.com/watch?v={value}", value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }:
        raise ValueError("expected one YouTube video URL or ID")
    video_id = (
        parsed.path.strip("/")
        if host == "youtu.be"
        else parse_qs(parsed.query).get("v", [""])[0]
    )
    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError("YouTube URL does not identify one video")
    return f"https://www.youtube.com/watch?v={video_id}", video_id


def _docker_client(docker_host: str | None = None) -> docker.DockerClient | None:
    """Connect through DOCKER_HOST or the active local Docker CLI context."""
    try:
        if docker_host:
            client = docker.DockerClient(base_url=docker_host, timeout=15)
        elif os.environ.get("DOCKER_HOST"):
            client = docker.from_env(timeout=15)
        else:
            config_path = Path.home() / ".docker" / "config.json"
            context = str(
                json.loads(config_path.read_text()).get("currentContext") or "default"
            )
            if context == "default":
                client = docker.from_env(timeout=15)
            else:
                digest = hashlib.sha256(context.encode()).hexdigest()
                metadata_path = (
                    Path.home() / ".docker" / "contexts" / "meta" / digest / "meta.json"
                )
                metadata = json.loads(metadata_path.read_text())
                client = docker.DockerClient(
                    base_url=str(metadata["Endpoints"]["docker"]["Host"]),
                    timeout=15,
                )
        client.ping()
        return client
    except (DockerException, OSError, KeyError, TypeError, ValueError):
        return None


class _Container:
    def __init__(
        self,
        client: docker.DockerClient | None,
        name: str,
        image: str,
        *,
        docker_command: str = "docker",
    ) -> None:
        self.client = client
        self.docker_command = docker_command
        self.name = name
        self.image = image
        self.container: Any | None = None
        self.base_url: str | None = None
        self.startup_timings: dict[str, float] = {}

    def start(
        self,
        *,
        download_dir: Path | None = None,
        cache_dir: Path | None = None,
        cookie_file: Path | None = None,
        input_video_dir: Path | None = None,
        environment: dict[str, str] | None = None,
        network: str | None = None,
    ) -> str:
        started = time.monotonic()
        self.startup_timings = {}
        volumes: dict[str, dict[str, str]] = {}
        if download_dir is not None:
            volumes[str(download_dir.resolve())] = {
                "bind": "/home/gem/Downloads",
                "mode": "rw",
            }
        if cache_dir is not None:
            volumes[str(cache_dir.resolve())] = {
                "bind": INTERNAL_YOUTUBE_CACHE,
                "mode": "rw",
            }
        if cookie_file is not None:
            volumes[str(cookie_file.resolve())] = {
                "bind": "/run/secrets/youtube-cookies.txt",
                "mode": "ro",
            }
        if input_video_dir is not None:
            volumes[str(input_video_dir.resolve())] = {
                "bind": INTERNAL_INPUT_VIDEO_ROOT,
                "mode": "ro",
            }
        env = {
            "WORKSPACE": "/home/gem",
            "PUBLIC_PORT": "8080",
            "WAIT_PORTS": "8091",
            **(environment or {}),
        }
        if self.client is not None:
            run_arguments: dict[str, Any] = {
                "name": self.name,
                "detach": True,
                "auto_remove": True,
                "extra_hosts": {"host.docker.internal": "host-gateway"},
                "shm_size": "2g",
                "ports": {"8080/tcp": ("127.0.0.1", None)},
                "environment": env,
                "volumes": volumes,
            }
            if network:
                run_arguments["network"] = network
            self.container = self.client.containers.run(
                self.image,
                **run_arguments,
            )
            self.container.reload()
            base_url = self._base_url_from_inspect(
                self.container.attrs,
                network=network,
            )
        else:
            command = [
                self.docker_command,
                "run",
                "--detach",
                "--rm",
                "--name",
                self.name,
                "--add-host",
                "host.docker.internal:host-gateway",
                "--shm-size",
                "2g",
                "--publish",
                "127.0.0.1::8080",
            ]
            if network:
                command += ["--network", network]
            for key, value in sorted(env.items()):
                command += ["--env", f"{key}={value}"]
            for source, mount in volumes.items():
                command += ["--volume", f"{source}:{mount['bind']}:{mount['mode']}"]
            command.append(self.image)
            subprocess.run(command, check=True, capture_output=True, text=True)
            inspect = subprocess.run(
                [self.docker_command, "inspect", self.name],
                check=True,
                capture_output=True,
                text=True,
            )
            details = json.loads(inspect.stdout)[0]
            base_url = self._base_url_from_inspect(details, network=network)
        self.base_url = base_url
        self.startup_timings["container_create_seconds"] = round(
            time.monotonic() - started, 3
        )
        if cache_dir is not None:
            ownership_started = time.monotonic()
            ownership_command = ["chown", "1000:1000", INTERNAL_YOUTUBE_CACHE]
            if self.container is not None:
                result = self.container.exec_run(ownership_command, user="0")
                if result.exit_code:
                    raise RuntimeError(
                        f"failed to prepare YouTube cache ownership: {result.output!r}"
                    )
            else:
                subprocess.run(
                    [
                        self.docker_command,
                        "exec",
                        "--user",
                        "0",
                        self.name,
                        *ownership_command,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.startup_timings["cache_ownership_seconds"] = round(
                time.monotonic() - ownership_started, 3
            )
        sandbox_started = time.monotonic()
        deadline = time.monotonic() + 180
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            try:
                with opener.open(
                    f"{self.base_url}/v1/sandbox", timeout=2
                ) as response:
                    if response.status == 200:
                        self.startup_timings["sandbox_ready_seconds"] = round(
                            time.monotonic() - sandbox_started, 3
                        )
                        self.startup_timings["total_seconds"] = round(
                            time.monotonic() - started, 3
                        )
                        return self.base_url
            except OSError:
                time.sleep(0.5)
        self.startup_timings["sandbox_ready_seconds"] = round(
            time.monotonic() - sandbox_started, 3
        )
        self.startup_timings["total_seconds"] = round(
            time.monotonic() - started, 3
        )
        self.stop()
        raise RuntimeError(f"container {self.name} did not become ready")

    @staticmethod
    def _base_url_from_inspect(
        details: dict[str, Any],
        *,
        network: str | None,
    ) -> str:
        network_settings = details.get("NetworkSettings") or {}
        bindings = (network_settings.get("Ports") or {}).get("8080/tcp") or []
        if bindings:
            host_port = str(bindings[0]["HostPort"])
            return f"http://127.0.0.1:{host_port}"
        if network:
            network_details = (network_settings.get("Networks") or {}).get(network) or {}
            address = str(network_details.get("IPAddress") or "")
            if address:
                return f"http://{address}:8080"
        raise RuntimeError(
            f"container {details.get('Name') or '<unknown>'} has no reachable port"
        )

    def stop(self) -> str | None:
        container, self.container = self.container, None
        if container is not None:
            try:
                container.kill()
            except (NotFound, APIError):
                pass
            except Exception as error:
                try:
                    completed = subprocess.run(
                        [self.docker_command, "rm", "--force", self.name],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired) as fallback_error:
                    return (
                        f"Docker SDK cleanup failed for {self.name}: {error}; "
                        f"CLI fallback failed: {fallback_error}"
                    )
                details = (completed.stderr or completed.stdout).strip()
                if completed.returncode and "No such container" not in details:
                    return (
                        f"Docker SDK cleanup failed for {self.name}: {error}; "
                        f"CLI fallback exited {completed.returncode}: {details}"
                    )
        elif self.client is None:
            try:
                completed = subprocess.run(
                    [self.docker_command, "rm", "--force", self.name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                return f"CLI cleanup failed for {self.name}: {error}"
            details = (completed.stderr or completed.stdout).strip()
            if completed.returncode and "No such container" not in details:
                return (
                    f"CLI cleanup failed for {self.name}: "
                    f"exit {completed.returncode}: {details}"
                )
        self.base_url = None
        return None


class _Shell:
    def __init__(self, client: Sandbox, *, working_dir: str) -> None:
        self.client = client
        self.working_dir = working_dir

    def run(self, args: list[str], timeout: float = 120) -> str:
        response = self.client.shell.exec_command(
            command=shlex.join(args),
            exec_dir=self.working_dir,
            hard_timeout=timeout,
            truncate=False,
            request_options={"timeout_in_seconds": int(timeout + 30)},
        )
        result = _data(response)
        if not isinstance(result, dict):
            raise RuntimeError("shell returned no result")
        if result.get("exit_code") not in (None, 0):
            output = str(result.get("output") or "command failed")[-2000:]
            output = re.sub(
                r"https?://[^\s'\"]*(?:googlevideo\.com|/videoplayback/)[^\s'\"]*",
                "<redacted-media-url>",
                output,
                flags=re.IGNORECASE,
            )
            raise RuntimeError(output)
        return str(result.get("output") or "")


class BrowserTools:
    def __init__(
        self,
        client: Sandbox,
        download_dir: Path,
        file_readers: FileReaderRegistry,
        navigation_attempts: int = BROWSER_NAVIGATION_ATTEMPTS,
        file_reader_config: dict | None = None,
    ) -> None:
        if navigation_attempts < 1:
            raise ValueError("navigation_attempts must be at least 1")
        self.client = client
        self.download_dir = download_dir
        self._file_readers = file_readers
        from .downloads import DownloadView
        self.download_view = DownloadView(download_dir, (file_reader_config or {}).get("text"))
        self._clickable_elements: dict[int, str] = {}
        self._download_paths: dict[str, str] = {}
        self._navigation_attempts = navigation_attempts

    def _call(
        self,
        name: str,
        request: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        request_options = (
            {"timeout_in_seconds": timeout_seconds}
            if timeout_seconds is not None
            else None
        )
        result = _data(
            self.client.mcp.execute_mcp_tool(
                "browser",
                name,
                request=request,
                request_options=request_options,
            )
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"{name} returned no MCP result")
        if result.get("is_error"):
            raise RuntimeError(self._text(result) or f"{name} failed")
        return result

    @staticmethod
    def _text(result: dict[str, Any]) -> str:
        return "\n".join(
            str(item.get("text") or "")
            for item in result.get("content") or []
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()

    def _remember_clickable_elements(self, text: str) -> str:
        elements = {}
        for line in text.splitlines():
            match = re.match(r"\[(\d+)\]", line)
            if match:
                elements[int(match.group(1))] = line
        if elements:
            self._clickable_elements = elements
        return text

    @staticmethod
    def _transient_navigation_reason(error: BaseException | str) -> str | None:
        if isinstance(error, (TimeoutError, httpx.TransportError)):
            message = str(error).strip()
            return message or type(error).__name__
        message = str(error).strip()
        match = TRANSIENT_BROWSER_FAILURE.search(message)
        return match.group(0) if match else None

    @staticmethod
    def _annotate_navigation_retry(
        result: str | dict[str, Any], attempts: int
    ) -> str | dict[str, Any]:
        note = (
            f"Navigation succeeded after {attempts} attempts "
            f"({attempts - 1} transient network retries)."
        )
        if isinstance(result, dict):
            annotated = dict(result)
            message = str(annotated.get("message") or "").strip()
            annotated["message"] = f"{note}\n{message}" if message else note
            return annotated
        return f"{note}\n{result}" if result else note

    def navigate(self, url: str) -> str | dict[str, Any]:
        """Navigate with bounded transient-network retries, including downloads."""
        last_reason = "transient network failure"
        for attempt in range(1, self._navigation_attempts + 1):
            try:
                result = self._navigate_once(url)
                if isinstance(result, str):
                    reason = self._transient_navigation_reason(result)
                    if reason is not None:
                        raise RuntimeError(reason)
                if attempt > 1:
                    return self._annotate_navigation_retry(result, attempt)
                return result
            except (RuntimeError, httpx.HTTPError, TimeoutError) as error:
                reason = self._transient_navigation_reason(error)
                if reason is None:
                    raise
                last_reason = reason
                if attempt >= self._navigation_attempts:
                    raise RuntimeError(
                        f"browser_navigate could not open {url!r} after {attempt} "
                        f"attempts ({attempt - 1} automatic retries for transient "
                        f"network failures). Last network error: {last_reason}"
                    ) from error
                time.sleep(
                    BROWSER_NAVIGATION_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                )
        raise AssertionError("unreachable navigation retry state")

    def _navigate_once(self, url: str) -> str | dict[str, Any]:
        """Perform one navigation attempt or complete one URL-triggered download."""
        result = self._download_interaction("browser_navigate", {"url": url}, "navigation")
        if isinstance(result, str):
            return self._remember_clickable_elements(result)
        return result

    def _download_interaction(self, tool: str, arguments: dict[str, Any], trigger: str):
        """Arm a listener for every action, including extensionless and redirected URLs."""
        before = {str(item["guid"]) for item in self._get_download_list_raw() if item.get("guid")}
        action_error = None
        message = ""
        response = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiter = executor.submit(
                self.client.browser_page.wait, type="download",
                timeout=DOWNLOAD_LISTENER_SECONDS,
                request_options={"timeout_in_seconds": int(DOWNLOAD_LISTENER_SECONDS + 15)},
            )
            time.sleep(0.25)
            try:
                message = self._text(self._call(tool, arguments))
            except (RuntimeError, httpx.HTTPError) as exc:
                # Chromium can report ERR_ABORTED when navigation becomes a download.
                action_error = exc
            try:
                response = _plain(waiter.result(timeout=DOWNLOAD_LISTENER_SECONDS + 15))
            except (RuntimeError, httpx.HTTPError, TimeoutError):
                pass  # Inspect the download registry even when the listener times out.
        data = response.get("data") if isinstance(response, dict) and response.get("success") else None
        if not isinstance(data, dict):
            data = self._new_download(before, timeout=DOWNLOAD_DETECTION_SECONDS)
        if data is not None:
            return self._completed_download_result(message, data, trigger=trigger)
        if action_error is not None:
            raise action_error
        return message

    def _wait_until_ready(self, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        last_error = "browser CDP is not ready"
        while time.monotonic() < deadline:
            try:
                remaining = max(1, int(deadline - time.monotonic()))
                self._call(
                    "browser_navigate",
                    {"url": "about:blank"},
                    timeout_seconds=min(5, remaining),
                )
                return
            except (RuntimeError, httpx.HTTPError) as error:
                last_error = str(error)
                time.sleep(0.25)
        raise RuntimeError(
            f"Browser did not become ready within {timeout:g}s: {last_error}"
        )

    def get_markdown(self) -> str:
        return self._text(self._call("browser_get_markdown", {}))

    def get_clickable_elements(self) -> str:
        return self._remember_clickable_elements(
            self._text(self._call("browser_get_clickable_elements", {}))
        )

    def click(self, index: int) -> dict[str, Any]:
        """Click an indexed element and describe the observed consequence."""
        if index not in self._clickable_elements:
            raise ValueError(
                "click index is not present in the current browser snapshot; "
                "call browser_get_clickable_elements first"
            )
        result = self._download_interaction("browser_click", {"index": index}, "click")
        if isinstance(result, dict):
            return result
        return {"action": "click", "status": "completed", "message": result}

    def _new_download(
        self, downloads_before: set[str], *, timeout: float
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while True:
            download = next(
                (
                    item
                    for item in reversed(self._get_download_list_raw())
                    if item.get("guid")
                    and str(item["guid"]) not in downloads_before
                ),
                None,
            )
            if download is not None or time.monotonic() >= deadline:
                return download
            time.sleep(0.1)

    def _completed_download_result(
        self, interaction_message: str, data: dict[str, Any], *, trigger: str
    ) -> dict[str, Any]:
        deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
        while data.get("state") not in (None, "completed"):
            if data.get("state") in {"canceled", "cancelled", "interrupted", "failed"}:
                raise RuntimeError(f"Download failed: {data.get('state')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("download did not complete within 180 seconds")
            guid = data.get("guid")
            if not guid:
                raise RuntimeError("incomplete download has no guid")
            time.sleep(0.25)
            updated = next((item for item in self._get_download_list_raw()
                            if item.get("guid") == guid), None)
            if updated is None:
                raise RuntimeError("download disappeared before completion")
            data = updated
        filename = PurePosixPath(
            str(data.get("suggested_filename") or data.get("suggestedFilename") or "")
        ).name
        if not filename:
            raise RuntimeError("completed download has no filename")
        saved_path = PurePosixPath(
            str(data.get("saved_path") or data.get("resourceUri") or "")
        )
        download_root = PurePosixPath("/home/gem/Downloads")
        try:
            relative_path = str(saved_path.relative_to(download_root))
        except ValueError:
            relative_path = filename
        target = self.download_dir / Path(*PurePosixPath(relative_path).parts)
        if ".." in PurePosixPath(relative_path).parts or not self._path_is_within(
            target.resolve(), self.download_dir.resolve()
        ):
            raise RuntimeError("download path must stay within current task Downloads")
        if target.suffix.lower() in {".crdownload", ".partial"}:
            raise RuntimeError("download is still a temporary file, not a completed download")
        if not target.is_file():
            source = str(data.get("saved_path") or data.get("resourceUri") or "")
            if not source:
                raise RuntimeError("completed download has no readable file path")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".partial",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    for chunk in self.client.file.download_file(
                        path=source,
                        request_options={"timeout_in_seconds": 240},
                    ):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, target)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
        with target.open("rb") as stream:
            is_pdf = stream.read(5) == b"%PDF-"
        if (target.suffix.lower() == ".pdf" or PurePosixPath(filename).suffix.lower() == ".pdf") and not is_pdf:
            raise RuntimeError("Downloaded .pdf is not a PDF (missing %PDF- header); it may be an error or login page")
        if is_pdf and target.suffix.lower() != ".pdf":
            # Readers dispatch by suffix; API downloads may not have a .pdf extension.
            renamed = target.with_name(target.name + ".pdf")
            counter = 1
            while renamed.exists():
                renamed = target.with_name(f"{target.name}.{counter}.pdf")
                counter += 1
            target.rename(renamed)
            target = renamed
            relative_path = target.relative_to(self.download_dir).as_posix()
        self._download_paths[filename] = relative_path
        return {
            "action": "download",
            "status": "completed",
            "path": relative_path,
            "filename": target.name,
            "size_bytes": target.stat().st_size,
            "verified_pdf": is_pdf,
            "message": (
                f"{interaction_message}\n"
                f"The {trigger} triggered a download and it completed: {filename}. "
                f"Read it with browser_get_download_content(path={relative_path!r})."
            ).strip(),
        }

    def form_input_fill(self, index: int, value: str, *, clear: bool = False) -> str:
        if index not in self._clickable_elements:
            raise ValueError(
                "form input index is not present in the current browser snapshot; "
                "call browser_get_clickable_elements first"
            )
        return self._text(
            self._call(
                "browser_form_input_fill",
                {"index": index, "value": value, "clear": clear},
            )
        )

    def press_key(self, key: str) -> str:
        return self._text(self._call("browser_press_key", {"key": key}))

    def scroll(self, amount: int | None = None) -> str:
        request = {} if amount is None else {"amount": amount}
        return self._text(self._call("browser_scroll", request))

    def screenshot(self, name: str = "screenshot") -> dict[str, str]:
        result = self._call("browser_screenshot", {"name": name})
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "image":
                return {
                    "type": "image",
                    "mime_type": str(item.get("mime_type") or "image/png"),
                    "data_base64": str(item.get("data") or ""),
                }
        raise RuntimeError("browser_screenshot returned no image")

    def _get_download_list_raw(self) -> list[dict[str, Any]]:
        result = self._call("browser_get_download_list", {})
        structured = result.get("structured_content")
        if isinstance(structured, dict) and isinstance(structured.get("list"), list):
            return [dict(item) for item in structured["list"]]
        payload = _json_output(self._text(result))
        return [dict(item) for item in payload.get("list") or []]

    def get_download_list(self, path: str | None = None) -> list[dict[str, Any]]:
        """List one task-local Downloads directory without following symlinks."""
        directory = self._resolve_download_path(path, expected="directory")
        root = self.download_dir.resolve()
        records: dict[str, dict[str, Any]] = {}
        if directory == root:
            for item in self._get_download_list_raw():
                filename = PurePosixPath(
                    str(item.get("suggestedFilename") or "")
                ).name
                relative_path = self._download_paths.get(filename, filename)
                records[relative_path] = item

        downloads = []
        for target in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if target.is_symlink():
                continue
            if target.is_file() and target.suffix.lower() in {".crdownload", ".partial"}:
                continue
            resolved = target.resolve()
            if not self._path_is_within(resolved, root):
                continue
            relative_path = resolved.relative_to(root).as_posix()
            record = records.get(relative_path, {})
            is_directory = resolved.is_dir()
            downloads.append(
                {
                    "filename": target.name,
                    "path": relative_path,
                    "type": "directory" if is_directory else "file",
                    "state": "available"
                    if is_directory
                    else str(record.get("state") or "completed"),
                    "progress": None if is_directory else record.get("progress", 1.0),
                    "size_bytes": None if is_directory else resolved.stat().st_size,
                }
            )
        return downloads

    def _resolve_download_path(
        self, path: str | None, *, expected: str
    ) -> Path:
        root = self.download_dir.resolve()
        if path in (None, "", "."):
            target = root
        else:
            relative = PurePosixPath(str(path))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("download path must stay within current task Downloads")
            target = (root / Path(*relative.parts)).resolve()
        if not self._path_is_within(target, root, allow_root=True):
            raise ValueError("download path must stay within current task Downloads")
        if expected == "directory" and not target.is_dir():
            raise ValueError("download directory not found")
        if expected == "file" and not target.is_file():
            raise ValueError("download file not found")
        return target

    @staticmethod
    def _path_is_within(
        target: Path, root: Path, *, allow_root: bool = False
    ) -> bool:
        try:
            target.relative_to(root)
        except ValueError:
            return False
        return allow_root or target != root

    def _get_download_bytes(self, path: str) -> bytes:
        if PurePosixPath(path).suffix.lower() in {".crdownload", ".partial"}:
            raise ValueError("download has not completed")
        content = self._resolve_download_path(path, expected="file").read_bytes()
        if PurePosixPath(path).suffix.lower() == ".pdf" and not content.startswith(b"%PDF-"):
            raise ValueError("Downloaded .pdf is not a PDF (missing %PDF- header); it may be an error or login page")
        return content

    def get_download_content(
        self, path: str, query: str | None = None, *, start_line: int | None = None,
        end_line: int | None = None, start_column: int | None = None,
    ) -> list[dict[str, str]]:
        """Read/select one explicit file; no prior open or selection is required."""
        from .downloads import DownloadView
        from .file_services import UnavailableReader
        coordinates = (start_line, end_line, start_column)
        if query is not None and any(value is not None for value in coordinates):
            raise ValueError("query cannot be combined with line coordinates")
        if start_line is None and (end_line is not None or start_column is not None):
            raise ValueError("line coordinates require start_line")
        for value in coordinates:
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("line coordinates must be positive integers")
        # A fresh view for every request: path, not previous tool calls, selects the file.
        self.download_view = DownloadView(self.download_dir, self.download_view.config)
        content = self._get_download_bytes(path)
        try:
            parts = self._file_readers.to_message_content(path, content)
        except UnavailableReader as exc:
            self.download_view.path, self.download_view.text_file = None, None
            self.download_view.metadata = {"path": path, "supported": False, "warning": str(exc)}
            return [{"type": "text", "text": f"Unsupported file type for this run: {path}. Optional reader unavailable: {exc}"}]
        shown = self.download_view.open(path, parts)
        if query is not None or start_line is not None:
            if self.download_view.text_file is None:
                raise ValueError("text selection requires a file with readable text")
            result = (self.download_view.search(query) if query is not None else
                      self.download_view.lines(start_line, end_line, start_column or 1))
            for key in ("next_line", "next_column"):
                self.download_view.metadata.pop(key, None)
            self.download_view.metadata["truncated"] = result.get("truncated", result.get("has_more", False))
            self.download_view.metadata.update(result)
            return [{"type": "text", "text": result["content"]}]
        return shown


class YouTubeSession:
    def __init__(self, tools: YouTubeTools, url: str, video_id: str) -> None:
        self.tools = tools
        self.url = url
        self.video_id = video_id
        output = tools.shell.run(
            [
                "yt-dlp",
                "--ignore-config",
                *tools.common_args,
                "--no-playlist",
                "--skip-download",
                "--format",
                VIDEO_FORMAT,
                "--dump-single-json",
                "--no-warnings",
                url,
            ]
        )
        self.metadata = _json_output(output)
        self.duration = float(self.metadata["duration"])
        self.format_id = str(self.metadata["format_id"])

    @staticmethod
    def _catalog(value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        return {
            str(language): sorted(
                {
                    str(item["ext"])
                    for item in formats
                    if isinstance(item, dict) and item.get("ext")
                }
            )
            for language, formats in value.items()
            if isinstance(formats, list)
        }

    def _list_subtitles(self) -> dict[str, dict[str, list[str]]]:
        return {
            "manual": self._catalog(self.metadata.get("subtitles")),
            "automatic": self._catalog(self.metadata.get("automatic_captions")),
        }

    @staticmethod
    def _family(catalog: dict[str, Any], family: str, source: str) -> list[str]:
        keys = set(catalog)
        if family == "en":
            candidates = {key for key in keys if key == "en" or key.startswith("en-")}
            order = (
                ["en", "en-orig", "en-US", "en-GB"]
                if source == "manual"
                else ["en-orig", "en", "en-US", "en-GB"]
            )
        elif family == "zh":
            candidates = {key for key in keys if key == "zh" or key.startswith("zh-")}
            order = ["zh-Hans", "zh-CN", "zh", "zh-Hant", "zh-TW"]
        else:
            candidates = {
                key
                for key in keys
                if not key.startswith(("en-", "zh-"))
                and key not in {"en", "zh", "live_chat"}
            }
            order = sorted(candidates)
        return [
            *filter(candidates.__contains__, order),
            *sorted(candidates.difference(order)),
        ]

    def _select_subtitle(
        self, language: str | None, source: str
    ) -> tuple[str, str] | None:
        catalogs = self._list_subtitles()
        sources = ["manual", "automatic"] if source == "preferred" else [source]
        if language is not None:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", language):
                raise ValueError("invalid subtitle language")
            for candidate_source in sources:
                if language in catalogs[candidate_source]:
                    return language, candidate_source
            return None
        for family in ("en", "zh", "other"):
            for candidate_source in sources:
                candidates = self._family(
                    catalogs[candidate_source], family, candidate_source
                )
                if candidates:
                    return candidates[0], candidate_source
        return None

    def _read_subtitle(
        self,
        language: str | None = None,
        *,
        source: str = "preferred",
    ) -> dict[str, Any]:
        if source not in {"preferred", "manual", "automatic"}:
            raise ValueError("source must be preferred, manual, or automatic")
        selected = self._select_subtitle(language, source)
        if selected is None:
            return {
                "video_id": self.video_id,
                "language": language,
                "source": source,
                "available": False,
                "reason": "no matching subtitles",
            }
        language, selected_source = selected
        root = f"{INTERNAL_YOUTUBE_CACHE}/{self.video_id}/subtitles/{selected_source}"
        lock = (
            self.tools.cache_dir
            / ".locks"
            / "subtitles"
            / f"{self.video_id}-{selected_source}-{language}.lock"
        )
        with _file_lock(lock):
            files = self.tools._glob(f"{root}/{self.video_id}.{language}.*")
            if not files:
                option = (
                    "--write-subs"
                    if selected_source == "manual"
                    else "--write-auto-subs"
                )
                self.tools.shell.run(
                    [
                        "yt-dlp",
                        "--ignore-config",
                        *self.tools.common_args,
                        "--no-playlist",
                        "--skip-download",
                        option,
                        "--sub-langs",
                        language,
                        "--sub-format",
                        "json3/vtt/best",
                        "--paths",
                        root,
                        "--output",
                        "%(id)s.%(ext)s",
                        self.url,
                    ],
                    180,
                )
                files = self.tools._glob(f"{root}/{self.video_id}.{language}.*")
        if not files:
            return {
                "video_id": self.video_id,
                "language": language,
                "source": selected_source,
                "available": False,
                "reason": "yt-dlp produced no subtitle file",
            }
        path = files[0]
        data = self.tools._read_binary(path)
        extension = Path(path).suffix.lstrip(".").lower()
        return {
            "available": True,
            "video_id": self.video_id,
            "language": language,
            "source": selected_source,
            "format": extension,
            "mime_type": {
                "json3": "application/json",
                "vtt": "text/vtt",
                "srt": "application/x-subrip",
            }.get(extension, "application/octet-stream"),
            "data_base64": base64.b64encode(data).decode(),
        }

    def _read_video(
        self,
        *,
        num_frames: int = 10,
        start: float | None = None,
        end: float | None = None,
    ) -> dict[str, Any]:
        if not 1 <= num_frames <= 100:
            raise ValueError("num_frames must be between 1 and 100")
        if start is None and end is None:
            mode = "global"
            selected_start, selected_end = 0.0, self.duration
            timestamps = [
                self.duration * (i + 0.5) / num_frames for i in range(num_frames)
            ]
        elif start is not None and end is not None:
            start = float(start)
            end = min(float(end), self.duration)
            if not 0 <= start < end:
                raise ValueError(
                    "local sampling requires "
                    "0 <= start_seconds < end_seconds <= duration_seconds; "
                    f"duration_seconds={self.duration:.3f}, "
                    f"start_seconds={start!r}, end_seconds={end!r}. "
                    "Provide both start_seconds and end_seconds, or omit both "
                    "for a global overview."
                )
            mode = "range"
            selected_start, selected_end = start, end
            timestamps = _range_frame_timestamps(
                selected_start, selected_end, num_frames
            )
        else:
            raise ValueError(
                "start_seconds and end_seconds must be provided together, "
                "or both omitted for a global overview"
            )
        source_url = str(self.metadata["url"])
        file_size = int(
            self.metadata.get("filesize") or self.metadata.get("filesize_approx") or 0
        )
        if file_size <= 0:
            raise RuntimeError("selected YouTube format has no usable size")
        fingerprint = hashlib.sha256(
            f"{self.video_id}\0{self.format_id}\0{file_size}\0{self.duration}".encode()
        ).hexdigest()
        payload = {
            "video_id": self.video_id,
            "format_id": self.format_id,
            "source_url": source_url,
            "file_size": file_size,
            "headers": self.metadata.get("http_headers") or {},
            "source_fingerprint": fingerprint,
            "timestamps": timestamps,
            "proxy_server": self.tools.proxy_server,
        }
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        result = _json_output(
            self.tools.shell.run(
                ["python3", "/opt/toolbox/range_cache_runtime.py", encoded],
                max(180, num_frames * 120),
            )
        )
        frames = []
        for frame in result["frames"]:
            data = self.tools._read_binary(frame["path"])
            frames.append(
                {
                    "timestamp": frame["timestamp"],
                    "mime_type": "image/jpeg",
                    "data_base64": base64.b64encode(data).decode(),
                }
            )
        return {
            "video_id": self.video_id,
            "source_type": "youtube",
            "mode": mode,
            "start": selected_start,
            "end": selected_end,
            "frames": frames,
        }


class LocalVideoSession:
    def __init__(
        self,
        tools: YouTubeTools,
        video_id: str,
        path: str,
        fingerprint: str,
        subtitle: str | None = None,
    ) -> None:
        self.tools = tools
        self.video_id = video_id
        self.path = path
        self.fingerprint = fingerprint
        self.subtitle = subtitle
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("invalid local input video fingerprint")
        output = tools.shell.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ]
        )
        self.duration = float(output.strip())

    def _read_video(
        self,
        *,
        num_frames: int = 10,
        start: float | None = None,
        end: float | None = None,
    ) -> dict[str, Any]:
        if not 1 <= num_frames <= 100:
            raise ValueError("num_frames must be between 1 and 100")
        if start is None and end is None:
            mode = "global"
            selected_start, selected_end = 0.0, self.duration
            timestamps = [
                self.duration * (index + 0.5) / num_frames
                for index in range(num_frames)
            ]
        elif start is not None and end is not None:
            start = float(start)
            end = min(float(end), self.duration)
            if not 0 <= start < end:
                raise ValueError(
                    "local sampling requires "
                    "0 <= start_seconds < end_seconds <= duration_seconds; "
                    f"duration_seconds={self.duration:.3f}, "
                    f"start_seconds={start!r}, end_seconds={end!r}."
                )
            mode = "range"
            selected_start, selected_end = start, end
            timestamps = _range_frame_timestamps(start, end, num_frames)
        else:
            raise ValueError(
                "start_seconds and end_seconds must be provided together, "
                "or both omitted for a global overview"
            )
        payload = {
            "video_id": self.video_id,
            "local_path": self.path,
            "source_fingerprint": self.fingerprint,
            "timestamps": timestamps,
        }
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        result = _json_output(
            self.tools.shell.run(
                ["python3", "/opt/toolbox/range_cache_runtime.py", encoded],
                max(180, num_frames * 120),
            )
        )
        frames = []
        for frame in result["frames"]:
            data = self.tools._read_binary(frame["path"])
            frames.append(
                {
                    "timestamp": frame["timestamp"],
                    "mime_type": "image/jpeg",
                    "data_base64": base64.b64encode(data).decode(),
                }
            )
        return {
            "video_id": self.video_id,
            "source_type": "input_video",
            "mode": mode,
            "start": selected_start,
            "end": selected_end,
            "frames": frames,
        }

    def _read_subtitle(self) -> dict[str, Any]:
        if not self.subtitle:
            return {
                "available": False,
                "video_id": self.video_id,
                "language": None,
                "source": "parquet",
                "reason": "no subtitle for this input video in configured Parquet",
            }
        return {
            "available": True,
            "video_id": self.video_id,
            "language": None,
            "source": "parquet",
            "format": "text",
            "mime_type": "text/plain",
            "data_base64": base64.b64encode(self.subtitle.encode()).decode(),
        }


class YouTubeTools:
    def __init__(
        self,
        shell: _Shell,
        *,
        cookie_file: str | None,
        proxy_server: str | None,
        cache_dir: Path,
        input_video: dict[str, Any] | None = None,
        subtitle_max_entries: int = 50,
    ) -> None:
        self.shell = shell
        self.proxy_server = proxy_server
        self.cache_dir = cache_dir
        self.input_video = dict(input_video or {})
        if isinstance(subtitle_max_entries, bool) or not isinstance(subtitle_max_entries, int) or subtitle_max_entries < 1:
            raise ValueError("subtitle_max_entries must be a positive integer")
        self.subtitle_max_entries = subtitle_max_entries
        self.common_args = []
        if cookie_file:
            self.common_args += ["--cookies", cookie_file]
        if proxy_server:
            self.common_args += ["--proxy", proxy_server]
        self.sessions: dict[str, YouTubeSession | LocalVideoSession] = {}

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or not 1 <= max_results <= 10:
            raise ValueError("query is required and max_results must be 1..10")
        data = _json_output(
            self.shell.run(
                [
                    "yt-dlp",
                    "--ignore-config",
                    *self.common_args,
                    "--flat-playlist",
                    "--dump-single-json",
                    "--no-warnings",
                    f"ytsearch{max_results}:{query.strip()}",
                ]
            )
        )
        return [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "url": item.get("webpage_url")
                or f"https://www.youtube.com/watch?v={item.get('id')}",
                "duration_seconds": item.get("duration"),
            }
            for item in (data.get("entries") or [])[:max_results]
        ]

    def _open(self, value: str) -> YouTubeSession | LocalVideoSession:
        if value == INPUT_VIDEO_ID:
            if not self.input_video:
                raise ValueError("this task has no local input video")
            if value not in self.sessions:
                self.sessions[value] = LocalVideoSession(
                    self,
                    value,
                    self.input_video["path"],
                    self.input_video["sha256"],
                    self.input_video.get("subtitle"),
                )
            return self.sessions[value]
        url, video_id = _youtube_url(value)
        if video_id not in self.sessions:
            self.sessions[video_id] = YouTubeSession(self, url, video_id)
        return self.sessions[video_id]

    def _glob(self, pattern: str) -> list[str]:
        encoded = base64.urlsafe_b64encode(pattern.encode()).decode()
        code = (
            "import base64,glob,json,sys;"
            "p=base64.urlsafe_b64decode(sys.argv[1]).decode();"
            "print(json.dumps(sorted(glob.glob(p))))"
        )
        return json.loads(self.shell.run(["python3", "-c", code, encoded]))

    def _read_binary(self, path: str) -> bytes:
        encoded = base64.urlsafe_b64encode(path.encode()).decode()
        code = (
            "import base64,sys;from pathlib import Path;"
            "p=Path(base64.urlsafe_b64decode(sys.argv[1]).decode()).resolve();"
            f"root=Path('{INTERNAL_YOUTUBE_CACHE}').resolve();"
            "assert root in p.parents and p.is_file();"
            "print(base64.b64encode(p.read_bytes()).decode())"
        )
        return base64.b64decode(
            self.shell.run(["python3", "-c", code, encoded]).strip(),
            validate=True,
        )

    def _read_input_binary(self, path: str) -> bytes:
        encoded = base64.urlsafe_b64encode(path.encode()).decode()
        code = (
            "import base64,sys;from pathlib import Path;"
            "p=Path(base64.urlsafe_b64decode(sys.argv[1]).decode()).resolve();"
            f"root=Path('{INTERNAL_INPUT_VIDEO_ROOT}').resolve();"
            "assert root in p.parents and p.is_file();"
            "print(base64.b64encode(p.read_bytes()).decode())"
        )
        return base64.b64decode(
            self.shell.run(["python3", "-c", code, encoded]).strip(),
            validate=True,
        )

    def read_video(
        self,
        value: str,
        *,
        num_frames: int = 10,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Sample globally, or honor a jointly supplied start/end range."""
        if (start_seconds is None) != (end_seconds is None):
            raise ValueError(
                "start_seconds and end_seconds must be provided together; "
                "omit both for a global overview"
            )
        session = self._open(value)
        requested_end = float(end_seconds) if end_seconds is not None else None
        result = session._read_video(
            num_frames=num_frames,
            start=start_seconds,
            end=end_seconds,
        )
        selected_start = float(result.pop("start"))
        selected_end = float(result.pop("end"))
        result["duration_seconds"] = session.duration
        result["start_seconds"] = selected_start
        result["end_seconds"] = selected_end
        result["end_was_clamped"] = bool(
            requested_end is not None and requested_end > session.duration
        )
        return result

    def read_subtitle(self, value: str, *, start_seconds: float | None = None,
                      end_seconds: float | None = None) -> dict[str, Any]:
        """Read one preferred subtitle track independently of video frames."""
        from .subtitles import select_subtitles, validate_range
        validate_range(start_seconds, end_seconds)
        session = self._open(value)
        return select_subtitles(session._read_subtitle(), start_seconds, end_seconds,
                                session.duration, self.subtitle_max_entries)


class Toolbox:
    """Manage one task-scoped Browser and YouTube container pair."""

    def __init__(
        self,
        task_id: str,
        *,
        data_dir: str | Path = "toolbox-data",
        browser_image: str = BROWSER_IMAGE,
        youtube_image: str = YOUTUBE_IMAGE,
        youtube_cache_dir: str | Path | None = None,
        youtube_cookie_file: str | Path | None = None,
        proxy_server: str | None = None,
        container_network: str | None = None,
        docker_host: str | None = None,
        docker_command: str = "docker",
        input_video_id: str | None = None,
        input_video_path: str | Path | None = None,
        input_video_sha256: str | None = None,
        input_video_subtitle: str | None = None,
        browser_navigation_attempts: int = BROWSER_NAVIGATION_ATTEMPTS,
        file_readers: dict | None = None,
        subtitle_max_entries: int = 50,
    ) -> None:
        if not SAFE_NAME.fullmatch(task_id):
            raise ValueError("task_id contains unsupported characters")
        self.task_id = task_id
        self.subtitle_max_entries = subtitle_max_entries
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.download_dir = self.data_dir / "downloads" / task_id
        self.youtube_cache_dir = (
            Path(youtube_cache_dir).expanduser().resolve()
            if youtube_cache_dir is not None
            else self.data_dir / "cache" / "youtube"
        )
        for path in (self.download_dir, self.youtube_cache_dir):
            path.mkdir(parents=True, exist_ok=True)
        # Containers run as `gem`, while the host process owns the bind mount.
        # The cache contains only reproducible public media data and must be
        # writable by every task-scoped YouTube container sharing it.
        self.youtube_cache_dir.chmod(0o777)
        cookie = (
            Path(youtube_cookie_file).expanduser().resolve()
            if youtube_cookie_file
            else None
        )
        if cookie is not None and not cookie.is_file():
            raise ValueError("youtube_cookie_file does not exist")
        input_values = (input_video_id, input_video_path, input_video_sha256)
        if any(input_values) and not all(input_values):
            raise ValueError("input video id, path, and sha256 are required together")
        if input_video_id is not None and input_video_id != INPUT_VIDEO_ID:
            raise ValueError(f"input_video_id must be {INPUT_VIDEO_ID!r}")
        self._input_video_source = (
            Path(input_video_path).expanduser().resolve()
            if input_video_path is not None
            else None
        )
        if self._input_video_source is not None and not self._input_video_source.is_file():
            raise ValueError("input_video_path does not exist")
        if input_video_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", input_video_sha256
        ):
            raise ValueError("invalid input_video_sha256")
        self._input_video_sha256 = input_video_sha256
        if input_video_subtitle is not None and not isinstance(input_video_subtitle, str):
            raise ValueError("input_video_subtitle must be a string")
        if input_video_subtitle is not None and self._input_video_source is None:
            raise ValueError("input_video_subtitle requires a local input video")
        self._input_video_subtitle = input_video_subtitle or None
        if browser_navigation_attempts < 1:
            raise ValueError("browser_navigation_attempts must be at least 1")
        self._browser_navigation_attempts = browser_navigation_attempts
        suffix = re.sub(r"[^a-z0-9_.-]", "-", task_id.lower())
        self._docker = _docker_client(docker_host)
        self._browser_container = _Container(
            self._docker,
            f"toolbox-browser-{suffix}",
            browser_image,
            docker_command=docker_command,
        )
        self._youtube_container = _Container(
            self._docker,
            f"toolbox-youtube-{suffix}",
            youtube_image,
            docker_command=docker_command,
        )
        self._cookie_source = cookie
        self._cookie_secret: Path | None = None
        self._cookie_temp_dir: tempfile.TemporaryDirectory | None = None
        self._proxy = proxy_server
        self._container_network = container_network
        self._http_clients: list[httpx.Client] = []
        self.cleanup_errors: list[str] = []
        self.startup_timings: dict[str, Any] = {}
        self._file_reader_config = file_readers or {}
        self.file_readers = create_file_reader_registry(self.download_dir, self._file_reader_config)
        self.browser: BrowserTools
        self.youtube: YouTubeTools

    def __enter__(self) -> Toolbox:
        toolbox_started = time.monotonic()
        try:
            proxy_environment: dict[str, str] = {}
            if self._proxy:
                proxy_environment = {
                    "PROXY_SERVER": self._proxy,
                    "PROXY_EXCLUDE": "127.0.0.1,localhost",
                    "HTTP_PROXY": self._proxy,
                    "HTTPS_PROXY": self._proxy,
                    "ALL_PROXY": self._proxy,
                    "http_proxy": self._proxy,
                    "https_proxy": self._proxy,
                    "all_proxy": self._proxy,
                    "NO_PROXY": "127.0.0.1,localhost",
                    "no_proxy": "127.0.0.1,localhost",
                }
            if self._cookie_source is not None:
                self._cookie_temp_dir = tempfile.TemporaryDirectory(
                    prefix=f"toolbox-{self.task_id}-"
                )
                secret_dir = Path(self._cookie_temp_dir.name)
                self._cookie_secret = secret_dir / "youtube-cookies.txt"
                shutil.copyfile(self._cookie_source, self._cookie_secret)
                self._cookie_secret.chmod(0o600)
            urls = {
                "browser": self._browser_container.start(
                    download_dir=self.download_dir,
                    environment={
                        "DISABLE_CODE_SERVER": "true",
                        "DISABLE_JUPYTER": "true",
                        "DISABLE_NODEJS_REPL": "true",
                        **proxy_environment,
                    },
                    network=self._container_network,
                )
            }
            self.startup_timings["browser_container"] = dict(
                self._browser_container.startup_timings
            )
            urls["youtube"] = self._youtube_container.start(
                cache_dir=self.youtube_cache_dir,
                cookie_file=self._cookie_secret,
                input_video_dir=(
                    self._input_video_source.parent
                    if self._input_video_source is not None
                    else None
                ),
                environment={
                    "DISABLE_BROWSER": "true",
                    "DISABLE_CODE_SERVER": "true",
                    "DISABLE_JUPYTER": "true",
                    "DISABLE_MCP_BROWSER": "true",
                    "DISABLE_NODEJS_REPL": "true",
                    "DISABLE_VNC": "true",
                    **proxy_environment,
                },
                network=self._container_network,
            )
            self.startup_timings["youtube_container"] = dict(
                self._youtube_container.startup_timings
            )
            clients: dict[str, Sandbox] = {}
            for service, url in urls.items():
                context_started = time.monotonic()
                http = httpx.Client(timeout=90, follow_redirects=True, trust_env=False)
                self._http_clients.append(http)
                client = Sandbox(base_url=url, httpx_client=http)
                _data(client.sandbox.get_context())
                clients[service] = client
                self.startup_timings[f"{service}_sdk_context_seconds"] = round(
                    time.monotonic() - context_started, 3
                )
            self.browser = BrowserTools(
                clients["browser"],
                self.download_dir,
                self.file_readers,
                navigation_attempts=self._browser_navigation_attempts,
                file_reader_config=self._file_reader_config,
            )
            cdp_started = time.monotonic()
            self.browser._wait_until_ready()
            self.startup_timings["browser_cdp_ready_seconds"] = round(
                time.monotonic() - cdp_started, 3
            )
            youtube_shell = _Shell(clients["youtube"], working_dir="/home/gem")
            cookie_path = None
            if self._cookie_secret is not None:
                youtube_shell.run(
                    [
                        "install",
                        "-D",
                        "-m",
                        "600",
                        "/run/secrets/youtube-cookies.txt",
                        INTERNAL_COOKIE,
                    ]
                )
                cookie_path = INTERNAL_COOKIE
                self._remove_cookie_secret()
            self.youtube = YouTubeTools(
                youtube_shell,
                cookie_file=cookie_path,
                proxy_server=self._proxy,
                cache_dir=self.youtube_cache_dir,
                subtitle_max_entries=self.subtitle_max_entries,
                input_video=(
                    {
                        "path": (
                            f"{INTERNAL_INPUT_VIDEO_ROOT}/"
                            f"{self._input_video_source.name}"
                        ),
                        "sha256": str(self._input_video_sha256),
                        "subtitle": self._input_video_subtitle,
                    }
                    if self._input_video_source is not None
                    else None
                ),
            )
            self.startup_timings["total_seconds"] = round(
                time.monotonic() - toolbox_started, 3
            )
            return self
        except BaseException:
            self.startup_timings.setdefault(
                "browser_container",
                dict(self._browser_container.startup_timings),
            )
            self.startup_timings.setdefault(
                "youtube_container",
                dict(self._youtube_container.startup_timings),
            )
            self.startup_timings["failed_after_seconds"] = round(
                time.monotonic() - toolbox_started, 3
            )
            self.close()
            raise

    def close(self) -> None:
        for client in self._http_clients:
            client.close()
        self._http_clients.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            errors = list(
                executor.map(
                    lambda container: container.stop(),
                    (self._youtube_container, self._browser_container),
                )
            )
        self.cleanup_errors.extend(error for error in errors if error)
        if self._docker is not None:
            self._docker.close()
        self._remove_cookie_secret()

    def _remove_cookie_secret(self) -> None:
        self._cookie_secret = None
        if self._cookie_temp_dir is not None:
            self._cookie_temp_dir.cleanup()
            self._cookie_temp_dir = None

    def __exit__(self, *_: object) -> None:
        self.close()
