"""Optional HTTP converters and a process-safe, explicitly managed MinerU service."""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import httpx


@contextlib.contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_text(path: Path, text: str) -> None:
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class UnavailableReader(RuntimeError):
    pass


def mineru_service(config: dict[str, Any], cache: Path, *, stop: bool = False) -> None:
    """Create/start once across xargs workers; only explicit stop removes our container."""
    docker_config = config.get("docker") or {}
    if not docker_config.get("enabled", False):
        return
    name = str(docker_config.get("name", "opends-world-mineru"))
    import re
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", name):
        raise ValueError("invalid MinerU container name")
    command = str(docker_config.get("command", "docker"))
    command_timeout = float(docker_config.get("command_timeout_seconds", 60))
    identity = hashlib.sha256(json.dumps(docker_config, sort_keys=True).encode()).hexdigest()
    with locked(cache / "services" / f"{name}.lock"):
        inspect = subprocess.run([command, "inspect", name], capture_output=True, text=True, check=False, timeout=command_timeout)
        details = None
        if inspect.returncode == 0:
            details = json.loads(inspect.stdout)[0]
            if (details.get("Config", {}).get("Labels") or {}).get("opends-world.mineru") != identity:
                raise UnavailableReader(f"MinerU container {name} exists with a different owner/configuration")
        if stop:
            if details is not None:
                subprocess.run([command, "stop", name], check=True, capture_output=True, timeout=command_timeout)
                subprocess.run([command, "rm", name], check=True, capture_output=True, timeout=command_timeout)
            return
        if details is None:
            data_dir = Path(docker_config.get("data_dir", cache / "mineru-service")).resolve()
            data_dir.mkdir(parents=True, exist_ok=True)
            args = [command, "create", "--name", name, "--label", f"opends-world.mineru={identity}",
                    "--restart", "unless-stopped", "-p", str(docker_config.get("publish", "127.0.0.1:18000:8000")),
                    "-v", f"{data_dir}:/data", "-e", "HF_HOME=/data/huggingface", "-e", "XDG_CACHE_HOME=/data/cache"]
            if docker_config.get("gpus"):
                args += ["--gpus", str(docker_config["gpus"])]
            for kind in ("pipeline", "vlm"):
                model_dir = docker_config.get(f"{kind}_models_dir")
                if model_dir:
                    source = Path(model_dir).resolve()
                    if not source.is_dir():
                        raise UnavailableReader(f"MinerU {kind} model directory does not exist: {source}")
                    args += ["--mount", f"type=bind,source={source},target=/models/{kind},readonly"]
            startup_args = docker_config.get("args")
            if startup_args is not None and (not isinstance(startup_args, list) or
                    not all(isinstance(arg, str) for arg in startup_args)):
                raise ValueError("MinerU docker.args must be a list of strings or null")
            # Preserve the image's ENTRYPOINT and CMD (including CPU/device options).
            args += [str(docker_config.get("image", "mineru-cpu:3.4.4"))]
            if startup_args is not None:
                args += startup_args
            result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=command_timeout)
            if result.returncode:
                raise UnavailableReader("Could not create MinerU container; build the configured image first")
        if details is None or not details.get("State", {}).get("Running"):
            subprocess.run([command, "start", name], check=True, capture_output=True, timeout=command_timeout)
        deadline = time.monotonic() + float(docker_config.get("startup_timeout_seconds", 300))
        while time.monotonic() < deadline:
            try:
                with httpx.Client(trust_env=False, timeout=5) as client:
                    response = client.get(str(config["base_url"]).rstrip("/") + "/health")
                    response.raise_for_status()
                    return
            except httpx.HTTPError:
                time.sleep(1)
        raise UnavailableReader("MinerU container did not become healthy")


class OptionalConverter:
    def __init__(self, kind: str, config: dict[str, Any], cache: Path):
        self.kind, self.config, self.cache = kind, dict(config), cache
        self.checked = False
        self.failure: str | None = None

    def disable(self, reason: str) -> None:
        if self.failure is None:
            self.failure = reason
            warnings.warn(f"{self.kind} reader unavailable: {reason}; files of this type are unsupported", RuntimeWarning)

    def headers(self) -> dict[str, str]:
        headers = dict(self.config.get("headers") or {})
        key = self.config.get("api_key") or os.environ.get(str(self.config.get("api_key_env", "")), "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def check(self) -> bool:
        if self.checked:
            return self.failure is None
        self.checked = True
        c = self.config
        if not c.get("enabled", False):
            self.disable("not configured/enabled")
            return False
        if not c.get("base_url") or (self.kind == "asr" and not c.get("model")):
            self.disable("base_url and (for ASR) model are required")
            return False
        try:
            if self.kind == "mineru":
                mineru_service(c, self.cache)
            if self.kind == "asr" and c.get("request_format", "multipart") not in {"multipart", "json"}:
                raise ValueError("ASR request_format must be multipart or json")
            if c.get("check_on_start", True):
                route = "/models" if self.kind == "asr" else "/health"
                with httpx.Client(trust_env=False, timeout=float(c.get("health_timeout_seconds", 10))) as client:
                    r = client.get(str(c["base_url"]).rstrip("/") + route, headers=self.headers())
                    r.raise_for_status()
                    data = r.json()
                    if self.kind == "asr" and str(c["model"]) not in {str(x.get("id")) for x in data.get("data", [])}:
                        raise ValueError("configured ASR model is not advertised by /models")
        except Exception as exc:
            # Do not log response bodies or headers: they may contain credentials.
            self.disable(f"preflight failed ({type(exc).__name__})")
        return self.failure is None

    def convert(self, path: Path, content: bytes) -> str:
        if not self.check():
            raise UnavailableReader(self.failure)
        chunk_seconds = self.config.get("chunk_seconds", 300) if self.kind == "asr" else None
        if chunk_seconds:
            import tempfile
            try:
                if float(chunk_seconds) <= 0:
                    raise ValueError("chunk_seconds must be positive or null")
                scratch = self.cache / "tmp"
                scratch.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=scratch) as temporary:
                    root = Path(temporary)
                    source = root / ("input" + path.suffix)
                    source.write_bytes(content)
                    subprocess.run(
                        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
                         "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "segment",
                         "-segment_time", str(chunk_seconds), "-reset_timestamps", "1", str(root / "chunk-%06d.wav")],
                        check=True, capture_output=True, timeout=float(self.config.get("timeout_seconds", 600)),
                    )
                    chunks = sorted(root.glob("chunk-*.wav"))
                    if not chunks:
                        raise ValueError("no audio stream decoded")
                    return "\n\n".join(
                        f"[chunk starts at {index * float(chunk_seconds):g}s]\n" + self._request(chunk, chunk.read_bytes())
                        for index, chunk in enumerate(chunks)
                    )
            except UnavailableReader:
                raise
            except Exception as exc:
                self.disable(f"audio decoding/chunking failed ({type(exc).__name__})")
                raise UnavailableReader(self.failure) from exc
        return self._request(path, content)

    def _request(self, path: Path, content: bytes) -> str:
        c = self.config
        try:
            with httpx.Client(trust_env=False, timeout=float(c.get("timeout_seconds", 600))) as client:
                base = str(c["base_url"]).rstrip("/")
                if self.kind == "asr":
                    parameters = {k: v for k, v in (c.get("parameters") or {}).items() if v is not None}
                    parameters["model"] = str(c["model"])
                    fmt = c.get("request_format", "multipart")
                    if fmt == "json":
                        parameters["input_audio"] = {"data": base64.b64encode(content).decode(), "format": path.suffix[1:].lower()}
                        response = client.post(base + "/audio/transcriptions", headers=self.headers(), json=parameters)
                    else:
                        data = {k: (json.dumps(v) if isinstance(v, (dict, list, bool)) else str(v)) for k,v in parameters.items()}
                        if isinstance(parameters.get("timestamp_granularities"), list):
                            data.pop("timestamp_granularities", None)
                            data["timestamp_granularities[]"] = parameters["timestamp_granularities"]
                        response = client.post(base + "/audio/transcriptions", headers=self.headers(), data=data,
                                               files={"file": (path.name, content)})
                    response.raise_for_status()
                    text = str(response.json().get("text") or "")
                else:
                    parameters = {"return_md": True, "return_images": False, "response_format_zip": False,
                                  **(c.get("parameters") or {})}
                    parameters.update(return_md=True, response_format_zip=False)
                    data = {k: (str(v).lower() if isinstance(v, bool) else v) for k,v in parameters.items() if v is not None}
                    response = client.post(base + "/file_parse", headers=self.headers(), data=data,
                                           files={"files": (path.name, content)})
                    response.raise_for_status()
                    payload = response.json()
                    text = "\n\n".join(str(v.get("md_content") or "") for v in payload.get("results", {}).values())
                if not text.strip():
                    raise ValueError("converter returned no text")
                return text
        except Exception as exc:
            self.disable(f"conversion failed ({type(exc).__name__})")
            raise UnavailableReader(self.failure) from exc
