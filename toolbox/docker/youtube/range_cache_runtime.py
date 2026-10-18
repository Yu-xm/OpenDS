#!/usr/bin/env python3
"""Small in-container HTTP Range cache used by YouTubeSession.read_video()."""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import threading
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024
CACHE_ROOT = Path("/home/gem/.cache/toolbox/youtube")
INPUT_VIDEO_ROOT = Path("/home/gem/input-video")


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ChunkCache:
    def __init__(self, root: Path, payload: dict[str, Any]) -> None:
        self.root = root / "ranges"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_root = root / "locks"
        self.url = str(payload["source_url"])
        self.headers = {str(k): str(v) for k, v in payload.get("headers", {}).items()}
        self.file_size = int(payload["file_size"])
        proxy = payload.get("proxy_server")
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {"http": proxy, "https": proxy} if proxy else {}
            )
        )
        self.stats_lock = threading.Lock()
        self.network_bytes = 0
        self.range_hits = 0
        self.range_misses = 0

    def get(self, index: int) -> bytes:
        start = index * CHUNK_SIZE
        end = min(self.file_size - 1, start + CHUNK_SIZE - 1)
        expected = end - start + 1
        name = f"range-{start:012d}-{end:012d}.bin"
        path = self.root / name
        lock_path = self.lock_root / f"{name}.lock"
        with file_lock(lock_path):
            if path.is_file() and path.stat().st_size == expected:
                with self.stats_lock:
                    self.range_hits += 1
                return path.read_bytes()
            request = urllib.request.Request(
                self.url,
                headers={**self.headers, "Range": f"bytes={start}-{end}"},
            )
            with self.opener.open(request, timeout=30) as response:
                data = response.read(end - start + 2)[: end - start + 1]
                content_range = response.headers.get("Content-Range", "")
            if len(data) != expected:
                raise RuntimeError("incomplete upstream byte range")
            if start and not content_range.lower().startswith("bytes "):
                raise RuntimeError("upstream ignored HTTP Range")
            atomic_write(path, data)
            with self.stats_lock:
                self.network_bytes += len(data)
                self.range_misses += 1
            return data


def parse_range(value: str | None, file_size: int) -> tuple[int, int, bool]:
    if value is None:
        return 0, file_size - 1, False
    match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
    if match is None:
        raise ValueError("unsupported range")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else file_size - 1
    if start >= file_size or end < start:
        raise ValueError("range outside media")
    return start, min(end, file_size - 1), True


def handler(cache: ChunkCache) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            return

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(cache.file_size))
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            try:
                start, end, partial = parse_range(
                    self.headers.get("Range"), cache.file_size
                )
            except ValueError:
                self.send_error(416)
                return
            self.send_response(206 if partial else 200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{cache.file_size}"
                )
            self.end_headers()
            try:
                for index in range(start // CHUNK_SIZE, end // CHUNK_SIZE + 1):
                    data = cache.get(index)
                    chunk_start = index * CHUNK_SIZE
                    left = max(start, chunk_start) - chunk_start
                    right = min(end + 1, chunk_start + len(data)) - chunk_start
                    self.wfile.write(data[left:right])
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(payload: dict[str, Any]) -> dict[str, Any]:
    video_id = str(payload["video_id"])
    fingerprint = str(payload["source_fingerprint"])
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid source fingerprint")
    local_path_value = payload.get("local_path")
    if local_path_value is not None:
        if video_id != "input_video":
            raise ValueError("invalid local input video id")
        local_path = Path(str(local_path_value)).resolve()
        if INPUT_VIDEO_ROOT.resolve() not in local_path.parents or not local_path.is_file():
            raise ValueError("local input video path is outside the read-only mount")
        root = (CACHE_ROOT / "input-videos" / fingerprint).resolve()
        media_url = str(local_path)
        cache: ChunkCache | None = None
        server = None
        thread = None
    else:
        format_id = str(payload["format_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("invalid video id")
        root = (CACHE_ROOT / video_id / format_id / fingerprint).resolve()
        cache = ChunkCache(root, payload)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(cache))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        media_url = f"http://127.0.0.1:{server.server_port}/media"
    if CACHE_ROOT.resolve() not in root.parents:
        raise ValueError("invalid cache path")
    root.mkdir(parents=True, exist_ok=True)
    frame_root = root / "frames"
    frame_root.mkdir(exist_ok=True)
    frame_lock_root = root / "locks"

    def extract(timestamp: float) -> dict[str, Any]:
        name = f"frame-t{timestamp:016.6f}s.jpg"
        destination = frame_root / name
        with file_lock(frame_lock_root / f"{name}.lock"):
            hit = destination.is_file() and destination.stat().st_size > 0
            if not hit:
                temporary = destination.with_name(
                    f".{destination.stem}.partial-{os.getpid()}-{uuid.uuid4().hex}.jpg"
                )
                try:
                    result = subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-ss",
                            f"{timestamp:.6f}",
                            "-i",
                            media_url,
                            "-frames:v",
                            "1",
                            "-vf",
                            "scale='min(640,iw)':-2",
                            "-q:v",
                            "3",
                            str(temporary),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    if (
                        result.returncode
                        or not temporary.is_file()
                        or not temporary.stat().st_size
                    ):
                        raise RuntimeError(
                            f"ffmpeg produced no frame at {timestamp:.3f}s"
                        )
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
        return {
            "timestamp": timestamp,
            "path": str(destination),
            "cache_key": str(destination.relative_to(CACHE_ROOT)),
            "sha256": sha256(destination),
            "cache_hit": hit,
        }

    timestamps = [float(value) for value in payload["timestamps"]]
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(timestamps))) as executor:
            frames = list(executor.map(extract, timestamps))
    finally:
        if server is not None and thread is not None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    return {
        "frames": frames,
        "cache_key": str(root.relative_to(CACHE_ROOT)),
        "network_bytes": cache.network_bytes if cache is not None else 0,
        "range_hits": cache.range_hits if cache is not None else 0,
        "range_misses": cache.range_misses if cache is not None else 0,
        "frame_cache_hits": sum(int(frame["cache_hit"]) for frame in frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload")
    args = parser.parse_args()
    payload = json.loads(base64.urlsafe_b64decode(args.payload))
    print(json.dumps(run(payload)))


if __name__ == "__main__":
    main()
