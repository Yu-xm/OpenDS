from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


def read_events(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """Read the durable journal, discarding an interrupted JSON record, never its fragments.

    A signal can interrupt a line write before its newline, so a subsequent run_end
    may be appended to that broken line. Only recover a fully decodable trailing
    envelope matching the original run identity; text inside payloads stays escaped.
    """
    events: list[dict[str, Any]] = []
    discarded: list[int] = []
    identity = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            discarded.append(number)
            record = None
            offset = line.find('{"run_id":', 1)
            while offset >= 0:
                try:
                    candidate = json.loads(line[offset:])
                    if (identity is not None and candidate.get("event") == "run_end"
                            and tuple(candidate.get(k) for k in ("run_id", "benchmark", "model", "sample_id")) == identity):
                        record = candidate
                        break
                except json.JSONDecodeError:
                    pass
                offset = line.find('{"run_id":', offset + 1)
            if record is None:
                continue
        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
            raise ValueError(f"Invalid trace envelope at {path}:{number}")
        if identity is None and record["event"] == "run_start":
            identity = tuple(record.get(k) for k in ("run_id", "benchmark", "model", "sample_id"))
        events.append(record)
    return events, discarded


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TraceWriter:
    def __init__(self, path: Path, common: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.common = common
        self.handle = path.open("a", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        record = {**self.common, "event": event, "timestamp": now_iso(), **payload}
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        previous_mask = None
        if threading.current_thread() is threading.main_thread() and hasattr(signal, "pthread_sigmask"):
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM, signal.SIGINT})
        try:
            self.handle.write(encoded)
            self.handle.flush()
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def save_artifact(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(path), "sha256": sha256_bytes(data)}
