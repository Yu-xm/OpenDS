from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


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
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.handle.flush()

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
        path.write_bytes(data)
    else:
        data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        path.write_bytes(data)
    return {"path": str(path), "sha256": sha256_bytes(data)}

