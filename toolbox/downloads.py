"""Persistent converted-file views. All model-facing text is bounded here."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .file_services import OptionalConverter, UnavailableReader, atomic_text, locked
from .plugins.base import FileReader


TEXT_DEFAULTS = {
    "max_chars": 10000,
    "preview_chars": 3000,
    "max_matches": 50,
    "max_lines": 100,
    "max_output_chars": 3000,
    "regex_timeout_seconds": 10,
}


class ConvertedReader(FileReader):
    def __init__(self, kind: str, config: dict, cache: Path):
        self.converter = OptionalConverter(kind, config, cache)
        self.cache = cache / "converted"
        self.suffixes = (
            (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".aac", ".opus", ".aiff")
            if kind == "asr" else (".pdf", ".docx", ".pptx", ".jpeg", ".jpg", ".tiff", ".bmp")
        )
        # Only hashes appear on disk; request headers are separately hashed at read time.
        settings = {k: config.get(k) for k in ("base_url", "model", "request_format", "parameters", "chunk_seconds")}
        self.identity = json.dumps(["v1", kind, settings], sort_keys=True).encode()

    def read(self, filename: str, content: bytes, mime_type: str) -> list[dict[str, str]]:
        del mime_type
        if not self.converter.check():
            raise UnavailableReader(self.converter.failure)
        headers = json.dumps(self.converter.headers(), sort_keys=True).encode()
        header_digest = hashlib.sha256(headers).digest()
        key = hashlib.sha256(self.identity + header_digest + Path(filename).suffix.lower().encode() + content).hexdigest()
        target = self.cache / (key + ".txt")
        with locked(self.cache / (key + ".lock")):
            if not target.exists():
                atomic_text(target, self.converter.convert(Path(filename), content))
            text = target.read_text(encoding="utf-8")
        return [{"type": "text", "text": text}]


class DownloadView:
    def __init__(self, root: Path, config: dict | None = None):
        self.root = root / ".converted-text"
        self.config = {**TEXT_DEFAULTS, **(config or {})}
        for key in TEXT_DEFAULTS:
            value = self.config[key]
            expected = (int, float) if key == "regex_timeout_seconds" else int
            if (isinstance(value, bool) or not isinstance(value, expected) or value <= 0
                    or (isinstance(value, float) and not math.isfinite(value))):
                requirement = "a positive finite number" if key == "regex_timeout_seconds" else "a positive integer"
                raise ValueError(f"file_readers.text.{key} must be {requirement}")
        if self.config["max_output_chars"] < 32:
            raise ValueError("file_readers.text.max_output_chars must be at least 32 for line-number pagination")
        self.path: str | None = None
        self.text_file: Path | None = None
        self.metadata: dict[str, Any] = {}

    def open(self, path: str, parts: list[dict[str, str]]) -> list[dict[str, str]]:
        self.path, self.text_file = path, None
        self.metadata = {"path": path, "supported": True}
        text_parts = [p["text"] for p in parts if p["type"] == "text"]
        if not text_parts:
            return parts
        text = "\n\n".join(text_parts).replace("\r\n", "\n").replace("\r", "\n")
        self.text_file = self.root / (hashlib.sha256(path.encode()).hexdigest() + ".txt")
        atomic_text(self.text_file, text)
        lines = text.splitlines(keepends=True)
        long = len(text) > self.config["max_chars"]
        self.metadata.update(total_chars=len(text), total_lines=len(lines), truncated=long)
        if long:
            limit = int(self.config["preview_chars"])
            shown = text[:limit]
            next_line = shown.count("\n") + 1
            next_column = len(shown.rsplit("\n", 1)[-1]) + 1
            self.metadata.update(next_line=next_line, next_column=next_column)
            text = shown + (
                f"\n\n[Preview truncated: {len(text)} characters, {len(lines)} lines. "
                f"Continue with browser_get_download_content(path={json.dumps(path)}, start_line={next_line}, start_column={next_column}) "
                "or use query in the same tool to search this file. Query is a ripgrep-compatible regex pattern; "
                f"at most {self.config['max_matches']} matching lines.]"
            )
        return [{"type": "text", "text": text}, *[p for p in parts if p["type"] != "text"]]

    def _require(self) -> Path:
        if self.text_file is None:
            raise ValueError("text selection requires a file with readable text")
        return self.text_file

    def search(self, query: str) -> dict[str, Any]:
        path = self._require()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty ripgrep regex")
        rg = shutil.which("rg")
        if rg is None:
            raise ValueError("query requires ripgrep (rg) installed on the runner host")
        # A real rg process, not Python regex: no shell, flags, paths or other files exposed.
        # Stream through a file so a single huge matching line cannot fill pipe RAM.
        import tempfile
        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(
                    [rg, "--no-config", "--color", "never", "--text", "--line-number", "--no-heading",
                     "--max-count", str(int(self.config["max_matches"])), "--", query, str(path)],
                    stdout=output, stderr=subprocess.PIPE,
                    timeout=float(self.config["regex_timeout_seconds"]), check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError("query exceeded regex_timeout_seconds") from exc
            if result.returncode not in (0, 1):
                raise ValueError("invalid query: " + result.stderr.decode(errors="replace")[:500])
            output.seek(0)
            # Read enough bytes for UTF-8 and impose the final character cap separately.
            limit = int(self.config["max_output_chars"])
            content = output.read(limit * 4 + 4).decode("utf-8", errors="replace")
            truncated = len(content) > limit
        return {"path": self.path, "query": query, "content": content[:limit] or "No matches found.",
                "truncated": truncated, "max_matches": int(self.config["max_matches"]),
                "note": "First matching lines only (line_number:text). Use browser_get_download_content with the same path and start_line/end_line for more context."}

    def lines(self, start_line: int = 1, end_line: int | None = None, start_column: int = 1) -> dict[str, Any]:
        path = self._require()
        maximum = int(self.config["max_lines"])
        if end_line is None:
            end_line = start_line + maximum - 1
        if start_line < 1 or end_line < start_line or start_column < 1:
            raise ValueError("line range and start_column must be positive and ordered")
        if end_line - start_line + 1 > maximum:
            raise ValueError(f"line range exceeds max_lines={maximum}")
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if start_line > len(lines):
            raise ValueError(f"start_line exceeds total_lines={len(lines)}")
        if start_column > len(lines[start_line - 1]) + 1:
            raise ValueError("start_column exceeds the selected line")
        remaining = int(self.config["max_output_chars"])
        output = []
        next_line, next_column = min(end_line, len(lines)) + 1, 1
        for number in range(start_line, min(end_line, len(lines)) + 1):
            column = start_column if number == start_line else 1
            line = lines[number - 1][column - 1:]
            prefix = f"{number}:"
            if remaining <= len(prefix):
                next_line, next_column = number, column
                break
            chunk = line[:remaining - len(prefix)]
            output.append(prefix + chunk)
            remaining -= len(prefix) + len(chunk)
            if len(chunk) < len(line):
                next_line, next_column = number, column + len(chunk)
                break
        more = next_line <= len(lines)
        return {"path": self.path, "content": "".join(output), "total_lines": len(lines),
                "has_more": more, "next_line": next_line if more else None,
                "next_column": next_column if more else None}
