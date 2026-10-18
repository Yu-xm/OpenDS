"""Explicit registry and return-schema validation for file reader plugins."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from .base import FileReader, MessageContent


class FileReaderRegistry:
    """Register reader objects and dispatch by the longest filename suffix."""

    def __init__(self) -> None:
        self._readers: dict[str, FileReader] = {}

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        return tuple(sorted(self._readers))

    def register(self, reader: FileReader, *, replace: bool = False) -> None:
        if not isinstance(reader, FileReader):
            raise TypeError("reader must inherit FileReader")
        if not reader.suffixes:
            raise ValueError("reader must declare at least one suffix")
        normalized = []
        for suffix in reader.suffixes:
            value = suffix.strip().lower()
            if not value.startswith(".") or len(value) < 2:
                raise ValueError("reader suffix must start with '.'")
            if value in self._readers and not replace:
                raise ValueError(f"reader already registered for {value!r}")
            normalized.append(value)
        for suffix in normalized:
            self._readers[suffix] = reader

    def supports(self, filename: str | Path) -> bool:
        """Return whether a registered reader matches the complete filename."""
        return self._matching_suffix(Path(filename).name) != ""

    def to_message_content(
        self, filename: str | Path, content: bytes
    ) -> MessageContent:
        name = Path(filename).name
        suffix = self._matching_suffix(name)
        reader = self._readers.get(suffix)
        if reader is None:
            raise ValueError(f"no message reader registered for {suffix or name!r}")
        mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        parts = reader.read(name, content, mime_type)
        self._validate(parts)
        return parts

    def _matching_suffix(self, filename: str) -> str:
        lowered = filename.lower()
        return next(
            (
                candidate
                for candidate in sorted(self._readers, key=len, reverse=True)
                if lowered.endswith(candidate)
            ),
            "",
        )

    @staticmethod
    def _validate(parts: MessageContent) -> None:
        if not isinstance(parts, list) or not parts:
            raise ValueError("reader must return at least one message content part")
        for part in parts:
            if not isinstance(part, dict):
                raise TypeError("reader parts must be dictionaries")
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("text"), str):
                continue
            if (
                part_type == "image"
                and isinstance(part.get("mime_type"), str)
                and isinstance(part.get("data_base64"), str)
            ):
                continue
            raise ValueError(
                "reader parts must be text{text} or image{mime_type,data_base64}"
            )
