"""Public contract implemented by every downloaded-file reader plugin."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


MessageContent = list[dict[str, str]]


class FileReader(ABC):
    """Convert one or more filename suffixes into LLM message content."""

    suffixes: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def read(
        self, filename: str, content: bytes, mime_type: str
    ) -> MessageContent:
        """Return validated text/image message content for one downloaded file."""


class Utf8TextReader(FileReader):
    """Reusable implementation for UTF-8-compatible text formats."""

    def read(
        self, filename: str, content: bytes, mime_type: str
    ) -> MessageContent:
        del filename, mime_type
        return [{"type": "text", "text": content.decode("utf-8", errors="replace")}]
