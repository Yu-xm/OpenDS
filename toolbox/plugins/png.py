"""Reader for PNG image downloads."""

from __future__ import annotations

import base64

from .base import FileReader, MessageContent


class PngReader(FileReader):
    suffixes = (".png",)

    def read(
        self, filename: str, content: bytes, mime_type: str
    ) -> MessageContent:
        del filename
        return [
            {
                "type": "image",
                "mime_type": mime_type,
                "data_base64": base64.b64encode(content).decode("ascii"),
            }
        ]
