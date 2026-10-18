"""Reader for Markdown downloads."""

from .base import Utf8TextReader


class MarkdownReader(Utf8TextReader):
    suffixes = (".md",)
