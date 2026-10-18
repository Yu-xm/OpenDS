"""Reader for plain-text downloads."""

from .base import Utf8TextReader


class TextReader(Utf8TextReader):
    suffixes = (".txt",)
