"""Reader for JSON downloads."""

from .base import Utf8TextReader


class JsonReader(Utf8TextReader):
    suffixes = (".json",)
