"""Reader for JSON Lines downloads."""

from .base import Utf8TextReader


class JsonLinesReader(Utf8TextReader):
    suffixes = (".jsonl",)
