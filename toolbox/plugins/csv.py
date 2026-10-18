"""Reader for comma-separated downloads."""

from .base import Utf8TextReader


class CsvReader(Utf8TextReader):
    suffixes = (".csv",)
