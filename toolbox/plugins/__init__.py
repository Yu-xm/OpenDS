"""Built-in downloaded-file plugins and their public extension contract."""

from pathlib import Path

from .base import FileReader, MessageContent
from .csv import CsvReader
from .json import JsonReader
from .jsonl import JsonLinesReader
from .markdown import MarkdownReader
from .png import PngReader
from .registry import FileReaderRegistry
from .tar_gz import TarGzReader
from .text import TextReader
from .xlsx import XlsxReader
from .zip import ZipReader


def create_file_reader_registry(
    download_root: str | Path | None = None,
    config: dict | None = None,
) -> FileReaderRegistry:
    """Build the explicit set of readers bundled with Toolbox."""
    resolved_root = Path(download_root) if download_root is not None else None
    registry = FileReaderRegistry()
    for reader in (
        TextReader(),
        MarkdownReader(),
        CsvReader(),
        JsonReader(),
        JsonLinesReader(),
        PngReader(),
        XlsxReader(),
    ):
        registry.register(reader)
    registry.register(TarGzReader(registry, resolved_root))
    registry.register(ZipReader(registry, resolved_root))
    if config is not None:
        from ..downloads import ConvertedReader
        cache = Path(config.get("cache_dir", ".cache/toolbox/files")).expanduser().resolve()
        for kind in ("asr", "mineru"):
            registry.register(ConvertedReader(kind, config.get(kind) or {}, cache))
    return registry


__all__ = [
    "FileReader",
    "FileReaderRegistry",
    "MessageContent",
    "create_file_reader_registry",
]
