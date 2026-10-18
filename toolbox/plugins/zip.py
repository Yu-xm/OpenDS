"""Reader for ZIP archives."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from stat import S_ISLNK

from .archive import ArchiveMember, ArchiveReader
from .registry import FileReaderRegistry


class ZipReader(ArchiveReader):
    """Extract registered regular files without following archive links."""

    suffixes = (".zip",)
    archive_label = "ZIP"

    def __init__(
        self, registry: FileReaderRegistry, download_root: Path | None = None
    ) -> None:
        super().__init__(registry, download_root)

    def iter_members(self, content: bytes):
        with zipfile.ZipFile(BytesIO(content)) as archive:
            for member in archive.infolist():
                unix_mode = member.external_attr >> 16
                yield ArchiveMember(
                    path=member.filename,
                    size=member.file_size,
                    read=lambda member=member: archive.read(member),
                    is_regular_file=(
                        not member.is_dir() and not S_ISLNK(unix_mode)
                    ),
                )
