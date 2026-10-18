"""Reader for gzip-compressed TAR archives."""

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path

from .archive import ArchiveMember, ArchiveReader
from .registry import FileReaderRegistry


class TarGzReader(ArchiveReader):
    suffixes = (".tar.gz",)
    archive_label = "TAR.GZ"

    def __init__(
        self, registry: FileReaderRegistry, download_root: Path | None = None
    ) -> None:
        super().__init__(registry, download_root)

    def iter_members(self, content: bytes):
        with tarfile.open(fileobj=BytesIO(content), mode="r:gz") as archive:
            for member in archive.getmembers():
                def read_member(member=member) -> bytes:
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("TAR member has no readable content")
                    return stream.read()

                yield ArchiveMember(
                    path=member.name,
                    size=member.size,
                    read=read_member,
                    is_regular_file=member.isfile(),
                )
