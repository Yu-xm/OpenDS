"""Shared extraction contract for archive reader plugins."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .base import FileReader, MessageContent
from .registry import FileReaderRegistry


@dataclass(frozen=True)
class ArchiveMember:
    """One archive member whose bytes remain owned by the concrete reader."""

    path: str
    size: int
    read: Callable[[], bytes]
    is_regular_file: bool = True


@dataclass(frozen=True)
class ExtractionResult:
    relative_path: str
    extracted_files: int
    skipped_files: int
    reused: bool


class ArchiveReader(FileReader):
    """Extract registered file types into a bounded task-local directory."""

    archive_label = "archive"
    max_extracted_files = 20
    max_member_bytes = 10_000_000
    max_total_bytes = 20_000_000

    def __init__(
        self,
        registry: FileReaderRegistry,
        download_root: Path | None,
    ) -> None:
        self.registry = registry
        self.download_root = download_root.resolve() if download_root else None

    @abstractmethod
    def iter_members(self, content: bytes) -> Iterator[ArchiveMember]:
        """Yield members without writing them to the filesystem."""

    def read(
        self, filename: str, content: bytes, mime_type: str
    ) -> MessageContent:
        if self.download_root is None:
            raise ValueError("archive extraction requires a download root")
        result = self._extract(filename, content)
        status = "reused" if result.reused else "completed"
        return [
            {
                "type": "text",
                "text": "\n".join(
                    [
                        f"{self.archive_label} extraction {status}.",
                        f"Archive: {filename}",
                        f"Archive size: {len(content)} bytes ({mime_type})",
                        f"Extracted readable files: {result.extracted_files}",
                        f"Skipped files: {result.skipped_files}",
                        f"Extracted path: {result.relative_path}",
                        "Use browser_get_download_list(path=...) to inspect it, then "
                        "browser_get_download_content(path) to read a selected file.",
                    ]
                ),
            }
        ]

    def _extract(self, filename: str, content: bytes) -> ExtractionResult:
        assert self.download_root is not None
        extraction_root = self.download_root / ".extracted"
        extraction_root.mkdir(parents=True, exist_ok=True)
        extraction_root = extraction_root.resolve()
        if not self._is_within(extraction_root, self.download_root):
            raise ValueError("archive extraction path escaped current task Downloads")
        digest = hashlib.sha256(content).hexdigest()[:12]
        archive_name = Path(filename).name[:80]
        destination = extraction_root / f"{archive_name}-{digest}"
        if destination.exists() and not self._is_within(
            destination.resolve(), extraction_root
        ):
            raise ValueError("archive destination escaped extraction directory")
        relative_path = destination.relative_to(self.download_root).as_posix()
        if destination.is_dir():
            return ExtractionResult(
                relative_path=relative_path,
                extracted_files=sum(
                    1 for path in destination.rglob("*") if path.is_file()
                ),
                skipped_files=0,
                reused=True,
            )

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{archive_name}-{digest}-", dir=extraction_root)
        )
        extracted_files = 0
        skipped_files = 0
        total_bytes = 0
        written_paths: set[Path] = set()
        try:
            for member in self.iter_members(content):
                relative_member = self._safe_member_path(member.path)
                if (
                    relative_member is None
                    or not member.is_regular_file
                    or not self.registry.supports(member.path)
                    or member.size > self.max_member_bytes
                    or extracted_files >= self.max_extracted_files
                    or total_bytes + member.size > self.max_total_bytes
                ):
                    skipped_files += 1
                    continue

                target = (temporary / Path(*relative_member.parts)).resolve()
                if not self._is_within(target, temporary) or target in written_paths:
                    skipped_files += 1
                    continue
                try:
                    member_content = member.read()
                except (OSError, RuntimeError, ValueError):
                    skipped_files += 1
                    continue
                if (
                    len(member_content) > self.max_member_bytes
                    or total_bytes + len(member_content) > self.max_total_bytes
                ):
                    skipped_files += 1
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(member_content)
                written_paths.add(target)
                extracted_files += 1
                total_bytes += len(member_content)

            try:
                temporary.rename(destination)
            except OSError:
                if not destination.is_dir() or not self._is_within(
                    destination.resolve(), extraction_root
                ):
                    raise
                shutil.rmtree(temporary)
                return ExtractionResult(
                    relative_path=relative_path,
                    extracted_files=sum(
                        1 for path in destination.rglob("*") if path.is_file()
                    ),
                    skipped_files=skipped_files,
                    reused=True,
                )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return ExtractionResult(
            relative_path=relative_path,
            extracted_files=extracted_files,
            skipped_files=skipped_files,
            reused=False,
        )

    @staticmethod
    def _safe_member_path(value: str) -> PurePosixPath | None:
        relative = PurePosixPath(value.replace("\\", "/"))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            return None
        return relative

    @staticmethod
    def _is_within(target: Path, root: Path) -> bool:
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return False
        return target != root.resolve()
