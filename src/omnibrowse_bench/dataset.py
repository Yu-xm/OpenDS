from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from .types import Sample


RESULT_COLUMNS = [
    "model_answer", "benchmark", "model", "run_id", "status", "latency_s",
    "tool_calls", "trace_path", "error",
]


def _media_path(csv_dir: Path, value: str) -> Path | None:
    if not value.strip():
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (csv_dir / path).resolve()


def load_samples(path: str | Path, benchmark: str) -> tuple[list[Sample], list[str]]:
    csv_path = Path(path).resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"id", "question"}
        if benchmark == "OmniBrowse":
            required |= {"video_path", "audio_path"}
        elif benchmark != "OmniBrowse-Text":
            raise ValueError("benchmark must be OmniBrowse or OmniBrowse-Text")
        missing = required - set(fields)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        samples: list[Sample] = []
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            sample_id = (row.get("id") or "").strip()
            question = (row.get("question") or "").strip()
            if not sample_id or not question:
                raise ValueError(f"CSV line {line}: id and question cannot be empty")
            if sample_id in seen:
                raise ValueError(f"CSV line {line}: duplicate id {sample_id!r}")
            seen.add(sample_id)
            sample = Sample(
                sample_id=sample_id,
                question=question,
                row=dict(row),
                csv_dir=csv_path.parent,
                video_path=_media_path(csv_path.parent, row.get("video_path", "")),
                audio_path=_media_path(csv_path.parent, row.get("audio_path", "")),
            )
            if benchmark == "OmniBrowse":
                for label, media in (("video", sample.video_path), ("audio", sample.audio_path)):
                    if media is None or not media.is_file():
                        raise ValueError(f"CSV line {line}: {label} file does not exist: {media}")
            samples.append(sample)
    return samples, fields


def write_predictions(path: Path, rows: list[dict[str, str]], source_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = source_fields + [field for field in RESULT_COLUMNS if field not in source_fields]
    fd, temp_name = tempfile.mkstemp(prefix="predictions-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: dict(row) for row in csv.DictReader(handle) if row.get("id")}
