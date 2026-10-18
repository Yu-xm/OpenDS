from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from .types import Sample


RESULT_COLUMNS = [
    "model_answer", "benchmark", "model", "run_id", "status", "finalization_status", "latency_s",
    "tool_calls", "text_search_calls", "image_search_calls", "open_page_calls",
    "toolbox_tool_calls", "blocked_tool_calls", "forced_final_reason", "trace_path", "error",
]

INPUT_VIDEO_ID = "input_video"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _media_path(csv_dir: Path, value: str) -> Path | None:
    if not value.strip():
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (csv_dir / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_mapping(config: dict[str, object] | None) -> dict[str, str]:
    mapping = {
        "id": "id",
        "question": "question",
        "answer": "gold_answer",
        "video": "video_path",
        "audio": "audio_path",
    }
    configured = (config or {}).get("columns", {})
    if not isinstance(configured, dict):
        raise ValueError("dataset.columns must be an object")
    allowed = {*mapping, "input_video"}
    unknown = set(configured) - allowed
    if unknown:
        raise ValueError(f"Unknown dataset.columns keys: {sorted(unknown)}")
    for logical_name, source_name in configured.items():
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError(f"dataset.columns.{logical_name} must be a non-empty string")
        mapping[str(logical_name)] = source_name.strip()
    return mapping


def _configured_input_video(
    root: Path, value: str, *, line: int
) -> Path | None:
    if not value.strip():
        return None
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"CSV line {line}: input video name must be relative")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"CSV line {line}: input video escapes input_video_dir: {value}")
    if not path.is_file():
        raise ValueError(f"CSV line {line}: input video file does not exist: {path}")
    return path


def _probe_input_video(path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError(
            "ffprobe is required to validate configured input videos before evaluation"
        )
    completed = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError(
            f"input video validation failed for {path}: "
            f"{completed.stderr.strip()[-500:]}"
        )
    try:
        payload = json.loads(completed.stdout)
        duration = float((payload.get("format") or {}).get("duration") or 0)
        has_video = any(
            stream.get("codec_type") == "video"
            for stream in (payload.get("streams") or [])
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe returned invalid metadata for {path}") from exc
    if not has_video or duration <= 0:
        raise ValueError(
            f"input video must contain a decodable video stream and positive duration: {path}"
        )


def _load_input_video_subtitles(
    dataset_config: dict[str, object] | None,
) -> dict[str, str]:
    value = (dataset_config or {}).get("input_video_subtitles")
    if value is None or not str(value).strip():
        return {}
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise ValueError(f"dataset.input_video_subtitles does not exist: {path}")
    if path.stat().st_size == 0:
        return {}
    try:
        table = pq.read_table(path, columns=["id", "subtitle"])
    except Exception as exc:
        raise ValueError(
            "dataset.input_video_subtitles must be a Parquet file with id and subtitle columns"
        ) from exc
    subtitles: dict[str, str] = {}
    seen: set[str] = set()
    for index, row in enumerate(table.to_pylist(), start=1):
        identifier = "" if row["id"] is None else str(row["id"]).strip()
        if not identifier:
            raise ValueError(f"subtitle Parquet row {index}: id cannot be empty")
        if identifier in seen:
            raise ValueError(f"subtitle Parquet has duplicate id: {identifier!r}")
        seen.add(identifier)
        subtitle = "" if row["subtitle"] is None else str(row["subtitle"]).strip()
        if subtitle:
            subtitles[identifier] = subtitle
    return subtitles


def load_samples(
    path: str | Path,
    benchmark: str,
    dataset_config: dict[str, object] | None = None,
    limit: int | None = None,
    index: int | None = None,
) -> tuple[list[Sample], list[str]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    if index is not None and (index < 1 or limit is not None):
        raise ValueError("index must be positive and cannot be combined with limit")
    csv_path = Path(path).resolve()
    if not csv_path.is_file():
        raise ValueError(f"dataset.path does not exist: {csv_path}")
    expected_count = (dataset_config or {}).get("expected_count")
    if expected_count is not None:
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError("dataset.expected_count must be a positive integer")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            actual_count = sum(1 for _ in csv.DictReader(handle))
        if actual_count != expected_count:
            raise ValueError(
                f"dataset.expected_count={expected_count}, but {csv_path} has "
                f"{actual_count} case rows; use the final benchmark export"
            )
    mapping = _column_mapping(dataset_config)
    input_video_subtitles = _load_input_video_subtitles(dataset_config)
    input_video_setting = (dataset_config or {}).get("input_video")
    auto_input_video = input_video_setting is True
    input_video_disabled = input_video_setting is False
    input_video_dir_value = (dataset_config or {}).get("input_video_dir")
    input_video_dir = (
        Path(str(input_video_dir_value)).resolve()
        if input_video_dir_value and not input_video_disabled else None
    )
    if auto_input_video and input_video_dir is None:
        raise ValueError(
            "dataset.input_video_dir is required when dataset.input_video is true"
        )
    if input_video_dir is not None and not input_video_dir.is_dir():
        raise ValueError(f"dataset.input_video_dir does not exist: {input_video_dir}")
    validate_input_videos = bool(
        (dataset_config or {}).get("validate_input_videos", True)
    )
    input_video_id_width_value = (dataset_config or {}).get("input_video_id_width")
    input_video_id_width = (
        int(input_video_id_width_value)
        if input_video_id_width_value is not None else 0
    )
    if input_video_id_width < 0:
        raise ValueError("dataset.input_video_id_width cannot be negative")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {mapping["id"], mapping["question"]}
        if dataset_config is not None and "columns" in dataset_config:
            required.add(mapping["answer"])
        if benchmark == "OpenDS-World":
            required |= {mapping["video"], mapping["audio"]}
        elif benchmark != "OpenDS-World-Text":
            raise ValueError("benchmark must be OpenDS-World or OpenDS-World-Text")
        if input_video_dir is not None and not auto_input_video:
            required.add(mapping["input_video"])
        missing = required - set(fields)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        samples: list[Sample] = []
        seen: set[str] = set()
        input_video_hashes: dict[Path, str] = {}
        validated_input_videos: set[Path] = set()
        for line, row in enumerate(reader, start=2):
            if index is not None and line - 1 != index:
                continue
            sample_id = (row.get(mapping["id"]) or "").strip()
            question = (row.get(mapping["question"]) or "").strip()
            if not sample_id or not question:
                raise ValueError(f"CSV line {line}: id and question cannot be empty")
            if sample_id in seen:
                raise ValueError(f"CSV line {line}: duplicate id {sample_id!r}")
            seen.add(sample_id)
            normalized_row = dict(row)
            normalized_row["id"] = sample_id
            normalized_row["question"] = question
            normalized_row["gold_answer"] = (
                row.get(mapping["answer"], "") or ""
            ).strip()
            if (
                dataset_config is not None
                and "columns" in dataset_config
                and not normalized_row["gold_answer"]
            ):
                raise ValueError(f"CSV line {line}: answer cannot be empty")
            configured_video = None
            configured_hash = None
            if input_video_dir is not None:
                if auto_input_video:
                    if Path(sample_id).name != sample_id or sample_id in {".", ".."}:
                        raise ValueError(
                            f"CSV line {line}: id cannot contain path separators "
                            "when dataset.input_video is true"
                        )
                    if input_video_id_width and not sample_id.isdecimal():
                        raise ValueError(
                            f"CSV line {line}: id must be numeric when "
                            "dataset.input_video_id_width is configured"
                        )
                    video_id = (
                        sample_id.zfill(input_video_id_width)
                        if input_video_id_width else sample_id
                    )
                    input_value = f"{video_id}.mp4"
                else:
                    input_value = row.get(mapping["input_video"], "") or ""
                    if not input_value.strip():
                        raise ValueError(f"CSV line {line}: input video cannot be empty")
                normalized_row["input_video_file"] = input_value.strip()
                configured_video = _configured_input_video(
                    input_video_dir, input_value, line=line
                )
                if configured_video is not None:
                    if validate_input_videos and configured_video not in validated_input_videos:
                        _probe_input_video(configured_video)
                        validated_input_videos.add(configured_video)
                    if configured_video not in input_video_hashes:
                        input_video_hashes[configured_video] = _sha256(configured_video)
                    configured_hash = input_video_hashes[configured_video]
            sample = Sample(
                sample_id=sample_id,
                question=question,
                row=normalized_row,
                csv_dir=csv_path.parent,
                video_path=(
                    _media_path(csv_path.parent, row.get(mapping["video"], ""))
                    if benchmark == "OpenDS-World" else None
                ),
                audio_path=(
                    _media_path(csv_path.parent, row.get(mapping["audio"], ""))
                    if benchmark == "OpenDS-World" else None
                ),
                input_video_id=(
                    INPUT_VIDEO_ID if configured_video else
                    (
                        (row.get("input_video_id") or "").strip() or None
                        if not input_video_disabled else None
                    )
                ),
                input_video_path=(
                    configured_video or _media_path(
                        csv_path.parent,
                        row.get("input_video_path", "")
                        if not input_video_disabled else "",
                    )
                ),
                input_video_sha256=(
                    configured_hash or
                    (
                        (row.get("input_video_sha256") or "").strip() or None
                        if not input_video_disabled else None
                    )
                ),
                input_video_subtitle=input_video_subtitles.get(sample_id),
            )
            if sample.input_video_path is None:
                sample.input_video_subtitle = None
            input_values = (
                sample.input_video_id,
                sample.input_video_path,
                sample.input_video_sha256,
            )
            if any(input_values):
                if not all(input_values):
                    raise ValueError(
                        f"CSV line {line}: input video id, path, and sha256 "
                        "must be provided together"
                    )
                if sample.input_video_id != INPUT_VIDEO_ID:
                    raise ValueError(
                        f"CSV line {line}: input_video_id must be {INPUT_VIDEO_ID!r}"
                    )
                if not SHA256.fullmatch(str(sample.input_video_sha256)):
                    raise ValueError(f"CSV line {line}: invalid input video sha256")
                if not sample.input_video_path or not sample.input_video_path.is_file():
                    raise ValueError(
                        f"CSV line {line}: input video file does not exist: "
                        f"{sample.input_video_path}"
                    )
                if sample.input_video_path not in input_video_hashes:
                    input_video_hashes[sample.input_video_path] = _sha256(
                        sample.input_video_path
                    )
                actual_sha256 = input_video_hashes[sample.input_video_path]
                if actual_sha256 != sample.input_video_sha256:
                    raise ValueError(
                        f"CSV line {line}: input video sha256 does not match file: "
                        f"{sample.input_video_path}"
                    )
            if benchmark == "OpenDS-World":
                for label, media in (("video", sample.video_path), ("audio", sample.audio_path)):
                    if media is None or not media.is_file():
                        raise ValueError(f"CSV line {line}: {label} file does not exist: {media}")
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"index {index} is outside the CSV data rows" if index is not None else "CSV must contain at least one sample")
    canonical_fields = ["id", "question", "gold_answer"]
    if input_video_dir is not None:
        canonical_fields.append("input_video_file")
    return samples, fields + [field for field in canonical_fields if field not in fields]


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
