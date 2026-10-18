#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


EXPECTED = {"text": 33, "video": 116}


def read_source(path: Path, dataset: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["query", "answer"]:
            raise ValueError(f"{path} must contain exactly query,answer")
        rows = list(reader)
    if len(rows) != EXPECTED[dataset]:
        raise ValueError(
            f"{path} has {len(rows)} rows; expected {EXPECTED[dataset]}"
        )
    prepared = []
    for index, row in enumerate(rows, 1):
        question = row["query"].strip()
        answer = row["answer"].strip()
        if not question or not answer:
            raise ValueError(f"{path} row {index + 1} has an empty value")
        prepared.append(
            {
                "id": f"{dataset}-{index:03d}",
                "question": question,
                "gold_answer": answer,
                "dataset": dataset,
                "source_row": str(index),
            }
        )
    return prepared


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "question", "gold_answer", "dataset", "source_row"]
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    text_rows = read_source(args.source_dir / "opends_world_bench_text.csv", "text")
    video_rows = read_source(args.source_dir / "opends_world_bench_video.csv", "video")
    atomic_csv(args.output_dir / "text.csv", text_rows)
    atomic_csv(args.output_dir / "video.csv", video_rows)
    for shard in range(3):
        atomic_csv(
            args.output_dir / f"video-{shard}.csv",
            [row for index, row in enumerate(video_rows) if index % 3 == shard],
        )
    print("prepared text=33 video=116 video_shards=39,39,38")


if __name__ == "__main__":
    main()
