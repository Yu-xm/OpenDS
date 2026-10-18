"""Timestamp-preserving subtitle selection for local text and online tracks."""
from __future__ import annotations

import base64
import html
import json
import math
import re
from typing import Any


def validate_range(start: float | None, end: float | None) -> None:
    if start is None and end is None:
        return
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (start, end)) or not 0 <= start < end:
        raise ValueError("provide both times with 0 <= start_seconds < end_seconds")


def parse_subtitles(data: str, format: str) -> list[dict[str, Any]]:
    entries = []

    def add(start, end, text):
        start, end = float(start), float(end)
        text = html.unescape(re.sub(r"<[^>]*>", "", text))
        text = " ".join(text.split())
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError("invalid subtitle timestamps")
        if text:
            entries.append({"start_seconds": start, "end_seconds": end, "text": text})

    if format == "json3":
        events = json.loads(data).get("events", [])
        for event in events:
            text = "".join(str(segment.get("utf8", "")) for segment in event.get("segs", []))
            if text.strip():
                start = float(event["tStartMs"]) / 1000
                # No invented duration when a provider omits it: retain a point cue.
                add(start, start + float(event.get("dDurationMs", 0)) / 1000, text)
    elif format in {"vtt", "srt"}:
        def seconds(value):
            parts = value.replace(",", ".").split(":")
            return sum(float(part) * 60 ** i for i, part in enumerate(reversed(parts)))
        timestamp = r"(?:\d+:)?\d{2}:\d{2}[.,]\d+"
        for block in re.split(r"\n\s*\n", data.replace("\r\n", "\n")):
            match = re.search(rf"(?m)^({timestamp})\s+-->\s+({timestamp})[^\n]*\n([\s\S]*)", block)
            if match:
                add(seconds(match[1]), seconds(match[2]), match[3])
    else:
        # Repository Parquet stores [0.080s-20.100s] text, with newline separators.
        matches = list(re.finditer(r"\[(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s\]", data))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(data)
            add(match[1], match[2], data[match.end():end].replace("\\n", "\n"))
    if data.strip() and not entries:
        raise ValueError("subtitle contains no readable timestamped entries")
    return sorted(entries, key=lambda entry: entry["start_seconds"])


def select_subtitles(subtitle: dict[str, Any], start: float | None, end: float | None,
                     duration: float, maximum: int) -> dict[str, Any]:
    validate_range(start, end)
    selected_start = 0.0 if start is None else float(start)
    selected_end = duration if end is None else min(float(end), duration)
    if not 0 <= selected_start < selected_end:
        raise ValueError("subtitle range must overlap the video duration")
    result = {**subtitle, "start_seconds": selected_start, "end_seconds": selected_end,
              "segments": [], "returned_entries": 0, "truncated": False}
    if not subtitle.get("available"):
        result["content"] = f"Subtitle unavailable: {subtitle.get('reason', 'unknown reason')}"
        return result
    data = base64.b64decode(subtitle["data_base64"], validate=True).decode("utf-8", errors="replace")
    entries = parse_subtitles(data, subtitle.get("format", "text"))
    def overlaps(entry, begin):
        return entry["start_seconds"] < selected_end and (
            entry["end_seconds"] > begin or
            entry["start_seconds"] == entry["end_seconds"] >= begin)
    matches = [e for e in entries if overlaps(e, selected_start)]
    shown = matches[:maximum]
    truncated = len(matches) > maximum
    result.update(segments=shown, returned_entries=len(shown), truncated=truncated)
    result["content"] = "\n".join(
        f"[{e['start_seconds']:.3f}s-{e['end_seconds']:.3f}s] {e['text']}" for e in shown
    ) or "No subtitles in the requested time range."
    if truncated:
        next_start = matches[maximum]["start_seconds"]
        next_page = [e for e in matches if overlaps(e, next_start)][:maximum]
        can_continue = next_start > selected_start and next_page != shown
        result["continuation_available"] = can_continue
        result["note"] = f"Subtitles truncated: returned the first {maximum} entries in the requested range. "
        if can_continue:
            result["next_start_seconds"] = next_start
            result["note"] += (
                f"To read later subtitles, call youtube_read_subtitle with the same url_or_id, "
                f"start_seconds={next_start!r}, and end_seconds={selected_end!r}. "
                "Subtitles overlapping the new start time may appear again."
            )
        else:
            result["note"] += (
                "Time-only continuation cannot retrieve all remaining entries: overlapping or "
                "same-start subtitles fill the result limit. No advancing continuation time is available."
            )
        result["content"] += "\n\n[" + result["note"] + "]"
    return result
