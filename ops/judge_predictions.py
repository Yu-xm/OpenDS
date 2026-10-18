#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opends_world.models import OpenAICompatibleBackend


SYSTEM_PROMPT = """You are a strict benchmark answer judge. Compare the candidate answer with the authoritative reference answer, using the question only to understand what must be answered.

Rules:
- Mark correct only when the candidate is semantically equivalent to the reference and answers every required part.
- Numbers, named entities, dates, directions, and list membership must be correct.
- Extra explanation is allowed only if it does not contradict the reference.
- A refusal, an insufficient-evidence response, an empty answer, or an answer from a failed execution is incorrect.
- Do not solve the question independently and do not use outside knowledge. Treat the reference answer as authoritative.

Return exactly one JSON object with this schema:
{"correct": true, "confidence": 0.0, "reason": "brief comparison"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge benchmark predictions with an OpenAI-compatible model")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", ""))
    parser.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", ""))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_judgment(text: str) -> tuple[bool, float, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"judge did not return JSON: {text[:300]!r}")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload.get("correct"), bool):
        raise ValueError("judge JSON field 'correct' must be boolean")
    confidence = float(payload.get("confidence", 0.5))
    confidence = min(1.0, max(0.0, confidence))
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        raise ValueError("judge JSON field 'reason' is empty")
    return payload["correct"], confidence, reason


def judge_messages(row: dict[str, str]) -> list[dict[str, str]]:
    content = {
        "question": row.get("question", ""),
        "reference_answer": row.get("gold_answer", ""),
        "candidate_answer": row.get("model_answer", "") or "(empty)",
        "execution_status": row.get("status", ""),
        "execution_error": row.get("error", "") or None,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
    ]


def backend_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "display_name": args.model_id,
        "model_id": args.model_id,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": 2,
        "retry_base_seconds": 2,
        "retry_max_seconds": 20,
        "parameters": {
            "temperature": 0,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "medium",
            },
            "stream": False,
        },
    }


def make_backend(args: argparse.Namespace) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(backend_config(args))


def resume_identity(args: argparse.Namespace) -> dict[str, Any]:
    config = backend_config(args)
    config.pop("api_key")  # Credential rotation does not change a judgment's identity.
    return {
        "schema_version": 1,
        "source_predictions_sha256": sha256(args.predictions),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "judge_script_sha256": sha256(Path(__file__)),
        "backend": config,
        "attempts": args.attempts,
        "limit": args.limit,
    }


def identity_hash(identity: dict[str, Any]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_local = threading.local()


def judge_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    backend = getattr(_local, "backend", None)
    if backend is None:
        backend = make_backend(args)
        _local.backend = backend
    started = time.monotonic()
    last_error = ""
    input_tokens = output_tokens = 0
    for attempt in range(1, args.attempts + 1):
        try:
            response = backend.generate(judge_messages(row), [])
            input_tokens += int(response.usage.get("input_tokens", 0))
            output_tokens += int(response.usage.get("output_tokens", 0))
            correct, confidence, reason = extract_judgment(response.text)
            return {
                "id": row["id"],
                "dataset": row.get("dataset", ""),
                "execution_status": row.get("status", ""),
                "correct": correct,
                "confidence": confidence,
                "reason": reason,
                "judge_status": "success",
                "judge_error": "",
                "judge_attempts": attempt,
                "judge_latency_s": round(time.monotonic() - started, 3),
                "judge_input_tokens": input_tokens,
                "judge_output_tokens": output_tokens,
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
    return {
        "id": row["id"],
        "dataset": row.get("dataset", ""),
        "execution_status": row.get("status", ""),
        "correct": False,
        "confidence": 0.0,
        "reason": "Judge failed after all retry attempts.",
        "judge_status": "error",
        "judge_error": last_error,
        "judge_attempts": args.attempts,
        "judge_latency_s": round(time.monotonic() - started, 3),
        "judge_input_tokens": input_tokens,
        "judge_output_tokens": output_tokens,
    }


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows[str(item["id"])] = item
    return rows


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_summary(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    def group(items: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(item["judge_status"] == "success" and item["correct"] for item in items)
        errors = sum(item["judge_status"] != "success" for item in items)
        count = len(items)
        return {
            "samples": count,
            "correct": correct,
            "incorrect": count - correct - errors,
            "judge_errors": errors,
            "accuracy": round(correct / count, 6) if count else None,
            "accuracy_percent": round(100 * correct / count, 2) if count else None,
            "judge_input_tokens": sum(int(item["judge_input_tokens"]) for item in items),
            "judge_output_tokens": sum(int(item["judge_output_tokens"]) for item in items),
            "judge_latency_s": round(sum(float(item["judge_latency_s"]) for item in items), 3),
        }

    datasets: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        datasets[dataset] = group(dataset_rows)
        datasets[dataset]["execution_success_only"] = group(
            [row for row in dataset_rows if row["execution_status"] == "success"]
        )

    return {
        "samples": total,
        "judgments": len(rows),
        "overall": group(rows),
        "execution_success_only": group(
            [row for row in rows if row["execution_status"] == "success"]
        ),
        "datasets": datasets,
    }


def write_outputs(
    output_dir: Path,
    source_rows: list[dict[str, str]],
    judgments: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    started_at: str,
    manifest: dict[str, Any],
) -> None:
    ordered = [judgments[row["id"]] for row in source_rows if row["id"] in judgments]
    fields = [
        "id", "dataset", "execution_status", "correct", "confidence", "reason",
        "judge_status", "judge_error", "judge_attempts", "judge_latency_s",
        "judge_input_tokens", "judge_output_tokens",
        "resume_sha256",
    ]
    csv_path = output_dir / "judgments.csv"
    fd, temporary = tempfile.mkstemp(prefix="judgments-", suffix=".csv", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ordered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    summary = build_summary(ordered, len(source_rows))
    summary.update(
        {
            "judge_model": args.model_id,
            "source_predictions": str(args.predictions),
            "source_predictions_sha256": manifest["source_predictions_sha256"],
            "finished_at": utc_now(),
        }
    )
    atomic_write(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    manifest = {**manifest,
        "started_at": started_at,
        "finished_at": summary["finished_at"],
        "judge_model": args.model_id,
        "judge_base_url": args.base_url,
        "source_predictions": str(args.predictions),
        "source_predictions_sha256": summary["source_predictions_sha256"],
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "parameters": {
            "temperature": 0,
            "max_tokens": None,
            "stop_condition": "model_end_token",
            "enable_thinking": True,
            "reasoning_effort": "medium",
            "concurrency": args.concurrency,
            "attempts": args.attempts,
            "timeout_seconds": args.timeout_seconds,
        },
    }
    atomic_write(output_dir / "judge_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if not args.model_id or not args.base_url:
        raise SystemExit("--model-id and --base-url are required")
    if args.concurrency < 1 or args.attempts < 1:
        raise SystemExit("--concurrency and --attempts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    identity = resume_identity(args)
    digest = identity_hash(identity)
    with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    if sha256(args.predictions) != identity["source_predictions_sha256"]:
        raise SystemExit("Predictions changed while loading; retry with a stable input file")
    ids = [row["id"] for row in source_rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("Prediction IDs must be unique")

    jsonl_path = args.output_dir / "judgments.jsonl"
    manifest_path = args.output_dir / "judge_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit("Cannot resume without a hashed judge manifest; use a new output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("resume_identity") != identity
                or manifest.get("resume_sha256") != digest):
            raise SystemExit("Resume hash mismatch or legacy manifest: predictions/judge settings changed; use a new output directory")
        judgments = read_existing(jsonl_path)
        if any(key not in ids or row.get("resume_sha256") != digest
               for key, row in judgments.items()):
            raise SystemExit("Judgment records do not match the resume hash; use a new output directory")
        started_at = manifest["started_at"]
    else:
        judgments = {}
        manifest = {
            "started_at": started_at,
            "source_predictions_sha256": identity["source_predictions_sha256"],
            "resume_identity": identity,
            "resume_sha256": digest,
        }
    if not args.resume and jsonl_path.exists():
        jsonl_path.unlink()
    # Persist before the first request so an interrupted first run is resumable.
    atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    pending = [row for row in source_rows if row["id"] not in judgments]
    with jsonl_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(judge_one, row, args): row["id"] for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                result["resume_sha256"] = digest
                judgments[result["id"]] = result
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(
                    f"[{len(judgments)}/{len(source_rows)}] {result['id']} "
                    f"judge={result['judge_status']} correct={result['correct']}",
                    flush=True,
                )

    write_outputs(args.output_dir, source_rows, judgments, args, started_at, manifest)
    print(json.dumps(build_summary(list(judgments.values()), len(source_rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
