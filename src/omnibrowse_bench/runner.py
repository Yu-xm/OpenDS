from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .agent import run_sample
from .config import public_config
from .dataset import load_samples, read_predictions, write_predictions
from .models import create_backend, initial_messages
from .trace import TraceWriter, now_iso


DEFAULT_PROMPT = """You are participating in a controlled live-web research benchmark.

Use only the supplied text_search, image_search, and open_page tools for web access. Conduct
thorough, autonomous research and use the available tools as you judge appropriate. Inspect
input media when present, verify the information needed to answer the question, and do not
fabricate unsupported facts. Decide your own research strategy and continue until you have
sufficient evidence or the available budget is exhausted.

Final-answer format: follow any explicit format requested in the question. Otherwise, output
only one short English paragraph, normally one to three sentences and no more than 80 words.
State the answer directly. Do not include headings, bullet points, a research narrative,
chain-of-thought, or labels such as "Final Answer"."""


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _safe_name(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return result.strip("._") or "unnamed"


def _unsupported_trace(trace_path: Path, sample: Any, benchmark: str, model_name: str, run_id: str, reason: str) -> None:
    common = {"run_id": run_id, "benchmark": benchmark, "model": model_name, "sample_id": sample.sample_id}
    input_record = {"question": sample.question}
    if sample.video_path:
        input_record["video_path"] = str(sample.video_path)
    if sample.audio_path:
        input_record["audio_path"] = str(sample.audio_path)
    with TraceWriter(trace_path, common) as trace:
        trace.write("run_start", input=input_record)
        trace.write("run_end", status="unsupported_input", error={"message": reason}, metrics={
            "total_latency_ms": 0, "model_calls": 0, "tool_calls": 0,
            "text_search_calls": 0, "image_search_calls": 0, "open_page_calls": 0,
            "input_tokens": 0, "output_tokens": 0,
        })


def run(config: dict[str, Any], resume: bool = False, limit: int | None = None) -> Path:
    benchmark = config["benchmark"]
    model_config = config["model"]
    model = create_backend(model_config)
    samples, source_fields = load_samples(config["dataset"]["path"], benchmark)
    if limit is not None:
        samples = samples[:limit]

    run_id = config.get("run_id") or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config.get("output_root", "results")).resolve()
    run_dir = output_root / _safe_name(benchmark) / _safe_name(model.display_name) / _safe_name(run_id)
    trace_dir, artifact_root = run_dir / "trajectories", run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.csv"

    prompt = config.get("agent", {}).get("system_prompt", DEFAULT_PROMPT)
    saved_config = public_config(config)
    dataset_path = Path(config["dataset"]["path"]).resolve()
    saved_config["dataset"]["path"] = str(dataset_path)
    saved_config["dataset"]["sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    saved_config.update({
        "run_id": run_id, "framework_version": __version__, "python_version": platform.python_version(),
        "git_commit": _git_commit(Path(config["_config_path"]).parent), "started_at": now_iso(),
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "finished_at": None,
    })
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8")

    previous_rows = read_predictions(predictions_path) if resume else {}
    rows: list[dict[str, str]] = []
    max_media_bytes = int(config.get("agent", {}).get("max_media_bytes", 100_000_000))
    for index, sample in enumerate(samples, 1):
        trace_path = trace_dir / f"{_safe_name(sample.sample_id)}.jsonl"
        if resume and trace_path.exists() and sample.sample_id in previous_rows:
            rows.append(previous_rows[sample.sample_id])
            continue
        print(f"[{index}/{len(samples)}] {benchmark} / {model.display_name} / {sample.sample_id}", flush=True)
        unsupported = []
        if sample.video_path and not model.supports_video:
            unsupported.append("video")
        if sample.audio_path and not model.supports_audio:
            unsupported.append("audio")
        if unsupported:
            reason = f"Model adapter does not declare support for: {', '.join(unsupported)}"
            _unsupported_trace(trace_path, sample, benchmark, model.display_name, run_id, reason)
            outcome = {"answer": "", "status": "unsupported_input", "error": reason, "latency_s": 0.0, "tool_calls": 0}
        else:
            try:
                messages = initial_messages(sample.question, sample.video_path, sample.audio_path, prompt, max_media_bytes)
                outcome = run_sample(
                    sample, benchmark, run_id, model, messages, config["search"], config.get("agent", {}),
                    trace_path, artifact_root / _safe_name(sample.sample_id),
                )
            except (OSError, ValueError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                _unsupported_trace(trace_path, sample, benchmark, model.display_name, run_id, reason)
                outcome = {"answer": "", "status": "unsupported_input", "error": reason, "latency_s": 0.0, "tool_calls": 0}
        row = dict(sample.row)
        row.update({
            "model_answer": outcome["answer"], "benchmark": benchmark, "model": model.display_name,
            "run_id": run_id, "status": outcome["status"], "latency_s": str(outcome["latency_s"]),
            "tool_calls": str(outcome["tool_calls"]), "trace_path": str(trace_path), "error": outcome["error"],
        })
        rows.append(row)
        write_predictions(predictions_path, rows, source_fields)

    saved_config["finished_at"] = now_iso()
    config_path.write_text(json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir
