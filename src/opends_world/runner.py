from __future__ import annotations

import hashlib
import csv
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

Use only the supplied tools for web, browser, download, and YouTube access. Conduct thorough,
autonomous research and use the available tools as you judge appropriate. Inspect input media
when present, verify the information needed to answer the question, and do not
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


def _git_worktree(root: Path) -> tuple[bool | None, str | None]:
    """Fingerprint tracked changes and untracked, non-ignored files without saving content."""
    try:
        repository = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=root,
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
        )
        if not status:
            return False, None
        digest = hashlib.sha256()
        digest.update(status)
        digest.update(
            subprocess.check_output(
                ["git", "diff", "--binary", "HEAD", "--"], cwd=repository
            )
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository,
        )
        for raw_name in untracked.split(b"\0"):
            if not raw_name:
                continue
            path = repository / raw_name.decode("utf-8", errors="surrogateescape")
            digest.update(raw_name)
            if path.is_file():
                digest.update(path.read_bytes())
        return True, digest.hexdigest()
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        return None, None


def _safe_name(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return result.strip("._") or "unnamed"


def _question_with_input_video(sample: Any) -> str:
    if not sample.input_video_path:
        return sample.question
    return (
        "Task-provided local video access ID: input_video\n"
        "Recommended: first use youtube_read_video with url_or_id=\"input_video\" "
        "to inspect "
        "this complete local video file. This reserved ID is not a YouTube "
        "search-result ID and does not require network access. Start with a "
        "global overview by omitting start_seconds and end_seconds. You may also "
        "use youtube_read_subtitle with the same reserved ID to read its "
        "subtitle track independently.\n\n"
        f"Question:\n{sample.question}"
    )


def _unsupported_trace(trace_path: Path, sample: Any, benchmark: str, model_name: str, run_id: str, reason: str) -> None:
    common = {"run_id": run_id, "benchmark": benchmark, "model": model_name, "sample_id": sample.sample_id}
    input_record = {"question": sample.question}
    if sample.video_path:
        input_record["video_path"] = str(sample.video_path)
    if sample.audio_path:
        input_record["audio_path"] = str(sample.audio_path)
    if sample.input_video_path:
        input_record["input_video"] = {
            "id": sample.input_video_id,
            "path": str(sample.input_video_path),
            "sha256": sample.input_video_sha256,
        }
    with TraceWriter(trace_path, common) as trace:
        trace.write("run_start", input=input_record)
        trace.write("run_end", status="unsupported_input", error={"message": reason}, metrics={
            "total_latency_ms": 0, "model_calls": 0, "model_attempts": 0, "model_errors": 0,
            "tool_calls": 0,
            "text_search_calls": 0, "image_search_calls": 0, "open_page_calls": 0,
            "toolbox_tool_calls": 0,
            "tool_requests": 0, "blocked_tool_calls": 0, "tool_errors": 0,
            "invalid_tool_calls": 0, "forced_finalizations": 0, "forced_final_reason": None,
            "input_tokens": 0, "output_tokens": 0,
        })


def _infrastructure_trace(
    trace_path: Path, sample: Any, benchmark: str, model_name: str,
    run_id: str, reason: str,
) -> None:
    common = {
        "run_id": run_id, "benchmark": benchmark, "model": model_name,
        "sample_id": sample.sample_id,
    }
    with TraceWriter(trace_path, common) as trace:
        trace.write("run_start", input={"question": sample.question})
        trace.write(
            "run_end", status="infrastructure_error",
            error={"message": reason}, metrics={
                "total_latency_ms": 0, "model_calls": 0, "model_attempts": 0,
                "model_errors": 0, "tool_calls": 0, "text_search_calls": 0,
                "image_search_calls": 0, "open_page_calls": 0,
                "toolbox_tool_calls": 0, "tool_requests": 0,
                "blocked_tool_calls": 0, "tool_errors": 0,
                "invalid_tool_calls": 0, "invalid_tool_argument_errors": 0,
                "infrastructure_errors": 1, "forced_finalizations": 0,
                "forced_final_reason": None, "input_tokens": 0,
                "output_tokens": 0,
            },
        )


def _trace_is_complete(path: Path, expected_status: str) -> bool:
    if not path.is_file():
        return False
    try:
        last = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = json.loads(line)
        return bool(
            last
            and last.get("event") == "run_end"
            and last.get("status") == expected_status
        )
    except (OSError, json.JSONDecodeError):
        return False


def execution_statuses(run_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with (run_dir / "predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            status = row.get("status") or "missing"
            counts[status] = counts.get(status, 0) + 1
    return counts


def _resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "benchmark", "dataset", "model", "search", "toolbox", "agent",
            "run_id", "framework_version", "python_version", "git_commit",
            "git_dirty", "git_worktree_sha256", "system_prompt_sha256", "limit", "index",
        )
    }


def run(config: dict[str, Any], resume: bool = False, limit: int | None = None, index: int | None = None) -> Path:
    """One lock/output namespace per run or one-based CSV row; safe across xargs processes."""
    import copy
    import fcntl
    config = copy.deepcopy(config)
    for field in ("benchmark", "dataset", "model", "search"):
        if field not in config:
            raise ValueError(f"Missing required evaluation config field: {field}")
    if index is not None and (index < 1 or limit is not None):
        raise ValueError("index must be positive and cannot be combined with limit")
    if index is not None and not config.get("run_id"):
        raise ValueError("parallel --index runs require an explicit shared run_id")
    config.setdefault("run_id", datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))
    if not config["run_id"]:
        config["run_id"] = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if index is not None:
        config["run_id"] = f"{config['run_id']}-index-{index:06d}"
    identity = json.dumps([_safe_name(config["benchmark"]), _safe_name(create_backend(config["model"]).display_name), _safe_name(config["run_id"])])
    lock_path = Path(config.get("output_root", "results")).resolve() / ".locks" / (hashlib.sha256(identity.encode()).hexdigest() + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This run/index already has an active worker") from exc
        try:
            return _run(config, resume, limit, index)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _run(config: dict[str, Any], resume: bool, limit: int | None, index: int | None) -> Path:
    benchmark = config["benchmark"]
    model_config = config["model"]
    model = create_backend(model_config)
    samples, source_fields = load_samples(
        config["dataset"]["path"], benchmark, config["dataset"], limit=limit, index=index
    )
    output_ids: dict[str, str] = {}
    for sample in samples:
        output_id = _safe_name(sample.sample_id)
        if output_id in output_ids:
            raise ValueError(
                f"Sample IDs {output_ids[output_id]!r} and {sample.sample_id!r} "
                f"map to the same output name {output_id!r}; use distinct safe IDs"
            )
        output_ids[output_id] = sample.sample_id

    run_id = config.get("run_id") or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config.get("output_root", "results")).resolve()
    run_dir = output_root / _safe_name(benchmark) / _safe_name(model.display_name) / _safe_name(run_id)
    trace_dir, artifact_root = run_dir / "trajectories", run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.csv"

    prompt = config.get("agent", {}).get("system_prompt", DEFAULT_PROMPT)
    toolbox_config = dict(config.get("toolbox") or {})
    toolbox_config.setdefault("enabled", False)
    if toolbox_config["enabled"]:
        toolbox_config.setdefault("browser_image", "toolbox-browser:pdf-v1")
        toolbox_config.setdefault("youtube_image", "toolbox-youtube:latest")
        toolbox_config.setdefault("docker_command", "docker")
    elif any(sample.input_video_path is not None for sample in samples):
        raise ValueError(
            "dataset contains input videos but toolbox.enabled is false; "
            "the model would have no tool capable of reading them"
        )
    config["toolbox"] = toolbox_config
    saved_config = public_config(config)
    dataset_path = Path(config["dataset"]["path"]).resolve()
    saved_config["dataset"]["path"] = str(dataset_path)
    saved_config["dataset"]["sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    subtitle_path_value = config["dataset"].get("input_video_subtitles")
    if subtitle_path_value:
        subtitle_path = Path(str(subtitle_path_value)).resolve()
        saved_config["dataset"]["input_video_subtitles"] = str(subtitle_path)
        saved_config["dataset"]["input_video_subtitles_sha256"] = hashlib.sha256(
            subtitle_path.read_bytes()
        ).hexdigest()
    input_video_root_value = config["dataset"].get("input_video_dir")
    input_video_root = (
        Path(input_video_root_value).resolve()
        if input_video_root_value and config["dataset"].get("input_video") is not False
        else None
    )
    input_video_assets: dict[str, str] = {}
    for sample in samples:
        if sample.input_video_path and sample.input_video_sha256:
            if input_video_root is not None:
                name = sample.input_video_path.relative_to(input_video_root).as_posix()
            else:
                name = str(sample.input_video_path)
            input_video_assets[name] = sample.input_video_sha256
    if input_video_assets:
        saved_config["dataset"]["input_video_assets"] = [
            {"path": path, "sha256": input_video_assets[path]}
            for path in sorted(input_video_assets)
        ]
    repository_context = Path(config["_config_path"]).parent
    git_dirty, git_worktree_sha256 = _git_worktree(repository_context)
    saved_config.update({
        "run_id": run_id, "framework_version": __version__, "python_version": platform.python_version(),
        "git_commit": _git_commit(repository_context), "git_dirty": git_dirty,
        "git_worktree_sha256": git_worktree_sha256, "started_at": now_iso(),
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "limit": limit, "index": index, "finished_at": None,
    })
    config_path = run_dir / "run_config.json"
    if config_path.exists() and not resume:
        raise ValueError(
            f"Run already exists: {run_dir}. Use --resume with the same config "
            "or choose a new run_id."
        )
    if resume:
        if not config_path.is_file():
            raise ValueError(f"Cannot resume without existing run_config.json: {config_path}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if _resume_signature(existing_config) != _resume_signature(saved_config):
            raise ValueError(
                "Resume configuration does not match the existing run. "
                "Use the original config or choose a new run_id."
            )
        saved_config["started_at"] = existing_config.get("started_at")
        saved_config["resumed_at"] = [
            *(existing_config.get("resumed_at") or []), now_iso(),
        ]
    config_path.write_text(json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8")

    previous_rows = read_predictions(predictions_path) if resume else {}
    rows: list[dict[str, str]] = []
    max_media_bytes = int(config.get("agent", {}).get("max_media_bytes", 100_000_000))
    for index, sample in enumerate(samples, 1):
        trace_path = trace_dir / f"{_safe_name(sample.sample_id)}.jsonl"
        previous = previous_rows.get(sample.sample_id)
        if resume and trace_path.exists():
            if (previous and previous.get("model_answer", "").strip()
                    and previous.get("finalization_status") not in {"pending", "failed", "partial", "not_started"}
                    and _trace_is_complete(trace_path, previous.get("status", ""))):
                rows.append(previous)
                continue
            raise ValueError(
                f"Case {sample.sample_id} has an existing trajectory without a verified completed answer. "
                f"Use ops/finalize_trace.py with --trace {trace_path} and a separate output directory; "
                "--resume does not rerun its research."
            )
        print(f"[{index}/{len(samples)}] {benchmark} / {model.display_name} / {sample.sample_id}", flush=True)
        unsupported = []
        if sample.video_path and not model.supports_video:
            unsupported.append("video")
        if sample.audio_path and not model.supports_audio:
            unsupported.append("audio")
        if unsupported:
            reason = f"Model adapter does not declare support for: {', '.join(unsupported)}"
            _unsupported_trace(trace_path, sample, benchmark, model.display_name, run_id, reason)
            outcome = {
                "answer": "", "status": "unsupported_input", "error": reason, "latency_s": 0.0,
                "tool_calls": 0, "text_search_calls": 0, "image_search_calls": 0,
                "open_page_calls": 0, "blocked_tool_calls": 0, "forced_final_reason": "",
                "toolbox_tool_calls": 0,
            }
        else:
            try:
                question = _question_with_input_video(sample)
                messages = initial_messages(
                    question,
                    sample.video_path,
                    sample.audio_path,
                    prompt,
                    max_media_bytes,
                )
                outcome = run_sample(
                    sample, benchmark, run_id, model, messages, config["search"], config.get("agent", {}),
                    trace_path, artifact_root / _safe_name(sample.sample_id), toolbox_config,
                )
            except ValueError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                _unsupported_trace(trace_path, sample, benchmark, model.display_name, run_id, reason)
                outcome = {
                    "answer": "", "status": "unsupported_input", "error": reason, "latency_s": 0.0,
                    "tool_calls": 0, "text_search_calls": 0, "image_search_calls": 0,
                    "open_page_calls": 0, "blocked_tool_calls": 0, "forced_final_reason": "",
                    "toolbox_tool_calls": 0,
                }
            except OSError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                _infrastructure_trace(
                    trace_path, sample, benchmark, model.display_name, run_id, reason
                )
                outcome = {
                    "answer": "", "status": "infrastructure_error",
                    "error": reason, "latency_s": 0.0, "tool_calls": 0,
                    "text_search_calls": 0, "image_search_calls": 0,
                    "open_page_calls": 0, "blocked_tool_calls": 0,
                    "forced_final_reason": "", "toolbox_tool_calls": 0,
                }
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if trace_path.exists():
                    with TraceWriter(trace_path, {
                        "run_id": run_id, "benchmark": benchmark,
                        "model": model.display_name, "sample_id": sample.sample_id,
                    }) as trace:
                        trace.write("run_end", status="infrastructure_error",
                                    finalization_status="pending",
                                    error={"type": type(exc).__name__, "message": str(exc)})
                else:
                    _infrastructure_trace(
                        trace_path, sample, benchmark, model.display_name, run_id, reason
                    )
                outcome = {
                    "answer": "", "status": "infrastructure_error",
                    "error": reason, "latency_s": 0.0, "tool_calls": 0,
                    "text_search_calls": 0, "image_search_calls": 0,
                    "open_page_calls": 0, "blocked_tool_calls": 0,
                    "forced_final_reason": "", "toolbox_tool_calls": 0,
                }
        row = dict(sample.row)
        row.update({
            "model_answer": outcome["answer"], "benchmark": benchmark, "model": model.display_name,
            "run_id": run_id, "status": outcome["status"], "latency_s": str(outcome["latency_s"]),
            "finalization_status": outcome.get("finalization_status", "not_started"),
            "tool_calls": str(outcome["tool_calls"]),
            "text_search_calls": str(outcome["text_search_calls"]),
            "image_search_calls": str(outcome["image_search_calls"]),
            "open_page_calls": str(outcome["open_page_calls"]),
            "toolbox_tool_calls": str(outcome["toolbox_tool_calls"]),
            "blocked_tool_calls": str(outcome["blocked_tool_calls"]),
            "forced_final_reason": outcome["forced_final_reason"],
            "trace_path": str(trace_path), "error": outcome["error"],
        })
        rows.append(row)
        write_predictions(predictions_path, rows, source_fields)

    saved_config["finished_at"] = now_iso()
    config_path.write_text(json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir
