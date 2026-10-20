from __future__ import annotations

import json
import math
import signal
import threading
from contextlib import contextmanager
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .serper import TOOL_DEFINITIONS, SerperTools
from .toolbox_tools import (
    TOOLBOX_TOOL_DEFINITIONS,
    TOOLBOX_TOOL_NAMES,
    ToolboxToolAdapter,
    create_toolbox,
    toolbox_task_id,
)
from .trace import TraceWriter, save_artifact, read_events
from .models import ModelBackend
from .types import Sample, ToolArgumentError, ToolInfrastructureError


class AgentError(RuntimeError):
    status = "api_error"


class AgentTimeoutError(AgentError):
    status = "timeout"


class ModelRequestError(AgentError):
    status = "api_error"


class InfrastructureError(AgentError):
    status = "infrastructure_error"


class _DeadlineExpired(BaseException):
    """Bypass tool/provider exception recovery when the case deadline expires."""


class _TerminationRequested(BaseException):
    """A graceful process stop during research can still persist and finalize evidence."""


@contextmanager
def _case_deadline(seconds: float):
    # Indexed workers run on the main thread of separate Linux/macOS processes.
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("run_sample requires a main-thread process for its case deadline")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("run_sample cannot replace an existing real-time alarm")
    previous = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    def expire(signum, frame):
        raise _DeadlineExpired()
    def terminate(signum, frame):
        raise _TerminationRequested()
    signal.signal(signal.SIGALRM, expire)
    signal.signal(signal.SIGTERM, terminate)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        signal.signal(signal.SIGTERM, previous_term)


from .context import _compact_history, _compact_images, _assistant_message, _visual_message
from .finalization import finalize, reconstruct_messages, is_context_overflow, finalization_config


def _generate_with_deadline(
    model: ModelBackend, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    remaining_seconds: float, disable_retries: bool,
) -> Any:
    """Temporarily bound adapters that expose timeout/retry attributes to this sample's deadline."""
    originals: dict[str, Any] = {}
    updates: dict[str, Any] = {}
    if hasattr(model, "timeout"):
        updates["timeout"] = min(float(getattr(model, "timeout")), max(0.001, remaining_seconds))
    if disable_retries and hasattr(model, "max_retries"):
        updates["max_retries"] = 0
    try:
        for name, value in updates.items():
            originals[name] = getattr(model, name)
            setattr(model, name, value)
        return model.generate(messages, tools)
    finally:
        for name, value in originals.items():
            setattr(model, name, value)


class ResearchStopped(AgentError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _safe_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)[:120] or "call"


def run_sample(
    sample: Sample, benchmark: str, run_id: str, model: ModelBackend,
    messages: list[dict[str, Any]], serper_config: dict[str, Any], agent_config: dict[str, Any],
    trace_path: Path, artifact_dir: Path, toolbox_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    total_timeout = agent_config.get("timeout_seconds", 3600)
    if (isinstance(total_timeout, bool) or not isinstance(total_timeout, (int, float))
            or not math.isfinite(total_timeout) or total_timeout <= 0):
        raise ValueError("agent.timeout_seconds must be a finite positive number")
    for key, default in (("max_tool_calls", 100), ("max_context_images", 250)):
        value = agent_config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"agent.{key} must be a positive integer")
    agent_config = finalization_config(agent_config)
    max_tools = agent_config.get("max_tool_calls", 100)
    max_images = agent_config.get("max_context_images", 250)
    toolbox_config = dict(toolbox_config or {})
    toolbox_enabled = toolbox_config.get("enabled", False)
    definitions = [*TOOL_DEFINITIONS, *(TOOLBOX_TOOL_DEFINITIONS if toolbox_enabled else [])]
    allowed = {d["function"]["name"] for d in definitions}
    tools = SerperTools(serper_config, artifact_dir)
    runtime = adapter = None
    startup_ms = cleanup_ms = 0
    counts: Counter[str] = Counter()
    infrastructure_failures = []
    answer = error = reason = ""
    status = "success"
    finalization_status = "complete"
    initial = json.loads(json.dumps(messages))
    step = 0
    common = {"run_id": run_id, "benchmark": benchmark, "model": model.display_name, "sample_id": sample.sample_id}
    with TraceWriter(trace_path, common) as trace:
        initial_artifact = save_artifact(artifact_dir / "initial_context.json", {"messages": initial, "agent_config": agent_config})
        input_record = {"question": sample.question}
        if sample.video_path:
            input_record["video_path"] = str(sample.video_path)
        if sample.audio_path:
            input_record["audio_path"] = str(sample.audio_path)
        if sample.input_video_path:
            input_record["input_video"] = {"id": sample.input_video_id,
                "path": str(sample.input_video_path), "sha256": sample.input_video_sha256}
        trace.write("run_start", input=input_record, initial_context_artifact=initial_artifact)
        try:
            with _case_deadline(total_timeout - (time.monotonic() - started)):
                if toolbox_enabled:
                    task_id = toolbox_task_id(str(artifact_dir.resolve()) + "\0" + run_id, benchmark, model.display_name, sample.sample_id)
                    trace.write("toolbox_start", task_id=task_id)
                    before = time.monotonic()
                    try:
                        runtime = create_toolbox(task_id, toolbox_config, artifact_dir,
                            input_video_id=sample.input_video_id, input_video_path=sample.input_video_path,
                            input_video_sha256=sample.input_video_sha256, input_video_subtitle=sample.input_video_subtitle)
                        runtime.__enter__()
                        adapter = ToolboxToolAdapter(runtime, artifact_dir)
                    finally:
                        startup_ms = round((time.monotonic() - before) * 1000)
                    trace.write("toolbox_ready", task_id=task_id, latency_ms=startup_ms,
                                startup=getattr(runtime, "startup_timings", {}))
                while True:
                    step += 1
                    if counts["tool_calls"] >= max_tools:
                        raise ResearchStopped("tool_budget")
                    remaining = total_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise _DeadlineExpired()
                    changed, before, after = _compact_history(messages,
                        int(agent_config.get("keep_recent_tool_observations", 4)),
                        int(agent_config.get("history_search_results", 2)), int(agent_config.get("history_page_chars", 2000)),
                        int(agent_config.get("keep_recent_reasoning_messages", 2)))
                    if changed:
                        trace.write("context_compaction", step=step, messages_compacted=changed,
                                    chars_before=before, chars_after=after, chars_removed=max(0, before-after))
                    before, after = _compact_images(messages, max_images)
                    counts["images_removed_from_context"] += before-after
                    if before != after:
                        trace.write("image_context_compaction", step=step, images_before=before,
                                    images_after=after, images_removed=before-after, limit=max_images)
                    counts["model_attempts"] += 1
                    call_started = time.monotonic()
                    try:
                        response = _generate_with_deadline(model, messages, definitions, remaining, False)
                    except _DeadlineExpired:
                        counts["model_errors"] += 1
                        trace.write("model_error", step=step, error={"type": "DeadlineExpired", "message": "Research model request interrupted"}, retrying_as_final=True)
                        raise
                    except Exception as exc:
                        counts["model_errors"] += 1
                        trace.write("model_error", step=step, error={"type": type(exc).__name__, "message": str(exc)}, retrying_as_final=True)
                        raise ResearchStopped("context_overflow" if is_context_overflow(exc) else "model_error") from exc
                    counts["model_calls"] += 1
                    # A discarded answer still consumed provider tokens.
                    counts["input_tokens"] += int(response.usage.get("input_tokens", 0))
                    counts["output_tokens"] += int(response.usage.get("output_tokens", 0))
                    if (response.finish_reason not in {"stop", "tool_calls"}
                            or (response.finish_reason == "tool_calls" and not response.tool_calls)):
                        trace.write("incomplete_model_response", step=step, finish_reason=response.finish_reason,
                                    usage=response.usage,
                                    latency_ms=round((time.monotonic()-call_started)*1000), discarded=True)
                        raise ResearchStopped("incomplete_response")
                    raw = save_artifact(artifact_dir / f"model_response_{step:03d}.json", response.raw) if response.raw is not None else None
                    trace.write("model_response", step=step,
                        response={"text": response.text, "reasoning_content": response.reasoning_content,
                                  "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls]},
                        usage=response.usage, finish_reason=response.finish_reason, raw_response_artifact=raw,
                        latency_ms=round((time.monotonic()-call_started)*1000))
                    messages.append(_assistant_message(response.text, response.reasoning_content, response.tool_calls))
                    if not response.tool_calls:
                        if not response.text.strip():
                            counts["empty_answer_retries"] += 1
                            trace.write("empty_answer_retry", step=step, attempt=1, finish_reason=response.finish_reason)
                            raise ResearchStopped("empty_answer")
                        answer = response.text.strip()
                        trace.write("final_answer", step=step, answer=answer, finish_reason=response.finish_reason, citations=[])
                        break
                    visuals = []
                    for call in response.tool_calls:
                        counts["tool_requests"] += 1
                        if call.name not in allowed or counts["tool_calls"] >= max_tools:
                            counts["blocked_tool_calls"] += 1
                            if call.name not in allowed:
                                counts["invalid_tool_calls"] += 1
                            observation = (
                                {"error": f"Unknown tool requested: {call.name}", "allowed_tools": sorted(allowed)}
                                if call.name not in allowed else
                                {"error": "Budget exhausted for all tools", "limit": max_tools}
                            )
                            trace.write("tool_result", step=step, tool_call_id=call.id, tool=call.name,
                                        success=False, blocked=True, observation=observation)
                            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(observation, ensure_ascii=False)})
                            continue
                        counts["tool_calls"] += 1
                        counts[call.name] += 1
                        trace.write("tool_start", step=step, tool_call_id=call.id, tool=call.name, arguments=call.arguments)
                        call_started = time.monotonic()
                        try:
                            executor = adapter if call.name in TOOLBOX_TOOL_NAMES else tools
                            result = executor.execute(call.name, call.arguments, call.id)
                            artifact = save_artifact(artifact_dir / f"{call.name}_{_safe_id(call.id)}.json", result.raw)
                            visual = _visual_message(result.media)
                            trace.write("tool_result", step=step, tool_call_id=call.id, tool=call.name, success=True,
                                        latency_ms=round((time.monotonic()-call_started)*1000), observation=result.observation,
                                        raw_artifact=artifact, media=result.media)
                            observation = result.observation
                            if visual:
                                visuals.append(visual)
                        except Exception as exc:
                            counts["tool_errors"] += 1
                            invalid = isinstance(exc, ToolArgumentError)
                            if invalid:
                                counts["invalid_tool_argument_errors"] += 1
                            else:
                                infrastructure_failures.append(f"{call.name}: {type(exc).__name__}: {exc}")
                            observation = {"error": str(exc), "type": type(exc).__name__}
                            trace.write("tool_result", step=step, tool_call_id=call.id, tool=call.name, success=False,
                                        latency_ms=round((time.monotonic()-call_started)*1000), observation=observation,
                                        failure_kind="invalid_arguments" if invalid else "infrastructure")
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(observation, ensure_ascii=False)})
                    messages.extend(visuals)
        except _DeadlineExpired:
            reason = "timeout"
            trace.write("research_timeout", step=step)
        except ResearchStopped as exc:
            reason = exc.reason
        except (KeyboardInterrupt, _TerminationRequested):
            reason = "interrupted"
        except Exception as exc:
            reason = "infrastructure_error"
            error = f"{type(exc).__name__}: {exc}"
            infrastructure_failures.append(error)
            trace.write("research_error", step=step, error={"type": type(exc).__name__, "message": str(exc)})
        try:
            if reason:
                trace.write("research_stopped", step=step, reason=reason)
                # Both online and offline finalization consume the recorded completed evidence.
                completed = reconstruct_messages(trace_path, initial, agent_config)
                records, discarded_lines = read_events(trace_path)
                if discarded_lines:
                    trace.write("discarded_incomplete_records", lines=discarded_lines)
                results = {e["tool_call_id"] for e in records if e["event"] == "tool_result"}
                starts = {e["tool_call_id"] for e in records if e["event"] == "tool_start"}
                calls = {c["id"]: c for e in records if e["event"] == "model_response" for c in e["response"].get("tool_calls", [])}
                for message in completed:
                    call_id = message.get("tool_call_id")
                    if message.get("role") == "tool" and call_id not in results:
                        interrupted = call_id in starts
                        if not interrupted:
                            counts["blocked_tool_calls"] += 1
                        trace.write("tool_result", step=step, tool_call_id=call_id, tool=calls[call_id]["name"],
                                    success=False, blocked=not interrupted, interrupted=interrupted,
                                    observation=json.loads(message["content"]), failure_kind="research_stopped")
                counts["tool_requests"] = max(counts["tool_requests"], len(calls))
                counts["forced_finalizations"] += 1
                trace.write("forced_final", step=step, reason=reason)
                final = finalize(model, completed, agent_config, artifact_dir, trace, reason)
                answer, status = final["answer"], final["status"]
                error = final["error"] if status != "success" else ""
                finalization_status = final["finalization_status"]
                for key in ("model_attempts", "model_calls", "model_errors", "input_tokens", "output_tokens"):
                    counts[key] += final[key]
        except Exception as exc:
            status, finalization_status = "infrastructure_error", "pending"
            error = f"{type(exc).__name__}: {exc}"
            trace.write("finalization_failed", reason=reason, error={"message": error}, recoverable=True)
        finally:
            if runtime is not None:
                before = time.monotonic()
                try:
                    runtime.close()
                except Exception as exc:
                    infrastructure_failures.append(f"Toolbox cleanup: {type(exc).__name__}: {exc}")
                    trace.write("toolbox_error", phase="cleanup", error={"type": type(exc).__name__, "message": str(exc)})
                cleanup_ms = round((time.monotonic()-before)*1000)
                cleanup_errors = list(getattr(runtime, "cleanup_errors", []))
                infrastructure_failures.extend(f"Toolbox cleanup: {e}" for e in cleanup_errors)
                trace.write("toolbox_closed", latency_ms=cleanup_ms, cleanup_errors=cleanup_errors)
        if reason == "infrastructure_error":
            if status == "success":
                status = "infrastructure_error"
            error = "; ".join(dict.fromkeys(([error] if error else []) + infrastructure_failures))
        total_ms = round((time.monotonic()-started)*1000)
        metrics = {key: counts[key] for key in (
            "model_calls", "model_attempts", "model_errors", "tool_calls", "tool_requests", "blocked_tool_calls",
            "tool_errors", "invalid_tool_calls", "invalid_tool_argument_errors", "input_tokens", "output_tokens",
            "empty_answer_retries", "forced_finalizations", "images_removed_from_context")}
        metrics.update(total_latency_ms=total_ms, text_search_calls=counts["text_search"], image_search_calls=counts["image_search"],
            open_page_calls=counts["open_page"], toolbox_tool_calls=sum(counts[n] for n in TOOLBOX_TOOL_NAMES),
            infrastructure_errors=len(infrastructure_failures), forced_final_reason=reason or None,
            max_context_images=max_images, toolbox_startup_ms=startup_ms, toolbox_cleanup_ms=cleanup_ms,
            agent_latency_ms=max(0,total_ms-startup_ms-cleanup_ms))
        trace.write("run_end", status=status, finalization_status=finalization_status, error={"message": error} if error else None, metrics=metrics)
    return {"answer": answer, "status": status, "finalization_status": finalization_status, "error": error,
            "latency_s": round(time.monotonic()-started,3), "tool_calls": counts["tool_calls"],
            "text_search_calls": counts["text_search"], "image_search_calls": counts["image_search"],
            "open_page_calls": counts["open_page"], "toolbox_tool_calls": sum(counts[n] for n in TOOLBOX_TOOL_NAMES),
            "blocked_tool_calls": counts["blocked_tool_calls"], "forced_final_reason": reason}
