from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .context import _assistant_message, _compact_history, _compact_images, _visual_message
from .http import HttpError
from .trace import TraceWriter, save_artifact, read_events
from .types import ToolCall


FINAL_PROMPT = (
    "Research has ended. Using only the completed evidence in this conversation, give the "
    "best-supported final answer now. Do not call tools or invent missing facts. If the evidence "
    "is insufficient, say so briefly. Follow any answer format specified in the question; otherwise "
    "give one short English paragraph of 1-3 sentences, at most 80 words. Do not include a research "
    "narrative or headings. Keep deliberation brief enough to leave room for the final answer."
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def compact(messages: list[dict], config: dict) -> None:
    _compact_history(messages, int(config.get("keep_recent_tool_observations", 4)),
                     int(config.get("history_search_results", 2)), int(config.get("history_page_chars", 2000)),
                     int(config.get("keep_recent_reasoning_messages", 2)))
    _compact_images(messages, config.get("max_context_images", 250))


def close_pending_tools(messages: list[dict]) -> list[str]:
    pending = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                pending[call["id"]] = call
        elif message.get("role") == "tool":
            pending.pop(message.get("tool_call_id"), None)
    for call_id in pending:
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps({"error": "Tool call interrupted or skipped when research stopped; no completed result is available", "type": "ResearchStopped"})})
    return list(pending)


def prepare_context(messages: list[dict], config: dict) -> list[dict]:
    messages = copy.deepcopy(messages)
    close_pending_tools(messages)
    compact(messages, config)
    messages.append({"role": "user", "content": FINAL_PROMPT})
    return messages


def remove_last_pair(messages: list[dict]) -> list[str]:
    """Delete exactly one assistant/tool exchange and its following visual messages."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            ids = [call["id"] for call in message["tool_calls"]]
            end = index + 1
            while end < len(messages):
                candidate = messages[end]
                if candidate.get("role") == "tool" and candidate.get("tool_call_id") in ids:
                    end += 1
                elif candidate.get("role") == "user" and isinstance(candidate.get("content"), list):
                    end += 1
                else:
                    break
            del messages[index:end]
            return ids
    return []


def is_context_overflow(exc: Exception) -> bool:
    if not isinstance(exc, HttpError) or exc.status_code not in {400, 413, 422}:
        return False
    text = (str(exc.response_text or "") + " " + str(exc)).lower()
    return any(term in text for term in (
        "context_length_exceeded", "maximum context length", "context window", "context length",
        "max_model_len", "too many tokens", "input is too long", "prompt is too long",
    ))


@contextmanager
def final_parameters(model: Any, allowance: int, temperature: float | None = None, overrides: dict | None = None):
    original = {}
    missing = object()
    parameters = copy.deepcopy(getattr(model, "parameters", {}))
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(parameters.get(key), dict):
            parameters[key] = {**parameters[key], **copy.deepcopy(value)}
        else:
            parameters[key] = copy.deepcopy(value)
    token_key = "max_completion_tokens" if "max_completion_tokens" in parameters else "max_tokens"
    for key in (
        "max_tokens", "max_completion_tokens", "max_output_tokens", "min_tokens", "tools", "tool_choice",
        "functions", "function_call", "parallel_tool_calls", "messages", "model", "n",
    ):
        parameters.pop(key, None)
    parameters[token_key] = allowance
    if temperature is not None:
        parameters["temperature"] = temperature
    for name, value in {"parameters": parameters, "timeout": None, "max_retries": 0}.items():
        original[name] = getattr(model, name, missing)
        setattr(model, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is missing:
                delattr(model, name)
            else:
                setattr(model, name, value)


def request_payload(model: Any, messages: list[dict]) -> dict:
    if hasattr(model, "build_payload"):
        return copy.deepcopy(model.build_payload(messages, []))
    return {"model": model.display_name, "messages": copy.deepcopy(messages),
            "tools": [], "parameters": copy.deepcopy(getattr(model, "parameters", {}))}


def finalization_config(config: dict) -> dict:
    budget = config.get("finalization_max_tokens", 5000)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("agent.finalization_max_tokens must be a positive integer")
    temperature = config.get("finalization_temperature", 0.0)
    if temperature is not None and (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2
    ):
        raise ValueError("agent.finalization_temperature must be null or a number between 0 and 2")
    config = {**config, "finalization_temperature": temperature}
    reserve = config.get("finalization_answer_tokens", min(1000, budget // 5))
    if isinstance(reserve, bool) or not isinstance(reserve, int) or not 0 <= reserve < budget:
        raise ValueError("agent.finalization_answer_tokens must be an integer from 0 to finalization_max_tokens - 1")
    fallback = config.get("finalization_fallback_parameters", {})
    if not isinstance(fallback, dict):
        raise ValueError("agent.finalization_fallback_parameters must be an object")
    parameters = config.get("finalization_parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("agent.finalization_parameters must be an object")
    config.update(finalization_max_tokens=budget, finalization_answer_tokens=reserve,
                  finalization_fallback_parameters=fallback, finalization_parameters=parameters)
    return config


def finalize(model: Any, messages: list[dict], config: dict, output_dir: Path,
             trace: TraceWriter, reason: str, expected_request_sha256: str | None = None) -> dict:
    config = finalization_config(config)
    budget = config["finalization_max_tokens"]
    temperature = config["finalization_temperature"]
    reserve = config["finalization_answer_tokens"]
    fallback = config["finalization_fallback_parameters"]
    parameters = config["finalization_parameters"]
    messages = prepare_context(messages, config)
    snapshot_messages = copy.deepcopy(messages)
    research_overflow = reason == "context_overflow"
    initially_removed = remove_last_pair(messages) if research_overflow else []
    with final_parameters(model, budget - reserve, temperature, parameters):
        initial_hash = fingerprint(request_payload(model, messages))
    if expected_request_sha256 is not None and initial_hash != expected_request_sha256:
        raise ValueError("Replay request differs from the recorded finalization request; check model parameters and context")
    context = {"schema_version": 4, "messages": snapshot_messages, "agent_config": config, "reason": reason,
               "request_sha256": initial_hash}
    snapshot = save_artifact(output_dir / "finalization_context.json", context)
    result = {"answer": "", "status": "api_error", "finalization_status": "pending",
              "error": "", "model_attempts": 0, "model_calls": 0, "model_errors": 0,
              "input_tokens": 0, "output_tokens": 0, "output_tokens_estimated": False,
              "request_hashes": [], "removed_pair": initially_removed}
    trace.write("finalization_start", reason=reason, context_artifact=snapshot, token_budget=budget)
    overflow_retried = research_overflow
    if research_overflow:
        trace.write("finalization_context_trim", removed_tool_call_ids=initially_removed,
                    triggered_by="research_context_overflow")
        if not initially_removed:
            result.update(finalization_status="failed", error="Context overflow with no assistant/tool pair to remove")
    # At most two model requests; retries share the one stage-wide token budget.
    for attempt in range(1, 3):
        if result["finalization_status"] == "failed":
            break
        allowance = budget - result["output_tokens"]
        if attempt == 1:
            allowance -= reserve
        if allowance <= 0:
            result["error"] = "Finalization token budget exhausted without a complete answer"
            break
        result["model_attempts"] += 1
        try:
            overrides = copy.deepcopy(parameters)
            if attempt > 1:
                for key, value in fallback.items():
                    overrides[key] = ({**overrides[key], **value}
                        if isinstance(value, dict) and isinstance(overrides.get(key), dict) else copy.deepcopy(value))
            with final_parameters(model, allowance, temperature, overrides):
                payload = request_payload(model, messages)
                digest = fingerprint(payload)
                request_artifact = save_artifact(output_dir / f"finalization_request_{attempt}.json", payload)
                result["request_hashes"].append(digest)
                trace.write("finalization_request", attempt=attempt, request_sha256=digest,
                            request_artifact=request_artifact, max_tokens=allowance)
                response = model.generate(messages, [])
            result["model_calls"] += 1
            usage = response.usage or {}
            result["input_tokens"] += int(usage.get("input_tokens", 0))
            # Missing usage must not silently grant another full generation budget.
            used = usage.get("output_tokens")
            if not isinstance(used, int) or isinstance(used, bool) or used < 0:
                used = allowance
                result["output_tokens_estimated"] = True
            result["output_tokens"] += used
            complete = bool(response.text.strip()) and not response.tool_calls and response.finish_reason == "stop"
            recorded = {"finish_reason": response.finish_reason, "usage": usage, "discarded": not complete}
            if complete:
                recorded.update(text=response.text, reasoning_content=response.reasoning_content, raw=response.raw)
            raw = save_artifact(output_dir / f"finalization_response_{attempt}.json", recorded)
            trace.write("finalization_response", attempt=attempt, response_artifact=raw,
                        finish_reason=response.finish_reason, usage=usage)
            if complete:
                result.update(answer=response.text.strip(), status="success", finalization_status="complete", error="")
                trace.write("final_answer", answer=result["answer"], finish_reason="stop", citations=[])
                break
            result["error"] = f"Incomplete final answer (finish_reason={response.finish_reason}, tool_calls={len(response.tool_calls)})"
            result["status"] = "empty_answer"
            # Do not append incomplete output. Retry the same completed evidence only.
        except Exception as exc:
            result["model_errors"] += 1
            # A stream can fail after generation began but before reporting usage.
            # Charge its allowance so an overflow retry cannot reset the token budget.
            if getattr(exc, "response_started", False):
                result["output_tokens"] += allowance
                result["output_tokens_estimated"] = True
            result["error"] = f"{type(exc).__name__}: {exc}"
            trace.write("finalization_error", attempt=attempt, error={"type": type(exc).__name__, "message": str(exc)})
            if is_context_overflow(exc):
                if not overflow_retried:
                    overflow_retried = True
                    removed = remove_last_pair(messages)
                    result["removed_pair"] = removed
                    trace.write("finalization_context_trim", removed_tool_call_ids=removed)
                    if removed:
                        continue
                partial = str(getattr(exc, "partial_text", "") or "").strip()
                result.update(answer=partial, status="partial" if partial else "api_error",
                              finalization_status="partial" if partial else "failed")
            break
    if result["finalization_status"] not in {"complete", "partial"}:
        trace.write("finalization_failed", reason=reason, status=result["status"],
                    error={"message": result["error"]}, recoverable=result["finalization_status"] == "pending")
    save_artifact(output_dir / "finalization_result.json", result)
    return result


def load_artifact(reference: dict) -> Any:
    path = Path(reference["path"])
    data = path.read_bytes()
    if reference.get("sha256") and hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return json.loads(data)


def reconstruct_messages(trace_path: Path, initial_messages: list[dict], config: dict) -> list[dict]:
    """Replay completed observations; no tools, ground truth or partial responses are used."""
    messages = copy.deepcopy(initial_messages)
    visuals = []
    for event in read_events(trace_path)[0]:
        kind = event["event"]
        if kind in {"finalization_start", "forced_final"}:
            # Old forced-final prompts/output are replaced by the shared current policy.
            break
        if kind in {"model_response", "context_compaction", "image_context_compaction"}:
            messages.extend(visuals)
            visuals.clear()
        if kind == "context_compaction":
            _compact_history(messages, int(config.get("keep_recent_tool_observations", 4)),
                             int(config.get("history_search_results", 2)), int(config.get("history_page_chars", 2000)),
                             int(config.get("keep_recent_reasoning_messages", 2)))
        elif kind == "image_context_compaction":
            _compact_images(messages, config.get("max_context_images", 250))
        elif kind == "model_response":
            response = event["response"]
            if event.get("finish_reason") not in {"stop", "tool_calls"}:
                break  # Its dependent tool observations also belong to the discarded turn.
            calls = [ToolCall(c["id"], c["name"], c["arguments"]) for c in response.get("tool_calls", [])]
            if not calls and response.get("text", "").strip():
                break  # Do not feed the previous final answer back to the model.
            messages.append(_assistant_message(response.get("text", ""), response.get("reasoning_content", ""), calls))
        elif kind == "tool_result":
            messages.append({"role": "tool", "tool_call_id": event["tool_call_id"],
                             "content": json.dumps(event["observation"], ensure_ascii=False)})
            media = event.get("media", [])
            for item in media:
                data = Path(item["path"]).read_bytes()
                if item.get("sha256") and hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError(f"Media hash mismatch: {item['path']}")
            visual = _visual_message(media)
            if visual:
                visuals.append(visual)
    # Close missing results before appending visuals from the completed portion of a batch.
    close_pending_tools(messages)
    messages.extend(visuals)
    return messages


def replay_context(snapshot: Path) -> tuple[list[dict], dict, str]:
    value = json.loads(snapshot.read_text())
    messages = value["messages"]
    if not messages or messages[-1] != {"role": "user", "content": FINAL_PROMPT}:
        raise ValueError("Unrecognized finalization context schema")
    # Snapshot is already compacted. prepare_context is idempotent apart from its final prompt.
    config = dict(value["agent_config"])
    if value.get("schema_version", 1) == 1:
        config.setdefault("finalization_temperature", None)
    if value.get("schema_version", 1) < 3:
        config.setdefault("finalization_answer_tokens", 0)
        config.setdefault("finalization_fallback_parameters", {})
    return messages[:-1], config, value["reason"]
