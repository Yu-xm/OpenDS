from __future__ import annotations

import base64
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .serper import TOOL_DEFINITIONS, SerperTools
from .trace import TraceWriter, save_artifact
from .models import ModelBackend
from .types import Sample


class AgentError(RuntimeError):
    status = "api_error"


class MaxStepsError(AgentError):
    status = "max_steps"


class InvalidToolCallError(AgentError):
    status = "invalid_tool_call"


class AgentTimeoutError(AgentError):
    status = "timeout"


_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def _search_tokens(value: Any) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(str(value).casefold()))


def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
    key = "query" if name in {"text_search", "image_search"} else "url"
    value = arguments.get(key, "")
    if name in {"text_search", "image_search"}:
        value = " ".join(sorted(_search_tokens(value)))
    else:
        value = str(value).strip().casefold().rstrip("/")
    return f"{name}:{value}"


def _query_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def _compact_history(
    messages: list[dict[str, Any]], keep_recent_tools: int, history_search_results: int,
    history_page_chars: int, keep_recent_reasoning: int,
) -> tuple[int, int, int]:
    """Deterministically shrink old observations without adding another model call."""
    tool_indexes = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    reasoning_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("reasoning_content")
    ]
    compact_tool_indexes = set(tool_indexes[:-max(0, keep_recent_tools)] if keep_recent_tools else tool_indexes)
    compact_reasoning_indexes = set(
        reasoning_indexes[:-max(0, keep_recent_reasoning)] if keep_recent_reasoning else reasoning_indexes
    )
    changed = before = after = 0
    for index in compact_reasoning_indexes:
        reasoning = str(messages[index].pop("reasoning_content", ""))
        if reasoning:
            changed += 1
            before += len(reasoning)
    for index in compact_tool_indexes:
        content = messages[index].get("content")
        if not isinstance(content, str):
            continue
        try:
            observation = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(observation, dict):
            continue
        if observation.get("_history_compacted"):
            continue
        compacted = dict(observation)
        if isinstance(compacted.get("results"), list):
            compacted["results"] = compacted["results"][:history_search_results]
        if isinstance(compacted.get("content"), str):
            compacted["content"] = compacted["content"][:history_page_chars]
            compacted["history_content_truncated"] = len(observation["content"]) > history_page_chars
        compacted["_history_compacted"] = True
        replacement = json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
        if len(replacement) < len(content):
            messages[index]["content"] = replacement
            changed += 1
            before += len(content)
            after += len(replacement)
    return changed, before, after


def _assistant_message(text: str, reasoning_content: str, calls: list[Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text or ""}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
            for call in calls
        ]
    return message


def _visual_message(media: list[dict[str, str]]) -> dict[str, Any] | None:
    content: list[dict[str, str]] = [{"type": "text", "text": "Visual observations downloaded from the preceding image_search results:"}]
    for item in media:
        path = Path(item["path"])
        if path.is_file() and item.get("mime_type", "").startswith("image/"):
            content.append({"type": "image", "mime_type": item["mime_type"], "data_base64": base64.b64encode(path.read_bytes()).decode("ascii")})
    return {"role": "user", "content": content} if len(content) > 1 else None


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)[:120] or "call"


def run_sample(
    sample: Sample,
    benchmark: str,
    run_id: str,
    model: ModelBackend,
    messages: list[dict[str, Any]],
    serper_config: dict[str, Any],
    agent_config: dict[str, Any],
    trace_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    common = {"run_id": run_id, "benchmark": benchmark, "model": model.display_name, "sample_id": sample.sample_id}
    started = time.monotonic()
    counts: Counter[str] = Counter()
    input_tokens = output_tokens = 0
    status, answer, error = "success", "", ""
    max_steps = int(agent_config.get("max_steps", 20))
    max_empty_answer_retries = int(agent_config.get("max_empty_answer_retries", 1))
    max_identical_tool_calls = int(agent_config.get("max_identical_tool_calls", 1))
    max_similar_searches = int(agent_config.get("max_similar_searches", 3))
    similar_search_threshold = float(agent_config.get("similar_search_threshold", 0.82))
    max_searches_without_open = int(agent_config.get("max_searches_without_open", 8))
    max_blocked_tool_steps = int(agent_config.get("max_blocked_tool_steps", 2))
    keep_recent_tools = int(agent_config.get("keep_recent_tool_observations", 4))
    history_search_results = int(agent_config.get("history_search_results", 2))
    history_page_chars = int(agent_config.get("history_page_chars", 2000))
    keep_recent_reasoning = int(agent_config.get("keep_recent_reasoning_messages", 2))
    total_timeout = float(agent_config.get("timeout_seconds", 600))
    budgets = {
        "text_search": int(agent_config.get("max_text_searches", 10)),
        "image_search": int(agent_config.get("max_image_searches", 5)),
        "open_page": int(agent_config.get("max_page_opens", 15)),
    }
    tools = SerperTools(serper_config, artifact_dir)
    force_final = False
    signatures: Counter[str] = Counter()
    search_history: dict[str, list[frozenset[str]]] = {"text_search": [], "image_search": []}
    searches_since_open = 0
    consecutive_blocked_steps = 0

    with TraceWriter(trace_path, common) as trace:
        input_record: dict[str, Any] = {"question": sample.question}
        if sample.video_path:
            input_record["video_path"] = str(sample.video_path)
        if sample.audio_path:
            input_record["audio_path"] = str(sample.audio_path)
        trace.write("run_start", input=input_record)
        try:
            for step in range(1, max_steps + 1):
                if time.monotonic() - started >= total_timeout:
                    raise AgentTimeoutError(f"Reached total timeout of {total_timeout:g} seconds")
                compacted, chars_before, chars_after = _compact_history(
                    messages, keep_recent_tools, history_search_results, history_page_chars,
                    keep_recent_reasoning,
                )
                if compacted:
                    trace.write(
                        "context_compaction", step=step, messages_compacted=compacted,
                        chars_before=chars_before, chars_after=chars_after,
                        chars_removed=max(0, chars_before - chars_after),
                    )
                call_started = time.monotonic()
                response = model.generate(messages, [] if force_final else TOOL_DEFINITIONS)
                latency_ms = round((time.monotonic() - call_started) * 1000)
                counts["model_calls"] += 1
                input_tokens += int(response.usage.get("input_tokens", 0))
                output_tokens += int(response.usage.get("output_tokens", 0))
                raw_response_artifact = None
                if response.raw is not None:
                    raw_response_artifact = save_artifact(
                        artifact_dir / f"model_response_{step:03d}.json", response.raw
                    )
                trace.write(
                    "model_response", step=step, response={
                        "text": response.text,
                        "reasoning_content": response.reasoning_content,
                        "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                    }, usage=response.usage, latency_ms=latency_ms, finish_reason=response.finish_reason,
                    raw_response_artifact=raw_response_artifact,
                )
                messages.append(_assistant_message(response.text, response.reasoning_content, response.tool_calls))
                if not response.tool_calls:
                    answer = response.text.strip()
                    if not answer:
                        if counts["empty_answer_retries"] < max_empty_answer_retries:
                            counts["empty_answer_retries"] += 1
                            force_final = True
                            messages.append({
                                "role": "user",
                                "content": (
                                    "Provide the final answer now. Output only the requested answer in the "
                                    "required format, without additional reasoning, and do not call tools."
                                ),
                            })
                            trace.write(
                                "empty_answer_retry", step=step,
                                attempt=counts["empty_answer_retries"],
                                finish_reason=response.finish_reason,
                            )
                            continue
                        status = "empty_answer"
                    trace.write("final_answer", step=step, answer=answer, citations=[], finish_reason=response.finish_reason)
                    break

                force_final = False

                visual_messages: list[dict[str, Any]] = []
                executed_this_step = 0
                blocked_this_step = 0
                for call in response.tool_calls:
                    counts["tool_requests"] += 1
                    if time.monotonic() - started >= total_timeout:
                        raise AgentTimeoutError(f"Reached total timeout of {total_timeout:g} seconds")
                    if call.name not in budgets:
                        raise InvalidToolCallError(f"Unknown tool requested: {call.name}")
                    if counts[call.name] >= budgets[call.name]:
                        observation = {"error": f"Budget exhausted for {call.name}", "limit": budgets[call.name]}
                        counts["blocked_tool_calls"] += 1
                        blocked_this_step += 1
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(observation)})
                        trace.write("tool_result", step=step, tool_call_id=call.id, tool=call.name, success=False, observation=observation)
                        continue
                    signature = _tool_signature(call.name, call.arguments)
                    block_reason = ""
                    if signatures[signature] >= max_identical_tool_calls:
                        block_reason = "This identical tool call has already been executed. Use existing evidence or take a materially different action."
                    elif call.name in search_history:
                        tokens = _search_tokens(call.arguments.get("query", ""))
                        similar_count = sum(
                            _query_similarity(tokens, previous) >= similar_search_threshold
                            for previous in search_history[call.name]
                        )
                        if similar_count >= max_similar_searches:
                            block_reason = "Too many substantially similar searches have already been executed. Open a promising result, change the research direction materially, or answer from existing evidence."
                        elif searches_since_open >= max_searches_without_open:
                            block_reason = "Search-without-opening limit reached. Use open_page on a promising result or answer from the evidence already collected."
                    if block_reason:
                        observation = {"error": "Tool loop prevented", "reason": block_reason}
                        counts["blocked_tool_calls"] += 1
                        blocked_this_step += 1
                        trace.write(
                            "tool_result", step=step, tool_call_id=call.id, tool=call.name,
                            success=False, blocked=True, observation=observation,
                        )
                        messages.append({
                            "role": "tool", "tool_call_id": call.id,
                            "content": json.dumps(observation, ensure_ascii=False),
                        })
                        continue
                    counts[call.name] += 1
                    counts["tool_calls"] += 1
                    executed_this_step += 1
                    trace.write("tool_start", step=step, tool_call_id=call.id, tool=call.name, arguments=call.arguments)
                    tool_started = time.monotonic()
                    try:
                        result = tools.execute(call.name, call.arguments, call.id)
                        signatures[signature] += 1
                        if call.name in search_history:
                            search_history[call.name].append(_search_tokens(call.arguments.get("query", "")))
                            searches_since_open += 1
                        elif call.name == "open_page":
                            searches_since_open = 0
                        artifact = save_artifact(artifact_dir / f"{call.name}_{_safe_id(call.id)}.json", result.raw)
                        tool_ms = round((time.monotonic() - tool_started) * 1000)
                        trace.write(
                            "tool_result", step=step, tool_call_id=call.id, tool=call.name,
                            success=True, latency_ms=tool_ms, observation=result.observation,
                            raw_artifact=artifact, media=result.media,
                        )
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result.observation, ensure_ascii=False)})
                        visual = _visual_message(result.media)
                        if visual:
                            visual_messages.append(visual)
                    except Exception as exc:
                        counts["tool_errors"] += 1
                        tool_ms = round((time.monotonic() - tool_started) * 1000)
                        observation = {"error": str(exc), "type": type(exc).__name__}
                        trace.write(
                            "tool_result", step=step, tool_call_id=call.id, tool=call.name,
                            success=False, latency_ms=tool_ms, observation=observation,
                        )
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(observation)})
                messages.extend(visual_messages)
                if blocked_this_step and not executed_this_step:
                    consecutive_blocked_steps += 1
                else:
                    consecutive_blocked_steps = 0
                if consecutive_blocked_steps >= max_blocked_tool_steps:
                    force_final = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "The tool loop guard has stopped repeated research. Give the best-supported "
                            "final answer now using the evidence already collected. Do not call tools."
                        ),
                    })
                    trace.write(
                        "loop_guard_final", step=step,
                        consecutive_blocked_steps=consecutive_blocked_steps,
                    )
            else:
                raise MaxStepsError(f"Reached maximum of {max_steps} agent steps")
        except AgentError as exc:
            status, error = exc.status, str(exc)
        except Exception as exc:
            status, error = "api_error", f"{type(exc).__name__}: {exc}"

        total_ms = round((time.monotonic() - started) * 1000)
        trace.write("run_end", status=status, error={"message": error} if error else None, metrics={
            "total_latency_ms": total_ms, "model_calls": counts["model_calls"], "tool_calls": counts["tool_calls"],
            "text_search_calls": counts["text_search"], "image_search_calls": counts["image_search"],
            "open_page_calls": counts["open_page"], "input_tokens": input_tokens, "output_tokens": output_tokens,
            "tool_requests": counts["tool_requests"], "blocked_tool_calls": counts["blocked_tool_calls"],
            "tool_errors": counts["tool_errors"], "empty_answer_retries": counts["empty_answer_retries"],
        })
    return {
        "answer": answer, "status": status, "error": error,
        "latency_s": round((time.monotonic() - started), 3),
        "tool_calls": counts["tool_calls"], "text_search_calls": counts["text_search"],
        "image_search_calls": counts["image_search"], "open_page_calls": counts["open_page"],
        "blocked_tool_calls": counts["blocked_tool_calls"],
    }
