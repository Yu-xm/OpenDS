from __future__ import annotations

import base64
import json
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


def _assistant_message(text: str, calls: list[Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text or ""}
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
    total_timeout = float(agent_config.get("timeout_seconds", 600))
    budgets = {
        "text_search": int(agent_config.get("max_text_searches", 10)),
        "image_search": int(agent_config.get("max_image_searches", 5)),
        "open_page": int(agent_config.get("max_page_opens", 15)),
    }
    tools = SerperTools(serper_config, artifact_dir)

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
                call_started = time.monotonic()
                response = model.generate(messages, TOOL_DEFINITIONS)
                latency_ms = round((time.monotonic() - call_started) * 1000)
                counts["model_calls"] += 1
                input_tokens += int(response.usage.get("input_tokens", 0))
                output_tokens += int(response.usage.get("output_tokens", 0))
                trace.write(
                    "model_response", step=step, response={
                        "text": response.text,
                        "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                    }, usage=response.usage, latency_ms=latency_ms, finish_reason=response.finish_reason,
                )
                messages.append(_assistant_message(response.text, response.tool_calls))
                if not response.tool_calls:
                    answer = response.text.strip()
                    if not answer:
                        status = "empty_answer"
                    trace.write("final_answer", step=step, answer=answer, citations=[], finish_reason=response.finish_reason)
                    break

                visual_messages: list[dict[str, Any]] = []
                for call in response.tool_calls:
                    if time.monotonic() - started >= total_timeout:
                        raise AgentTimeoutError(f"Reached total timeout of {total_timeout:g} seconds")
                    if call.name not in budgets:
                        raise InvalidToolCallError(f"Unknown tool requested: {call.name}")
                    if counts[call.name] >= budgets[call.name]:
                        observation = {"error": f"Budget exhausted for {call.name}", "limit": budgets[call.name]}
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(observation)})
                        trace.write("tool_result", step=step, tool_call_id=call.id, tool=call.name, success=False, observation=observation)
                        continue
                    counts[call.name] += 1
                    counts["tool_calls"] += 1
                    trace.write("tool_start", step=step, tool_call_id=call.id, tool=call.name, arguments=call.arguments)
                    tool_started = time.monotonic()
                    try:
                        result = tools.execute(call.name, call.arguments, call.id)
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
                        tool_ms = round((time.monotonic() - tool_started) * 1000)
                        observation = {"error": str(exc), "type": type(exc).__name__}
                        trace.write(
                            "tool_result", step=step, tool_call_id=call.id, tool=call.name,
                            success=False, latency_ms=tool_ms, observation=observation,
                        )
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(observation)})
                messages.extend(visual_messages)
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
        })
    return {"answer": answer, "status": status, "error": error, "latency_s": round((time.monotonic() - started), 3), "tool_calls": counts["tool_calls"]}
