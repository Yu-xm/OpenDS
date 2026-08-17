from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

from omnibrowse_bench.agent import _compact_history, run_sample
from omnibrowse_bench.http import post_json
from omnibrowse_bench.models import OpenAICompatibleBackend
from omnibrowse_bench.runner import run
from omnibrowse_bench.types import ModelResponse, Sample, ToolCall, ToolResult


class EmptyThenAnswerBackend:
    display_name = "reasoning-model"
    supports_video = False
    supports_audio = False

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        self.calls.append((list(messages), list(tools)))
        if len(self.calls) == 1:
            return ModelResponse(
                reasoning_content="The answer follows from the evidence.",
                usage={"input_tokens": 10, "output_tokens": 8},
                finish_reason="stop", raw={"choices": [{"message": {"reasoning_content": "reasoning"}}]},
            )
        return ModelResponse(
            text="University of Oxford", usage={"input_tokens": 12, "output_tokens": 3},
            finish_reason="stop", raw={"choices": [{"message": {"content": "University of Oxford"}}]},
        )


class RepeatingSearchBackend:
    display_name = "looping-model"
    supports_video = False
    supports_audio = False

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        self.calls.append(list(tools))
        if len(self.calls) <= 2:
            return ModelResponse(
                tool_calls=[ToolCall(f"call-{len(self.calls)}", "text_search", {"query": "Oxford energy founder"})],
                usage={"input_tokens": 10, "output_tokens": 2}, finish_reason="tool_calls",
            )
        return ModelResponse(
            text="University of Oxford", usage={"input_tokens": 10, "output_tokens": 3},
            finish_reason="stop",
        )


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class PipelineTest(unittest.TestCase):
    def test_openai_adapter_reads_reasoning_content(self) -> None:
        backend = OpenAICompatibleBackend({
            "display_name": "test", "model_id": "test", "base_url": "https://example.test/v1",
        })
        raw = {
            "choices": [{
                "message": {"content": "", "reasoning_content": "private visible reasoning", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        }
        with patch("omnibrowse_bench.models.post_json", return_value=raw) as post:
            response = backend.generate([{"role": "user", "content": "question"}], [])
        self.assertEqual(response.text, "")
        self.assertEqual(response.reasoning_content, "private visible reasoning")
        payload = post.call_args.args[1]
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_openai_adapter_assembles_streamed_reasoning_and_tool_call(self) -> None:
        backend = OpenAICompatibleBackend({
            "display_name": "test", "model_id": "test", "base_url": "https://example.test/v1",
            "parameters": {"stream": True},
        })
        chunks = [
            {"id": "r1", "model": "test", "choices": [{
                "index": 0, "delta": {"reasoning_content": "Search "}, "finish_reason": None,
            }]},
            {"id": "r1", "model": "test", "choices": [{
                "index": 0, "delta": {"reasoning_content": "now.", "tool_calls": [{
                    "index": 0, "id": "call_1", "function": {
                        "name": "text_search", "arguments": "{\"query\":",
                    },
                }]}, "finish_reason": None,
            }]},
            {"id": "r1", "model": "test", "choices": [{
                "index": 0, "delta": {"tool_calls": [{
                    "index": 0, "function": {"arguments": "\"Oxford\"}"},
                }]}, "finish_reason": "tool_calls",
            }], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
        ]
        with patch("omnibrowse_bench.models.post_sse_json", return_value=chunks) as post:
            response = backend.generate([{"role": "user", "content": "question"}], [{"type": "function"}])
        self.assertEqual(response.reasoning_content, "Search now.")
        self.assertEqual(response.tool_calls[0].name, "text_search")
        self.assertEqual(response.tool_calls[0].arguments, {"query": "Oxford"})
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertTrue(post.call_args.args[1]["stream"])

    def test_empty_answer_gets_one_forced_final_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = EmptyThenAnswerBackend()
            sample = Sample("s1", "question", {}, root)
            trace_path = root / "trajectory.jsonl"
            outcome = run_sample(
                sample, "OmniBrowse-Text", "test-run", backend,
                [{"role": "user", "content": "question"}], {"api_key": "unused"},
                {"max_steps": 3, "max_empty_answer_retries": 1}, trace_path, root / "artifacts",
            )
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["answer"], "University of Oxford")
            self.assertTrue(backend.calls[0][1])
            self.assertEqual(backend.calls[1][1], [])
            self.assertEqual(backend.calls[1][0][-2]["reasoning_content"], "The answer follows from the evidence.")
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertIn("empty_answer_retry", [event["event"] for event in events])
            self.assertEqual(events[-1]["metrics"]["empty_answer_retries"], 1)

    def test_repeated_tool_call_is_blocked_and_forces_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RepeatingSearchBackend()
            sample = Sample("s1", "question", {}, root)
            trace_path = root / "trajectory.jsonl"
            result = ToolResult(
                {"query": "Oxford energy founder", "results": [{"title": "Oxford", "url": "https://example.test"}]},
                {"organic": []},
            )
            with patch("omnibrowse_bench.agent.SerperTools.execute", return_value=result) as execute:
                outcome = run_sample(
                    sample, "OmniBrowse-Text", "test-run", backend,
                    [{"role": "user", "content": "question"}], {"api_key": "unused"},
                    {"max_steps": 4, "max_identical_tool_calls": 1, "max_blocked_tool_steps": 1},
                    trace_path, root / "artifacts",
                )
            self.assertEqual(outcome["answer"], "University of Oxford")
            self.assertEqual(outcome["text_search_calls"], 1)
            self.assertEqual(outcome["blocked_tool_calls"], 1)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(backend.calls[-1], [])
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            blocked = [event for event in events if event.get("blocked")]
            self.assertEqual(len(blocked), 1)
            self.assertEqual(events[-1]["metrics"]["tool_requests"], 2)

    def test_http_429_is_retried_after_retry_after(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test", 429, "rate limited", {"Retry-After": "0"},
            io.BytesIO(b'{"message":"TPM limit reached"}'),
        )
        with patch("omnibrowse_bench.http.urllib.request.urlopen", side_effect=[error, FakeResponse(b'{"ok":true}')]) as request:
            with patch("omnibrowse_bench.http.time.sleep") as sleep:
                result = post_json("https://example.test", {}, {}, 1, max_retries=1)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_old_tool_history_is_compacted_deterministically(self) -> None:
        messages = [
            {"role": "assistant", "content": "", "reasoning_content": "old reasoning"},
            {"role": "tool", "content": json.dumps({
                "query": "q1", "results": [
                    {"rank": 1, "snippet": "a" * 100},
                    {"rank": 2, "snippet": "b" * 100},
                    {"rank": 3, "snippet": "c" * 100},
                ],
            })},
            {"role": "assistant", "content": "", "reasoning_content": "new reasoning"},
            {"role": "tool", "content": json.dumps({"query": "q2", "results": [{"rank": 1}]})},
        ]
        changed, before, after = _compact_history(messages, 1, 2, 100, 1)
        self.assertGreaterEqual(changed, 2)
        self.assertGreater(before, after)
        self.assertNotIn("reasoning_content", messages[0])
        self.assertEqual(messages[2]["reasoning_content"], "new reasoning")
        compacted = json.loads(messages[1]["content"])
        self.assertEqual(len(compacted["results"]), 2)
        self.assertTrue(compacted["_history_compacted"])
        self.assertNotIn("_history_compacted", json.loads(messages[3]["content"]))

    def test_mock_text_run_writes_csv_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "questions.csv"
            csv_path.write_text("id,question,gold_answer\ns1,What is this?,gold\n", encoding="utf-8")
            fake_config_path = root / "config.json"
            fake_config_path.write_text("{}", encoding="utf-8")
            config = {
                "_config_path": str(fake_config_path),
                "benchmark": "OmniBrowse-Text",
                "run_id": "test-run",
                "dataset": {"path": str(csv_path), "version": "test"},
                "model": {"adapter": "mock", "display_name": "mock", "model_id": "mock"},
                "search": {"api_key": "not-used"},
                "agent": {"max_steps": 2},
                "output_root": str(root / "results"),
            }
            result = run(config)
            predictions = result / "predictions.csv"
            trace = result / "trajectories" / "s1.jsonl"
            self.assertTrue(predictions.is_file())
            self.assertTrue(trace.is_file())
            with predictions.open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "success")
            self.assertEqual(row["benchmark"], "OmniBrowse-Text")
            self.assertEqual(row["model"], "mock")
            self.assertEqual(row["text_search_calls"], "0")
            self.assertEqual(row["image_search_calls"], "0")
            self.assertEqual(row["open_page_calls"], "0")
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], ["run_start", "model_response", "final_answer", "run_end"])
            self.assertTrue(all(event["benchmark"] == "OmniBrowse-Text" for event in events))


if __name__ == "__main__":
    unittest.main()
