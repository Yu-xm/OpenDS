import contextlib
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from opends_world.agent import run_sample
from opends_world.config import load_config
from opends_world.runner import run
from opends_world.types import ModelResponse, Sample, ToolCall, ToolInfrastructureError


ROOT = Path(__file__).resolve().parents[1]


class TruncatedModel:
    display_name = "test-model"

    def __init__(self, finish_reason):
        self.finish_reason = finish_reason
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(text="discard this incomplete answer",
                                 finish_reason=self.finish_reason,
                                 usage={"input_tokens": 123, "output_tokens": 45})
        assert "discard this incomplete answer" not in json.dumps(messages)
        return ModelResponse(text="final answer", finish_reason="stop",
                             usage={"input_tokens": 10, "output_tokens": 5})


class ExecutionIntegrityTests(unittest.TestCase):
    def test_discarded_response_usage_is_counted_and_preserved(self):
        for reason in ("length", "content_filter", "tool_calls"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "trace.jsonl"
                result = run_sample(
                    Sample("1", "question", {}, root), "OpenDS-World-Text", "test",
                    TruncatedModel(reason), [{"role": "user", "content": "question"}],
                    {"api_key": "unused"}, {"timeout_seconds": 10}, trace, root / "artifacts",
                )
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                metrics = events[-1]["metrics"]
                self.assertEqual(result["status"], "success")
                self.assertEqual((metrics["input_tokens"], metrics["output_tokens"]), (133, 50))
                discarded = next(e for e in events if e["event"] == "incomplete_model_response")
                self.assertEqual(discarded["usage"], {"input_tokens": 123, "output_tokens": 45})
                self.assertNotIn("discard this incomplete answer", trace.read_text())

    def test_tool_failure_preserves_completed_answer_and_trace(self):
        class RecoveringModel:
            display_name = "test-model"
            calls = 0

            def generate(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return ModelResponse(tool_calls=[ToolCall("call-1", "text_search", {"query": "test"})], finish_reason="tool_calls")
                return ModelResponse(text="final answer", finish_reason="stop")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("opends_world.agent.SerperTools.execute", side_effect=ToolInfrastructureError("service unavailable")):
                result = run_sample(
                    Sample("1", "question", {}, root), "OpenDS-World-Text", "test",
                    RecoveringModel(), [{"role": "user", "content": "question"}],
                    {"api_key": "unused"}, {"timeout_seconds": 10}, root / "trace.jsonl", root / "artifacts",
                )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["answer"], "final answer")
            self.assertEqual(result["error"], "")
            events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            failure = next(e for e in events if e["event"] == "tool_result" and not e["success"])
            self.assertEqual(failure["observation"]["error"], "service unavailable")
            self.assertNotIn("warnings", result)
            self.assertEqual(events[-1]["metrics"]["tool_errors"], 1)
            self.assertEqual(events[-1]["metrics"]["infrastructure_errors"], 1)

    def test_startup_failure_remains_infrastructure_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("opends_world.agent.create_toolbox", side_effect=RuntimeError("startup failed")):
                result = run_sample(
                    Sample("1", "question", {}, root), "OpenDS-World-Text", "test",
                    TruncatedModel("length"), [{"role": "user", "content": "question"}],
                    {"api_key": "unused"}, {"timeout_seconds": 10}, root / "trace.jsonl", root / "artifacts",
                    {"enabled": True},
                )
            self.assertEqual(result["status"], "infrastructure_error")
            self.assertIn("startup failed", result["error"])

    def config(self, root, ids):
        path = root / "cases.csv"
        with path.open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "question", "gold_answer"])
            writer.writerows((identifier, "Question " + identifier, "answer") for identifier in ids)
        config = load_config(ROOT / "configs/mock-text.json")
        config.update(run_id="test", output_root=str(root / "results"))
        config["dataset"] = {"path": str(path)}
        return config

    def test_output_name_collision_rejected_before_any_case_runs(self):
        for ids in (("a/b", "a_b"), (".x", "x"), ("/", "_")):
            with self.subTest(ids=ids), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(ValueError, "same output name"):
                    run(self.config(root, ids))
                self.assertFalse(list((root / "results").rglob("*.jsonl")))
                self.assertFalse(list((root / "results").rglob("initial_context.json")))

    def test_normal_ids_keep_separate_traces_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = self.config(root, ("001", "002"))
            output = run(config)
            before = {p.name: p.read_bytes() for p in (output / "trajectories").glob("*.jsonl")}
            self.assertEqual(set(before), {"001.jsonl", "002.jsonl"})
            run(config, resume=True)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (output / "trajectories").glob("*.jsonl")})


if __name__ == "__main__":
    unittest.main()
