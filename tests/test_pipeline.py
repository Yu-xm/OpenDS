from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from omnibrowse_bench.runner import run


class PipelineTest(unittest.TestCase):
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
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], ["run_start", "model_response", "final_answer", "run_end"])
            self.assertTrue(all(event["benchmark"] == "OmniBrowse-Text" for event in events))


if __name__ == "__main__":
    unittest.main()

