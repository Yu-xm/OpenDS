import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'ops/judge_predictions.py'
spec = importlib.util.spec_from_file_location('judge', SCRIPT)
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pred = self.root / 'predictions.csv'
        self.pred.write_text('id,question,gold_answer,model_answer,status\n1,One plus one?,2,2,success\n')
        self.args = argparse.Namespace(predictions=self.pred, output_dir=self.root/'out',
            model_id='model-a', base_url='http://localhost:1/v1', api_key='secret-a',
            concurrency=1, timeout_seconds=30, attempts=1, limit=None, resume=False)

    def result(self, row, args):
        return dict(id=row['id'], dataset='', execution_status='success', correct=True,
            confidence=1, reason='Matches', judge_status='success', judge_error='',
            judge_attempts=1, judge_latency_s=0, judge_input_tokens=1, judge_output_tokens=1)

    def run_cli(self, effect=None):
        with patch.object(judge, 'parse_args', return_value=self.args), \
             patch.object(judge, 'judge_one', side_effect=effect or self.result) as call, \
             contextlib.redirect_stdout(io.StringIO()):
            judge.main()
        return call.call_count

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.args.output_dir.iterdir() if p.is_file()}

    def seed(self):
        self.assertEqual(self.run_cli(), 1)
        self.args.resume = True

    def test_same_input_resumes_without_model_call_and_allows_key_rotation(self):
        self.seed()
        self.args.api_key = 'secret-b'
        self.args.concurrency = 2
        self.assertEqual(self.run_cli(), 0)
        self.assertNotIn('secret-', (self.args.output_dir/'judge_manifest.json').read_text())

    def test_changed_prediction_rejected_without_modifying_outputs(self):
        self.seed()
        before = self.snapshot()
        self.pred.write_text(self.pred.read_text().replace(',2,success', ',WRONG,success'))
        with self.assertRaisesRegex(SystemExit, 'hash mismatch'):
            self.run_cli()
        self.assertEqual(before, self.snapshot())

    def test_changed_model_endpoint_limit_and_parameters_rejected(self):
        self.seed()
        before = self.snapshot()
        for field, value in [('model_id','model-b'),('base_url','http://elsewhere/v1'),
                             ('limit',1),('timeout_seconds',60),('attempts',2)]:
            with self.subTest(field=field):
                original = getattr(self.args, field)
                setattr(self.args, field, value)
                with self.assertRaisesRegex(SystemExit, 'hash mismatch'):
                    self.run_cli()
                self.assertEqual(before, self.snapshot())
                setattr(self.args, field, original)

    def test_changed_prompt_rejected(self):
        self.seed()
        with patch.object(judge, 'SYSTEM_PROMPT', 'Different rubric'):
            with self.assertRaisesRegex(SystemExit, 'hash mismatch'):
                self.run_cli()

    def test_legacy_manifest_rejected_without_changes(self):
        self.seed()
        (self.args.output_dir/'judge_manifest.json').write_text('{"judge_model":"model-a"}')
        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, 'legacy manifest'):
            self.run_cli()
        self.assertEqual(before, self.snapshot())

    def test_missing_manifest_rejected(self):
        self.seed()
        (self.args.output_dir/'judge_manifest.json').unlink()
        with self.assertRaisesRegex(SystemExit, 'without a hashed'):
            self.run_cli()

    def test_mixed_judgment_rejected(self):
        self.seed()
        path = self.args.output_dir/'judgments.jsonl'
        row = json.loads(path.read_text())
        row['resume_sha256'] = 'old-run'
        path.write_text(json.dumps(row)+'\n')
        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, 'records do not match'):
            self.run_cli()
        self.assertEqual(before, self.snapshot())

    def test_interrupted_first_run_has_manifest_and_can_resume(self):
        with self.assertRaises(RuntimeError):
            self.run_cli(lambda *_: (_ for _ in ()).throw(RuntimeError('interrupted')))
        self.assertTrue((self.args.output_dir/'judge_manifest.json').is_file())
        self.args.resume = True
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.run_cli(), 0)


if __name__ == '__main__':
    unittest.main()
