# Evaluation guide

[← Paper overview](../README.md) · [Default harness](harness.md)

Score answers from either the default agent or your own harness with the same standalone LLM-as-a-Judge.

Complete [Installation](../README.md#1-installation) first. Run commands from the repository root with an OpenAI-compatible judge endpoint.

## Prediction format

Use a UTF-8 CSV with one row per task (200 rows), including failed and unanswered tasks. The default harness exports this format.

| Column | Content |
| --- | --- |
| `id` | Unique benchmark task ID. |
| `question` | Original task question. |
| `gold_answer` | Reference answer matched by task ID. |
| `model_answer` | Final agent answer; leave empty if unanswered. |
| `status` | Overall task execution status. |
| `dataset` | Optional dataset label for grouped statistics. |
| `error` | Optional execution failure details. |

Reference answers are used only for judging. Set `status` to `success` for completed task runs; retain the execution failure status otherwise. Tool diagnostics are available in the default harness traces.

## Answer scoring

Set `MODEL_API_KEY` to the judge API key and run:

```bash
python ops/judge_predictions.py \
  --predictions /path/to/predictions.csv \
  --output-dir /path/to/judgments \
  --model-id your-judge-model \
  --base-url https://your-endpoint/v1
```

The judge checks semantic equivalence to the reference answer. The paper uses Qwen3.8-27B.

## Results

| File | Contents |
| --- | --- |
| `judgments.csv`, `judgments.jsonl` | Per-task judgments |
| `summary.json` | Accuracy, correct counts, and `judge_errors` |

Accuracy is correct judgments divided by judged rows, including judge errors. The judge treats failed executions as incorrect. Report the evaluated task count, accuracy, judge errors, model, judge, and search budget.

## Resume scoring

Add `--resume` to the scoring command to continue with the same inputs, judge settings, and output directory. Keep the generated `judge_manifest.json`; use a new output directory when inputs or judge settings change.
