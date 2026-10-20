# Evaluation guide

[← Paper overview](../README.md) · [Default harness](harness.md)

Score answers from either the default agent or your own harness with the same standalone LLM-as-a-Judge.

## Install the judge

The standalone judge needs Python 3.10–3.13 and an OpenAI-compatible model endpoint. It does not require the browser/video Docker images, FFmpeg, or Serper.

```bash
git clone https://github.com/Yu-xm/OpenDS.git
cd OpenDS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Prediction format

The default harness already exports this format. For your own harness, write a UTF-8 CSV with one row per benchmark task, retaining failed and unanswered tasks:

| Column | Content |
| --- | --- |
| `id` | Unique benchmark task ID; preserve it when joining reference answers. |
| `question` | Original task question. |
| `gold_answer` | Matching reference answer, joined only after answer generation. |
| `model_answer` | Final agent answer; leave empty if unanswered. |
| `status` | Overall task execution status. |
| `dataset` | Optional dataset label for grouped statistics. |
| `error` | Optional execution failure details. |

Synthetic format example:

```csv
id,question,gold_answer,model_answer,status,dataset,error
example-001,What is two plus two?,4,Four,success,format-example,
```

Keep reference answers out of the solving agent's inputs. Export all 200 tasks for a full benchmark result; omitted tasks are not automatically added to the denominator by the judge. Join references by task ID, not row order.

### Execution status and tool failures

For an external harness, use `success` when the task run completes normally, including after recoverable tool failures. Preserve a failure status when the task run itself fails. `status` does not encode whether the answer is correct.

The default harness records tool failures separately from task completion. The existing `trace_path` points to per-task traces containing tool failure details, cleanup events, and error counts. The CSV `error` field describes overall execution failures. A completed answer can have `status=success` alongside these diagnostics. Failures that interrupt the overall research runtime, such as toolbox startup failure, still produce `infrastructure_error`. Review diagnostics when reporting results.

The judge receives the execution status and error alongside the answer. Its prompt treats failed executions as incorrect, and summaries also include an `execution_success_only` subset. Accurate run statuses therefore matter for interpreting the score.

## Answer scoring

Answer generation and scoring are separate steps. `opends-world --config ...` writes answers and trajectories; it does **not** automatically run the judge. Its `success` status describes execution, not answer correctness.

After installing the project, score the generated `predictions.csv` from the repository root:

```bash
python ops/judge_predictions.py \
  --predictions /path/to/predictions.csv \
  --output-dir /path/to/judgments \
  --model-id your-judge-model \
  --base-url https://your-endpoint/v1
```

Set `MODEL_API_KEY` to the judge endpoint's credential in your shell. The judge compares the question, reference answer, and candidate answer for semantic equivalence. The paper uses Qwen3.8-27B; record the actual judge used when reporting your results.

Outputs include per-case `judgments.jsonl` and `judgments.csv`, plus `summary.json` with correct counts, accuracy, and `judge_errors`. Accuracy is the number of correct judgments divided by the number of judged rows, **including rows whose judge calls failed**. These failures also appear separately as `judge_errors`; report them alongside accuracy. A partial run or `--limit` result is not a full 200-task benchmark score.

### Using an external harness

You can evaluate in your own agent environment without installing our browser/video stack. Report the evaluated task count, correct count, accuracy, judge errors, model and judge identities, and search budget; note any deviations from the paper's protocol. Final aggregate statistics are sufficient to share results.

To optionally reuse this judge, install the Python project and export a CSV with unique `id` values and columns `question`, `gold_answer`, `model_answer`, and `status` (the overall task-run status). Optional `dataset` and `error` columns supply grouping and failure context. Never provide reference answers to the solving agent; they are inputs to the judge only.

## Resume scoring

For `ops/judge_predictions.py --resume`, the output directory must contain a
hashed `judge_manifest.json`. The SHA-256 identity binds the predictions file,
judge model and endpoint, prompt, inference settings, sample limit, and judge
script version. The manifest must match the current inputs and judge configuration. Use a new
output directory to re-score changed inputs.
The manifest is saved before the first request so interrupted runs can resume.
API keys are excluded from the identity, allowing credential rotation.

