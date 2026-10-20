# Default harness guide

[← OpenDS-World paper overview](../README.md)

This repository provides a reference template for the agent harness and evaluation workflow used in the OpenDS-World study. This guide covers the default agent implementation: environment setup, dataset preparation, execution, outputs, and recovery. Run commands from the repository root.

The paper evaluates models with shared task inputs, prompts, tools, and search budgets. A run searches for evidence, produces a final answer from the collected observations, and is scored against the reference answer. Paper results are reported in the [README](../README.md#benchmark-results); reproducing them also requires the corresponding data, model settings, and external services.

## Quick start

The local diagnostic requires **Python 3.10–3.13**; Python 3.12 is recommended. A real web/video run also needs Docker, FFmpeg (`ffprobe`), `rg`, an OpenAI-compatible model endpoint, and a Serper API key.

For video cases, the model endpoint must accept image inputs returned by the video tools.

```bash
git clone https://github.com/Yu-xm/OpenDS.git
cd OpenDS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# One end-to-end local diagnostic; no API key, Docker, or paid model call.
opends-world --config configs/mock-text.json --limit 1
```

The diagnostic verifies installation, configuration loading, and output generation. It prints `success=1` and writes mock outputs to `results/OpenDS-World-Text/mock-model/mock-smoke/`. To run it again, set a new `run_id` in `configs/mock-text.json`.

### Run a real case

1. Build the browser and video images.
2. Download the 200 videos from [Yu2020/OpenDS-World](https://huggingface.co/datasets/Yu2020/OpenDS-World) and prepare the matching 200-row question/answer CSV.
3. Export the dataset paths, model endpoint, and search credentials, then run the configuration.

```bash
docker build -f toolbox/docker/browser/Dockerfile -t toolbox-browser:pdf-v1 toolbox
docker build -f toolbox/docker/youtube/Dockerfile -t toolbox-youtube:latest toolbox

python -m pip install 'huggingface_hub[cli]'
hf download Yu2020/OpenDS-World --repo-type dataset \
  --include 'videos/*.mp4' --local-dir data/opends-world

export BENCHMARK_DATASET_CSV='/absolute/path/to/final-200.csv'
export BENCHMARK_VIDEO_DIR="$PWD/data/opends-world/videos"
export MODEL_ID='your-model-id'
export MODEL_BASE_URL='https://your-openai-compatible-service/v1'
export MODEL_API_KEY='your-key'  # use '' when the endpoint needs no authentication
export SERPER_API_KEY='your-serper-key'

# Give this experiment a unique run_id in evaluation.yaml first.
opends-world --config evaluation.yaml --index 1
```

For all cases, run `opends-world --config evaluation.yaml` after supplying every matching video. The configuration rejects a CSV unless it has exactly 200 data rows, including when `--index` selects one case. `--index N` selects the Nth CSV data row, starting at 1; it is not the sample ID. `--resume` uses the same configuration and run ID after an interruption. See [`evaluation_all.yaml`](../evaluation_all.yaml) for the complete option reference.

## Benchmark data

OpenDS-World contains **200 cases**. Obtain the benchmark data from [Yu2020/OpenDS-World](https://huggingface.co/datasets/Yu2020/OpenDS-World). Each evaluation input consists of a question/answer row and its matching video (`videos/001.mp4` through `videos/200.mp4`).

Point `BENCHMARK_DATASET_CSV` to the downloaded CSV and `BENCHMARK_VIDEO_DIR` to its matching local videos. The configuration expects `id`, `question`, and `answer` columns and exactly 200 rows. It maps numeric ID `58` to `058.mp4`; adjust `input_video_id_width` or use an explicit video filename column if the published data uses another naming scheme. Set `input_video_subtitles` to a Parquet file with aligned `id` and `subtitle` columns when using subtitles; its IDs must match the 200-case dataset. [`examples/opends_world_text.csv`](../examples/opends_world_text.csv) is a synthetic one-row mock input, not benchmark data.

> **External evidence changes.** Search rankings, pages, and video availability can change between runs. Preserve the generated trajectories and artifacts alongside reported scores. Video use remains subject to the source platform's terms.

## Evaluation protocol

| Parameter | Default | Why it matters |
| --- | ---: | --- |
| Research deadline | 3,600 s | Per case, including model and tool work |
| Shared tool budget | 100 calls | Failed executions also consume a call |
| Images per model request | 250 | Shared by search, browser, files, and video frames |
| Finalization budget | 5,000 output tokens | Tool-free answer generation after research stops |
| Second-answer reserve | 1,000 tokens | Available if the first finalization needs a retry |
| Model sampling | Provider-specific | Set `model.parameters` explicitly for reported experiments |

If research stops because of a deadline, exhausted budget, model error, or empty answer, finalization consumes **only completed observations** from the durable trace. Interrupted model output and unfinished tool responses are discarded. The run records `status` and `finalization_status` separately; failed cases do not silently become successes.

For a reproducible report, retain the code commit, model and inference settings, dataset hashes, execution time, tool responses, and scoring procedure. Optional ASR and MinerU readers are disabled in [`evaluation.yaml`](../evaluation.yaml); enable them only when their services or images are ready.

## Results and recovery

```text
results/<benchmark>/<model>/<run_id>/
├── predictions.csv    # answer, status, latency, tool counts, error
├── run_config.json    # redacted configuration and provenance metadata
├── trajectories/      # per-case JSONL events
└── artifacts/         # pages, screenshots, files, and saved model context
```

The CLI exits with code `2` when any case has a non-success status. Investigate `infrastructure_error` before scoring; it reflects an environment failure, not an incorrect model answer. For answer-only recovery from an existing trace, run `python ops/finalize_trace.py --help`. Recovery writes to a **new** directory and does not repeat research tools.

## Score the answers

Pass the generated `predictions.csv` to the standalone judge using the [evaluation guide](evaluation.md). Scoring is a separate step; execution success does not imply a correct answer.

### When a run fails

| Symptom | First check |
| --- | --- |
| Missing `${...}` variable | Export the variable named in the YAML field-path error. |
| Missing video or `ffprobe` error | Download the matching MP4; install FFmpeg. |
| Docker image or daemon error | Start Docker and build both images above. |
| HTTP 401/403 or 429 | Check credentials, service permissions, or rate limits. |
| Existing run ID | Use a new ID for a new experiment; use `--resume` only for the original run. |
| Sample IDs map to the same output name | Rename colliding IDs (for example `a/b` and `a_b`) before running; the runner rejects them before writing case artifacts. |
| `infrastructure_error` | Read the row's `error` and the matching JSONL trace before scoring. |

Token totals include provider-reported usage from incomplete research responses,
even though their unfinished answer text is discarded. The corresponding
`incomplete_model_response` event retains usage for auditing.

Run the regression tests after installing the project:

```bash
python -m unittest discover -s tests -v
```

