# Default harness guide

[← OpenDS-World paper overview](../README.md)

Complete [Installation](../README.md#1-installation) and [dataset download](../README.md#dataset) first. Run commands from the repository root.

## Tool setup

The default harness requires Docker, FFmpeg (`ffprobe`), `rg`, an OpenAI-compatible model endpoint, and a Serper API key.

For video cases, the model endpoint must accept image inputs returned by the video tools.

### Local diagnostic

```bash
opends-world --config configs/mock-text.json --limit 1
```

Use a new `run_id` in `configs/mock-text.json` for each diagnostic run.

### Run the benchmark

1. Build the browser and video images.
2. Set the paths to the 200-row question/answer CSV and corresponding videos.
3. Export the dataset paths, model endpoint, and search credentials, then run the configuration.

```bash
docker build -f toolbox/docker/browser/Dockerfile -t toolbox-browser:pdf-v1 toolbox
docker build -f toolbox/docker/youtube/Dockerfile -t toolbox-youtube:latest toolbox

export BENCHMARK_DATASET_CSV='/absolute/path/to/final-200.csv'
export BENCHMARK_VIDEO_DIR="$PWD/data/opends-world/videos"
export MODEL_ID='your-model-id'
export MODEL_BASE_URL='https://your-openai-compatible-service/v1'
export MODEL_API_KEY='your-key'  # use '' when the endpoint needs no authentication
export SERPER_API_KEY='your-serper-key'

# Give this experiment a unique run_id in evaluation.yaml first.
opends-world --config evaluation.yaml --index 1
```

Run all tasks with `opends-world --config evaluation.yaml`. Use `--index N` for CSV row N (starting at 1), or `--resume` to continue the same run. Full configuration reference: [`evaluation_all.yaml`](../evaluation_all.yaml).

## Benchmark data

| Setting | Value |
| --- | --- |
| `BENCHMARK_DATASET_CSV` | CSV with 200 rows and columns `id`, `question`, `answer` |
| `BENCHMARK_VIDEO_DIR` | Directory containing the corresponding videos |
| `input_video_subtitles` (optional) | Parquet file with matching `id` and `subtitle` columns |

## Evaluation protocol

| Parameter | Default | Why it matters |
| --- | ---: | --- |
| Research deadline | 3,600 s | Per case, including model and tool work |
| Shared tool budget | 100 calls | Failed executions also consume a call |
| Images per model request | 250 | Shared by search, browser, files, and video frames |
| Finalization budget | 5,000 output tokens | Tool-free answer generation after research stops |
| Second-answer reserve | 1,000 tokens | Available if the first finalization needs a retry |
| Model sampling | Provider-specific | Set `model.parameters` explicitly for reported experiments |

Optional ASR and MinerU readers can be configured in [`evaluation.yaml`](../evaluation.yaml).

## Results and recovery

```text
results/<benchmark>/<model>/<run_id>/
├── predictions.csv    # answer, status, latency, tool counts, error
├── run_config.json    # redacted configuration and provenance metadata
├── trajectories/      # per-case JSONL events
└── artifacts/         # pages, screenshots, files, and saved model context
```

For answer recovery from a saved trace, see `python ops/finalize_trace.py --help`.

## Score the answers

Pass the generated `predictions.csv` to the standalone judge using the [evaluation guide](evaluation.md).

### When a run fails

| Symptom | First check |
| --- | --- |
| Missing `${...}` variable | Export the variable named in the YAML field-path error. |
| Missing video or `ffprobe` error | Download the matching MP4; install FFmpeg. |
| Docker image or daemon error | Start Docker and build both images above. |
| HTTP 401/403 or 429 | Check credentials, service permissions, or rate limits. |
| Existing run ID | Use a new ID for a new experiment; use `--resume` only for the original run. |
| Sample IDs map to the same output name | Use unique output-safe sample IDs. |
| `infrastructure_error` | Read the row's `error` and the matching JSONL trace before scoring. |
