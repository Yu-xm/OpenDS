<div align="center">

# OpenDS-World

[English](README.md) · [简体中文](README.zh-CN.md)

### Deep Search in Real-World Web Environments

<p>
  <a href="https://huggingface.co/datasets/Yu2020/OpenDS-World"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-OpenDS--World-FFD21E?style=flat-square" alt="Hugging Face dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2563EB?style=flat-square" alt="License: Apache-2.0"></a>
  <a href="https://github.com/Yu-xm/OpenDS"><img src="https://img.shields.io/github/stars/Yu-xm/OpenDS?style=flat-square&logo=github&label=Stars" alt="GitHub stars"></a>
</p>

<p>
  <a href="#benchmark-results">📊 Benchmark results</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/Yu2020/OpenDS-World">🤗 Dataset</a> &nbsp;·&nbsp;
  <a href="docs/evaluation.md">🛠 Evaluation guide</a>
</p>

<img src="assets/OpenDS.png" width="880" alt="OpenDS: Deep Search in Real-World Web Environments">

</div>

**OpenDS-World** is a benchmark for deep search on the open Web, with 200 tasks across eight domains. Agents retrieve, access, and integrate multimodal evidence in real-world Web environments to answer factual questions.

<a id="benchmark-results"></a>

## News

- **[2026.10]** 🔥 Our paper **[Deep Search in Real-World Web Environments](https://arxiv.org/abs/2610.30181)** is now available on arXiv! 🎉

## 📊 Benchmark Results

> [!NOTE]
> Across 12 models evaluated on the same 200 tasks with a shared tool environment and search budget, the highest accuracy is **39.0%** (GPT-6 Astra).

<img src="assets/paper-results.png" width="100%" alt="Paper results: model accuracy ranges from 12.0% to 39.0%; the independent human study reports 21.1% per attempt">

<p align="center"><sub>Figure 1 from the paper. Model accuracy is measured over 200 tasks; human accuracy is measured over 1,000 independent attempts.</sub></p>

The human study includes 50 participants, with five independent attempts per task. Participants achieve **21.1% accuracy per attempt**; **42.5% of tasks were solved by none of their five assigned participants**. Humans use conventional browsers without LLM assistance and have no fixed per-question time limit. Model search is limited to one hour and 100 tool calls per task.

<a id="dataset"></a>

## 🤗 Dataset

Download the dataset from **[🤗 Yu2020/OpenDS-World](https://huggingface.co/datasets/Yu2020/OpenDS-World)**:

```bash
pip install huggingface_hub
hf download Yu2020/OpenDS-World --repo-type dataset --local-dir data/opends-world
```

Evaluation uses a **200-row question-and-answer CSV and the corresponding videos**. See [data preparation](docs/harness.md#benchmark-data) for the expected format and paths.

<a id="quick-start"></a>

## 🚀 Quick start

### 1. Installation

Requires Python 3.10–3.13.

```bash
git clone https://github.com/Yu-xm/OpenDS.git
cd OpenDS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 2. Running

We support evaluation with either a custom model and harness or the provided evaluation harness.

**Use your own harness (Recommended)**

Run your agent on the [benchmark tasks](#dataset) and export its answers to `predictions.csv`:

| Fields | Content |
| --- | --- |
| `id`, `question` | Original task ID and question |
| `gold_answer` | Reference answer matched by ID |
| `model_answer` | Final agent answer |
| `status` | Overall task execution status |

Export one row per task (200 rows total).

**Use the default harness**

Configure the model, tools, and dataset using the [harness guide](docs/harness.md), then run:

```bash
opends-world --config evaluation.yaml
```

Outputs are saved to `results/<benchmark>/<model>/<run_id>/`: `predictions.csv` contains answers and `trajectories/` contains per-task traces.

### 3. Evaluate answers

Evaluate `predictions.csv` from either harness. Set the judge API key in `MODEL_API_KEY`, then run from the repository root:

```bash
python ops/judge_predictions.py \
  --predictions /path/to/predictions.csv \
  --output-dir /path/to/judgments \
  --model-id your-judge-model \
  --base-url https://your-endpoint/v1
```

`summary.json` contains accuracy, the number of correct answers, and judge errors. Per-task judgments are saved in `judgments.csv` and `judgments.jsonl`.

See the [evaluation guide](docs/evaluation.md) for input format, scoring details, and resuming evaluation.

## 📖 Citation & License

If you find this work useful, please cite

The code and included examples are licensed under Apache-2.0.

## 📬 Contact

For questions or feedback, please contact us at `yuxm02@gmail.com`.

## 💬 Discussion Group

Join our WeChat discussion group for discussions and updates.

<p align="center">
  <img src="assets/discussion-group.png" width="320" alt="OpenDS WeChat discussion group QR code, valid until October 27">
</p>
