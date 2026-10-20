<div align="center">

# OpenDS-World

[English](README.md) · [简体中文](README.zh-CN.md)

### Deep Search in Real-World Web Environments

**From a video clue to a verifiable answer across the open Web.**

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

**OpenDS-World** studies whether agents can carry out multi-step research in real-world Web environments. Given a question and an initial video, an agent must discover relevant sources, access their content, and connect evidence to determine a unique factual answer. The locations of the evidence and the path to the answer are not supplied in advance.

| **200** challenging tasks | **8** domains | **12** frontier models | **1,000** human attempts |
| :---: | :---: | :---: | :---: |
| Independently verified evidence and answers | From culture and history to science and technology | Evaluated under a shared protocol | Five independent solvers per task |

Open-Web Deep Search (**OpenDS**) brings together three challenges that arise throughout a research task:

| Diverse modalities | Heterogeneous formats | Interactive access |
| --- | --- | --- |
| Connect clues across text, images, audio, and video. | Follow evidence through webpages, documents, media, and structured datasets. | Navigate interfaces, inspect visual scenes, adjust filters, and retrieve resources. |

<a id="benchmark-results"></a>

## 📊 Benchmark Results

> [!NOTE]
> **Even the best-performing model solves fewer than 40% of the tasks.** GPT-6 Astra achieves **39.0%**, followed by Claude Opus 5.5 at **38.0%** and Gemini 3.8 Flash at **36.5%**. All 12 models are evaluated on the same 200 tasks with a shared tool environment and search budget.

<img src="assets/paper-results.png" width="100%" alt="Paper results: model accuracy ranges from 12.0% to 39.0%; the independent human study reports 21.1% per attempt">

<p align="center"><sub>Figure 1 from the paper. Model accuracy is measured over 200 tasks; human accuracy is measured over 1,000 independent attempts.</sub></p>

The human study includes 50 participants, with five independent attempts per task. Participants achieve **21.1% accuracy per attempt**; **42.5% of tasks remain unsolved by all five assigned solvers**. Humans use conventional browsers without LLM assistance and have no fixed per-question time limit. Model search is limited to one hour and 100 tool calls per task.

<details>
<summary><strong>Full model results · Table 1</strong></summary>

| Model | Accuracy | Correct / 200 |
| --- | ---: | ---: |
| **GPT-6 Astra** | **39.0%** | **78** |
| Claude Opus 5.5 | 38.0% | 76 |
| Gemini 3.8 Flash | 36.5% | 73 |
| Claude Fable 5.1 | 33.5% | 67 |
| GPT-5.6 Sol | 28.0% | 56 |
| Qwen3.8-Max | 25.5% | 51 |
| GLM-5.3-Flash | 23.5% | 47 |
| DeepSeek-V4.1-Flash | 20.5% | 41 |
| Kimi K3 | 17.5% | 35 |
| GLM-5.3 (text-only) | 13.5% | 27 |
| DeepSeek-V4-Pro (text-only) | 12.5% | 25 |
| Qwen3.8-Flash | 12.0% | 24 |

Results are reported in Table 1 of *Deep Search in Real-World Web Environments*. Answers are judged for semantic equivalence to the reference using Qwen3.8-27B. The human study is described in Section 2.3.

</details>

The paper also analyzes search cost, tool use, failure recovery, and budget exhaustion. More tool calls and longer runtimes do not consistently yield higher accuracy: coordinating tools and managing long research trajectories remain central challenges.

<a id="dataset"></a>

## 🤗 Dataset

**[🤗 Yu2020/OpenDS-World](https://huggingface.co/datasets/Yu2020/OpenDS-World)** hosts the benchmark release. Download the dataset with:

```bash
pip install huggingface_hub
hf download Yu2020/OpenDS-World --repo-type dataset --local-dir data/opends-world
```

Evaluation requires the matching **200-case question/answer CSV and videos**. See [data preparation](docs/harness.md#benchmark-data) for the expected format and paths.

<a id="quick-start"></a>

## 🚀 Quick start

### 1. Run your agent

**Use your own harness**

Run the [dataset](#dataset) in your own agent environment, then export `predictions.csv` for shared scoring:

| Fields | Content |
| --- | --- |
| `id`, `question` | Original task ID and question |
| `gold_answer` | Reference answer matched by ID, used only for scoring |
| `model_answer` | Final agent answer |
| `status` | `success` for completed execution; retain failure statuses otherwise |

Keep all 200 tasks, including failed and unanswered tasks; our browser and video tool environment is not required. Continue to the evaluation step below after exporting predictions.

**Use the default harness**

Complete installation, model/tool configuration, and data preparation in the [harness guide](docs/harness.md), then run:

```bash
opends-world --config evaluation.yaml
```

Outputs are saved to `results/<benchmark>/<model>/<run_id>/`: `predictions.csv` feeds the next evaluation step, while `trajectories/` contains per-task traces.

### 2. Evaluate answers

Both harness paths use the `predictions.csv` produced above. After installing this repository's Python package, run the standalone LLM-as-a-Judge from the repository root, with the judge credential set through the `MODEL_API_KEY` environment variable:

```bash
python ops/judge_predictions.py \
  --predictions /path/to/predictions.csv \
  --output-dir /path/to/judgments \
  --model-id your-judge-model \
  --base-url https://your-endpoint/v1
```

The output directory contains `summary.json` with correct counts, accuracy, and `judge_errors`, plus per-task judgments in `judgments.csv` and `judgments.jsonl`. Failed judge calls remain in the accuracy denominator. Report the model, judge, and search budget alongside your results.

See the [evaluation guide](docs/evaluation.md) for judge installation, the full CSV format, metric definitions, and resuming interrupted scoring.

> [!IMPORTANT]
> Answer generation and scoring run separately. `success` indicates execution completion; the judge determines answer correctness.

## 📖 Citation & License

If you find this work useful, please cite:

```bibtex
@misc{yu2026opendsworld,
  title  = {Deep Search in Real-World Web Environments},
  author = {Xiaomin Yu and Hongliang Chen and Zhe Huang and Shilin Yan and
            Shao-Rong Su and Xuqiao Ma and Feiyang Wang and Jiayu Chen and
            Jiwei Fang and Yu Yan and Yini Huang and Chen Chen and
            Ruoxi Cheng and Yuzhi He and Yuhui Zhang and Xiaoxing Hu and
            Chen Liu and Jianbo Lin and Yijiang Li and Xiaobin Hu and
            Menglong Li and Zhuoran Lu and Weishi Wang and Zhongqi Yue and
            Chengwei Qin},
  year   = {2026}
}
```

Repository-authored code and included examples are released under [Apache-2.0](LICENSE). Videos are distributed separately on Hugging Face; third-party videos are outside this code license grant.

---

<p align="center">
  <a href="LICENSE">Apache-2.0 · code and examples</a> &nbsp;·&nbsp;
  <a href="mailto:yuxm02@gmail.com">Research contact</a>
</p>
