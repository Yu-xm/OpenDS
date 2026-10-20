<div align="center">

# OpenDS-World

[English](README.md) · [简体中文](README.zh-CN.md)

### Deep Search in Real-World Web Environments

### 真实 Web 环境中的深度搜索

**从一段视频线索出发，在开放 Web 中寻找可核实的答案。**

<p>
  <a href="https://huggingface.co/datasets/Yu2020/OpenDS-World"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-OpenDS--World-FFD21E?style=flat-square" alt="Hugging Face dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2563EB?style=flat-square" alt="License: Apache-2.0"></a>
  <a href="https://github.com/Yu-xm/OpenDS"><img src="https://img.shields.io/github/stars/Yu-xm/OpenDS?style=flat-square&logo=github&label=Stars" alt="GitHub stars"></a>
</p>

<p>
  <a href="#基准测试结果">📊 基准测试结果</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/Yu2020/OpenDS-World">🤗 数据集</a> &nbsp;·&nbsp;
  <a href="docs/evaluation.md">🛠 评估指南（英文）</a>
</p>

<img src="assets/OpenDS.png" width="880" alt="OpenDS：真实 Web 环境中的深度搜索">

</div>

**OpenDS-World** 研究智能体能否在真实 Web 环境中完成多步骤研究任务。给定一个问题和一段初始视频，智能体需要自行发现相关来源、访问其内容，并串联证据，确定唯一的事实性答案。任务不会预先提供证据的位置或通往答案的路径。

| **200** 道挑战性任务 | **8** 个领域 | **12** 个前沿模型 | **1,000** 次人工作答 |
| :---: | :---: | :---: | :---: |
| 证据与答案经过独立核验 | 覆盖文化、历史、科学、技术等领域 | 采用统一评估协议 | 每题由五位参与者独立作答 |

开放 Web 深度搜索（Open-Web Deep Search，**OpenDS**）将三类挑战贯穿于同一个研究任务：

| 多模态线索 | 多种内容格式 | 交互式访问 |
| --- | --- | --- |
| 关联文本、图像、音频和视频中的线索。 | 在网页、文档、媒体和结构化数据集中追踪证据。 | 操作界面、观察视觉场景、调整筛选条件并获取资源。 |

<a id="基准测试结果"></a>

## 📊 基准测试结果

> [!NOTE]
> **表现最好的模型也只能解决不到 40% 的任务。** GPT-6 Astra 的准确率为 **39.0%**，其次是 Claude Opus 5.5（**38.0%**）和 Gemini 3.8 Flash（**36.5%**）。12 个模型均在相同的 200 道任务上，使用统一的工具环境和搜索预算进行评估。

<img src="assets/paper-results.png" width="100%" alt="论文结果：模型准确率为 12.0%–39.0%；独立人工实验的单次作答准确率为 21.1%">

<p align="center"><sub>结果取自论文图 1。模型准确率按 200 道任务统计，人工准确率按 1,000 次独立作答统计。</sub></p>

人工实验共有 50 位参与者，每道任务由五人独立作答。**单次作答准确率为 21.1%**；**42.5% 的任务没有被分配到的任何一位参与者解出**。参与者使用常规浏览器，不使用大语言模型辅助，每题不设固定时间限制。模型每题的搜索预算为一小时、100 次工具调用。

<details>
<summary><strong>完整模型结果 · 论文表 1</strong></summary>

| 模型 | 准确率 | 答对题数 / 200 |
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
| GLM-5.3（纯文本） | 13.5% | 27 |
| DeepSeek-V4-Pro（纯文本） | 12.5% | 25 |
| Qwen3.8-Flash | 12.0% | 24 |

完整结果见论文 *Deep Search in Real-World Web Environments* 表 1。评估使用 Qwen3.8-27B 判断模型答案与参考答案是否语义等价。人工实验见论文第 2.3 节。

</details>

论文还分析了搜索成本、工具使用、失败恢复和预算耗尽等问题。更多的工具调用和更长的运行时间并不总能带来更高的准确率：工具协同与长程研究过程的管理仍是关键挑战。

<a id="数据集"></a>

## 🤗 数据集

基准数据发布于 **[🤗 Yu2020/OpenDS-World](https://huggingface.co/datasets/Yu2020/OpenDS-World)**。使用以下命令下载数据集：

```bash
pip install huggingface_hub
hf download Yu2020/OpenDS-World --repo-type dataset --local-dir data/opends-world
```

评估需要相互匹配的 **200 道题目及答案 CSV 和视频文件**。文件格式与路径配置见 [数据准备（英文）](docs/harness.md#benchmark-data)。

<a id="快速开始"></a>

## 🚀 快速开始

### 1. 运行智能体

**使用自己的 Harness**

在自己的智能体环境中运行[数据集](#数据集)，然后导出 `predictions.csv`，接入统一判分：

| 字段 | 内容 |
| --- | --- |
| `id`、`question` | 原始题目 ID 与问题 |
| `gold_answer` | 按 ID 匹配的标准答案，仅供判分使用 |
| `model_answer` | 智能体最终答案 |
| `status` | 执行完成填 `success`；失败时保留失败状态 |

保留全部 200 道题，包括失败和未作答的题目；无需部署我们的浏览器与视频工具环境。导出后进入下方评估步骤。

**使用默认 Harness**

按照 [Harness 指南（英文）](docs/harness.md)完成安装、模型与工具配置、数据准备，然后运行：

```bash
opends-world --config evaluation.yaml
```

输出位于 `results/<benchmark>/<model>/<run_id>/`，其中 `predictions.csv` 用于下一步评估，`trajectories/` 保存逐题轨迹。

### 2. 评估答案

两种 Harness 均使用上一步的 `predictions.csv`。安装本仓库的 Python 包后，在仓库根目录运行独立的 LLM-as-a-Judge，并通过环境变量 `MODEL_API_KEY` 配置 Judge 密钥：

```bash
python ops/judge_predictions.py \
  --predictions /path/to/predictions.csv \
  --output-dir /path/to/judgments \
  --model-id your-judge-model \
  --base-url https://your-endpoint/v1
```

输出目录中的 `summary.json` 包含答对题数、准确率和 `judge_errors`，逐题判分保存为 `judgments.csv` 与 `judgments.jsonl`。Judge 调用失败的题目仍计入准确率分母。分享结果时请同时注明模型、Judge 和搜索预算。

Judge 安装、完整 CSV 格式、统计口径与断点恢复见[评估指南（英文）](docs/evaluation.md)。

> [!IMPORTANT]
> 答题与判分分开运行。`success` 表示执行完成，答案是否正确由 Judge 判定。

## 📖 引用与许可

如果本研究对你的工作有所帮助，欢迎引用：

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

本仓库原创代码与随附示例采用 [Apache-2.0](LICENSE) 许可。视频文件在 Hugging Face 单独分发；第三方视频不在此代码许可的授权范围内。

---

<p align="center">
  <a href="LICENSE">Apache-2.0 · 代码与示例</a> &nbsp;·&nbsp;
  <a href="mailto:yuxm02@gmail.com">学术联系</a>
</p>
