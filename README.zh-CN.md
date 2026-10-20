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

例如，一道任务从博物馆视频出发，依次涉及虚拟导览、化石展品、播客、研究论文和官方数据集。每个来源都提供下一步所需的线索，最终共同指向答案。

<a href="assets/paper-task-example.png"><img src="assets/paper-task-example.png" width="100%" alt="论文图 2：从博物馆视频到官方数据集标识符的多步骤证据链"></a>

<p align="center"><sub>跨越视频、交互环境、文档与数据的研究路径。</sub></p>

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

结果摘自 *Deep Search in Real-World Web Environments*（2026 年 10 月 20 日版）表 1。评估使用 Qwen3.8-27B 判断模型答案与参考答案是否语义等价。人工实验见论文第 2.3 节。

</details>

论文还分析了搜索成本、工具使用、失败恢复和预算耗尽等问题。更多的工具调用和更长的运行时间并不总能带来更高的准确率：工具协同与长程研究过程的管理仍是关键挑战。

<a id="数据集"></a>

## 🤗 数据集

基准数据发布于 **[🤗 Yu2020/OpenDS-World](https://huggingface.co/datasets/Yu2020/OpenDS-World)**。使用以下命令下载已发布的文件：

```bash
pip install huggingface_hub
hf download Yu2020/OpenDS-World --repo-type dataset --local-dir data/opends-world
```

评估需要相互匹配的 **200 道题目及答案 CSV 和视频文件**。文件格式与路径配置见 [数据准备（英文）](docs/evaluation.md#benchmark-data)。

<a id="快速开始"></a>

## 🚀 快速开始

### 使用自己的 Harness

下载[数据集](#数据集)，在自己的智能体环境中运行 200 道任务，分享最终统计信息即可：评估题数、答对题数、准确率和 Judge 错误数，并注明模型、Judge 和搜索预算。无需安装本仓库的工具环境。为便于比较，请保持判分口径一致；我们的 LLM-as-a-Judge 流程见[判分指南（英文）](docs/evaluation.md#answer-scoring)。

### 使用默认智能体

安装参考评估框架（Python 3.10–3.13）：

```bash
git clone https://github.com/Yu-xm/OpenDS.git
cd OpenDS
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

按照[环境与数据配置（英文）](docs/evaluation.md#run-a-real-case)完成准备后，**分别运行答题与判分**：

```bash
opends-world --config evaluation.yaml

python ops/judge_predictions.py \
  --predictions /path/to/predictions.csv \
  --output-dir /path/to/judgments \
  --model-id your-judge-model \
  --base-url https://your-endpoint/v1
```

> [!IMPORTANT]
> Judge 从 `MODEL_API_KEY` 读取密钥，并将准确率写入 `summary.json`。智能体的 `success` 状态表示执行完成，是否答对由独立运行的 Judge 判定。输出说明与错误统计口径见[判分指南（英文）](docs/evaluation.md#answer-scoring)。

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
