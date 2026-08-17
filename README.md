# OmniBrowse Bench

一个尽量小的 Controlled / Live Search benchmark runner。它把两种数据集、不同模型和每次实验统一到同一条执行链：

```text
CSV + 可选视频/音频
        ↓
统一模型接口（不启用厂商内置搜索）
        ↓
Serper: text_search / image_search / open_page
        ↓
predictions.csv + 每题 observable trajectory.jsonl + 原始 artifacts
```

主榜只适用于能关闭 native search、接受本框架三个函数工具的模型。只能使用厂商私有搜索的产品应放入单独的 Native/Uncontrolled 对照，不应和 Controlled Serper 结果混排。

## 1. 环境

只需要 Python 3.10+，运行时没有第三方依赖：

```bash
cd omnibrowse-bench
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

也可以不安装，直接设置源码路径运行：

```bash
PYTHONPATH=src python3 -m omnibrowse_bench.cli --config configs/mock-text.json
```

## 2. 输入 CSV

`OmniBrowse-Text` 至少包含：

```csv
id,question,gold_answer
text_001,Who published the report?,Example Institute
```

`OmniBrowse` 至少包含：

```csv
id,question,video_path,audio_path,gold_answer
omni_001,Who published the report?,media/001.mp4,media/001.wav,Example Institute
```

媒体路径相对于 CSV 所在目录解析。`id` 必须唯一。框架不会修改源 CSV。

## 3. 配置模型

复制一个配置文件并填写：

- `configs/openai-compatible-text.json`：OpenAI-compatible `/chat/completions`，适用于很多自部署服务和兼容 API。
- `configs/bridge-omni.json`：统一 HTTP bridge，适用于官方 Claude、Gemini 等非兼容 API，也适用于你自己的推理服务。
- `configs/mock-text.json`：不联网，只检查整条保存流程。

设置密钥：

```bash
export SERPER_API_KEY="..."
export MODEL_API_KEY="..."
```

配置中的 `${ENV_NAME}` 会在内存中展开；写入 `run_config.json` 前，key/token/secret/password 字段会被替换为 `[REDACTED]`。
配置中的数据集和输出路径相对于该 JSON 配置文件所在目录解析，因此可以从任意工作目录启动命令。

### OpenAI-compatible 注意事项

该适配器会发送标准 function calling。文本和图片兼容性通常较好；音频被编码为 `input_audio`，视频被编码为 `video_url` data URL。不同兼容服务对音视频字段支持并不一致。如果某个官方 API 的格式不同，请使用 bridge，而不要修改 benchmark 核心循环。

对于持续数分钟的推理模型，可在 `model.parameters` 中设置 `"stream": true`。runner 会解析 OpenAI-compatible SSE，持续接收 `content`、`reasoning_content` 和分片 function call，从而避免非流式请求长时间无数据时被服务商网关断开。流式响应会组装为统一的模型响应 artifact。

模型请求遇到 HTTP 429、部分 5xx 或临时连接错误时默认最多重试 2 次。429 若没有服务端 `Retry-After`，每次最多等待 30 秒；可通过模型配置中的 `max_retries`、`retry_base_seconds` 和 `retry_max_seconds` 调整。单次 Serper 工具失败不会在 HTTP 层自动重试，而是作为可观察的失败工具结果交还模型，避免不透明地增加搜索调用数。

### Bridge 请求协议

框架向 `model.endpoint` POST：

```json
{
  "model": "provider-model-id",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "question"},
        {"type": "video", "mime_type": "video/mp4", "data_base64": "...", "filename": "x.mp4"},
        {"type": "audio", "mime_type": "audio/wav", "data_base64": "...", "filename": "x.wav"}
      ]
    }
  ],
  "tools": [],
  "tool_choice": "auto",
  "parameters": {"temperature": 0}
}
```

bridge 必须统一返回以下格式之一。

请求工具：

```json
{
  "text": "",
  "tool_calls": [
    {"id": "call_1", "name": "text_search", "arguments": {"query": "..."}}
  ],
  "usage": {"input_tokens": 100, "output_tokens": 20},
  "finish_reason": "tool_calls"
}
```

最终回答：

```json
{
  "text": "final answer",
  "reasoning_content": "optional provider-visible reasoning",
  "tool_calls": [],
  "usage": {"input_tokens": 500, "output_tokens": 50},
  "finish_reason": "stop"
}
```

bridge 的职责仅是格式转换：把同样的媒体、消息和 function tools 传给模型，再把模型响应转换回来。不要在 bridge 内增加搜索、转录、OCR 或额外模型，否则不同模型获得的能力不再一致。

## 4. 运行

先验证保存流程：

```bash
PYTHONPATH=src python3 -m omnibrowse_bench.cli \
  --config configs/mock-text.json
```

正式运行：

```bash
PYTHONPATH=src python3 -m omnibrowse_bench.cli \
  --config configs/my-run.json
```

只跑前两题：

```bash
PYTHONPATH=src python3 -m omnibrowse_bench.cli \
  --config configs/my-run.json --limit 2
```

断点续跑需要配置固定的 `run_id`，然后加入 `--resume`。已有 trace 且已有 prediction 的题会被跳过：

```json
{"run_id": "paper_run_01"}
```

```bash
PYTHONPATH=src python3 -m omnibrowse_bench.cli \
  --config configs/my-run.json --resume
```

## 5. 输出

```text
results/
└── OmniBrowse/
    └── model-name/
        └── run-id/
            ├── predictions.csv
            ├── run_config.json
            ├── trajectories/
            │   └── sample-id.jsonl
            └── artifacts/
                └── sample-id/
                    ├── model_response_001.json
                    ├── text_search_call-id.json
                    ├── open_page_call-id.json
                    └── image_call-id_01.jpg
```

`OmniBrowse-Text` 自动写入单独的顶层目录。每个 JSONL 事件都冗余写入 `run_id`、`benchmark`、`model` 和 `sample_id`。

轨迹事件为：

- `run_start`：问题及媒体路径
- `model_response`：API 可见文本、工具调用、token、延迟
- `empty_answer_retry`：模型只返回思考内容或空正文时触发的一次统一最终回答请求
- `tool_start`：工具参数
- `tool_result`：实际传给模型的标准化观察、原始 artifact 路径与哈希
- `final_answer`：最终答案
- `run_end`：状态、总用时、调用次数与 token

这是 **observable agent trajectory**。如果兼容 API 明确返回 `reasoning_content`，框架会将其保存并在多轮调用中原样传回；它不会把思考内容当作最终答案。框架不会记录 API key，也不会尝试访问服务端未返回的隐藏状态。原始模型响应另存为 artifact，并在事件中记录路径和哈希。

推理模型如果以 `stop` 或 `length` 结束但没有返回正文，runner 默认会追加一次不带工具的统一最终回答请求。可用 `agent.max_empty_answer_retries` 调整；正式比较时应对所有模型固定为相同值。

### 防止搜索循环和 TPM 膨胀

Prompt 会要求模型避免重复搜索，但 runner 还提供确定性的框架级保护，默认行为如下：

- 完全相同的工具参数只实际执行 1 次；之后返回 `Tool loop prevented`。
- 同一种搜索已有 3 个高度相似查询后，继续提交相似查询会被拦截。
- 连续进行 8 次搜索却没有 `open_page` 后，新的搜索会被拦截，并提示模型打开已有结果或作答。
- 连续 2 个步骤只有被拦截的调用时，下一轮关闭工具并要求基于已有证据给出答案。
- 只保留最近 4 个完整工具观察；更早的搜索结果确定性缩减到前 2 条、页面正文缩减到 2000 字符，并移除较早的 provider reasoning。完整观察仍保存在 JSONL 和 artifact 中。

对应配置项为 `max_identical_tool_calls`、`max_similar_searches`、`similar_search_threshold`、`max_searches_without_open`、`max_blocked_tool_steps`、`keep_recent_tool_observations`、`history_search_results`、`history_page_chars` 和 `keep_recent_reasoning_messages`。这些参数会写入 `run_config.json`；正式对比时必须对所有模型保持一致。

`predictions.csv` 会分别写入 `tool_calls`（实际执行总数）、`text_search_calls`、`image_search_calls`、`open_page_calls` 和 `blocked_tool_calls`。模型发起但被循环保护或预算限制拦截的调用只计入 `blocked_tool_calls`，不计入实际 Serper 调用数。轨迹的 `run_end.metrics` 还会记录 `tool_requests` 和 `tool_errors`。

### 超时前的统一作答

示例配置把每道题的总运行时间设为 360 秒，单次模型请求设为 300 秒。剩余 60 秒、进入最后一个 agent step、普通研究轮的模型请求失败、请求的工具预算已耗尽、出现未知工具调用或触发循环保护时，runner 会关闭工具并要求模型基于已收集证据作答。`predictions.csv` 的 `forced_final_reason` 和轨迹中的 `forced_final` 事件会记录触发原因。

这是一种提前收尾机制，不可能挽救已经彻底不可用的模型 API。如果最终回答请求本身在重试后仍然超时或被限流，结果仍会是 `api_error`；此时轨迹和 artifact 会保留已有研究内容，但框架不会把 provider reasoning 冒充为答案。

## 6. 公平性约束

正式比较时固定以下配置，并把生成的 `run_config.json` 和结果一起保存：

- 同一个 Serper key/account 与 Search/Images/Scrape 后端
- 相同 `gl`、`hl`、结果数和正文截断长度
- 相同三种工具定义与调用预算
- 相同最大 agent steps、温度和输出上限
- OmniBrowse 输入完全相同的视频、音频和问题
- OmniBrowse-Text 只移除初始音视频，其他条件不变
- 不支持必需输入模态的模型标记 `unsupported_input` 或 N/A，不私自追加转录/OCR

`run_config.json` 会额外记录数据集绝对路径与 SHA-256、系统提示哈希、Python 版本、框架版本、Git commit（若可用）和起止时间，便于之后核对实验。

## 7. 状态字段

当前 runner 会产生：

- `success`
- `empty_answer`
- `max_steps`
- `api_error`
- `timeout`
- `unsupported_input`

单次搜索或抓取失败会作为失败的 `tool_result` 返回模型，模型仍可改写查询继续研究；如果模型最终作答，整题仍可成功，同时轨迹完整保留该工具错误。
