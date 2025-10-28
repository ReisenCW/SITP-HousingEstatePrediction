# 房价走势预测（基于大语言模型 + 进化/反思）

## 项目简介
一个轻量级实验性项目，使用大语言模型结合“进化”思路：先对窗口期（前半期）进行预测并与实际结果比对、评分、生成反思（reflection），在反思的帮助下迭代改进，最终给出对目标时间段的预测。

## 核心目标
- 将 LLM 的推理与外部网络信息（新闻/政策）结合成可审计的思维链（COT），并用小型评估/反思闭环提高短期预测稳定性。
- 支持可重复的本地运行流程：搜索 -> 预测 -> 获取实际 -> 评分 -> 反思 -> （重试） -> 最终预测。

## 环境构建

```shell
# 创建并激活 venv
python -m venv .venv
.\.venv\Scripts\activate.ps1
# 安装依赖(requirements中存在一些不必要的依赖, 为先前尝试不同方法时残留的依赖)
pip install -r requirements.txt
```

## 主要模块与文件
- `api.py` — 提供完整工作流接口（搜索 -> 预测 -> 获取实际 -> 评分 -> 反思，支持演化循环与后续时间段预测）。
- `agent.py` — `HousePriceAgent`：封装与 LLM 的交互、COT、轨迹与持久化反思。 
- `evaluator.py` — 用于解析 LLM 输出的趋势并计算预测分数。
- `prompts.py` — 所有 LLM prompt 模板。
- `config.py` — 配置与阈值（API KEY、模型名、是否启用演化等）。
- `reflection_history.json`, `cot_trajectory.json`, `answer/` — 持久化反思、COT 轨迹与最终答案目录。


## 配置说明
- 将 DashScope API key 放入环境变量 `DASHSCOPE_API_KEY`，或在 `config.py` 中设置。
- 主要可调项（位于 `config.py`）：
	- `ENABLE_EVOLUTION`：是否启用演化（获取实际结果并做反思/重试）。
	- `MAX_RETRIES`, `MAX_ITERATIONS`：重试和搜索轮数上限。
	- `SCORE_THRESHOLD`：评分阈值，低于该值触发反思并重试。

## 使用方法
调用`api.py` 中提供的 `predict_region`, 例如:
```python
# test
region = "徐汇区滨江"
time_range = "2025年上半年"
follow_up_time_range = "2025年下半年"

first_pred, follow_up_pred = predict_region(
    region,
    time_range,
    follow_up_time_range,
    debug=True
)
print(f"First Prediction: {first_pred}")
print(f"Follow-up Prediction: {follow_up_pred}")
```
