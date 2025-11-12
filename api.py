"""
轻量级异步接口，用于区域级别的预测工作流。

用法示例：
        from api import predict_region
        # 同步包装器
        pred = predict_region(region="徐汇区滨江", time_range="2025年上半年", follow_up_time_range="2025年下半年")

或直接使用异步函数：
        from api import predict_region_async
        await predict_region_async(...)

流程说明：
    - 搜索相关信息 -> 预测前半期
    - 若启用进化（ENABLE_EVOLUTION）：获取实际趋势、计算评分、生成反思并重试直到评分达标或达到最大重试次数
    - 成功后，可选择利用持久记忆/反思预测后续时间段

此模块不修改已有智能体行为，仅编排对 `HousePriceAgent` 的方法调用。
"""
import asyncio
from typing import Optional, Tuple

from config import Config
from agent import HousePriceAgent
from evaluator import Evaluator


async def predict_region_async(region: str,
                               time_range: str,
                               follow_up_time_range: Optional[str] = None,
                               max_retries: Optional[int] = None,
                               debug: bool = False) -> Tuple[str, Optional[str]]:
    """以异步方式运行所需的工作流。

    返回一个元组 (first_prediction, follow_up_prediction_or_None)。
    """
    config = Config()
    
    if max_retries is None:
        max_retries = config.MAX_RETRIES

    agent = HousePriceAgent(config)

    # 确保 agent 带有区域/时间信息，便于文件命名和追踪
    agent.region = region
    agent.time_range = time_range
    # 若没有通过 parse_query 初始化 current_cot，则在此创建一个占位对象，
    # 以便后续的 predict_trend / record_trajectory 等方法可以安全写入。
    if getattr(agent, 'current_cot', None) is None:
        agent.current_cot = {
            "query": f"{region} {time_range}",
            "predict_trend": None,
            "steps": [],
            "accurate": "unknown"
        }

    retries = 0
    first_prediction = None
    # 循环流程：搜索 -> 预测 -> （若启用）获取实际 -> 评估 -> 反思 -> 可能重试
    while True:
        info = await agent.search_related_info(region, time_range)
        pred = await agent.predict_trend(region, time_range, info)
        first_prediction = pred

        if debug:
            print(f"[api] First prediction for {region} {time_range}: {pred}")

        if not config.ENABLE_EVOLUTION:
            # 未启用进化：保存结果并返回
            agent.save_answer(f"{region} {time_range}", pred, "未获取实际趋势", -1)
            break

        # 获取实际趋势并计算评分
        actual = await agent.get_actual_trend(region, time_range)
        score = Evaluator.score(pred, actual)

        if debug:
            print(f"[api] Actual: {actual}\n[api] Score: {score}")

        # 生成反思（此操作也会将反思持久化记录）
        _ = await agent.generate_reflection(f"{region} {time_range}", pred, actual, score, info)

        if score >= config.SCORE_THRESHOLD:
            agent.save_answer(f"{region} {time_range}", pred, actual, score)
            break

        retries += 1
        if retries > max_retries:
            # 达到最大重试次数，放弃并保存当前最佳结果
            agent.save_answer(f"{region} {time_range}", pred, actual, score)
            break

        if debug:
            print(f"[api] Retry {retries}/{max_retries} for {region} {time_range}")

        # 否则继续循环重试（agent.generate_reflection 已更新持久记忆）

    # 前半段预测结束，进行后半段预测(真正要预测的时间段)
    follow_up_prediction = None
    if follow_up_time_range:
        # 预测后续时间段时，利用反思/持久记忆进行辅助
        info2 = await agent.search_related_info(region, follow_up_time_range)
        follow_up_prediction = await agent.predict_trend(region, follow_up_time_range, info2)
        agent.save_answer(f"{region} {follow_up_time_range}", follow_up_prediction, "未获取实际趋势", -1)
        if debug:
            print(f"[api] Follow-up prediction for {region} {follow_up_time_range}: {follow_up_prediction}")

    return first_prediction, follow_up_prediction


def predict_region(region: str,
                   time_range: str,
                   follow_up_time_range: Optional[str] = None,
                   max_retries: Optional[int] = None,
                   debug: bool = False) -> Tuple[str, Optional[str]]:
    """predict_region_async 的同步包装器。"""
    return asyncio.run(predict_region_async(region, time_range, follow_up_time_range, max_retries, debug))

if __name__ == "__main__":
    # 简单测试
    region = "杨浦区五角场"
    time_range = "2025年上半年"
    follow_up_time_range = "2025年下半年"

    first_pred, follow_up_pred = predict_region(region, time_range, follow_up_time_range, debug=True)
    print(f"First Prediction: {first_pred}")
    print(f"Follow-up Prediction: {follow_up_pred}")